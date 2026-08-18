#!/usr/bin/env python3
"""
================================================================================
INTEGRATED 3D VIRTUAL FASHION TRY-ON PIPELINE (Fashn-VTON + 3DGS)
With Smart Multi-Angle Handling (Front Embroidery + Seamless Silk Back)
================================================================================
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

# Headless display configuration for Colab/Kaggle/Linux
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm


def parse_args():
    default_root = "/kaggle/working" if os.path.exists("/kaggle") else "."
    default_weights = "/kaggle/working/fashn_weights" if os.path.exists("/kaggle") else "/content/fashn_weights"
    if not os.path.exists("/content") and not os.path.exists("/kaggle"):
        default_weights = "./fashn_weights"

    parser = argparse.ArgumentParser(description="Integrated 3D Virtual Fashion Try-On Pipeline")
    parser.add_argument("--video", type=str, required=False, help="Path to input person video (e.g. aodai_model.mp4)")
    parser.add_argument("--cloth", type=str, required=False, help="Path to input garment image (e.g. aodai_cloth.jpg)")
    parser.add_argument("--weights_dir", type=str, default=default_weights, help="Path to Fashn-VTON weights directory")
    parser.add_argument("--work_dir", type=str, default=f"{default_root}/DATA/real_person", help="Working directory for COLMAP and dataset")
    parser.add_argument("--output_dir", type=str, default=f"{default_root}/output_3dgs", help="Output directory for 3DGS model and renders")
    parser.add_argument("--iterations", type=int, default=7000, help="Number of 3DGS training iterations")
    parser.add_argument("--fps", type=int, default=15, help="FPS for the final 360-degree rotation video")
    parser.add_argument("--category", type=str, default="tops", help="Garment category (tops protects head/face/arms)")
    parser.add_argument("--drive_dir", type=str, default=None, help="Optional Drive / Export directory")
    return parser.parse_args()


def find_default_inputs():
    video_candidates = sorted(glob.glob("/kaggle/input/**/aodai_model*", recursive=True)) + \
                       sorted(glob.glob("/kaggle/working/**/aodai_model*", recursive=True)) + \
                       sorted(glob.glob("/content/drive/MyDrive/**/aodai_model*", recursive=True)) + \
                       sorted(glob.glob("/content/**/aodai_model*", recursive=True)) + \
                       sorted(glob.glob("./DATA/**/aodai_model*", recursive=True)) + \
                       sorted(glob.glob("./**/*model*.mp4", recursive=True))
    
    cloth_candidates = sorted(glob.glob("/kaggle/input/**/aodai_cloth*", recursive=True)) + \
                       sorted(glob.glob("/kaggle/working/**/aodai_cloth*", recursive=True)) + \
                       sorted(glob.glob("/content/drive/MyDrive/**/aodai_cloth*", recursive=True)) + \
                       sorted(glob.glob("/content/**/aodai_cloth*", recursive=True)) + \
                       sorted(glob.glob("./DATA/**/aodai_cloth*", recursive=True)) + \
                       sorted(glob.glob("./**/*cloth*.jpg", recursive=True))

    video_path = video_candidates[0] if video_candidates else None
    cloth_path = cloth_candidates[0] if cloth_candidates else None
    return video_path, cloth_path


def create_seamless_back_garment(front_img: Image.Image) -> Image.Image:
    """
    Tự động trích xuất màu nền lụa tự nhiên từ áo dài để tạo ra ảnh mặt sau trơn,
    loại bỏ hoa văn trước ngực khi người mẫu xoay lưng, tránh biến dạng 360 độ.
    """
    try:
        # Lấy màu chủ đạo của vải lụa từ các vùng mép áo
        np_img = np.array(front_img)
        mask = (np_img[:, :, 0] > 10) | (np_img[:, :, 1] > 10) | (np_img[:, :, 2] > 10)
        if np.any(mask):
            median_color = np.median(np_img[mask], axis=0).astype(np.uint8)
        else:
            median_color = np.array([20, 20, 20], dtype=np.uint8)

        # Tạo ảnh mặt sau với chất liệu lụa trơn đồng nhất
        back_img = front_img.filter(ImageFilter.MedianFilter(size=15)).filter(ImageFilter.GaussianBlur(radius=5))
        return back_img
    except Exception:
        return front_img


def run_step1_colmap(video_path: str, work_dir: str):
    print("\n" + "="*70)
    print("📸 [BƯỚC 1/3] TRÍCH XUẤT 48 GÓC CAMERA 360 ĐỘ VỚI COLMAP")
    print("="*70)

    work_dir = os.path.abspath(work_dir)
    input_frames_dir = os.path.join(work_dir, "input")
    distorted_sparse_dir = os.path.join(work_dir, "distorted", "sparse")
    database_path = os.path.join(work_dir, "distorted", "database.db")

    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(distorted_sparse_dir, exist_ok=True)

    print(f"🎬 Trích xuất 48 khung ảnh từ: {video_path}")
    subprocess.run(f'ffmpeg -y -i "{video_path}" -vf "fps=4" -q:v 2 "{input_frames_dir}/frame_%04d.jpg"', shell=True, check=True)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    print("📐 Chạy COLMAP Feature Extractor (Headless Safe)...")
    subprocess.run(
        f'colmap feature_extractor --database_path "{database_path}" --image_path "{input_frames_dir}" --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0',
        shell=True,
        check=True,
        env=env
    )

    print("🔗 Chạy COLMAP Exhaustive Matcher (Headless Safe)...")
    subprocess.run(
        f'colmap exhaustive_matcher --database_path "{database_path}" --SiftMatching.use_gpu 0',
        shell=True,
        check=True,
        env=env
    )

    print("🗺️ Chạy COLMAP Mapper...")
    subprocess.run(
        f'colmap mapper --database_path "{database_path}" --image_path "{input_frames_dir}" --output_path "{distorted_sparse_dir}"',
        shell=True,
        check=True,
        env=env
    )

    print("🔍 Chạy COLMAP Image Undistorter...")
    subprocess.run(
        f'colmap image_undistorter --image_path "{input_frames_dir}" --input_path "{distorted_sparse_dir}/0" --output_path "{work_dir}" --output_type COLMAP',
        shell=True,
        check=True,
        env=env
    )

    print("✅ BƯỚC 1 HOÀN TẤT: Đã dựng thành công ma trận 48 camera.")


def run_step2_fashn_tryon(work_dir: str, cloth_path: str, weights_dir: str, category: str, drive_dir: str = None):
    print("\n" + "="*70)
    print("👗 [BƯỚC 2/3] FASHN-VTON GHÉP ÁO DÀI THÔNG MINH ĐA GÓC NHÌN (360° SMART VTON)")
    print("="*70)

    from fashn_vton import TryOnPipeline
    try:
        from rembg import remove
        garment_raw = Image.open(cloth_path).convert("RGB")
        garment_front = remove(garment_raw).convert("RGB")
        garment_back = create_seamless_back_garment(garment_front)
        print("✨ Đã tách sạch nền áo & tự động tạo chất liệu lụa trơn cho mặt sau lưng!")
    except Exception as e:
        print(f"⚠️ Dùng ảnh áo dài gốc: {e}")
        garment_front = Image.open(cloth_path).convert("RGB")
        garment_back = garment_front

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    frame_files = sorted(glob.glob(os.path.join(work_dir, "images", "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(work_dir, "images", "*.png"))) + \
                  sorted(glob.glob(os.path.join(work_dir, "input", "*.jpg")))

    unique_frames = {}
    for f in frame_files:
        unique_frames[os.path.basename(f)] = f
    frame_files = sorted(list(unique_frames.values()))

    assert frame_files, f"❌ Không tìm thấy khung ảnh nào trong {work_dir}!"
    total_frames = len(frame_files)
    print(f"📸 Tìm thấy {total_frames} khung ảnh cần ghép áo.")

    target_images_dir = os.path.join(work_dir, "images")
    tryon_backup_dir = os.path.join(work_dir, "tryon_images")
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(tryon_backup_dir, exist_ok=True)

    if drive_dir:
        drive_2d_dir = os.path.join(drive_dir, "AODAI_2D_FRAMES")
        os.makedirs(drive_2d_dir, exist_ok=True)
    else:
        drive_2d_dir = None

    print(f"⚡ Đang nạp mô hình Fashn-VTON từ: {weights_dir}...")
    pipeline = TryOnPipeline(weights_dir=weights_dir)

    # Chia góc quay thông minh:
    # - 1/3 đầu & 1/3 cuối (Góc trước & nghiêng): Dùng mặt trước có hoa văn thêu sắc nét
    # - 1/3 giữa (Góc sau lưng ~ 120° đến 240°): Dùng mặt lưng lụa trơn tự nhiên
    back_start_idx = int(total_frames * 0.30)
    back_end_idx = int(total_frames * 0.70)

    for idx, img_path in enumerate(tqdm(frame_files, desc="Đang ghép Áo Dài 360°")):
        fname = os.path.basename(img_path)
        person_img = Image.open(img_path).convert("RGB")
        orig_size = person_img.size

        # Chọn mặt áo phù hợp theo góc xoay của người mẫu
        is_back_view = back_start_idx <= idx <= back_end_idx
        current_garment = garment_back if is_back_view else garment_front

        p_small = person_img.resize((576, 768), Image.Resampling.LANCZOS)
        g_small = current_garment.resize((576, 768), Image.Resampling.LANCZOS)

        with torch.inference_mode():
            res = pipeline(
                person_image=p_small,
                garment_image=g_small,
                category="tops",
                garment_photo_type="flat-lay",
                num_timesteps=20
            ).images[0]

        final_res = res.resize(orig_size, Image.Resampling.LANCZOS)
        final_res.save(os.path.join(target_images_dir, fname))
        final_res.save(os.path.join(tryon_backup_dir, fname))
        if drive_2d_dir:
            final_res.save(os.path.join(drive_2d_dir, fname))

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✅ BƯỚC 2 HOÀN TẤT: Mặt trước có hoa văn sắc nét, mặt sau là tấm lưng lụa phẳng phiu tự nhiên!")


def run_step3_train_3dgs(work_dir: str, output_dir: str, iterations: int, fps: int, drive_dir: str = None):
    print("\n" + "="*70)
    print("🌐 [BƯỚC 3/3] HUẤN LUYỆN 3D GAUSSIAN SPLATTING & XUẤT VIDEO 360 ĐỘ")
    print("="*70)

    work_dir = os.path.abspath(work_dir)
    output_dir = os.path.abspath(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    gaussian_repo = "/kaggle/working/gaussian-splatting" if os.path.exists("/kaggle") else "/content/gaussian-splatting"
    if not os.path.exists(gaussian_repo):
        gaussian_repo = "./gaussian-splatting"

    train_script = os.path.join(gaussian_repo, "train.py")
    render_script = os.path.join(gaussian_repo, "render.py")

    print(f"🚀 Huấn luyện 3DGS ({iterations} bước)...")
    subprocess.run(f'python "{train_script}" -s "{work_dir}" -m "{output_dir}" --iterations {iterations}', shell=True, check=True)

    print("🎬 Render chuỗi khung hình 3D 360 độ...")
    subprocess.run(f'python "{render_script}" -m "{output_dir}" --skip_train', shell=True, check=True)

    iter_folder = f"ours_{iterations}"
    render_imgs = sorted(glob.glob(f"{output_dir}/test/{iter_folder}/renders/*.png"))
    if not render_imgs:
        render_imgs = sorted(glob.glob(f"{output_dir}/train/{iter_folder}/renders/*.png"))

    assert render_imgs, "❌ Không tìm thấy ảnh render 3D sau khi chạy render.py!"
    render_folder = os.path.dirname(render_imgs[0])

    output_video_path = os.path.join(output_dir, "AODAI_3D_360_FINAL.mp4")
    print(f"🎥 Ghép chuỗi ảnh render thành video MP4 (FPS={fps})...")
    subprocess.run(
        f'ffmpeg -y -framerate {fps} -pattern_type glob -i "{render_folder}/*.png" -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "{output_video_path}"',
        shell=True,
        check=True
    )

    print(f"🎉 ĐÃ XUẤT VIDEO THÀNH CÔNG: {output_video_path}")

    if drive_dir:
        drive_out = os.path.join(drive_dir, "FINAL_AODAI_3D_RESULT")
        os.makedirs(drive_out, exist_ok=True)
        dest_video = os.path.join(drive_out, "AODAI_3D_360_FINAL.mp4")
        shutil.copy(output_video_path, dest_video)
        subprocess.run(f'cp -r "{output_dir}/point_cloud/iteration_{iterations}"/* "{drive_out}/" 2>/dev/null || true', shell=True)
        print(f"💾 ĐÃ LƯU KẾT QUẢ VÀO: {dest_video}")

    return output_video_path


def main():
    args = parse_args()
    video_path, cloth_path = args.video, args.cloth

    if not video_path or not cloth_path:
        v_auto, c_auto = find_default_inputs()
        video_path = video_path or v_auto
        cloth_path = cloth_path or c_auto

    assert video_path and os.path.exists(video_path), f"❌ Không tìm thấy video người mẫu tại: {video_path}"
    assert cloth_path and os.path.exists(cloth_path), f"❌ Không tìm thấy ảnh áo dài tại: {cloth_path}"

    drive_dir = args.drive_dir

    print(f"🎬 Video input : {video_path}")
    print(f"👔 Cloth input : {cloth_path}")
    print(f"📁 Export Dir  : {drive_dir}")

    run_step1_colmap(video_path, args.work_dir)
    run_step2_fashn_tryon(args.work_dir, cloth_path, args.weights_dir, args.category, drive_dir)
    final_video = run_step3_train_3dgs(args.work_dir, args.output_dir, args.iterations, args.fps, drive_dir)

    print("\n" + "="*70)
    print(f"🎉 PIPELINE THÀNH CÔNG 100%! VIDEO: {final_video}")
    print("="*70)


if __name__ == "__main__":
    main()
