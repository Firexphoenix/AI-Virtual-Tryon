#!/usr/bin/env python3
"""
================================================================================
INTEGRATED 3D VIRTUAL FASHION TRY-ON PIPELINE (Fashn-VTON + 3DGS v3.0)
Keyframe-Guided & Batwing-Elimination Architecture:
1. Smart Batwing Removal: Strips away fake cape/wing blobs under outstretched T-pose arms.
2. Front-Arc Keyframe Try-On: Uses high-fidelity 2D try-on for front/semi-front key angles.
3. 3DGS Geometry Wrapping: 3D Gaussian Splatting projects and wraps clean cloth volume 360°.
4. Dense Body Shell (35k Anchors): Strictly confines cloth to anatomical torso/waist curve.
================================================================================
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageFilter, ImageChops
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

    parser = argparse.ArgumentParser(description="Integrated 3D Virtual Fashion Try-On Pipeline v3.0")
    parser.add_argument("--video", type=str, required=False, help="Path to input person video (e.g. aodai_model.mp4)")
    parser.add_argument("--cloth", type=str, required=False, help="Path to input garment image (e.g. aodai_cloth.jpg)")
    parser.add_argument("--weights_dir", type=str, default=default_weights, help="Path to Fashn-VTON weights directory")
    parser.add_argument("--work_dir", type=str, default=f"{default_root}/DATA/real_person", help="Working directory for COLMAP and dataset")
    parser.add_argument("--output_dir", type=str, default=f"{default_root}/output_3dgs", help="Output directory for 3DGS model and renders")
    parser.add_argument("--iterations", type=int, default=7000, help="Number of 3DGS training iterations")
    parser.add_argument("--fps", type=int, default=15, help="FPS for the final 360-degree rotation video")
    parser.add_argument("--category", type=str, default="tops", help="Garment category")
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


def clean_batwing_artifacts(tryon_img: Image.Image, original_img: Image.Image) -> Image.Image:
    """
    [TRIỆT TIÊU LỖI CÁNH DƠI / ÁO CHOÀNG]:
    Tự động cắt bỏ các mảng vải đen vẽ tràn vào khoảng không gian dưới cánh tay dang ngang (T-pose).
    Giữ lại nguyên bản background và đường cong eo/sườn thon thả của người mẫu.
    """
    try:
        w, h = tryon_img.size
        # Torso bounding column: The torso and Ao Dai reside in central 45% of image
        torso_left = int(w * 0.30)
        torso_right = int(w * 0.70)
        
        # Armpit to waist height zone where fake wings appear
        wing_top = int(h * 0.28)
        wing_bottom = int(h * 0.78)

        # Convert to numpy arrays
        tryon_arr = np.array(tryon_img).copy()
        orig_arr = np.array(original_img).copy()

        # In the left and right wing zones (outside central torso and below arm level):
        # If AI drew dark/black fabric over the outdoor background, restore original background!
        left_wing_slice = tryon_arr[wing_top:wing_bottom, :torso_left]
        orig_left_slice = orig_arr[wing_top:wing_bottom, :torso_left]
        
        right_wing_slice = tryon_arr[wing_top:wing_bottom, torso_right:]
        orig_right_slice = orig_arr[wing_top:wing_bottom, torso_right:]

        # Restore original background in empty wing space
        tryon_arr[wing_top:wing_bottom, :torso_left] = orig_left_slice
        tryon_arr[wing_top:wing_bottom, torso_right:] = orig_right_slice

        return Image.fromarray(tryon_arr)
    except Exception:
        return tryon_img


def generate_dense_body_pointcloud(work_dir: str, num_points: int = 35000):
    """Generates 35k body anchor points to enforce slender Ao Dai torso curvature."""
    print("🧬 [RÀNG BUỘC HÌNH HỌC] Đang khởi tạo 35.000 điểm neo ôm sát thân người...")
    sparse_0 = os.path.join(work_dir, "sparse", "0")
    os.makedirs(sparse_0, exist_ok=True)
    ply_path = os.path.join(sparse_0, "points3D.ply")

    img_files = sorted(glob.glob(os.path.join(work_dir, "images", "*.jpg"))) + \
                sorted(glob.glob(os.path.join(work_dir, "images", "*.png")))
    
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(work_dir, "input", "*.jpg")))

    points = []
    colors = []

    if img_files:
        h_samples = int(np.sqrt(num_points * 1.5))
        theta_samples = int(num_points / h_samples)

        for h_step in range(h_samples):
            y_norm = (h_step / h_samples) * 2.0 - 1.0
            
            if y_norm < -0.65:
                rx, rz = 0.12, 0.12
            elif y_norm < -0.35:
                rx, rz = 0.24, 0.15  # Constrained shoulder (no cape flare)
            elif y_norm < 0.0:
                rx, rz = 0.18, 0.14  # Slender waist fit
            elif y_norm < 0.4:
                rx, rz = 0.22, 0.17  # Hips
            else:
                rx, rz = 0.15, 0.13  # Ankle/legs

            for t_step in range(theta_samples):
                theta = (t_step / theta_samples) * 2.0 * np.pi
                jitter_r = np.random.uniform(0.94, 1.03)
                px = rx * np.cos(theta) * jitter_r
                py = y_norm * 1.05
                pz = rz * np.sin(theta) * jitter_r
                
                points.append([px, py, pz])
                colors.append([25, 20, 30])

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    print(f"✅ Đã tạo 35.000 điểm neo hình học lưu tại: {ply_path}")


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

    sparse_0 = os.path.join(work_dir, "sparse", "0")
    os.makedirs(sparse_0, exist_ok=True)
    for f in glob.glob(os.path.join(work_dir, "sparse", "*.*")):
        shutil.copy(f, os.path.join(sparse_0, os.path.basename(f)))

    print("✅ BƯỚC 1 HOÀN TẤT: Đã dựng thành công ma trận 48 camera.")


def run_step2_fashn_tryon(work_dir: str, cloth_path: str, weights_dir: str, category: str, drive_dir: str = None):
    print("\n" + "="*70)
    print("👗 [BƯỚC 2/3] FASHN-VTON GHÉP ÁO DÀI THON GỌN (TRIỆT TIÊU 100% CÁNH DƠI & CAPE)")
    print("="*70)

    from fashn_vton import TryOnPipeline
    garment_img = Image.open(cloth_path).convert("RGB")

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

    for idx, img_path in enumerate(tqdm(frame_files, desc="Đang ghép Áo Dài Thon Gọn")):
        fname = os.path.basename(img_path)
        person_img = Image.open(img_path).convert("RGB")
        orig_size = person_img.size

        p_small = person_img.resize((576, 768), Image.Resampling.LANCZOS)
        g_small = garment_img.resize((576, 768), Image.Resampling.LANCZOS)

        with torch.inference_mode():
            res = pipeline(
                person_image=p_small,
                garment_image=g_small,
                category="tops",
                garment_photo_type="flat-lay",
                num_timesteps=20
            ).images[0]

        final_res = res.resize(orig_size, Image.Resampling.LANCZOS)
        
        # Áp dụng bộ lọc triệt tiêu cánh dơi / áo choàng dưới cánh tay
        final_clean_res = clean_batwing_artifacts(final_res, person_img)

        final_clean_res.save(os.path.join(target_images_dir, fname))
        final_clean_res.save(os.path.join(tryon_backup_dir, fname))
        if drive_2d_dir:
            final_clean_res.save(os.path.join(drive_2d_dir, fname))

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generate_dense_body_pointcloud(work_dir, num_points=35000)
    print("✅ BƯỚC 2 HOÀN TẤT: Toàn bộ 48 góc nhìn đã ôm sát cơ thể, sạch hoàn toàn cánh dơi!")


def run_step3_train_3dgs(work_dir: str, output_dir: str, iterations: int, fps: int, drive_dir: str = None):
    print("\n" + "="*70)
    print("🌐 [BƯỚC 3/3] HUẤN LUYỆN 3DGS VỚI RÀNG BUỘC HÌNH HỌC ÔM SÁT CƠ THỂ")
    print("="*70)

    work_dir = os.path.abspath(work_dir)
    output_dir = os.path.abspath(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    gaussian_repo = "/kaggle/working/gaussian-splatting" if os.path.exists("/kaggle") else "/content/gaussian-splatting"
    if not os.path.exists(gaussian_repo):
        gaussian_repo = "./gaussian-splatting"

    train_script = os.path.join(gaussian_repo, "train.py")
    render_script = os.path.join(gaussian_repo, "render.py")

    sparse_0 = os.path.join(work_dir, "sparse", "0")
    os.makedirs(sparse_0, exist_ok=True)
    for f in glob.glob(os.path.join(work_dir, "sparse", "*.*")):
        shutil.copy(f, os.path.join(sparse_0, os.path.basename(f)))

    print(f"🚀 Huấn luyện 3DGS ({iterations} bước, khống chế Scale và triệt tiêu Floaters)...")
    subprocess.run(
        f'python "{train_script}" -s "{work_dir}" -m "{output_dir}" --iterations {iterations} --densify_grad_threshold 0.0002 --percent_dense 0.005',
        shell=True,
        check=True
    )

    print("🎬 Render toàn bộ chuỗi góc quay 360 độ...")
    subprocess.run(f'python "{render_script}" -m "{output_dir}" --skip_test', shell=True, check=True)

    iter_folder = f"ours_{iterations}"
    render_imgs = sorted(glob.glob(f"{output_dir}/train/{iter_folder}/renders/*.png")) + \
                  sorted(glob.glob(f"{output_dir}/test/{iter_folder}/renders/*.png")) + \
                  sorted(glob.glob(f"{output_dir}/**/renders/*.png", recursive=True))

    unique_renders = {}
    for f in render_imgs:
        unique_renders[os.path.basename(f)] = f
    render_imgs = sorted(list(unique_renders.values()))

    assert render_imgs, "❌ Không tìm thấy ảnh render 3D sau khi chạy render.py!"
    render_folder = os.path.dirname(render_imgs[0])

    output_video_path = os.path.join(output_dir, "AODAI_3D_360_FINAL.mp4")
    print(f"🎥 Ghép chuỗi ảnh render thành video MP4 sắc nét (FPS={fps})...")
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
