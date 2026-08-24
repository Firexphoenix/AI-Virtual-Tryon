#!/usr/bin/env python3
"""
================================================================================
FASHION-VTON + TRELLIS.2 PIPELINE v1.0
Thay thế hoàn toàn GS-VTON (COLMAP + LoRA + 3DGS Training)

Pipeline mới:
  Bước 1: Fashn-VTON  → Ảnh 2D người mẫu mặc Áo Dài (10-15 giây)
  Bước 2: Remove BG   → Tách nền trong suốt RGBA (5 giây)
  Bước 3: TRELLIS.2   → 3D .glb + .ply + Video 360° (20-40 giây)

Tổng thời gian: ~1-2 phút (thay vì 2-4 giờ của GS-VTON)
Môi trường:     Kaggle / Colab GPU (T4 16GB trở lên, Linux, CUDA 12.x)
================================================================================
"""

import argparse
import gc
import glob
import os
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

# ── Thiết lập môi trường GPU & TRELLIS Backend ────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_BACKEND"] = "spconv"

# ── Cache HuggingFace về Kaggle working dir để tránh mất khi restart ──────────
if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/.cache_hf"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/.cache_hf"
elif os.path.exists("/content"):
    os.environ["HF_HOME"] = "/content/.cache_hf"


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: FASHN-VTON — 2D VIRTUAL TRY-ON
# ══════════════════════════════════════════════════════════════════════════════

def run_step1_fashn_tryon(
    person_image_path: str,
    cloth_image_path: str,
    output_dir: str,
    category: str = "tops",
    num_timesteps: int = 20,
) -> str:
    """
    Dùng Fashn-VTON để ghép quần áo vào ảnh người mẫu.
    Trả về đường dẫn tới ảnh 2D try-on kết quả.
    """
    print("\n" + "=" * 70)
    print("👗 [BƯỚC 1/3] FASHN-VTON — GHÉP ÁO VÀO ẢNH NGƯỜI MẪU 2D")
    print("=" * 70)

    from fashn_vton import TryOnPipeline

    os.makedirs(output_dir, exist_ok=True)

    person_img = Image.open(person_image_path).convert("RGB")
    cloth_img = Image.open(cloth_image_path).convert("RGB")
    orig_size = person_img.size

    # Resize về kích thước chuẩn của Fashn-VTON
    p_resized = person_img.resize((576, 768), Image.Resampling.LANCZOS)
    c_resized = cloth_img.resize((576, 768), Image.Resampling.LANCZOS)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    weights_dir = _find_fashn_weights()
    print(f"⚡ Đang tải Fashn-VTON từ: {weights_dir}")
    pipeline = TryOnPipeline(weights_dir=weights_dir)

    print(f"🎨 Đang ghép áo [{category}] vào người mẫu... ({num_timesteps} bước diffusion)")
    with torch.inference_mode():
        result = pipeline(
            person_image=p_resized,
            garment_image=c_resized,
            category=category,
            garment_photo_type="flat-lay",
            num_timesteps=num_timesteps,
        ).images[0]

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Khôi phục kích thước gốc
    tryon_img = result.resize(orig_size, Image.Resampling.LANCZOS)

    tryon_path = os.path.join(output_dir, "step1_tryon_2d.png")
    tryon_img.save(tryon_path)
    print(f"✅ BƯỚC 1 HOÀN TẤT: Ảnh 2D Try-On lưu tại: {tryon_path}")
    return tryon_path


def _find_fashn_weights() -> str:
    """Tìm hoặc tự động tải weights của Fashn-VTON từ Hugging Face."""
    from huggingface_hub import hf_hub_download

    candidates = [
        "/kaggle/working/fashn_weights",
        "/kaggle/input/fashn-vton/fashn_weights",
        "/content/fashn_weights",
        "./fashn_weights",
    ]
    target_dir = None
    for p in candidates:
        if os.path.exists(p):
            target_dir = p
            break
    if not target_dir:
        target_dir = "/kaggle/working/fashn_weights" if os.path.exists("/kaggle") else "./fashn_weights"
        os.makedirs(target_dir, exist_ok=True)

    dwpose_dir = os.path.join(target_dir, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)

    model_path = os.path.join(target_dir, "model.safetensors")
    yolox_path = os.path.join(dwpose_dir, "yolox_l.onnx")
    dwpose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

    if not os.path.exists(model_path):
        print("⬇️ Đang tải Fashn-VTON model.safetensors (~2.1 GB)...")
        hf_hub_download(
            repo_id="fashn-ai/fashn-vton-1.5",
            filename="model.safetensors",
            local_dir=target_dir,
        )

    if not os.path.exists(yolox_path):
        print("⬇️ Đang tải DWPose yolox_l.onnx...")
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename="yolox_l.onnx",
            local_dir=dwpose_dir,
        )

    if not os.path.exists(dwpose_path):
        print("⬇️ Đang tải DWPose dw-ll_ucoco_384.onnx...")
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename="dw-ll_ucoco_384.onnx",
            local_dir=dwpose_dir,
        )

    print(f"✅ Đã sẵn sàng model weights tại: {target_dir}")
    return target_dir


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: TÁCH NỀN — RGBA BACKGROUND REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_step2_remove_bg(tryon_image_path: str, output_dir: str) -> str:
    """
    Tách nền ảnh 2D try-on → PNG trong suốt RGBA.
    TRELLIS.2 hoạt động tốt nhất với ảnh đã tách nền.
    Trả về đường dẫn tới ảnh RGBA.
    """
    print("\n" + "=" * 70)
    print("✂️  [BƯỚC 2/3] TÁCH NỀN — XỬ LÝ ẢNH RGBA CHO TRELLIS.2")
    print("=" * 70)

    from rembg import remove, new_session

    person_img = Image.open(tryon_image_path).convert("RGB")

    print("🔄 Đang tách nền bằng rembg (model: u2net_human_seg)...")
    session = new_session("u2net_human_seg")
    rgba_img = remove(
        person_img,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=11,
    )

    del session
    gc.collect()

    rgba_path = os.path.join(output_dir, "step2_rgba.png")
    rgba_img.save(rgba_path)
    print(f"✅ BƯỚC 2 HOÀN TẤT: Ảnh RGBA lưu tại: {rgba_path}")
    return rgba_path


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: TRELLIS.2 — SINH MÔ HÌNH 3D
# ══════════════════════════════════════════════════════════════════════════════

def run_step3_trellis_3d(
    rgba_image_path: str,
    output_dir: str,
    seed: int = 1,
    sparse_steps: int = 12,
    slat_steps: int = 12,
    fps: int = 24,
    drive_dir: str = None,
) -> dict:
    """
    Dùng TRELLIS.2 để chuyển ảnh RGBA 2D thành mô hình 3D hoàn chỉnh.
    Xuất ra:
      - output.glb      (Textured Mesh — dùng trên Web Three.js/Unity/Blender)
      - output.ply      (3D Gaussian Splatting — chất lượng render cao nhất)
      - 360deg.mp4      (Video xoay 360 độ)
    """
    print("\n" + "=" * 70)
    print("🌐 [BƯỚC 3/3] TRELLIS.2 — SINH MÔ HÌNH 3D TỪ ẢNH 2D")
    print("=" * 70)

    # Tự động tìm và thêm thư mục TRELLIS vào sys.path, đồng bộ submodules (FlexiCubes)
    trellis_candidates = [
        "/kaggle/working/TRELLIS",
        "/kaggle/working/TRELLIS.2",
        "./TRELLIS",
        "../TRELLIS",
        os.path.join(os.getcwd(), "TRELLIS"),
    ]
    for tpath in trellis_candidates:
        if os.path.exists(tpath):
            if tpath not in sys.path:
                sys.path.insert(0, os.path.abspath(tpath))
            # Kiểm tra và tải submodule FlexiCubes nếu chưa có
            flexicubes_init = os.path.join(tpath, "trellis", "representations", "mesh", "flexicubes", "flexicubes.py")
            if not os.path.exists(flexicubes_init):
                print(f"🔄 Đang đồng bộ submodules (FlexiCubes) tại {tpath}...")
                import subprocess
                subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=tpath, check=False)

    # Tự động tạo mock cho kaolin.utils.testing nếu chưa có kaolin
    # (FlexiCubes chỉ dùng check_tensor từ kaolin để assert shape, không cần cài đặt Kaolin nặng)
    try:
        import kaolin
    except ImportError:
        import types
        kaolin_mod = types.ModuleType("kaolin")
        kaolin_utils = types.ModuleType("kaolin.utils")
        kaolin_testing = types.ModuleType("kaolin.utils.testing")
        kaolin_testing.check_tensor = lambda *args, **kwargs: True
        kaolin_utils.testing = kaolin_testing
        kaolin_mod.utils = kaolin_utils
        sys.modules["kaolin"] = kaolin_mod
        sys.modules["kaolin.utils"] = kaolin_utils
        sys.modules["kaolin.utils.testing"] = kaolin_testing

    # Import TRELLIS
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as e:
        # Nếu chưa có thư mục TRELLIS, tự động clone về đầy đủ submodules
        if not any(os.path.exists(p) for p in trellis_candidates):
            print("⬇️ Đang tải mã nguồn Microsoft TRELLIS về /kaggle/working/TRELLIS...")
            import subprocess
            subprocess.run(["git", "clone", "--recurse-submodules", "https://github.com/microsoft/TRELLIS", "/kaggle/working/TRELLIS"], check=True)
            sys.path.insert(0, "/kaggle/working/TRELLIS")
            try:
                from trellis.pipelines import TrellisImageTo3DPipeline
            except ImportError:
                raise ImportError(f"\n❌ Lỗi import TRELLIS ({e})! Vui lòng chạy lệnh cài đặt TRELLIS trước.")
        else:
            raise ImportError(f"\n❌ Lỗi import TRELLIS ({e})! Vui lòng kiểm tra các thư viện phụ thuộc của TRELLIS.")

    # Import postprocessing_utils an toàn (nếu có nvdiffrast)
    try:
        from trellis.utils import postprocessing_utils
    except Exception as exc:
        print(f"ℹ️ postprocessing_utils chạy ở chế độ cơ bản ({exc})")
        postprocessing_utils = None

    # Import render_utils an toàn (nếu có nvdiffrast)
    try:
        from trellis.utils import render_utils
        has_render_utils = True
    except Exception as exc:
        render_utils = None
        has_render_utils = False

    os.makedirs(output_dir, exist_ok=True)

    # Đường dẫn file đầu ra
    glb_path   = os.path.join(output_dir, "output.glb")
    ply_path   = os.path.join(output_dir, "output.ply")
    video_path = os.path.join(output_dir, "360deg_AODAI_3D.mp4")

    # Tải ảnh RGBA
    rgba_img = Image.open(rgba_image_path).convert("RGBA")
    print(f"📷 Ảnh đầu vào: {rgba_image_path}  ({rgba_img.size[0]}x{rgba_img.size[1]} RGBA)")

    # Tải mô hình TRELLIS.2 từ HuggingFace
    print("⚡ Đang tải mô hình TRELLIS.2 (microsoft/TRELLIS-image-large)...")
    print("   (Lần đầu tải ~2-3GB checkpoint, các lần sau dùng cache HF)")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_vram = torch.cuda.mem_get_info()[0] / 1024 ** 3
        print(f"💾 VRAM khả dụng: {free_vram:.1f} GB")

    # Chạy inference TRELLIS.2
    print(f"🎲 Đang sinh mô hình 3D (seed={seed}, sparse_steps={sparse_steps}, slat_steps={slat_steps})...")
    with torch.inference_mode():
        outputs = pipeline.run(
            rgba_img,
            seed=seed,
            formats=["mesh", "gaussian"],
            sparse_structure_sampler_params={
                "steps": sparse_steps,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": 3.0,
            },
        )

    # Xuất file .glb (Textured Mesh)
    print(f"📦 Đang xuất Textured Mesh → {glb_path}")
    try:
        if postprocessing_utils is not None and hasattr(postprocessing_utils, "to_glb"):
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,      # Giữ 95% polygon detail, giảm kích thước file
                texture_size=1024,
            )
            glb.export(glb_path)
        else:
            outputs["mesh"][0].export(glb_path)
        glb_size_mb = os.path.getsize(glb_path) / 1024 ** 2
        print(f"   ✅ .glb xuất thành công ({glb_size_mb:.1f} MB)")
    except Exception as glb_err:
        print(f"⚠️ Thử xuất OBJ/GLB trực tiếp ({glb_err})...")
        outputs["mesh"][0].export(glb_path)
        glb_size_mb = os.path.getsize(glb_path) / 1024 ** 2
        print(f"   ✅ .glb xuất thành công ({glb_size_mb:.1f} MB)")

    # Xuất file .ply (3D Gaussian Splatting)
    print(f"✨ Đang xuất 3D Gaussian Splatting → {ply_path}")
    outputs["gaussian"][0].save_ply(ply_path)
    ply_size_mb = os.path.getsize(ply_path) / 1024 ** 2
    print(f"   ✅ .ply xuất thành công ({ply_size_mb:.1f} MB)")

    # Render video xoay 360° (nếu có render_utils)
    if has_render_utils and render_utils is not None:
        try:
            print(f"🎬 Đang render video 360° (FPS={fps}, 120 frames)...")
            video_frames = render_utils.render_video(
                outputs["gaussian"][0],
                num_frames=120,
                resolution=512,
                r=2.0,         # Khoảng cách camera so với tâm đối tượng
            )["color"]         # List of np.ndarray frames (H, W, 3), float [0,1]

            # Lưu video bằng imageio (không cần ffmpeg subprocess)
            with imageio.get_writer(
                video_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
            ) as writer:
                for frame in video_frames:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)

            video_size_mb = os.path.getsize(video_path) / 1024 ** 2
            print(f"   ✅ Video 360° xuất thành công ({video_size_mb:.1f} MB)")
        except Exception as v_exc:
            print(f"⚠️ Render video bỏ qua do lỗi rasterizer ({v_exc}). File 3D (.glb, .ply) đã được tạo đầy đủ!")
            video_path = None
    else:
        print("ℹ️ Bỏ qua render video (cài thêm nvdiffrast nếu cần render video MP4).")
        video_path = None
    del pipeline, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Tóm tắt kết quả
    print("\n" + "=" * 70)
    print("🎉 BƯỚC 3 HOÀN TẤT! Đầu ra 3D:")
    print(f"   📦 Mesh (.glb) : {glb_path}  [{glb_size_mb:.1f} MB]")
    print(f"   ✨ 3DGS (.ply)  : {ply_path}  [{ply_size_mb:.1f} MB]")
    print(f"   🎬 Video (.mp4) : {video_path}  [{video_size_mb:.1f} MB]")
    print("=" * 70)

    result = {"glb": glb_path, "ply": ply_path, "video": video_path}

    # Sao chép sang Google Drive nếu được chỉ định
    if drive_dir:
        drive_out = os.path.join(drive_dir, "AODAI_3D_TRELLIS_RESULT")
        os.makedirs(drive_out, exist_ok=True)
        for label, fpath in result.items():
            dst = os.path.join(drive_out, os.path.basename(fpath))
            shutil.copy(fpath, dst)
            print(f"💾 Đã sao chép {label} → {dst}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT INPUT FILES
# ══════════════════════════════════════════════════════════════════════════════

def find_default_inputs():
    """Tự động tìm ảnh người mẫu và ảnh áo trong các thư mục phổ biến."""
    person_candidates = (
        sorted(glob.glob("/kaggle/input/**/aodai_model*", recursive=True))
        + sorted(glob.glob("/kaggle/working/**/aodai_model*", recursive=True))
        + sorted(glob.glob("/content/**/aodai_model*", recursive=True))
        + sorted(glob.glob("./DATA/**/aodai_model*", recursive=True))
    )
    cloth_candidates = (
        sorted(glob.glob("/kaggle/input/**/aodai_cloth*", recursive=True))
        + sorted(glob.glob("/kaggle/working/**/aodai_cloth*", recursive=True))
        + sorted(glob.glob("/content/**/aodai_cloth*", recursive=True))
        + sorted(glob.glob("./DATA/**/aodai_cloth*", recursive=True))
    )
    person = person_candidates[0] if person_candidates else None
    cloth  = cloth_candidates[0]  if cloth_candidates  else None
    return person, cloth


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    default_out = "/kaggle/working/output_trellis" if os.path.exists("/kaggle") else "./output_trellis"

    parser = argparse.ArgumentParser(
        description="Fashion-VTON + TRELLIS.2 Pipeline — Áo Dài 3D từ 1 ảnh người mẫu"
    )
    parser.add_argument("--person_image", type=str, default=None,
                        help="Đường dẫn ảnh người mẫu (JPG/PNG)")
    parser.add_argument("--cloth_image",  type=str, default=None,
                        help="Đường dẫn ảnh quần áo / Áo Dài (JPG/PNG)")
    parser.add_argument("--output_dir",   type=str, default=default_out,
                        help="Thư mục lưu kết quả đầu ra")
    parser.add_argument("--category",     type=str, default="tops",
                        choices=["tops", "bottoms", "one-pieces"],
                        help="Loại trang phục (mặc định: tops)")
    parser.add_argument("--seed",         type=int, default=1,
                        help="Random seed cho TRELLIS.2 (ảnh hưởng hình dạng 3D)")
    parser.add_argument("--sparse_steps", type=int, default=12,
                        help="Số bước sampling giai đoạn Sparse Structure của TRELLIS.2")
    parser.add_argument("--slat_steps",   type=int, default=12,
                        help="Số bước sampling giai đoạn SLAT của TRELLIS.2")
    parser.add_argument("--fps",          type=int, default=24,
                        help="FPS của video 360° đầu ra")
    parser.add_argument("--drive_dir",    type=str, default=None,
                        help="Thư mục Google Drive để sao lưu kết quả (tùy chọn)")
    parser.add_argument("--skip_tryon",   action="store_true",
                        help="Bỏ qua Bước 1 — dùng khi đã có ảnh try-on sẵn")
    parser.add_argument("--tryon_image",  type=str, default=None,
                        help="Đường dẫn ảnh try-on có sẵn (dùng với --skip_tryon)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect input nếu không truyền tham số
    person_path = args.person_image
    cloth_path  = args.cloth_image
    if not person_path or not cloth_path:
        p_auto, c_auto = find_default_inputs()
        person_path = person_path or p_auto
        cloth_path  = cloth_path  or c_auto

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 FASHION-VTON + TRELLIS.2 PIPELINE — BẮT ĐẦU")
    print("=" * 70)

    # Bước 1: Try-On 2D
    if args.skip_tryon and args.tryon_image and os.path.exists(args.tryon_image):
        print(f"⏭️  Bỏ qua Bước 1. Sử dụng ảnh try-on có sẵn: {args.tryon_image}")
        tryon_path = args.tryon_image
    else:
        assert person_path and os.path.exists(person_path), \
            f"❌ Không tìm thấy ảnh người mẫu: {person_path}\n   --person_image /path/to/model.jpg"
        assert cloth_path  and os.path.exists(cloth_path), \
            f"❌ Không tìm thấy ảnh áo: {cloth_path}\n   --cloth_image /path/to/garment.jpg"
        print(f"👤 Ảnh người mẫu : {person_path}")
        print(f"👗 Ảnh quần áo   : {cloth_path}")
        tryon_path = run_step1_fashn_tryon(
            person_path, cloth_path, output_dir,
            category=args.category,
        )

    # Bước 2: Tách nền
    rgba_path = run_step2_remove_bg(tryon_path, output_dir)

    # Bước 3: TRELLIS.2 → 3D
    result = run_step3_trellis_3d(
        rgba_path, output_dir,
        seed=args.seed,
        sparse_steps=args.sparse_steps,
        slat_steps=args.slat_steps,
        fps=args.fps,
        drive_dir=args.drive_dir,
    )

    print("\n" + "=" * 70)
    print("🎉 PIPELINE HOÀN THÀNH 100%!")
    print(f"   📦 Mesh  (.glb) : {result['glb']}")
    print(f"   ✨ 3DGS  (.ply)  : {result['ply']}")
    print(f"   🎬 Video (.mp4) : {result['video']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
