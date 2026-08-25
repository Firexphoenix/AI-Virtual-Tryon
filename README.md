<div align="center">

# 🎽 AI Virtual Try-On 3D
### Fashion-VTON + Microsoft TRELLIS — High-Fidelity 3D Virtual Try-On from a Single Photo

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-orange)](https://kaggle.com)

**Biến 1 ảnh người mẫu + 1 ảnh trang phục → Mô hình 3D hoàn chỉnh trong ~2 phút**

</div>

---

## 📖 Giới thiệu

Repo này tổng hợp và tối ưu hóa toàn bộ pipeline **AI Virtual Try-On 3D**, thay thế kiến trúc GS-VTON cũ (COLMAP + LoRA Training + 3DGS, mất 2–4 giờ) bằng quy trình **2D-to-3D Decoupled Pipeline** hiện đại:

```
[Ảnh Người mẫu] + [Ảnh Trang phục]
        ↓
[Bước 1: Fashn-VTON]          ──► Ghép quần áo 2D siêu nét    (~10-15s)
        ↓
[Bước 2: Rembg / BiRefNet]    ──► Tách nền trong suốt RGBA     (~5s)
        ↓
[Bước 3: Microsoft TRELLIS]   ──► Sinh vật thể 3D hoàn chỉnh  (~20-40s)
        ↓
        ├──► 📦 output.glb   (Textured Mesh — Web / Unity / Blender)
        ├──► ✨ output.ply   (3D Gaussian Splatting)
        └──► 🎬 360deg.mp4  (Video xoay 360°)
```

### ⚡ So sánh hiệu năng

| Tiêu chí | GS-VTON (cũ) | Pipeline mới (Fashion-VTON + TRELLIS) |
|:---|:---|:---|
| **Đầu vào** | Video quay vòng 360° + ảnh áo | **1 ảnh người mẫu** + ảnh áo |
| **Thời gian chạy** | 2 – 4 giờ (Training nặng) | **~1 – 2 phút (Inference only)** |
| **Cài đặt** | Cực kỳ phức tạp (CUDA ext, COLMAP) | Đơn giản, đóng gói sẵn trên Linux/Kaggle |
| **Đầu ra** | Chỉ có video render 3DGS | **.glb (Mesh PBR) + .ply (3DGS) + Video MP4** |
| **Phần cứng** | GPU A100+ (≥40GB VRAM) | **T4 16GB (Kaggle Free)** |

---

## 📚 Nguồn tham khảo (Credits)

Repo này được xây dựng dựa trên 3 nguồn mã nguồn mở sau:

| Repo | Tác giả | Đóng góp trong pipeline |
|:---|:---|:---|
| **[GS-VTON](https://github.com/yukangcao/GS-VTON)** | Yukang Cao, Masoud Hadi et al. (NTU) | Kiến trúc gốc: IDM-VTON + LoRA + GaussianEditor + 3DGS pipeline (stage1, stage2) |
| **[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)** | Microsoft Research | 3D Generative Model (SLAT + Sparse Diffusion) để sinh mô hình 3D từ ảnh 2D |
| **[fashn-AI/fashn-vton-1.5](https://github.com/fashn-AI/fashn-vton-1.5)** | Fashn AI | 2D Virtual Try-On model siêu nhanh (thay thế IDM-VTON) |

> ⚠️ **Tuyên bố bản quyền**: Toàn bộ code chính của TRELLIS (`trellis/`) được giữ nguyên bản từ Microsoft Research theo giấy phép MIT. Repo này chỉ tích hợp và tối ưu hóa pipeline chạy trên Kaggle — không sửa đổi kiến trúc model gốc.

---

## 🚀 Hướng dẫn chạy trên Kaggle (ALL-IN-ONE)

### Yêu cầu
- **GPU**: NVIDIA T4 x1 hoặc x2 (Kaggle Free), P100, hoặc A100
- **RAM**: ≥ 16GB
- **Disk**: ≥ 20GB

### Chỉ cần 1 ô lệnh duy nhất (copy & paste vào Kaggle):

```python
import os, sys, shutil, base64
from IPython.display import HTML, display

# ── Bước 1: Cài đặt thư viện (~1 phút) ──────────────────────────────────────
print("📦 Đang cài đặt thư viện...")
os.system('pip install "numpy<2" "spconv-cu121" "cumm-cu121" xformers --no-deps -q')

# ── Bước 2: Tải mã nguồn ─────────────────────────────────────────────────────
print("🔄 Đang tải mã nguồn...")
if not os.path.exists("/kaggle/working/AI-Virtual-Tryon"):
    os.system("git clone https://github.com/Firexphoenix/AI-Virtual-Tryon.git /kaggle/working/AI-Virtual-Tryon")
else:
    os.system("cd /kaggle/working/AI-Virtual-Tryon && git pull origin main")

if not os.path.exists("/kaggle/working/TRELLIS"):
    os.system("git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /kaggle/working/TRELLIS")

os.system("pip install -r /kaggle/working/AI-Virtual-Tryon/requirements.txt -q")

# ── Bước 3: Chạy Pipeline ────────────────────────────────────────────────────
# 👇 Thay đường dẫn ảnh của bạn vào đây:
PERSON_IMAGE = "/kaggle/input/your-dataset/person_image.jpg"
CLOTH_IMAGE  = "/kaggle/input/your-dataset/cloth_image.jpg"
CATEGORY     = "one-pieces"   # tops | bottoms | one-pieces
OUTPUT_DIR   = "/kaggle/working/output_trellis"

os.chdir("/kaggle/working/AI-Virtual-Tryon")
os.system(f"""python run_integrated_3d_pipeline.py \
    --person_image {PERSON_IMAGE} \
    --cloth_image  {CLOTH_IMAGE} \
    --category     {CATEGORY} \
    --output_dir   {OUTPUT_DIR}""")

# ── Bước 4: Tải kết quả về máy ───────────────────────────────────────────────
if os.path.exists(f"{OUTPUT_DIR}/output.glb"):
    shutil.make_archive("/kaggle/working/AODAI_3D_RESULT", 'zip', OUTPUT_DIR)
    with open("/kaggle/working/AODAI_3D_RESULT.zip", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    display(HTML(f'''
    <a download="AODAI_3D_RESULT.zip" href="data:application/zip;base64,{b64}"
       style="padding:12px 24px;background:#2e7d32;color:white;font-size:16px;
              font-weight:bold;border-radius:6px;text-decoration:none;">
       ⬇️ TẢI KẾT QUẢ 3D VỀ MÁY
    </a>
    <script>
    var a=document.createElement("a");a.href="data:application/zip;base64,{b64}";
    a.download="AODAI_3D_RESULT.zip";document.body.appendChild(a);a.click();
    </script>'''))
```

---

## 🔧 Tùy chọn nâng cao

### Bỏ qua Bước 1 (đã có ảnh try-on sẵn):
```bash
python run_integrated_3d_pipeline.py \
    --skip_tryon \
    --tryon_image /kaggle/working/output_trellis/step1_tryon_2d.png \
    --output_dir  /kaggle/working/output_trellis
```

### Tùy chỉnh chất lượng 3D:
```bash
python run_integrated_3d_pipeline.py \
    --person_image model.jpg \
    --cloth_image  aodai.jpg \
    --category     one-pieces \
    --sparse_steps 25 \        # Tăng lên để chất lượng tốt hơn (mặc định 12)
    --slat_steps   25 \        # Tăng lên để chi tiết texture hơn (mặc định 12)
    --seed         42 \        # Seed ngẫu nhiên để tái tạo kết quả
    --output_dir   ./output
```

### Tham số đầy đủ:
```
--person_image   Đường dẫn ảnh người mẫu (bắt buộc nếu không --skip_tryon)
--cloth_image    Đường dẫn ảnh trang phục (bắt buộc nếu không --skip_tryon)
--category       Loại trang phục: tops | bottoms | one-pieces (mặc định: tops)
--output_dir     Thư mục lưu kết quả (mặc định: ./output_trellis)
--skip_tryon     Bỏ qua bước 1 (ghép áo), dùng ảnh try-on có sẵn
--tryon_image    Ảnh try-on sẵn (dùng với --skip_tryon)
--seed           Seed sinh 3D (mặc định: 1)
--sparse_steps   Số bước Sparse Diffusion (mặc định: 12, tối đa: 50)
--slat_steps     Số bước SLAT Diffusion (mặc định: 12, tối đa: 50)
--fps            FPS của video đầu ra (mặc định: 24)
--drive_dir      Sao chép kết quả sang Google Drive
```

---

## 📦 Kết quả đầu ra

Sau khi chạy xong, thư mục `--output_dir` sẽ chứa:

| File | Mô tả | Cách xem |
|:---|:---|:---|
| `step1_tryon_2d.png` | Ảnh 2D ghép trang phục | Xem trực tiếp |
| `step2_rgba.png` | Ảnh RGBA đã tách nền | Xem trực tiếp |
| `output.glb` | 3D Textured Mesh tiêu chuẩn | [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com) hoặc Windows 3D Viewer |
| `output.ply` | 3D Gaussian Splatting | [SuperSplat Viewer](https://playcanvas.com/supersplat/editor) |
| `360deg_AODAI_3D.mp4` | Video xoay 360° | Mọi trình phát video |

---

## 🏗️ Cấu trúc repo

```
AI-Virtual-Tryon/
├── run_integrated_3d_pipeline.py   # ← Pipeline chính (ALL-IN-ONE)
├── main.py                         # ← Entry point đơn giản
├── requirements.txt                # Thư viện Python cần thiết
├── stage1/                         # GS-VTON Stage 1 (IDM-VTON legacy)
├── stage2/                         # GS-VTON Stage 2 (GaussianEditor legacy)
├── DATA/                           # Thư mục dữ liệu mẫu
└── docs/                           # Tài liệu bổ sung
```

> **Ghi chú**: Thư mục `stage1/` và `stage2/` là code gốc từ **GS-VTON** (pipeline cũ, giữ lại để tham khảo). Pipeline mới hoàn toàn không phụ thuộc vào chúng — chỉ cần `run_integrated_3d_pipeline.py`.

---

## 🔬 Giải thích kỹ thuật

### Tại sao pipeline mới nhanh hơn 100x?

**GS-VTON (cũ):**
1. Chụp video 360° người mẫu → COLMAP → 3DGS từ đầu (train ~2h)
2. IDM-VTON ghép áo từng frame → LoRA fine-tune (~1h)
3. GaussianEditor chỉnh sửa 3DGS (~30 phút)

**Pipeline mới:**
1. Fashn-VTON ghép áo vào 1 ảnh tĩnh (20 bước Diffusion, ~15s)
2. Rembg tách nền → RGBA PNG (~5s)
3. TRELLIS từ 1 ảnh RGBA → 3D mesh+Gaussian hoàn chỉnh (~40s)

### Tối ưu hóa cho GPU T4 x2 trên Kaggle:
- **FP16 Tensor Cores**: T4 đạt ~65 TFLOPs FP16, nhanh gấp 2-3x so với FP32
- **spconv-cu121**: Sparse 3D Convolution được tối ưu JIT compile với CPATH tự động
- **xFormers attention**: Thay Flash Attention (không hỗ trợ T4 sm_75) bằng xFormers
- **cupy mock**: Bypass lỗi ABI của CuPy trên Kaggle để rembg/pymatting chạy ổn định

---

## 📝 Giấy phép

Dự án này sử dụng mã nguồn mở từ nhiều tác giả. Vui lòng tôn trọng giấy phép gốc:
- **GS-VTON**: [arXiv 2410.05259](https://arxiv.org/abs/2410.05259) — Yukang Cao et al. (NTU)
- **Microsoft TRELLIS**: MIT License — Microsoft Research
- **Fashn-VTON**: Xem tại [fashn-AI/fashn-vton-1.5](https://github.com/fashn-AI/fashn-vton-1.5)

Phần tích hợp và tối ưu hóa pipeline trong repo này: **MIT License**.
