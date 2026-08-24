<div align="center">

# 👗 AI Virtual Try-On 3D (Fashion-VTON + Microsoft TRELLIS.2)

### High-Fidelity 3D Virtual Try-On from a Single Photo

**Fast 2-Step Architecture: 2D Virtual Try-On + 3D Generative Synthesis**

---

</div>

## 🌟 Overview & Architecture

Hệ thống này thay thế kiến trúc GS-VTON cũ (COLMAP + LoRA + 3DGS Training kéo dài 2-4 tiếng) bằng quy trình **2D-to-3D Decoupled Pipeline** thế hệ mới:

```
[Ảnh Người mẫu] + [Ảnh Áo Dài/Quần áo]
           │
           ▼
  [BƯỚC 1: Fashn-VTON]          ──> Ghép quần áo 2D siêu nét (~10-15s)
           │
           ▼
  [BƯỚC 2: Rembg / BiRefNet]    ──> Tách nền trong suốt RGBA (~5s)
           │
           ▼
  [BƯỚC 3: Microsoft TRELLIS.2] ──> Sinh vật thể 3D hoàn chỉnh (~20-40s)
           │
           ├──> 📦 output.glb  (Textured Mesh - Web / Unity / Blender)
           ├──> ✨ output.ply  (3D Gaussian Splatting)
           └──> 🎬 360deg.mp4  (Video xoay 360°)
```

### ⚡ So sánh hiệu năng:

| Tiêu chí | GS-VTON cũ | Pipeline mới (Fashion + TRELLIS.2) |
|:---|:---|:---|
| **Đầu vào** | Video quay vòng 360° + ảnh áo | **1 ảnh người mẫu duy nhất** + ảnh áo |
| **Thời gian chạy** | 2 - 4 tiếng (Training nặng) | **~1 - 2 phút (Inference only)** |
| **Cài đặt & Build** | Cực kỳ phức tạp (CUDA extensions, COLMAP) | Đơn giản, đóng gói sẵn trên Linux/Kaggle |
| **Đầu ra** | Chỉ có video render 3DGS | **.glb (Mesh PBR) + .ply (3DGS) + Video MP4** |
| **Khả năng thương mại**| Khó tích hợp | **Dễ dàng làm REST API / Web App** |

---

## 🚀 Hướng dẫn cài đặt và chạy trên Kaggle

Kaggle cung cấp GPU miễn phí (T4 / P100 16GB VRAM) hoàn toàn phù hợp để chạy pipeline này.

### 1. Cài đặt môi trường (Chạy trong Kaggle Notebook cell):

```bash
# 1. Clone và cài đặt TRELLIS
!git clone https://github.com/microsoft/TRELLIS /kaggle/working/TRELLIS
%cd /kaggle/working/TRELLIS
!bash setup.sh
!pip install -e .
%cd /kaggle/working

# 2. Cài đặt các thư viện cần thiết
!pip install rembg[gpu] imageio imageio-ffmpeg trimesh pygltflib plyfile
!pip install git+https://github.com/fashn-AI/fashn-vton-1.5.git
```

### 2. Chạy Pipeline:

```bash
# Chạy trực tiếp từ ảnh đầu vào
python run_integrated_3d_pipeline.py \
    --person_image /kaggle/input/your-dataset/model.jpg \
    --cloth_image  /kaggle/input/your-dataset/aodai.jpg \
    --output_dir   /kaggle/working/output_trellis \
    --category     tops
```

### 3. Tùy chọn nâng cao:

```bash
# Nếu bạn đã có sẵn ảnh ghép 2D (muốn bỏ qua Bước 1 để test nhanh 3D):
python run_integrated_3d_pipeline.py \
    --skip_tryon \
    --tryon_image /kaggle/working/output_trellis/step1_tryon_2d.png \
    --output_dir  /kaggle/working/output_trellis
```

---

## 📦 Định dạng đầu ra

Sau khi chạy xong, thư mục `--output_dir` sẽ chứa:
1. `step1_tryon_2d.png`: Ảnh 2D người mẫu sau khi ghép trang phục.
2. `step2_rgba.png`: Ảnh người mẫu đã tách sạch phông nền.
3. `output.glb`: Mô hình 3D Textured Mesh tiêu chuẩn (mở được ngay trên Windows 3D Viewer, Blender, hoặc nhúng Web bằng Three.js / `<model-viewer>`).
4. `output.ply`: File 3D Gaussian Splatting 360 độ.
5. `360deg_AODAI_3D.mp4`: Video xoay vòng 360 độ độ nét cao.

---

## 💡 Acknowledgements
- [Fashn-VTON](https://github.com/fashn-AI/fashn-vton-1.5)
- [Microsoft TRELLIS & TRELLIS.2](https://github.com/microsoft/TRELLIS)
- [Rembg](https://github.com/danielgatis/rembg)
