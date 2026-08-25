#!/usr/bin/env python3
"""
================================================================================
AI VIRTUAL TRY-ON 3D — FASTAPI SERVER (api_server.py)
Cung cấp REST API công khai để kết nối trực tiếp với website đồ án ATPP.

Endpoints:
  - GET  /health              → Kiểm tra trạng thái server & GPU
  - POST /api/v1/tryon-3d     → Nhận 2 ảnh (người mẫu + áo) → Trả về file 3D (.glb)
  - POST /api/v1/tryon-3d/json → Nhận 2 ảnh → Trả về JSON (gồm Base64 .glb + ảnh 2D)
  - GET  /docs                → Giao diện Swagger UI test API trực quan
================================================================================
"""

import os
import sys
import uuid
import shutil
import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Khởi tạo FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="ATPP AI Virtual Try-On 3D API",
    description="Backend API chuyển đổi ảnh thời trang 2D thành mô hình 3D Áo Dài hoàn chỉnh",
    version="2.0.0",
)

# Cho phép Website ATPP gọi API từ bất kỳ domain nào (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import pipeline từ module chính
from run_integrated_3d_pipeline import (
    run_step1_fashn_tryon,
    run_step2_remove_bg,
    run_step3_trellis_3d,
)


from fastapi.responses import HTMLResponse

HTML_UI = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATPP AI 3D Virtual Try-On Studio</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #ec4899;
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.15) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            background: rgba(15, 23, 42, 0.6);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .logo {
            font-size: 22px;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .badge {
            font-size: 11px;
            padding: 4px 10px;
            background: rgba(99, 102, 241, 0.2);
            border: 1px solid rgba(99, 102, 241, 0.4);
            color: #c7d2fe;
            border-radius: 20px;
            font-weight: 600;
        }

        .main-container {
            flex: 1;
            max-width: 1300px;
            width: 100%;
            margin: 0 auto;
            padding: 30px 20px;
            display: grid;
            grid-template-columns: 460px 1fr;
            gap: 30px;
        }

        @media (max-width: 1024px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 26px;
            backdrop-filter: blur(16px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #f1f5f9;
        }

        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }

        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: rgba(15, 23, 42, 0.4);
            position: relative;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .dropzone:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
            transform: translateY(-2px);
        }

        .dropzone.has-file {
            border-style: solid;
            border-color: rgba(99, 102, 241, 0.5);
        }

        .dropzone img.preview {
            width: 100%;
            height: 100%;
            object-fit: cover;
            position: absolute;
            top: 0;
            left: 0;
            border-radius: 12px;
        }

        .drop-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .drop-label {
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
        }

        .drop-hint {
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 4px;
        }

        .form-group {
            margin-bottom: 18px;
        }

        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 8px;
        }

        select, input[type="range"] {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
            color: var(--text-main);
            font-size: 14px;
            outline: none;
            transition: all 0.2s;
        }

        select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        .btn-generate {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-generate:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(236, 72, 153, 0.5);
        }

        .btn-generate:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            filter: grayscale(0.5);
        }

        /* 3D Viewer Container */
        .viewer-container {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 520px;
            position: relative;
        }

        model-viewer {
            width: 100%;
            height: 100%;
            flex: 1;
            background: radial-gradient(circle at 50% 50%, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
            border-radius: 16px;
            --poster-color: transparent;
        }

        .placeholder-viewer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 16px;
            border: 1px dashed rgba(255, 255, 255, 0.1);
            text-align: center;
            padding: 20px;
        }

        .placeholder-icon {
            font-size: 48px;
            margin-bottom: 12px;
            opacity: 0.8;
        }

        /* Progress & Loader */
        .progress-box {
            display: none;
            margin-top: 16px;
            padding: 14px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 10px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .progress-bar-bg {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-bar-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            transition: width 0.4s ease;
        }

        .status-text {
            font-size: 13px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .action-bar {
            margin-top: 16px;
            display: flex;
            gap: 12px;
        }

        .btn-action {
            flex: 1;
            padding: 10px 16px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #f8fafc;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            text-decoration: none;
        }

        .btn-action:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .btn-download {
            background: rgba(34, 197, 94, 0.2);
            border-color: rgba(34, 197, 94, 0.4);
            color: #86efac;
        }

        .btn-download:hover {
            background: rgba(34, 197, 94, 0.3);
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <span>🎽 ATPP 3D VIRTUAL TRY-ON</span>
            <span class="badge">GPU AI v2.0</span>
        </div>
        <div>
            <a href="/docs" target="_blank" style="color: var(--text-sub); text-decoration: none; font-size: 14px; font-weight: 600;">📖 API Docs</a>
        </div>
    </header>

    <div class="main-container">
        <!-- Panel Điều Khiển & Upload -->
        <div class="card">
            <div class="section-title">
                <span>📸</span> Tải Lên Ảnh Đầu Vào
            </div>

            <div class="upload-grid">
                <!-- Dropzone Người Mẫu -->
                <div class="dropzone" id="personDropzone" onclick="document.getElementById('personInput').click()">
                    <input type="file" id="personInput" accept="image/*" style="display: none;" onchange="handleFile(this, 'person')">
                    <div id="personEmpty">
                        <div class="drop-icon">👤</div>
                        <div class="drop-label">Ảnh Người Mẫu</div>
                        <div class="drop-hint">Chụp toàn thân/nửa người</div>
                    </div>
                    <img id="personPreview" class="preview" style="display: none;">
                </div>

                <!-- Dropzone Trang Phục -->
                <div class="dropzone" id="clothDropzone" onclick="document.getElementById('clothInput').click()">
                    <input type="file" id="clothInput" accept="image/*" style="display: none;" onchange="handleFile(this, 'cloth')">
                    <div id="clothEmpty">
                        <div class="drop-icon">👗</div>
                        <div class="drop-label">Ảnh Trang Phục</div>
                        <div class="drop-hint">Áo Dài hoặc quần áo</div>
                    </div>
                    <img id="clothPreview" class="preview" style="display: none;">
                </div>
            </div>

            <div class="form-group">
                <label for="categorySelect">Loại Trang Phục (Category)</label>
                <select id="categorySelect">
                    <option value="one-pieces" selected>👗 Áo Dài / Đầm Nguyên Bộ (One-pieces)</option>
                    <option value="tops">👕 Áo ngắn / Áo sơ mi (Tops)</option>
                    <option value="bottoms">👖 Quần / Váy ngắn (Bottoms)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="qualitySelect">Chế Độ Tạo 3D (Quality & Speed)</label>
                <select id="qualitySelect">
                    <option value="turbo" selected>⚡ Turbo Mode (~15-20s, Khuyên dùng)</option>
                    <option value="hd">💎 High Detail Mode (~30-40s, Chi tiết cao)</option>
                </select>
            </div>

            <button class="btn-generate" id="generateBtn" onclick="startTryOn()">
                <span>🚀 Thử Đồ & Sinh Mô Hình 3D</span>
            </button>

            <!-- Tiến trình xử lý -->
            <div class="progress-box" id="progressBox">
                <div class="status-text">
                    <span id="statusLabel">Đang kết nối tới GPU...</span>
                    <span id="percentLabel">0%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" id="progressBar"></div>
                </div>
            </div>
        </div>

        <!-- Panel 3D Viewer Trực Quan -->
        <div class="card viewer-container">
            <div class="section-title" style="justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span>🌐</span> Mô Hình 3D Interactive
                </div>
                <span class="badge" id="renderTimeBadge" style="display: none;">18.5s</span>
            </div>

            <div style="position: relative; flex: 1; display: flex;">
                <model-viewer id="modelViewer" 
                    camera-controls 
                    touch-action="pan-y" 
                    auto-rotate 
                    shadow-intensity="1.5" 
                    exposure="1" 
                    loading="eager"
                    ar
                    style="display: none;">
                </model-viewer>

                <div class="placeholder-viewer" id="viewerPlaceholder">
                    <div class="placeholder-icon">👗✨</div>
                    <h3 style="font-size: 16px; margin-bottom: 6px;">Chưa Có Mô Hình 3D</h3>
                    <p style="font-size: 13px; color: var(--text-sub); max-width: 320px;">
                        Tải lên ảnh người mẫu và trang phục ở bên trái, sau đó bấm <b>"Thử Đồ & Sinh Mô Hình 3D"</b> để xem kết quả.
                    </p>
                </div>
            </div>

            <div class="action-bar" id="actionBar" style="display: none;">
                <button class="btn-action" onclick="toggleRotate()">🔄 Bật/Tắt Xoay</button>
                <button class="btn-action" onclick="resetCamera()">🎯 Đặt Lại Góc Nhìn</button>
                <a id="downloadGlbBtn" class="btn-action btn-download" download="AODAI_3D_MODEL.glb">
                    ⬇️ Tải File .GLB (3D)
                </a>
            </div>
        </div>
    </div>

    <script>
        let personFile = null;
        let clothFile = null;

        function handleFile(input, type) {
            if (input.files && input.files[0]) {
                const file = input.files[0];
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (type === 'person') {
                        personFile = file;
                        document.getElementById('personPreview').src = e.target.result;
                        document.getElementById('personPreview').style.display = 'block';
                        document.getElementById('personEmpty').style.display = 'none';
                        document.getElementById('personDropzone').classList.add('has-file');
                    } else {
                        clothFile = file;
                        document.getElementById('clothPreview').src = e.target.result;
                        document.getElementById('clothPreview').style.display = 'block';
                        document.getElementById('clothEmpty').style.display = 'none';
                        document.getElementById('clothDropzone').classList.add('has-file');
                    }
                }
                reader.readAsDataURL(file);
            }
        }

        async function startTryOn() {
            if (!personFile || !clothFile) {
                alert('Vui lòng tải lên đầy đủ cả 2 ảnh: Người mẫu và Trang phục!');
                return;
            }

            const btn = document.getElementById('generateBtn');
            const progressBox = document.getElementById('progressBox');
            const statusLabel = document.getElementById('statusLabel');
            const percentLabel = document.getElementById('percentLabel');
            const progressBar = document.getElementById('progressBar');

            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div> Đang xử lý AI trên GPU...';
            progressBox.style.display = 'block';

            // Cập nhật tiến độ giả lập trực quan
            let progress = 10;
            progressBar.style.width = '10%';
            statusLabel.innerText = 'Bước 1/3: Fashn-VTON ghép áo vào người mẫu...';
            percentLabel.innerText = '15%';

            const timer = setInterval(() => {
                if (progress < 45) {
                    progress += 5;
                    statusLabel.innerText = 'Bước 1/3: Đang chạy 2D Diffusion...';
                } else if (progress < 75) {
                    progress += 3;
                    statusLabel.innerText = 'Bước 2/3: Rembg tách nền trong suốt RGBA...';
                } else if (progress < 92) {
                    progress += 2;
                    statusLabel.innerText = 'Bước 3/3: TRELLIS sinh mô hình 3D Mesh...';
                }
                progressBar.style.width = progress + '%';
                percentLabel.innerText = progress + '%';
            }, 600);

            const formData = new FormData();
            formData.append('person_image', personFile);
            formData.append('cloth_image', clothFile);
            formData.append('category', document.getElementById('categorySelect').value);

            const quality = document.getElementById('qualitySelect').value;
            if (quality === 'turbo') {
                formData.append('sparse_steps', '8');
                formData.append('slat_steps', '8');
            } else {
                formData.append('sparse_steps', '12');
                formData.append('slat_steps', '12');
            }

            const startTime = performance.now();

            try {
                const response = await fetch('/api/v1/tryon-3d', {
                    method: 'POST',
                    body: formData,
                });

                clearInterval(timer);

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Lỗi server');
                }

                progressBar.style.width = '100%';
                percentLabel.innerText = '100%';
                statusLabel.innerText = '✅ Hoàn tất! Đang nạp mô hình 3D...';

                const blob = await response.blob();
                const modelUrl = URL.createObjectURL(blob);

                const viewer = document.getElementById('modelViewer');
                viewer.src = modelUrl;
                viewer.style.display = 'block';
                document.getElementById('viewerPlaceholder').style.display = 'none';
                document.getElementById('actionBar').style.display = 'flex';

                const downloadBtn = document.getElementById('downloadGlbBtn');
                downloadBtn.href = modelUrl;

                const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
                const badge = document.getElementById('renderTimeBadge');
                badge.innerText = `⏱️ ${elapsed}s`;
                badge.style.display = 'inline-block';

            } catch (error) {
                clearInterval(timer);
                alert('Lỗi khi sinh 3D: ' + error.message);
                statusLabel.innerText = '❌ Có lỗi xảy ra!';
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>🚀 Thử Đồ & Sinh Mô Hình 3D</span>';
            }
        }

        function toggleRotate() {
            const viewer = document.getElementById('modelViewer');
            viewer.autoRotate = !viewer.autoRotate;
        }

        function resetCamera() {
            const viewer = document.getElementById('modelViewer');
            viewer.cameraOrbit = '0deg 75deg 105%';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root_ui():
    """Trả về giao diện web studio 3D ảo trực tiếp trên trình duyệt"""
    return HTMLResponse(content=HTML_UI)


@app.get("/health")
def health_check():
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
    }


@app.post("/api/v1/tryon-3d")
async def generate_3d_file(
    person_image: UploadFile = File(..., description="Ảnh người mẫu toàn thân hoặc nửa người"),
    cloth_image: UploadFile = File(..., description="Ảnh trang phục / Áo Dài"),
    category: str = Form("one-pieces", description="Loại trang phục: tops | bottoms | one-pieces"),
    sparse_steps: int = Form(8, description="Số bước Sparse Diffusion (khuyên dùng 8-12)"),
    slat_steps: int = Form(8, description="Số bước SLAT Diffusion (khuyên dùng 8-12)"),
    seed: int = Form(1, description="Seed ngẫu nhiên"),
):
    """
    Nhận 2 ảnh và trả về trực tiếp file 3D binary (.glb) để nhúng vào thẻ <model-viewer>
    """
    job_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(tempfile.gettempdir(), f"tryon_job_{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        # 1. Lưu ảnh upload vào thư mục tạm
        person_path = os.path.join(work_dir, f"person_{person_image.filename}")
        cloth_path = os.path.join(work_dir, f"cloth_{cloth_image.filename}")

        with open(person_path, "wb") as f:
            f.write(await person_image.read())
        with open(cloth_path, "wb") as f:
            f.write(await cloth_image.read())

        # 2. Bước 1: Fashn-VTON
        tryon_path = run_step1_fashn_tryon(
            person_image_path=person_path,
            cloth_image_path=cloth_path,
            output_dir=work_dir,
            category=category,
            num_timesteps=12,
        )

        # 3. Bước 2: Tách nền
        rgba_path = run_step2_remove_bg(
            tryon_image_path=tryon_path,
            output_dir=work_dir,
        )

        # 4. Bước 3: TRELLIS 3D
        res_3d = run_step3_trellis_3d(
            rgba_image_path=rgba_path,
            output_dir=work_dir,
            seed=seed,
            sparse_steps=sparse_steps,
            slat_steps=slat_steps,
        )

        glb_path = res_3d.get("glb")
        if not glb_path or not os.path.exists(glb_path):
            raise HTTPException(status_code=500, detail="Không thể tạo file 3D .glb")

        return FileResponse(
            path=glb_path,
            media_type="model/gltf-binary",
            filename="tryon_model.glb",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")


@app.post("/api/v1/tryon-3d/json")
async def generate_3d_json(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    category: str = Form("one-pieces"),
    sparse_steps: int = Form(8),
    slat_steps: int = Form(8),
    seed: int = Form(1),
):
    """
    Nhận 2 ảnh và trả về JSON chứa Base64 của:
      - File 3D (.glb)
      - Ảnh 2D try-on kết quả
      - Ảnh RGBA đã tách nền
    """
    job_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(tempfile.gettempdir(), f"tryon_job_{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        person_path = os.path.join(work_dir, f"person_{person_image.filename}")
        cloth_path = os.path.join(work_dir, f"cloth_{cloth_image.filename}")

        with open(person_path, "wb") as f:
            f.write(await person_image.read())
        with open(cloth_path, "wb") as f:
            f.write(await cloth_image.read())

        tryon_path = run_step1_fashn_tryon(
            person_image_path=person_path,
            cloth_image_path=cloth_path,
            output_dir=work_dir,
            category=category,
            num_timesteps=12,
        )

        rgba_path = run_step2_remove_bg(
            tryon_image_path=tryon_path,
            output_dir=work_dir,
        )

        res_3d = run_step3_trellis_3d(
            rgba_image_path=rgba_path,
            output_dir=work_dir,
            seed=seed,
            sparse_steps=sparse_steps,
            slat_steps=slat_steps,
        )

        glb_path = res_3d.get("glb")
        if not glb_path or not os.path.exists(glb_path):
            raise HTTPException(status_code=500, detail="Không thể tạo file 3D .glb")

        with open(glb_path, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode()

        with open(tryon_path, "rb") as f:
            tryon_b64 = base64.b64encode(f.read()).decode()

        return {
            "status": "success",
            "job_id": job_id,
            "glb_base64": glb_b64,
            "preview_2d_base64": f"data:image/png;base64,{tryon_b64}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")
    finally:
        # Dọn dẹp bộ nhớ tạm sau khi hoàn tất
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Hàm chạy Server với Cloudflare Tunnel ──────────────────────────────────────
def start_server_with_tunnel(port: int = 8000):
    import subprocess
    import threading
    import time
    import re

    print("=" * 70)
    print("🚀 ĐANG KHỞI CHẠY ATPP AI TRY-ON 3D API SERVER...")
    print("=" * 70)

    # 1. Tải binary cloudflared nếu chưa có
    cloudflared_path = "/kaggle/working/cloudflared"
    if not os.path.exists(cloudflared_path):
        print("⬇️ Đang tải Cloudflare Tunnel (miễn phí, bảo mật HTTPS)...")
        os.system(f"wget -q -O {cloudflared_path} https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
        os.system(f"chmod +x {cloudflared_path}")

    # 2. Khởi động Cloudflare Tunnel ở background thread
    def run_tunnel():
        process = subprocess.Popen(
            [cloudflared_path, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in process.stderr:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                tunnel_url = match.group(0)
                print("\n" + "=" * 70)
                print("🎉 SERVER ĐÃ SẴN SÀNG KẾT NỐI VỚI WEBSITE ATPP!")
                print(f"🔗 Public API URL:  {tunnel_url}")
                print(f"📖 Swagger API Docs: {tunnel_url}/docs")
                print("=" * 70 + "\n")

    threading.Thread(target=run_tunnel, daemon=True).start()

    # 3. Chạy Uvicorn Server
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start_server_with_tunnel(port=8000)
