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


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ATPP 3D Fashion Try-On AI",
        "version": "2.0.0",
        "docs_url": "/docs",
    }


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
