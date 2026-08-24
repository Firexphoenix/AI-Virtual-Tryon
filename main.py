"""
================================================================================
FASHION-VTON + TRELLIS.2 — WRAPPER CLI (main.py)
Đây là điểm khởi chạy chính của pipeline.

Cách sử dụng:
  python main.py --person_image model.jpg --cloth_image aodai.jpg

Hoặc để pipeline tự tìm ảnh trong thư mục DATA/:
  python main.py

Xem thêm tham số:
  python main.py --help
================================================================================
"""

import sys
import os

# Chuyển hướng sang pipeline chính
if __name__ == "__main__":
    # Chạy trực tiếp run_integrated_3d_pipeline.py với tất cả args được truyền vào
    from run_integrated_3d_pipeline import main
    main()
