# cut_paper.py
import cv2
import numpy as np
import os

def get_pixel_per_cm_from_a4(image_path, real_width_cm=29.7, show_debug=False):
    # ★ 修正：支援中文路徑讀取
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = None

    if img is None:
        print(f"[CutPaper] 讀取失敗: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[CutPaper] 找不到輪廓")
        return None

    a4_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(a4_contour, True)
    approx = cv2.approxPolyDP(a4_contour, epsilon, True)

    if len(approx) != 4:
        print("[CutPaper] 無法偵測 A4 四邊形")
        # 如果找不到四邊形，回傳原圖，避免程式掛掉
        return img 

    # 整理四個角點
    pts = approx.reshape(4, 2).astype(np.float32)
    # 排序：先依 x 排序，再依 y 排序區分左右
    pts = sorted(pts, key=lambda p: p[0]) 
    left = sorted(pts[0:2], key=lambda p: p[1])
    right = sorted(pts[2:4], key=lambda p: p[1])
    tl, bl = left
    tr, br = right

    # 定義目標尺寸（標準化為A4比例 297mm x 210mm -> 轉像素）
    # 這裡設定高解析度一點以便後續畫圖
    target_width = 842 
    target_height = 595 

    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, transform_matrix, (target_width, target_height))

    return warped