# correct.py
import cv2
import numpy as np
import os

def analyze_image(img_input, dot_distance_cm=10.0, show_windows=False):
    """
    接收圖片路徑或 numpy array，回傳 (score, result_img)
    """
    # 1. 讀取圖片 (支援中文路徑)
    if isinstance(img_input, str):
        try:
            img = cv2.imdecode(np.fromfile(img_input, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            img = None
    else:
        img = img_input

    if img is None:
        return 0, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_disp = img.copy()

    # === 小工具：偵測圓點 ===
    def detect_dots(gray_img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray_img)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 簡單過濾
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000: # 面積篩選
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
        
        # 如果找到多於2個點，取距離最遠的兩個
        if len(dots) > 2:
            max_d = 0
            pair = (dots[0], dots[1])
            for i in range(len(dots)):
                for j in range(i+1, len(dots)):
                    d = np.hypot(dots[i][0]-dots[j][0], dots[i][1]-dots[j][1])
                    if d > max_d:
                        max_d = d
                        pair = (dots[i], dots[j])
            return list(pair)
        
        return dots

    # === 偵測兩點 ===
    dots = detect_dots(gray)
    
    if len(dots) != 2:
        cv2.putText(img_disp, f"Error: Found {len(dots)} dots (Need 2)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return 0, img_disp

    (x1, y1), (x2, y2) = dots
    
    # === 計算比例尺 ===
    dist_px = np.hypot(x1 - x2, y1 - y2)
    px_per_cm = dist_px / dot_distance_cm
    
    # === 建立遮罩：只看兩點連線附近 ===
    mask = np.zeros_like(gray)
    cv2.line(mask, (x1, y1), (x2, y2), 255, int(px_per_cm * 1.5)) # 寬度約 1.5cm
    
    # === 找出連線 (自適應二值化) ===
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    # 去除圓點本身 (避免干擾)
    cv2.circle(bw, (x1, y1), int(px_per_cm * 0.4), 0, -1)
    cv2.circle(bw, (x2, y2), int(px_per_cm * 0.4), 0, -1)
    
    # 只保留遮罩內的線
    bw = cv2.bitwise_and(bw, mask)
    
    # === 分析連通性 ===
    # 檢查是否有像素連接這兩點的區域
    # 這裡簡化邏輯：計算兩點連線上的像素覆蓋率
    line_mask = np.zeros_like(gray)
    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3) # 理想直線
    
    overlap = cv2.bitwise_and(bw, line_mask)
    ideal_pixels = cv2.countNonZero(line_mask)
    actual_pixels = cv2.countNonZero(overlap)
    
    connectivity = actual_pixels / ideal_pixels if ideal_pixels > 0 else 0
    
    # === 計算最大偏差 ===
    # 找出所有黑色像素點
    ys, xs = np.where(bw > 0)
    max_dev_cm = 0
    
    if len(xs) > 0:
        # 直線方程式 Ax + By + C = 0
        # (y1-y2)x + (x2-x1)y + x1y2 - x2y1 = 0
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        denom = np.sqrt(A*A + B*B)
        
        if denom > 0:
            distances = np.abs(A * xs + B * ys + C) / denom
            max_dev_px = np.max(distances)
            max_dev_cm = max_dev_px / px_per_cm

    # === 評分邏輯 ===
    score = 0
    msg = ""
    
    if connectivity < 0.5:
        score = 0
        msg = "未連接 (Not Connected)"
    elif max_dev_cm > 1.25:
        score = 0
        msg = "偏差過大 (>1.25cm)"
    elif max_dev_cm > 0.65:
        score = 1
        msg = "偏差稍大 (0.65~1.25cm)"
    else:
        score = 2
        msg = "連接良好 (<=0.65cm)"

    # === 繪圖 ===
    # 畫出兩點
    cv2.circle(img_disp, (x1, y1), 10, (0, 0, 255), -1)
    cv2.circle(img_disp, (x2, y2), 10, (0, 0, 255), -1)
    # 畫出理想直線
    cv2.line(img_disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # 寫上結果
    cv2.putText(img_disp, f"Score: {score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_disp, f"Dev: {max_dev_cm:.2f}cm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img_disp, msg, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return score, img_disp