import cv2
import numpy as np
from ultralytics import YOLO

# 讀取 YOLO segmentation 模型
model = YOLO(r'model\toybrick.pt')
CONF = 0.8

# 攝影機初始化
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

CROP_RATIO = 0.5  # 中央區域比例

print("按下 's' 拍照並檢測，按下 'q' 退出")

def analyze_image(frame):
    """分析影像並返回結果"""
    H, W, _ = frame.shape
    crop_w, crop_h = int(W * CROP_RATIO), int(H * CROP_RATIO)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2, y2 = x1 + crop_w, y1 + crop_h

    # 裁切中央區域
    cropped = frame[y1:y2, x1:x2]
    results = model.predict(source=cropped, conf=CONF, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    centers = []
    angles = []
    max_mask_side = 0
    rotate_ok_list = []
    
    for mask in masks:
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            max_mask_side = max(max_mask_side, max(w, h))

            mask_H, mask_W = binary_mask.shape
            scale_x = crop_w / mask_W
            scale_y = crop_h / mask_H

            # 中心點位置
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] * scale_x) 
                cy = int(M["m01"] / M["m00"] * scale_y) 
                centers.append((cx, cy))
                cv2.circle(cropped, (cx, cy), 5, (0, 0, 0), -1)

            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box[:, 0] = box[:, 0] * scale_x 
                box[:, 1] = box[:, 1] * scale_y 
                box = np.intp(box)

                # === 找出主邊方向（最長的邊）===
                max_len = -1
                main_angle = 0
                for i in range(4):
                    pt1 = box[i]
                    pt2 = box[(i + 1) % 4]
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    length = dx**2 + dy**2
                    angle = np.arctan2(dy, dx) * 180 / np.pi

                    if length > max_len:
                        max_len = length
                        main_angle = angle

                main_angle = abs(main_angle)
            
                # === 比對是否「近似水平」或「近似垂直」 ===
                angle_diff_to_horizontal = abs(main_angle)   # 跟 0 度比
                angle_diff_to_vertical = abs(main_angle - 90) # 跟 90 度比

                rotate_ok = (angle_diff_to_horizontal <= 10 or angle_diff_to_vertical <= 10)
                rotate_ok_list.append(rotate_ok)

                # === 畫框、標角度 ===
                color = (0, 255, 0) if rotate_ok else (0, 0, 255)  # OK = 綠，NG = 紅
                cv2.drawContours(cropped, [box], 0, color, 2)

    # === 判斷邏輯 ===
    offset = False
    if len(centers) >= 2 and max_mask_side > 0:
        threshold = max_mask_side // 8

        x_vals = [pt[0] for pt in centers]
        y_vals = [pt[1] for pt in centers]
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)

        offset = (std_x < threshold or std_y < threshold)

        cv2.putText(cropped, f"std_x = {std_x:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(cropped, f"std_y = {std_y:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(cropped, f"threshold = {threshold:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # 顯示總體狀態
    status_rotate = "?"
    if rotate_ok_list:
        status_rotate = "No Rotate" if all(rotate_ok_list) else "Rotate !"

    status_offset = "No Offset" if offset else "Offset !"
    summary = f"{status_offset} | {status_rotate}"

    if status_offset == 'Offset !' or status_rotate == 'Rotate !':
        color = (0, 0, 255)
    else:
        color = (0, 0, 0)
    
    cv2.putText(cropped, summary, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    return cropped, summary

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 顯示原始預覽畫面（未檢測）
    H, W, _ = frame.shape
    crop_w, crop_h = int(W * CROP_RATIO), int(H * CROP_RATIO)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2, y2 = x1 + crop_w, y1 + crop_h
    
    # 顯示裁切區域
    preview_frame = frame[y1:y2, x1:x2].copy()
    
    # 在預覽畫面上加上提示文字
    cv2.putText(preview_frame, "Press 's' to capture & detect", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(preview_frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Camera Preview - Press 's' to detect", preview_frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # 按下 's' 拍照並檢測
        print("拍照並檢測中...")
        
        # 拍照並分析
        captured_frame = frame.copy()
        analyzed_frame, summary = analyze_image(captured_frame)
        
        # 顯示分析結果
        cv2.imshow("Detection Result", analyzed_frame)
        
        print(f"檢測結果: {summary}")
        print("按下任意鍵返回預覽模式...")
        
        # 等待用戶按鍵後返回預覽模式
        cv2.waitKey(0)
        cv2.destroyWindow("Detection Result")
        
    elif key == ord('q'):  # 按下 'q' 退出
        break

cap.release()
cv2.destroyAllWindows()
print("程式結束")