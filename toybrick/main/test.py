import cv2
import numpy as np
from ultralytics import YOLO

# ================== YOLO 模型 ==================
model = YOLO(r'model\toybrick.pt')
CONF = 0.5

# ================== YOLO 偵測方塊 & 取得 mask ==================
def detect_blocks_mask(frame, CONF=0.5):
    results = model.predict(source=frame, conf=CONF, verbose=False)
    boxes = []
    masks = []

    for r in results:
        if r.boxes is None:
            continue
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == 0:  # 方塊類別
                boxes.append((x1, y1, x2, y2))
                if r.masks is not None:
                    mask = r.masks.data.cpu().numpy()[i]
                    masks.append(mask)
    return boxes, masks, results

# ================== 遮掉方塊 ==================
def remove_blocks_with_mask(binary, masks, extra_px=10):
    h, w = binary.shape
    for mask in masks:
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 膨脹 mask，增加遮擋範圍
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_px*2, extra_px*2))
        mask_dilated = cv2.dilate((mask_resized > 0).astype(np.uint8), kernel)

        binary[mask_dilated > 0] = 0
    return binary

# ================== 骨架化 ==================
def extract_line_skeleton(binary):
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = np.copy(binary)
    while True:
        open_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
        temp2 = cv2.subtract(temp, open_img)
        eroded = cv2.erode(temp, element)
        skeleton = cv2.bitwise_or(skeleton, temp2)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton

# ================== 畫骨架紅點 ==================
def draw_skeleton_points(skeleton, frame):
    points = np.column_stack(np.where(skeleton > 0))
    for (y, x) in points:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # 紅色小點
    return frame

# ================== 判斷是否在骨架線點附近 ==================
def is_mask_near_skeleton(mask, skeleton, tol=5):
    """
    mask: 二值 mask (0/1 或 0/255)
    skeleton: 骨架二值圖
    tol: 搜尋半徑
    """
    # resize mask 到骨架尺寸
    mask_resized = cv2.resize(mask, (skeleton.shape[1], skeleton.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 取得 mask 非零座標
    ys, xs = np.where(mask_resized > 0)
    
    h, w = skeleton.shape
    for x, y in zip(xs, ys):
        x_start = max(0, x - tol)
        x_end   = min(w, x + tol + 1)
        y_start = max(0, y - tol)
        y_end   = min(h, y + tol + 1)
        if np.any(skeleton[y_start:y_end, x_start:x_end] > 0):
            return True  # 任意一點靠近骨架就算
    return False

# ================== 主程式 ==================
# 開啟攝影機
cap = cv2.VideoCapture(1)  

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

print("按下 's' 鍵拍照並分析")
print("按下 'q' 鍵退出")

while True:
    # 讀取影像
    ret, frame = cap.read()
    
    if not ret:
        print("無法讀取影像")
        break
    
    # 顯示即時影像
    cv2.imshow('Live Video - Press "s" to capture, "q" to quit', frame)
    
    # 等待按鍵
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("拍照中...")
        
        # 複製當前幅面進行分析
        img = frame.copy()
        display_frame = img.copy()

        # 灰階 + 二值化
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21,21), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

        # 閉運算去雜點
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # YOLO 偵測方塊 & 取得 mask
        boxes, masks, results = detect_blocks_mask(display_frame)

        # 畫方塊邊框與中心點
        annotated = display_frame.copy()
        correct_num = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if masks:
                mask = masks[i]
                mask_resized = cv2.resize(mask, (display_frame.shape[1], display_frame.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                ys, xs = np.where(mask_resized > CONF)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    cv2.circle(annotated, (cx, cy), 3, (0, 0, 0), -1)
                    # cv2.putText(annotated, f"({cx},{cy})", (cx, cy + 15),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # 遮掉方塊
        binary = remove_blocks_with_mask(binary, masks)

        # 骨架化
        skeleton = extract_line_skeleton(binary)

        # 畫骨架紅點
        final_frame = draw_skeleton_points(skeleton, annotated)

        # 檢查每個方塊是否靠近骨架
        for i, mask in enumerate(masks):
            if is_mask_near_skeleton(mask, skeleton, tol=10):
                correct_num += 1
                print(f"Box {i} is near skeleton")
            else:
                print(f"Box {i} is NOT near skeleton")

        # 添加文字資訊
        cv2.putText(final_frame, f"Cube count : {len(boxes)}", (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(final_frame, f"Correct Cube count : {correct_num}", (300, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 縮放顯示
        binary_display = cv2.resize(binary, (0, 0), fx=0.4, fy=0.4)
        blurred_display = cv2.resize(blurred, (0, 0), fx=0.4, fy=0.4)
        final_frame_display = cv2.resize(final_frame, (0, 0), fx=0.4, fy=0.4)

        # 顯示分析結果

        SCALE = 2
        blurred_display = cv2.resize(blurred_display, (0, 0), fx=SCALE, fy=SCALE)
        cv2.imshow('Blurred', blurred_display)

        binary_display = cv2.resize(binary_display, (0, 0), fx=SCALE, fy=SCALE)
        cv2.imshow('Binary', binary_display)

        final_frame_display = cv2.resize(final_frame_display, (0, 0), fx=SCALE, fy=SCALE)
        cv2.imshow("Frame", final_frame_display)
        
        print(f"分析完成 - 找到 {len(boxes)} 個方塊，{correct_num} 個正確位置")
        print("按任意鍵關閉分析視窗，繼續錄影...")
        
        # 等待按鍵關閉分析視窗
        cv2.waitKey(0)
        
        # 關閉分析視窗
        cv2.destroyWindow('Blurred')
        cv2.destroyWindow('Binary') 
        cv2.destroyWindow("Frame")
        
    elif key == ord('q'):
        print("退出程式")
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()