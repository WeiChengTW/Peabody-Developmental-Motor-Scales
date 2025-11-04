from ultralytics import YOLO
import cv2
import numpy as np
import time

# 初始化模型
model = YOLO(r'bean_model.pt')

cap = cv2.VideoCapture(0)  # 開啟攝影機

frame_count = 0
PER_FRAME = 3
prev_box_count = 0

# 計時相關
game_duration = 3
start_time = None
game_started = False

# 分數相關
WARNING = False
SCORE = -1

# 按鈕設定
button_x, button_y, button_w, button_h = 10, 300, 150, 50

# 滑鼠回調函數
def mouse_callback(event, x, y, flags, param):
    global game_started, start_time
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 檢查是否點擊按鈕區域
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            if not game_started:
                game_started = True
                start_time = time.time()
            

cv2.namedWindow('Pick Bean Game')
cv2.setMouseCallback('Pick Bean Game', mouse_callback)

def calculate_score(cur_count, remain_time, WARNING):
    
    if cur_count >= 10 and remain_time >= 30:
        return 2 if not WARNING else 1
    elif cur_count >= 5 and remain_time >= 0:
        return 1
    elif remain_time == 0:
        return 0
    else:
        return -1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 裁切中間區域
    scale = 0.8
    height, width = frame.shape[:2]

    crop_height = int(height * scale)
    crop_width = int(width * scale)

    # 計算起始位置(讓裁切區域置中) [丟棄部分][   保留區域   ][丟棄部分]
    #                          start_x 
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    # 裁切畫面
    frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # 計算剩餘時間
    if game_started:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        SCORE = calculate_score(current_box_count, remaining_time, WARNING)
        if SCORE != -1:
            cv2.putText(frame, f"GAME OVER !", (start_x + 5, start_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
            cv2.imshow('Game Over !', frame)
            cv2.waitKey(3000)
            break
            #return SCORE


    results = model.predict(source=frame, conf=0.5, verbose=False)

    current_box_count = 0
    for result in results:
        boxes = result.boxes
        current_box_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # 檢查是否突然增加 2 個或以上
    if game_started and prev_box_count > 0:
        increase = current_box_count - prev_box_count
        if frame_count % PER_FRAME == 0 and increase >= 2:
            WARNING = True
    
    prev_box_count = current_box_count

    # 顯示資訊
    cv2.putText(frame, f'Beans: {current_box_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if game_started:
        cv2.putText(frame, f'Time: {int(remaining_time)}s', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 繪製按鈕
    if not game_started:
        # 按鈕背景
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     (0, 255, 0), -1)
        # 按鈕邊框
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     (0, 0, 0), 2)
        # 按鈕文字
        cv2.putText(frame, 'START', (button_x + 25, button_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Pick Bean Game', frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()