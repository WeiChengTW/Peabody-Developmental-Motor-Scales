from ultralytics import YOLO
import cv2
import numpy as np
import time

# 初始化模型
model = YOLO(r'bean_model.pt')
cap = cv2.VideoCapture(4)

CONF = 0.45
DIST_THRESHOLD = 5  # 中心點合併距離閾值
CHECK_INTERVAL = 0.5  # 檢查間隔秒數

previous_count = 0
last_check_time = time.time()

# 遊戲計時
GAME_DURATION = 120
start_time = time.time()

# 狀態記錄
warning_flag = False   # 是否有違規
total_count = 0        # 豆子總數

def calculate_score(total_count, warning_flag, elapsed):
    """依規則計算分數"""
    if elapsed <= 30:  # 前 30 秒
        if total_count >= 10:
            return 1 if warning_flag else 2
        else:
            return 0
    else:  # 31–60 秒
        if total_count >= 5:
            return 1
        else:
            return 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 偵測
    results = model.predict(source=frame, conf=CONF, verbose=False)
    annotated = frame.copy()
    centers = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            ys, xs = np.where(mask > CONF)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                centers.append((cx, cy))

    # 合併過近的中心點
    merged = []
    used = set()
    for i, (x1, y1) in enumerate(centers):
        if i in used:
            continue
        group = [(x1, y1)]
        for j, (x2, y2) in enumerate(centers):
            if i != j and j not in used:
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < DIST_THRESHOLD:
                    group.append((x2, y2))
                    used.add(j)
        merged.append(np.mean(group, axis=0))  # 取平均當新的中心點

    # 畫合併後的中心點
    for (cx, cy) in merged:
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # 紅點

    count = len(merged)

    # 每隔一段時間檢查一次
    current_time = time.time()
    if current_time - last_check_time >= CHECK_INTERVAL:
        if count > previous_count:  # 有新增豆子
            added = count - previous_count
            
            if added > 1:  # 違規
                warning_flag = True
                cv2.putText(annotated, "Warning !", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if count > total_count:
            total_count = count

        previous_count = count
        last_check_time = current_time

    # 計算剩餘時間
    elapsed = current_time - start_time
    remaining = max(0, GAME_DURATION - int(elapsed))

    cv2.putText(annotated, f'SoyBean count: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f'Total placed: {total_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(annotated, f'Time Left: {remaining}s', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # 放大顯示
    combined = cv2.resize(annotated, (0, 0), fx=2, fy=2)
    cv2.imshow("SoyBean", combined)

    # === 結束條件 ===
    end_game = False
    # 1. 累積達 10 顆就結束（但分數要依秒數判斷）
    if total_count >= 10:
        end_game = True
    # 2. 時間結束
    elif elapsed >= GAME_DURATION:
        end_game = True

    if end_game:
        score = calculate_score(total_count, warning_flag, elapsed)
        print(f"遊戲結束！總數: {total_count}, 警告: {warning_flag}, 分數: {score}, 用時: {int(elapsed)} 秒")

        # 畫面顯示分數
        end_screen = frame.copy()
        
        cv2.putText(end_screen, "Game Over!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(end_screen, f"Total: {total_count}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(end_screen, f"Score: {score}", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(end_screen, f"Time: {int(elapsed)}s", (50, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.imshow("SoyBean", end_screen)
        cv2.waitKey(5000)  # 顯示 5 秒
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


