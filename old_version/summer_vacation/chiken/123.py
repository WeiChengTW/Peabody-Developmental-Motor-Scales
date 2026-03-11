import time
from ultralytics import YOLO
import cv2
from collections import deque

# 載入模型
model = YOLO(r"runs\detect\train\weights\best.pt")

# 開啟攝影機或影片檔案
cap = cv2.VideoCapture(1)  # 改成 "chicken_video.mp4" 也可

# 設定前幀追蹤緩衝區
buffer_size = 5
head_history = deque(maxlen=buffer_size)

# 時間與狀態變數
start_time = time.time()
pull_time = None
release_time = None
last_switch_time = start_time
first_state = None
switch_count = 0
cooldown = 2
reached = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False, conf=0.4, verbose=False)

    zip_body_box = None
    zip_head_box = None

    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:
            zip_body_box = (x1, y1, x2, y2)
        elif cls == 1:
            zip_head_box = (x1, y1, x2, y2)

        # 繪製框與標籤
        label = f"{model.names[cls]} {float(box.conf[0]):.2f}"
        color = (0, 255, 0) if cls == 0 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # 若目前偵測不到 zip_head，使用歷史座標補上
    if zip_head_box is not None:
        head_history.append(zip_head_box)
    elif len(head_history) > 0:
        zip_head_box = head_history[-1]

    # 僅更新這段以適應側面鏡頭（垂直方向）
    if zip_body_box and zip_head_box:
        body_x1, body_y1, body_x2, body_y2 = zip_body_box
        head_x1, head_y1, head_x2, head_y2 = zip_head_box
        head_center_y = (head_y1 + head_y2) // 2
        now = time.time()

        # 判斷 zip_head 靠近哪一邊（由上到下）
        if abs(head_center_y - body_y2) < 10:  # 靠近 body 下方 → 拉上
            current_state = "pull"
        elif abs(head_center_y - body_y1) < 10:  # 靠近 body 上方 → 拉開
            current_state = "release"
        else:
            current_state = None

        if not hasattr(model, "last_action"):
            model.last_action = None

        if first_state is None and current_state:
            first_state = current_state
            model.last_action = current_state
            last_switch_time = now

        # 狀態切換判斷
        if (
            current_state
            and current_state != model.last_action
            and (not reached or now - reached > cooldown)
            and switch_count < 2
        ):
            elapsed = now - last_switch_time
            if switch_count == 0:
                if current_state == "pull":
                    pull_time = elapsed
                    print(f"第一次從拉開轉拉上，花費: {elapsed:.2f} 秒")
                else:
                    release_time = elapsed
                    print(f"第一次從拉上轉拉開，花費: {elapsed:.2f} 秒")
            elif switch_count == 1:
                if current_state == "pull":
                    pull_time = elapsed
                    print(f"第二次轉拉上，花費: {elapsed:.2f} 秒")
                else:
                    release_time = elapsed
                    print(f"第二次轉拉開，花費: {elapsed:.2f} 秒")
            switch_count += 1
            reached = now
            model.last_action = current_state
            last_switch_time = now
        elif current_state:
            model.last_action = current_state

    cv2.imshow("YOLOv8 Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# 結束後輸出結果與評分
if pull_time is not None:
    print(f"拉上所花時間: {pull_time:.2f} 秒")
else:
    print("未偵測到拉上")

if release_time is not None:
    print(f"拉開所花時間: {release_time:.2f} 秒")
else:
    print("未偵測到拉開")

score = None
if pull_time is not None and release_time is not None:
    total_time = pull_time + release_time
    print(f"總共花費時間: {total_time:.2f} 秒")
    if total_time <= 10:
        score = 3
        print("3 10秒內完成拉開拉關")
    elif total_time <= 20:
        score = 1
        print("1 11-20秒完成")
    else:
        score = 0
        print("0 超過20秒完成")
elif release_time is not None:
    if release_time <= 10:
        score = 2
        print("2 10秒內完成拉開")
    elif release_time <= 20:
        score = 1
        print("1 11-20秒完成")
    else:
        score = 0
        print("0 超過20秒完成")
else:
    print("未完成拉開，無法評分")

# 印出星星得分
if score is not None:
    print("⭐ 得分：" + "★" * score + "☆" * (3 - score))
