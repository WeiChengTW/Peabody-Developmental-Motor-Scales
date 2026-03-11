import time
from ultralytics import YOLO
import cv2

# 載入模型
model = YOLO(r"runs\detect\train\weights\best.pt")

# cap = cv2.VideoCapture("test1.mp4")


cap = cv2.VideoCapture(0)
# print("請按下 ENTER 開始計時...")
# input()
start_time = time.time()
reached = False
prev_head_center_x = None  # 上一幀 zip_head 的中心 x 座標
pull_time = None  # 第一次拉上所花時間
release_time = None  # 第一次拉開所花時間
last_switch_time = start_time
first_state = None  # 記錄一開始的狀態
switch_count = 0  # 狀態轉換次數
x_range = 5  # 判斷 head 中心在 body 哪個邊附近的範圍
y_range = 10  # 判斷 head 中心在 body 哪個邊

# 初始化旗標，第一次偵測到狀態時不計入時間
first_action = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False, conf=0.4, verbose=False)

    colors = {
        0: (0, 255, 0),
        1: (0, 255, 255),
    }

    zip_body_box = None
    zip_head_box = None

    # 取得 zip_body 和 zip_head 的框
    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:  # zip_body
            zip_body_box = (x1, y1, x2, y2)
        elif cls == 1:  # zip_head
            zip_head_box = (x1, y1, x2, y2)

        label = f"{model.names[cls]} {float(box.conf[0]):.2f}"
        color = colors.get(cls, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        # 用不同顏色標示 x1, y1, x2, y2
        cv2.circle(frame, (x1, y1), 6, (0, 0, 255), -1)  # x1, y1 - 紅色
        cv2.circle(frame, (x1, y2), 6, (0, 255, 255), -1)  # x1, y2 - 黃色

        cv2.circle(frame, (x2, y1), 6, (0, 255, 0), -1)  # x2, y1 - 綠色
        cv2.circle(frame, (x2, y2), 6, (255, 0, 0), -1)  # x2, y2 - 藍色

    # 判斷 head 中心在 body 哪個邊附近
    cooldown = 2  # 秒
    if zip_body_box and zip_head_box:
        body_x1, body_y1, body_x2, body_y2 = zip_body_box
        head_x1, head_y1, head_x2, head_y2 = zip_head_box
        head_center_x = (head_x1 + head_x2) // 2
        head_center_y = (head_y1 + head_y2) // 2
        now = time.time()
        if not hasattr(model, "last_action"):
            model.last_action = None  # "pull" or "release"
        # 判斷目前狀態
        if (
            abs(head_center_x - body_x1) < x_range
            and abs(head_center_y - body_y1) < y_range
        ) or (
            abs(head_center_x - body_x1) < x_range
            and abs(head_center_y - body_y2) < y_range
        ):
            current_state = "pull"
        elif (
            abs(head_center_x - body_x2) < x_range
            and abs(head_center_y - body_y1) < y_range
        ) or (
            abs(head_center_x - body_x2) < x_range
            and abs(head_center_y - body_y2) < y_range
        ):
            current_state = "release"
        else:
            current_state = None

        # 記錄一開始的狀態，但不計時
        if first_state is None and current_state is not None:
            first_state = current_state
            model.last_action = current_state
            last_switch_time = now
            # print(f"初始狀態: {first_state}")

        # 狀態轉換，僅記錄前兩次
        if (
            current_state is not None
            and current_state != model.last_action
            and (not reached or now - reached > cooldown)
            and switch_count < 2
        ):
            elapsed = now - last_switch_time
            if (
                switch_count == 0
                and first_state == "pull"
                and current_state == "release"
            ):
                release_time = elapsed
                print(f"第一次從拉上轉拉開，花費: {elapsed:.2f} 秒")
            elif (
                switch_count == 0
                and first_state == "release"
                and current_state == "pull"
            ):
                pull_time = elapsed
                print(f"第一次從拉開轉拉上，花費: {elapsed:.2f} 秒")
            elif switch_count == 1 and current_state == "pull":
                pull_time = elapsed
                print(f"第二次轉拉上，花費: {elapsed:.2f} 秒")
            elif switch_count == 1 and current_state == "release":
                release_time = elapsed
                print(f"第二次轉拉開，花費: {elapsed:.2f} 秒")
            switch_count += 1
            reached = now
            model.last_action = current_state
            last_switch_time = now
        elif current_state is not None:
            model.last_action = current_state
        prev_head_center_x = head_center_x  # 更新上一幀中心 x

    cv2.imshow("YOLOv8 Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# 結束時印出兩次狀態轉換所花的時間
if pull_time is not None:
    print(f"拉上所花時間: {pull_time:.2f} 秒")
else:
    print("未偵測到拉上")

if release_time is not None:
    print(f"拉開所花時間: {release_time:.2f} 秒")
else:
    print("未偵測到拉開")
# 評分機制
score = None
if pull_time is not None and release_time is not None:
    total_time = pull_time + release_time
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
