from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("runs/detect/yolov8n8/weights/best.pt")
cap = cv2.VideoCapture(1)

CONF = 0.5
DIST_THRESHOLD = 2
DEBUG_DRAW = False  # ✅ 切換是否畫 debug 線與距離

def is_inside(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2

hole_positions = []  # ✅ 一次初始化

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False, conf=CONF, verbose=False)

    ladybug_box = None

    # === Step 1：找出瓢蟲區塊 ===
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if name == "ladybug_case":
            ladybug_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # === Step 2：初始化洞口座標（只做一次）===
    if ladybug_box and not hole_positions:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if name == "hole" and is_inside((x1, y1, x2, y2), ladybug_box):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                hole_positions.append({
                    "pos": (cx, cy),
                    "status": "NO CAP",
                    "assigned": False
                })

        hole_positions.sort(key=lambda h: (h["pos"][1], h["pos"][0]))

    # === Step 3：重設每次的洞狀態 ===
    for hole in hole_positions:
        hole["status"] = "NO CAP"
        hole["assigned"] = False

    cap_ok_count = 0
    cap_ng_count = 0

    # === Step 4：逐一處理瓶蓋配對 ===
    for box in results[0].boxes:

        cls = int(box.cls[0])
        name = model.names[cls]
        if name not in ["cap_ok", "cap_ng"]:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 預設不畫（除非配對成功）
        matched = False

        # 各自對應顏色
        if name == "cap_ok":
            cap_color = (0, 255, 0)      # 綠色
        elif name == "cap_ng":
            cap_color = (0, 165, 255)    # 橘色
        else:
            cap_color = (255, 255, 255)  # fallback

        for i, hole in enumerate(hole_positions):
            hx, hy = hole["pos"]
            dist = math.hypot(hx - cx, hy - cy)

            if dist <= 10 and not hole["assigned"]:  # 距離門檻 5
                hole["assigned"] = True
                hole["status"] = "OK" if name == "cap_ok" else "NG"

                if name == "cap_ok":
                    cap_ok_count += 1
                elif name == 'cap_ng':
                    cap_ng_count += 1
                matched = True

                # if DEBUG_DRAW:
                #     cv2.line(frame, (cx, cy), (hx, hy), (0, 100, 255), 1)
                #     mid_x, mid_y = (cx + hx) // 2, (cy + hy) // 2
                #     cv2.putText(frame, f"{dist:.1f}px", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
                break  # ✅ 一顆蓋子只對一個洞

        # 只有成功配對的才畫出來
        if matched:
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), cap_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cap_color, 2)
            cv2.circle(frame, (cx, cy), 3, (200, 200, 0), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)


    # === Step 5：畫出所有洞口狀態 ===
    # for i, hole in enumerate(hole_positions):
    #     hx, hy = hole["pos"]
    #     status = hole["status"]
    #     color = (0, 255, 0) if status == "OK" else (255, 255, 255)
    #     cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1)
        # cv2.putText(frame, f"H{i+1} {status}", (hx + 10, hy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.putText(frame, f"({hx},{hy})", (hx + 10, hy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 200), 1)

    cv2.putText(frame, f"cap_ok: {cap_ok_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - Precise Cap-on-Hole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2
