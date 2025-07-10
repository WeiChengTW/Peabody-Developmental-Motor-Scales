from ultralytics import YOLO
import cv2
import numpy as np
import math
import time

#側邊攝影機 只顯示cap_ng

# 初始化 YOLO 模型
model = YOLO("runs\\detect\\yolov8n8\\weights\\best.pt")

# 開啟攝影機
cap = cv2.VideoCapture(1)


# 初始參數
CONF = 0.5

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    cap_ng_count = 0

    frame = cv2.resize(frame, (0, 0), fx = 1, fy = 1)

    # YOLO 偵測瓶蓋
    results = model.predict(source=frame, show=False, conf=CONF, verbose = False)
    output = frame.copy()

    colors = {
        0: (255, 180, 0),
        1: (0, 255, 0)
    }

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if model.names[cls] == 'cap_ng':
            label = f"{model.names[cls]} {conf:.2f}"
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cap_ng_count += 1
        
    cv2.putText(frame, f"cap_ng: {cap_ng_count}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("YOLOv8 Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("離開程式")
        break

cap.release()
cv2.destroyAllWindows()
