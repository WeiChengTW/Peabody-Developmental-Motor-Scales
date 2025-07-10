from ultralytics import YOLO
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter

model = YOLO('runs\\detect\\train3\\weights\\best.pt')
cap = cv2.VideoCapture(1)
CONF = 0.3
new_count = -1
while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    scale = 1

    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

  
    H, W, _ = frame.shape

    # 計算裁切區域（中央區域）
    scope = 0.7
    crop_w, crop_h = int(W * scope), int(H * scope)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # 裁切並放大
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    # 接下來就用 zoomed 當作 frame 去顯示或做 YOLO 偵測
    frame = zoomed


    results = model.predict(source = frame, conf = CONF, verbose = False, show = False)
    result = results[0]
    boxes = result.boxes

    bead_count = 0
    bead_center = []
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]#取得物件名稱

        if class_name == 'bead':
            x1, y1, x2, y2 = map(int, box.xyxy[0])#取得座標 (左上到右下)
            cx = (x1+x2) // 2
            cy = (y1+y2) // 2
            bead_center.append([cx, cy])

            cv2.circle(frame, (cx, cy), 3, (0, 0, 0), -1)#畫中心點
            conf = float(box.conf[0])

            bead_count += 1
            #畫框和文字
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if bead_count > new_count:
        new_count = bead_count
    # print(bead_count)
    cv2.putText(frame, f"bead_count : {bead_count}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.imshow('detect', frame)

    if cv2.waitKey(1) == ord('q') :
        break       

print(f"最多串起 {new_count} 顆串珠!")
cap.release()
cv2.destroyAllWindows()
