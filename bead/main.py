from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO(r'model.pt')

cap = cv2.VideoCapture(1)

CONF = 0.5

while True:
    ret, frame = cap.read()

    if not ret:
        break
    count = 0
    results = model.predict(source=frame, conf=CONF, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    # 自定義繪圖：畫 mask 與 conf
    annotated = frame.copy()

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # shape: (N, H, W)
        confs = results[0].boxes.conf.cpu().numpy()  # 信心值
        names = results[0].names
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):
            color = (255, 255, 255) 

            # 畫半透明 mask
            colored_mask = np.zeros_like(annotated, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.3, 0)

            # 找中心點顯示 conf
            ys, xs = np.where(mask > CONF)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                label = f"{names[classes[i]]} {confs[i]:.2f}"
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 0), -1)
                cv2.putText(annotated, label, (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (98, 111, 19), 2)

                count += 1

    cv2.putText(annotated, f'Beads count : {count}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (98, 111, 19), 2)

    combined = cv2.addWeighted(annotated, 0.8, frame, 0.2, 0)
    combined = cv2.resize(combined, (0, 0), fx=2, fy=2)
        
    cv2.imshow("Bead cap", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()