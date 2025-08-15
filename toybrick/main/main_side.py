import cv2
from ultralytics import YOLO
import numpy as np
from check_gap import CheckGap
from MaskAnalyzer import MaskAnalyzer
from StairChecker import StairChecker
from PyramidChecker import PyramidCheck
from LayerGrouping import LayerGrouping
# 0 = 階梯, 1 = 金字塔(一定要有空隙)
MODE = 0

# === 初始化模型與攝影機 ===
cap = cv2.VideoCapture(1)
model = YOLO(r'model\toybrick.pt')
CONF = 0.8
GAP_THRESHOLD_RATIO = 1.05

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(source=frame, conf=CONF, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    centroids = MaskAnalyzer.get_centroids(masks)
    IS_GAP = False
    if len(masks) != 6:
        cv2.putText(frame, 'Plz place 6 bricks', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2)
    else:     
        if len(centroids) >= 2:
            
            bbox_widths = [box[2] - box[0] for box in results[0].boxes.xyxy.cpu().numpy()]
            avg_width = np.mean(bbox_widths) if bbox_widths else 1
            GAP_THRESHOLD = GAP_THRESHOLD_RATIO * avg_width

            gap_checker = CheckGap(gap_threshold=GAP_THRESHOLD, y_layer_threshold=30)
            gap_pairs = gap_checker.check(centroids)

            if gap_pairs:
                for idx, (p1, p2, d) in enumerate(gap_pairs):
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                    mid_x = int((p1[0] + p2[0]) / 2)
                    mid_y = int((p1[1] + p2[1]) / 2)
                    cv2.putText(frame, f"{d:.1f}", (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                IS_GAP = True if(len(gap_pairs) // 2 == 3) else False

                cv2.putText(frame, f"Gap Detected ({len(gap_pairs) // 2})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"No Gap", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if MODE == 0:
        stair_checker = StairChecker()
        result, msg = stair_checker.check(centroids)

        cv2.putText(frame, msg, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255) if result else (0, 0, 255), 2)
    elif MODE == 1:
        block_width = np.mean([box[2] - box[0] for box in results[0].boxes.xyxy.cpu().numpy()]) // 3.6
        grouper = LayerGrouping()
        layers = grouper.group_by_y(centroids)
        
        pyramid_checker = PyramidCheck()
        is_pyramid, pyramid_msg = pyramid_checker.check_pyramid(centroids, block_width, IS_GAP)

        cv2.putText(frame, pyramid_msg, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 0) if is_pyramid else (0, 0, 255), 2)
    

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

            # 找中心點顯示 confq
            ys, xs = np.where(mask > CONF)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                label = f"{names[classes[i]]} {confs[i]:.2f}"
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 0), -1)
                cv2.putText(annotated, label, (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (98, 111, 19), 2)
        
    combined = cv2.addWeighted(annotated, 0.8, frame, 0.2, 0)
    combined = cv2.resize(combined, (0, 0), fx=2, fy=2)

    cv2.imshow("YOLOv8-segmentation with Gap Detection", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()