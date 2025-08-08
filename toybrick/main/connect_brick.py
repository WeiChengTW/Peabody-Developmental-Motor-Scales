import cv2
from ultralytics import YOLO
import numpy as np
from check_gap import CheckGap
from MaskAnalyzer import MaskAnalyzer
from LayerGrouping import LayerGrouping
from check_online import check_on_line

#串積木 拉起來用相同x判斷是否為同一直線 + y空隙判斷是否相鄰, >= 2個積木才開始評分

# 0 = 階梯, 1 = 金字塔, 2 = 串積木
MODE = 1

cap = cv2.VideoCapture(1)
model = YOLO(r'model\toybrick.pt')

#參數
CONF = 0.7
GAP_THRESHOLD_RATIO = 1.22
offset_RATIO = 2.2
layer_threshold = 30
col = check_on_line()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    results = model.predict(source=frame, conf=CONF, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    #偵測到的積木
    centroids = MaskAnalyzer.get_centroids(masks)

    bbox_widths = [box[2] - box[0] for box in results[0].boxes.xyxy.cpu().numpy()]
    avg_width = np.mean(bbox_widths) if bbox_widths else 1

    grouper = LayerGrouping(layer_threshold)
    layers = grouper.group_by_y(centroids)

    OL = col.check_x(layers=layers, offset = avg_width // offset_RATIO) 
    count_brick = col.check_y(centroids)
    gap_pairs = col.check_gap(centroids=centroids)

    if gap_pairs:
        
            for idx, (p1, p2, d) in enumerate(gap_pairs):
                cv2.line(frame, p1, p2, (0, 0, 255), 2)
                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2)
                cv2.putText(frame, f"{d:.1f}", (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Gap Detected ({len(gap_pairs) // 2}) ", (0, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"No Gap", (0, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Correct : {count_brick} ", (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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
                label = f"({cx}, {cy}) {names[classes[i]]} {confs[i]:.2f}"
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 0), -1)
                cv2.putText(annotated, label, (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (98, 111, 19), 2)
    
    line_msg = f"ON-Line, have {OL} brick(s)" if OL else "NO ON-Line"

    cv2.putText(annotated, line_msg, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    combined = cv2.addWeighted(annotated, 0.8, frame, 0.2, 0)
    combined = cv2.resize(combined, (0, 0), fx=2, fy=2)
        
    cv2.imshow("YOLOv8-segmentation with Gap Detection", combined)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()