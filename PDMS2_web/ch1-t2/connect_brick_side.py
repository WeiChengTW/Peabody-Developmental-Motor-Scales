import cv2
from ultralytics import YOLO
import numpy as np
from MaskAnalyzer import MaskAnalyzer
from LayerGrouping import LayerGrouping
from check_online import check_on_line

#串積木 >= 2個積木才開始評分 (只要判斷有幾個積木, 可以有空隙)

# 0 = 階梯, 1 = 金字塔, 2 = 串積木
MODE = 1

cap = cv2.VideoCapture(1)
model = YOLO(r'model\toybrick.pt')

#參數
CONF = 0.6
offset_RATIO = 2.2
layer_threshold = 30
col = check_on_line()
lg = LayerGrouping()

print("按下 's' 鍵拍照並分析")
print("按下 'q' 鍵退出")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # 顯示即時影像
    live_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    cv2.imshow("Live Video - Press 's' to capture, 'q' to quit", live_frame)
    
    # 等待按鍵
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("拍照中...")
        
        # 進行YOLO預測和分析
        results = model.predict(source=frame, conf=CONF, verbose=False)
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

        #偵測到的積木
        centroids = MaskAnalyzer.get_centroids(masks)

        bbox_widths = [box[2] - box[0] for box in results[0].boxes.xyxy.cpu().numpy()]
        avg_width = np.mean(bbox_widths) if bbox_widths else 1
        OFFSET = offset_RATIO * avg_width

        layers = lg.group_by_y(centroids=centroids)
        cor_num = col.check_x(layers=layers, offset=OFFSET)

        # print(f"偵測到 {len(centroids)} 個積木")
        
        # 自定義繪圖：畫 mask 與 conf
        annotated = frame.copy()

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # shape: (N, H, W)
            confs = results[0].boxes.conf.cpu().numpy()  # 信心值
            names = results[0].names
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # 取得邊界框座標

            for i, mask in enumerate(masks):
                # 畫邊界框而不是填充mask
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # 找中心點顯示 conf
                ys, xs = np.where(mask > CONF)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    label = f"({cx}, {cy}) {names[classes[i]]} {confs[i]:.2f}"
                    cv2.circle(annotated, (cx, cy), 3, (0, 0, 0), -1)
                    cv2.putText(annotated, label, (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (98, 111, 19), 2)
        
        cv2.putText(annotated, f"Detected Bricks: {len(centroids)}, Correct Bricks: {cor_num - 2}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        combined = cv2.addWeighted(annotated, 0.8, frame, 0.2, 0)
        # combined = cv2.resize(combined, (0, 0), fx=2, fy=2)
        
        # 顯示分析結果
        cv2.imshow("Connect Brick Analysis", combined)
        
        print(f"分析完成 - 偵測到 {cor_num} 個積木")
        print("按任意鍵關閉分析視窗，繼續錄影...")
        
        # 等待按鍵關閉分析視窗
        cv2.waitKey(0)
        
        # 關閉分析視窗
        cv2.destroyWindow("Connect Brick Analysis")
        
    elif key == ord('q'):
        print("退出程式")
        break

cap.release()
cv2.destroyAllWindows()