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
CAM_INDEX = 1  # 你的相機索引
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # 指定用 DirectShow（Windows 較穩）

# 先指定壓縮格式，很多鏡頭 1080p 需要 MJPG 才能上 30fps
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 設定期望解析度與 FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

model = YOLO(r'toybrick.pt')
CONF = 0.8
GAP_THRESHOLD_RATIO = 0.366

SCORE = 2
print("按下 's' 鍵拍照並分析")
print("按下 'q' 鍵退出")
print(f"當前模式: {['階梯', '金字塔'][MODE]}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 顯示即時影像
    live_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Live Video - Press 's' to capture, 'q' to quit", live_frame)
    
    # 等待按鍵
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("拍照中...")
        
        # 進行YOLO預測和分析
        results = model.predict(source=frame, conf=CONF, verbose=False)
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

        centroids = MaskAnalyzer.get_centroids(masks)
        IS_GAP = False
        
        # 複製frame進行標註
        analysis_frame = frame.copy()
        
        if len(masks) != 6:
            cv2.putText(analysis_frame, 'Plz place 6 bricks', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2)
            print("請放置6個積木")
        else:     
            if len(centroids) >= 2:
                
                bbox_widths = [box[2] - box[0] for box in results[0].boxes.xyxy.cpu().numpy()]
                avg_width = np.mean(bbox_widths) if bbox_widths else 1
                GAP_THRESHOLD = GAP_THRESHOLD_RATIO * avg_width

                gap_checker = CheckGap(gap_threshold=GAP_THRESHOLD, y_layer_threshold=30)
                gap_pairs = gap_checker.check(centroids)

                if gap_pairs:
                    for idx, (p1, p2, d) in enumerate(gap_pairs):
                        cv2.line(analysis_frame, p1, p2, (0, 0, 255), 2)
                        mid_x = int((p1[0] + p2[0]) / 2)
                        mid_y = int((p1[1] + p2[1]) / 2)
                        cv2.putText(analysis_frame, f"{d:.1f}", (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    IS_GAP = True if(len(gap_pairs) // 2 == 3) else False

                    cv2.putText(analysis_frame, f"Gap Detected ({len(gap_pairs) // 2})", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    print(f"偵測到空隙: {len(gap_pairs) // 2} 個")

                    if MODE == 0:
                        SCORE = 1
                else:
                    cv2.putText(analysis_frame, f"No Gap", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    if MODE == 1:
                        SCORE = 1
                    print("無空隙")
        
        # 模式檢測
        if MODE == 0:  # 階梯模式

            # Left Stair, Right Stair
            ORI_TYPE = "Left Stair"

            # 使用自適應LayerGrouping
            boxes = results[0].boxes.xyxy.cpu().numpy()
            grouper = LayerGrouping(layer_ratio=0.2)
            layers = grouper.group_by_y(centroids, boxes=boxes)

            stair_checker = StairChecker()
            result, msg = stair_checker.check(layers)

            TYPE = msg

            cv2.putText(analysis_frame, msg, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255) if result else (0, 0, 255), 2)
            print(f"階梯檢測結果: {msg}")
            
            if not result or TYPE != ORI_TYPE:
                SCORE = 0
            # 顯示分層資訊（除錯用）
            print(f"分層結果: {len(layers)}層")
            for i, layer in enumerate(layers):
                avg_y = np.mean([point[1] for point in layer])
                print(f"  第{i+1}層 (Y={avg_y:.1f}): {len(layer)}個積木")
            
        elif MODE == 1:  # 金字塔模式
            # 使用自適應LayerGrouping
            boxes = results[0].boxes.xyxy.cpu().numpy()
            grouper = LayerGrouping(layer_ratio=0.2)
            layers = grouper.group_by_y(centroids, boxes=boxes)
            
            # 計算block_width（用於金字塔檢測）
            block_width = np.mean([box[2] - box[0] for box in boxes]) // 2
            
            pyramid_checker = PyramidCheck()
            is_pyramid, pyramid_msg = pyramid_checker.check_pyramid(layers, block_width, IS_GAP)

            cv2.putText(analysis_frame, pyramid_msg, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 0) if is_pyramid else (0, 0, 255), 2)
            print(f"金字塔檢測結果: {pyramid_msg}")
            
            if not is_pyramid:
                SCORE = 0
            # # 顯示分層資訊（除錯用）
            # print(f"分層結果: {len(layers)}層")
            # for i, layer in enumerate(layers):
            #     avg_y = np.mean([point[1] for point in layer])
            #     print(f"  第{i+1}層 (Y={avg_y:.1f}): {len(layer)}個積木")

        # 自定義繪圖：畫邊界框與資訊
        annotated = analysis_frame.copy()

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # shape: (N, H, W)
            confs = results[0].boxes.conf.cpu().numpy()  # 信心值
            names = results[0].names
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # 取得邊界框座標

            for i, mask in enumerate(masks):
                # 畫邊界框而不是填充mask
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # 找中心點顯示 conf
                ys, xs = np.where(mask > CONF)
                if len(xs) > 0 and len(ys) > 0:

                    label = f"{names[classes[i]]} {confs[i]:.2f}"
                    cv2.circle(annotated, (cx, cy), 3, (0, 0, 0), -1)
                    cv2.putText(annotated, label, (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (98, 111, 19), 2)
        
        # 添加模式資訊
        cv2.putText(annotated, f"Mode: {['Stair', 'Pyramid'][MODE]}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(annotated, f"Detected: {len(centroids)} bricks", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        combined = cv2.addWeighted(annotated, 0.8, analysis_frame, 0.2, 0)
        combined = cv2.resize(combined, (0, 0), fx=1, fy=1)

        # 顯示分析結果
        cv2.imshow("Analysis Result", combined)
        
        print(f"分析完成 - 偵測到 {len(centroids)} 個積木")
        print("按任意鍵關閉分析視窗，繼續錄影...")
        print(SCORE)
        # 等待按鍵關閉分析視窗
        cv2.waitKey(0)
        
        # 關閉分析視窗
        cv2.destroyWindow("Analysis Result")
        
    elif key == ord('q'):
        print("退出程式")
        break

cap.release()
cv2.destroyAllWindows()