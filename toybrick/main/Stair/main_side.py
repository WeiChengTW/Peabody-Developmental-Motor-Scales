import cv2
from ultralytics import YOLO
import numpy as np
from check_gap import CheckGap
from MaskAnalyzer import MaskAnalyzer
from StairChecker import StairChecker
from PyramidChecker import PyramidCheck
from LayerGrouping import LayerGrouping
import os



def main(img_path, score, ORI_TYPE):
    # 0 = 階梯, 1 = 金字塔(一定要有空隙)
    MODE = 0

    # === 設定圖片路徑 ===
    IMAGE_PATH = rf"{img_path}" # 讀取照片

    # 檢查圖片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤: 找不到圖片檔案 {IMAGE_PATH}")
        print("請確認圖片路徑是否正確")
        exit()

    # 載入圖片
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"錯誤: 無法讀取圖片 {IMAGE_PATH}")
        print("請確認圖片格式是否正確 (支援 jpg, png, bmp 等)")
        exit()

    print(f"成功載入圖片: {IMAGE_PATH}")
    print(f"圖片尺寸: {frame.shape[1]} x {frame.shape[0]}")

    # 初始化模型
    model = YOLO(r'toybrick.pt')
    CONF = 0.8
    GAP_THRESHOLD_RATIO = 0.7

    SCORE = score
    print(f"當前模式: {['階梯', '金字塔'][MODE]}")
    print("開始分析...")

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
        ORI_TYPE = ORI_TYPE

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

    # 顯示分析結果
    cv2.imshow("Analysis Result", combined)
    cv2.waitKey(0)
    # print(f"分析完成 - 偵測到 {len(centroids)} 個積木")
    # print(f"最終分數: {SCORE}")
    # print("按任意鍵關閉視窗...")

    # 儲存結果圖片
    # output_path = "analysis_result.jpg"
    # cv2.imwrite(output_path, combined)
    # print(f"分析結果已儲存至: {output_path}")

    # 等待按鍵關閉視窗
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return SCORE
if __name__ == "__main__":

    #圖片路徑
    img_path = "left_stair.jpg"

    # Left Stair, Right Stair
    ORI_TYPE = 'Left Stair'
    score = main(img_path, 2, ORI_TYPE)
    print(score)