import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# 載入 main_side.py 所需的模組
# 確保這些檔案 (check_gap.py, MaskAnalyzer.py, StairChecker.py, PyramidChecker.py, LayerGrouping.py) 存在
try:
    from check_gap import CheckGap
    from MaskAnalyzer import MaskAnalyzer
    from StairChecker import StairChecker
    from PyramidChecker import PyramidCheck
    from LayerGrouping import LayerGrouping
except ImportError as e:
    print(f"錯誤：缺少側視圖分析所需的模組，請確保這些檔案存在並在同一路徑下：{e}")
    # 設置一個標誌，在 main 區塊進行側視圖分析時會跳過
    SIDE_ANALYSIS_AVAILABLE = False
else:
    SIDE_ANALYSIS_AVAILABLE = True


def return_score(score):
    sys.exit(int(score))


# ====================================================================
# === 側視圖 (SIDE View) 分析函數 - 來自 main_side.py 的 main 函數核心邏輯
# ====================================================================


# def analyze_image_side(img_path, initial_score, ori_type, model):
#     """
#     分析側視圖 (SIDE View) 影像，檢查空隙和結構。

#     :param img_path: 圖片路徑
#     :param initial_score: 初始得分 (例如 2)
#     :param ori_type: 期望的結構類型 ('Left Stair' 或 'Right Stair')
#     :param model: 已載入的 YOLO 模型
#     :return: score_side (0, 1, 或 2, 失敗返回 -1)
#     """
#     if not SIDE_ANALYSIS_AVAILABLE:
#         print("側視圖分析模組缺失，跳過分析並返回 -1。")
#         return -1

#     MODE = 0  # 0 = 階梯
#     CONF = 0.8
#     GAP_THRESHOLD_RATIO = 0.7
#     SCORE = initial_score

#     # 檢查圖片是否存在
#     if not os.path.exists(img_path):
#         print(f"側視圖分析錯誤: 找不到圖片檔案 {img_path}")
#         return -1

#     # 載入圖片
#     frame = cv2.imread(img_path)
#     if frame is None:
#         print(f"側視圖分析錯誤: 無法讀取圖片 {img_path}")
#         return -1

#     print(f"側視圖分析 - 圖片: {img_path}")
#     print(f"當前模式: {['階梯', '金字塔'][MODE]}")

#     # 進行YOLO預測和分析
#     results = model.predict(source=frame, conf=CONF, verbose=False)
#     masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
#     boxes = (
#         results[0].boxes.xyxy.cpu().numpy()
#         if results[0].boxes is not None
#         else np.empty((0, 4))
#     )

#     centroids = MaskAnalyzer.get_centroids(masks)
#     IS_GAP = False

#     # 核心邏輯開始
#     if len(masks) != 6:
#         print(f"側視圖警告: 偵測到 {len(masks)} 個積木，非預期的 6 個。")

#     if len(centroids) >= 2:
#         bbox_widths = [box[2] - box[0] for box in boxes] if len(boxes) > 0 else []
#         avg_width = np.mean(bbox_widths) if bbox_widths else 1
#         GAP_THRESHOLD = GAP_THRESHOLD_RATIO * avg_width

#         gap_checker = CheckGap(gap_threshold=GAP_THRESHOLD, y_layer_threshold=30)
#         gap_pairs = gap_checker.check(centroids)

#         if gap_pairs:
#             IS_GAP = True if (len(gap_pairs) // 2 == 3) else False
#             print(f"偵測到空隙: {len(gap_pairs) // 2} 組")
#             if MODE == 0:
#                 SCORE = 1  # 階梯模式遇到空隙，扣分
#         else:
#             print("無空隙")
#             if MODE == 1:
#                 SCORE = 1  # 金字塔模式沒空隙，扣分

#     # 模式檢測
#     if MODE == 0:  # 階梯模式
#         grouper = LayerGrouping(layer_ratio=0.2)
#         layers = grouper.group_by_y(centroids, boxes=boxes)

#         stair_checker = StairChecker()
#         result, msg = stair_checker.check(layers)
#         TYPE = msg

#         print(f"階梯檢測結果: {msg}")

#         if not result or TYPE != ori_type:
#             SCORE = 0  # 結構不對或類型不符，最低分

#     elif MODE == 1:  # 金字塔模式 (保留原邏輯，雖然原 main_side.py 預設 MODE=0)
#         grouper = LayerGrouping(layer_ratio=0.2)
#         layers = grouper.group_by_y(centroids, boxes=boxes)
#         block_width = (
#             np.mean([box[2] - box[0] for box in boxes]) // 2 if len(boxes) > 0 else 0
#         )

#         pyramid_checker = PyramidCheck()
#         is_pyramid, pyramid_msg = pyramid_checker.check_pyramid(
#             layers, block_width, IS_GAP
#         )
#         print(f"金字塔檢測結果: {pyramid_msg}")

#         if not is_pyramid:
#             SCORE = 0

#     # 註釋掉原有的 cv2.imshow 呼叫
#     # cv2.imshow("Side Analysis Result", annotated_frame)
#     # cv2.waitKey(0) # 只留一幀，不阻塞

#     return SCORE

def analyze_image_side(img_path, initial_score, ori_type, model):
    """
    分析側視圖 (SIDE View) 影像，檢查空隙和結構。

    :param img_path: 圖片路徑
    :param initial_score: 初始得分 (例如 2)
    :param ori_type: 期望的結構類型 ('Left Stair' 或 'Right Stair')
    :param model: 已載入的 YOLO 模型
    :return: score_side (0, 1, 或 2, 失敗返回 -1)
    """
    if not SIDE_ANALYSIS_AVAILABLE:
        print("側視圖分析模組缺失，跳過分析並返回 -1。")
        return -1

    MODE = 0  # 0 = 階梯
    CONF = 0.8
    GAP_THRESHOLD_RATIO = 0.7
    SCORE = initial_score

    # 檢查圖片是否存在
    if not os.path.exists(img_path):
        print(f"側視圖分析錯誤: 找不到圖片檔案 {img_path}")
        return -1

    # 載入圖片
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"側視圖分析錯誤: 無法讀取圖片 {img_path}")
        return -1

    annotated_frame = frame.copy()  # 用來繪製的副本
    
    print(f"側視圖分析 - 圖片: {img_path}")
    print(f"當前模式: {['階梯', '金字塔'][MODE]}")

    # 進行YOLO預測和分析
    results = model.predict(source=frame, conf=CONF, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
    boxes = (
        results[0].boxes.xyxy.cpu().numpy()
        if results[0].boxes is not None
        else np.empty((0, 4))
    )

    centroids = MaskAnalyzer.get_centroids(masks)
    IS_GAP = False

    # ========== 繪製所有偵測到的框框 ==========
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 綠色框
        cv2.putText(annotated_frame, f"Block {i+1}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # ========== 繪製質心 ==========
    for i, (cx, cy) in enumerate(centroids):
        cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)  # 藍色點
        cv2.putText(annotated_frame, f"C{i+1}", (int(cx)+5, int(cy)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 核心邏輯開始
    if len(masks) != 6:
        print(f"側視圖警告: 偵測到 {len(masks)} 個積木，非預期的 6 個。")

    if len(centroids) >= 2:
        bbox_widths = [box[2] - box[0] for box in boxes] if len(boxes) > 0 else []
        avg_width = np.mean(bbox_widths) if bbox_widths else 1
        GAP_THRESHOLD = GAP_THRESHOLD_RATIO * avg_width

        gap_checker = CheckGap(gap_threshold=GAP_THRESHOLD, y_layer_threshold=30)
        gap_pairs = gap_checker.check(centroids)

        # ========== 繪製空隙 ==========
        if gap_pairs:

            print(f"gap_pairs 內容: {gap_pairs}")
            print(f"gap_pairs[0] 型別: {type(gap_pairs[0])}")

            IS_GAP = True if (len(gap_pairs) // 2 == 3) else False
            print(f"偵測到空隙: {len(gap_pairs) // 2} 組")
            
            # gap_pairs 應該是 [(c1, c2), (c3, c4), ...] 的格式
            for idx in range(0, len(gap_pairs), 2):
                if idx + 1 < len(gap_pairs):
                    c1_idx, c2_idx = gap_pairs[idx], gap_pairs[idx + 1]
                    if c1_idx < len(centroids) and c2_idx < len(centroids):
                        cx1, cy1 = centroids[c1_idx]
                        cx2, cy2 = centroids[c2_idx]
                        # 在空隙點之間畫紅線
                        cv2.line(annotated_frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), 
                                (0, 0, 255), 2)  # 紅色線
                        # 標記空隙
                        mid_x, mid_y = int((cx1 + cx2) / 2), int((cy1 + cy2) / 2)
                        cv2.putText(annotated_frame, "GAP", (mid_x, mid_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if MODE == 0:
                SCORE = 1  # 階梯模式遇到空隙，扣分
        else:
            print("無空隙")
            if MODE == 1:
                SCORE = 1  # 金字塔模式沒空隙，扣分

    # 模式檢測
    if MODE == 0:  # 階梯模式
        grouper = LayerGrouping(layer_ratio=0.2)
        layers = grouper.group_by_y(centroids, boxes=boxes)


        print(f"layers 內容: {layers}")
        print(f"layers[0] 型別: {type(layers[0])}")
        if len(layers) > 0 and len(layers[0]) > 0:
            print(f"layers[0][0] 型別: {type(layers[0][0])}")
            
        # ========== 繪製分層 ==========
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255)]
        for layer_idx, layer in enumerate(layers):
            color = colors[layer_idx % len(colors)]
            
            # ✅ layer 裡面是質心座標 (cx, cy)
            for centroid in layer:
                if isinstance(centroid, tuple) and len(centroid) == 2:
                    cx, cy = centroid
                    
                    # 找到這個質心對應的 box
                    for box_idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        box_cx = (x1 + x2) / 2
                        box_cy = (y1 + y2) / 2
                        
                        # 如果質心和 box 中心很接近，就是對應的 box
                        if abs(box_cx - cx) < 30 and abs(box_cy - cy) < 30:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(annotated_frame, f"L{layer_idx+1}", (x1, y2+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            break  # 找到就跳出，避免重複繪製

        stair_checker = StairChecker()
        result, msg = stair_checker.check(layers)
        TYPE = msg

        print(f"階梯檢測結果: {msg}")

        # ========== 在左上角顯示檢測結果 ==========
        result_text = f"Type: {TYPE} | Expected: {ori_type} | Match: {'✓' if TYPE == ori_type else '✗'}"
        cv2.putText(annotated_frame, result_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not result or TYPE != ori_type:
            SCORE = 0  # 結構不對或類型不符，最低分

    elif MODE == 1:  # 金字塔模式
        grouper = LayerGrouping(layer_ratio=0.2)
        layers = grouper.group_by_y(centroids, boxes=boxes)
        block_width = (
            np.mean([box[2] - box[0] for box in boxes]) // 2 if len(boxes) > 0 else 0
        )

        pyramid_checker = PyramidCheck()
        is_pyramid, pyramid_msg = pyramid_checker.check_pyramid(
            layers, block_width, IS_GAP
        )
        print(f"金字塔檢測結果: {pyramid_msg}")

        if not is_pyramid:
            SCORE = 0

    # ========== 顯示得分 ==========
    score_text = f"Score: {SCORE}/2"
    cv2.putText(annotated_frame, score_text, (10, annotated_frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 保存標註後的圖片（可選）
    output_path = img_path.replace(".jpg", "_annotated.jpg")
    cv2.imwrite(output_path, annotated_frame)
    print(f"標註結果已保存至: {output_path}")

    # 如果要顯示（開發時用），可解註：
    # cv2.imshow("Side Analysis Result", annotated_frame)
    # cv2.waitKey(0)

    return SCORE

# ====================================================================
# === 俯視圖 (TOP View) 分析函數 - 來自 main_top.py 的 analyze_image 函數核心邏輯
# ====================================================================


def analyze_image_top(frame, initial_score, model):
    """
    分析俯視圖 (TOP View) 影像，檢查旋轉和偏移。

    :param frame: 影像幀
    :param initial_score: 初始得分 (例如 2)
    :param model: 已載入的 YOLO 模型
    :return: (summary_string, final_score, analyzed_frame)
    """
    CONF = 0.8
    CROP_RATIO = 0.5  # 中央區域比例
    GET_POINT = initial_score

    H, W, _ = frame.shape
    crop_w, crop_h = int(W * CROP_RATIO), int(H * CROP_RATIO)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2, y2 = x1 + crop_w, y1 + crop_h

    # 裁切中央區域
    cropped = frame[y1:y2, x1:x2].copy()  # 使用 copy 以便在上面繪圖
    results = model.predict(source=cropped, conf=CONF, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    centers = []
    max_mask_side = 0
    rotate_ok_list = []

    for mask in masks:
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            # x, y, w, h = cv2.boundingRect(cnt) # 從原碼繼承的 max_mask_side 計算
            bbox = cv2.boundingRect(cnt)
            max_mask_side = max(max_mask_side, max(bbox[2], bbox[3]))

            mask_H, mask_W = binary_mask.shape
            scale_x = crop_w / mask_W
            scale_y = crop_h / mask_H

            # 中心點位置
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] * scale_x)
                cy = int(M["m01"] / M["m00"] * scale_y)
                centers.append((cx, cy))
                cv2.circle(cropped, (cx, cy), 5, (0, 0, 0), -1)  # 繪製中心點

            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box[:, 0] = box[:, 0] * scale_x
                box[:, 1] = box[:, 1] * scale_y
                box = np.intp(box)

                # === 找出主邊方向（最長的邊）===
                max_len = -1
                main_angle = 0
                for i in range(4):
                    pt1 = box[i]
                    pt2 = box[(i + 1) % 4]
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    length = dx**2 + dy**2
                    angle = np.arctan2(dy, dx) * 180 / np.pi

                    if length > max_len:
                        max_len = length
                        main_angle = angle

                main_angle = abs(main_angle)

                # === 比對是否「近似水平」或「近似垂直」 (<= 10 度) ===
                angle_diff_to_horizontal = abs(main_angle)
                angle_diff_to_vertical = abs(main_angle - 90)

                rotate_ok = (
                    angle_diff_to_horizontal <= 10 or angle_diff_to_vertical <= 10
                )
                rotate_ok_list.append(rotate_ok)

                # === 畫框、標角度 ===
                color = (0, 255, 0) if rotate_ok else (0, 0, 255)
                cv2.drawContours(cropped, [box], 0, color, 2)

    # === 判斷邏輯：偏移 (Offset) ===
    offset = False  # True = No Offset, False = Offset !
    if len(centers) >= 2 and max_mask_side > 0:
        threshold = max_mask_side // 8

        x_vals = [pt[0] for pt in centers]
        y_vals = [pt[1] for pt in centers]
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)

        # 原碼判斷邏輯：std_x < threshold 或 std_y < threshold 表示 No Offset
        offset = std_x < threshold or std_y < threshold

        # 繪製除錯資訊
        cv2.putText(
            cropped,
            f"std_x = {std_x:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 100),
            2,
        )
        cv2.putText(
            cropped,
            f"std_y = {std_y:.2f}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 255),
            2,
        )
        cv2.putText(
            cropped,
            f"threshold = {threshold:.2f}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2,
        )

    # 顯示總體狀態
    status_rotate = "No Bricks"
    if rotate_ok_list:
        status_rotate = "No Rotate" if all(rotate_ok_list) else "Rotate !"

    status_offset = "No Offset" if offset else "Offset !"
    summary = f"{status_offset} | {status_rotate}"

    if (
        status_offset == "Offset !"
        or status_rotate == "Rotate !"
        or status_rotate == "No Bricks"
    ):
        GET_POINT = 1
        color = (0, 0, 255)
    else:
        color = (0, 0, 0)

    cv2.putText(cropped, summary, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return summary, GET_POINT, cropped


# ====================================================================
# === 主程式執行區塊
# ====================================================================

if __name__ == "__main__":
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        stair_type = sys.argv[3] if len(sys.argv) > 3 else None
        # uid = "lull222"
        # img_id = "ch3-t1"
        # image_path = rf"kid\{uid}\{img_id}.jpg"
        # --- 圖片路徑設定 ---
        # 請根據您的實際情況修改這些路徑
        SIDE_IMG_PATH = rf"kid\{uid}\{img_id}-side.jpg"
        TOP_IMG_PATH = rf"kid\{uid}\{img_id}-top.jpg"
        MODEL_PATH = r"ch1-t3/toybrick.pt"

    # 側視圖的期望結構類型 (來自 main_side.py 的 if __name__ 區塊)
    SIDE_ORI_TYPE = "Left Stair" if stair_type == "L" else "Right Stair"
    print(f"側視圖期望結構類型: {SIDE_ORI_TYPE}")
    INITIAL_SCORE = 2

    # --- 載入模型 ---
    print("--- 載入 YOLO 模型 ---")
    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"錯誤：載入 YOLO 模型失敗 (路徑: {MODEL_PATH})。請確保檔案存在。")
        sys.exit(1)

    # --- 1. 執行側視圖分析 ---
    score_side = -1
    print("\n--- 1. 執行側視圖分析 ---")
    try:
        score_side = analyze_image_side(
            img_path=SIDE_IMG_PATH,
            initial_score=INITIAL_SCORE,
            ori_type=SIDE_ORI_TYPE,
            model=yolo_model,
        )
        print(f"側視圖得分 (score_side): {score_side}")
    except Exception as e:
        print(f"側視圖分析時發生嚴重錯誤: {e}")

    # --- 2. 執行俯視圖分析 ---
    score_top = -1
    analyzed_frame_top = None
    print("\n--- 2. 執行俯視圖分析 ---")
    try:
        frame_top = cv2.imread(TOP_IMG_PATH)
        if frame_top is None:
            raise FileNotFoundError(f"讀取俯視圖失敗，路徑：{TOP_IMG_PATH}")

        summary, score_top, analyzed_frame_top = analyze_image_top(
            frame_top, INITIAL_SCORE, yolo_model
        )

        print(f"俯視圖檢測結果: {summary}")
        print(f"俯視圖得分 (score_top): {score_top}")

        # 顯示結果 (可選)
        # cv2.imshow("TOP Detection Result", analyzed_frame_top)
        # cv2.waitKey(1)

    except FileNotFoundError as e:
        print(f"俯視圖分析失敗: {e}")
    except Exception as e:
        print(f"俯視圖分析時發生錯誤: {e}")

    # --- 3. 輸出最低得分 ---

    valid_scores = [s for s in [score_side, score_top] if s != -1]

    print("\n===============================")
    if not valid_scores:
        final_score = -1
        print("總結：兩項分析皆失敗或未執行，無法計算最低得分。")
    else:
        final_score = min(valid_scores)
        print(f"最低得分回傳：{final_score}")
    print("===============================")

    # 等待按鍵關閉所有視窗
    # if analyzed_frame_top is not None:
    #      cv2.waitKey(0)
    #      cv2.destroyAllWindows()
    return_score(final_score)
