import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# 載入 main_side.py 所需的模組 (假設它們在同一資料夾)
# 如果這些模組不存在，程式會報錯。
try:
    from check_gap import CheckGap
    from MaskAnalyzer import MaskAnalyzer
    from StairChecker import StairChecker
    from PyramidChecker import PyramidCheck
    from LayerGrouping import LayerGrouping
except ImportError as e:
    print(f"錯誤：缺少側視圖分析所需的模組，請確保這些檔案存在：{e}")
    sys.exit(1)


# ====================================================================
# === 俯視圖 (TOP View) 分析函數
# ====================================================================


CONF_TOP = 0.8
CROP_RATIO = 0.5  # 中央區域比例

def analyze_image_top(frame, model, initial_get_point=2):
    """
    分析俯視圖 (TOP View) 影像，檢查旋轉和偏移。
    返回: (cropped_frame, summary_string, final_score)
    """
    H, W, _ = frame.shape
    crop_w, crop_h = int(W * CROP_RATIO), int(H * CROP_RATIO)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2, y2 = x1 + crop_w, y1 + crop_h

    # 裁切中央區域
    cropped = frame[y1:y2, x1:x2].copy() # 使用 .copy() 確保操作的是裁剪區域的獨立副本

    # YOLO 推理
    results = model.predict(source=cropped, conf=CONF_TOP, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    centers = []
    max_mask_side = 0
    rotate_ok_list = []
    GET_POINT = initial_get_point # 初始得分

    for mask in masks:
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            max_mask_side = max(max_mask_side, max(w, h))

            mask_H, mask_W = binary_mask.shape
            scale_x = crop_w / mask_W
            scale_y = crop_h / mask_H

            # 中心點位置
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                # 注意：這裡的 cx, cy 是在 cropped 座標系中的縮放位置
                cx = int(M["m10"] / M["m00"] * scale_x) 
                cy = int(M["m01"] / M["m00"] * scale_y) 
                centers.append((cx, cy))
                # 繪製中心點 (在 cropped 圖像上)
                cv2.circle(cropped, (cx, cy), 5, (0, 0, 0), -1)

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
            
                # === 比對是否「近似水平」或「近似垂直」 ===
                angle_diff_to_horizontal = abs(main_angle)   # 跟 0 度比
                angle_diff_to_vertical = abs(main_angle - 90) # 跟 90 度比

                rotate_ok = (angle_diff_to_horizontal <= 10 or angle_diff_to_vertical <= 10)
                rotate_ok_list.append(rotate_ok)

                # === 畫框、標角度 ===
                color = (0, 255, 0) if rotate_ok else (0, 0, 255)  # OK = 綠，NG = 紅
                cv2.drawContours(cropped, [box], 0, color, 2)

    # === 判斷邏輯：偏移 (Offset) ===
    offset = False
    if len(centers) >= 2 and max_mask_side > 0:
        threshold = max_mask_side // 8

        x_vals = [pt[0] for pt in centers]
        y_vals = [pt[1] for pt in centers]
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)

        offset = (std_x < threshold or std_y < threshold) # std 越小代表越集中 (越對齊)，即 No Offset

        # 繪製 std/threshold (在 cropped 圖像上)
        cv2.putText(cropped, f"std_x = {std_x:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(cropped, f"std_y = {std_y:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(cropped, f"threshold = {threshold:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # === 判斷邏輯：旋轉 (Rotate) ===
    status_rotate = "?"
    is_rotate_ng = False
    if rotate_ok_list:
        if all(rotate_ok_list):
            status_rotate = "No Rotate"
        else:
            status_rotate = "Rotate !"
            is_rotate_ng = True

    status_offset = "Offset !" if not offset else "No Offset" # 這裡與原碼邏輯 'offset = (std_x < threshold or std_y < threshold)' 相反
    is_offset_ng = not offset # 'offset' 變數在原碼中為 True 表示 No Offset

    summary = f"{status_offset} | {status_rotate}"

    # === 判斷總分 ===
    if is_offset_ng or is_rotate_ng:
        GET_POINT = 1 # 只要有問題，就扣一分
        color = (0, 0, 255)
    else:
        color = (0, 0, 0)
    
    # 繪製總體狀態 (在 cropped 圖像上)
    cv2.putText(cropped, summary, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    return cropped, summary, GET_POINT


# ====================================================================
# === 側視圖 (SIDE View) 分析函數
# ====================================================================

# ===== 參數 =====
MODE_SIDE = 1  # 0 = 階梯, 1 = 金字塔(一定要有空隙)
CONF_SIDE = 0.8
GAP_THRESHOLD_RATIO = 0.222  # 用平均 bbox 寬度 * 這個比例

def analyze_image_side(IMG_PATH, model):
    """
    分析側視圖 (SIDE View) 影像，檢查間隙和結構。
    返回: score_side (0, 1, 或 2)
    """

    # ===== 讀圖 =====
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise ValueError(f"讀不到圖片：{IMG_PATH}")

    # ===== YOLO 偵測 =====
    # model = YOLO(MODEL_PATH) # 已在 main 區塊讀取
    results = model.predict(source=frame, conf=CONF_SIDE, verbose=False)
    r0 = results[0]

    masks = r0.masks.data.cpu().numpy() if r0.masks is not None else []
    boxes = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
    centroids = MaskAnalyzer.get_centroids(masks)

    # ===== 計分 =====
    SCORE = 2
    IS_GAP = False

    # 空隙檢測（需要物件 bbox 寬度估計 gap threshold）
    bbox_widths = [b[2] - b[0] for b in boxes] if len(boxes) > 0 else []
    avg_width = np.mean(bbox_widths) if len(bbox_widths) > 0 else 1.0
    GAP_THRESHOLD = GAP_THRESHOLD_RATIO * avg_width

    if len(centroids) >= 2:
        gap_checker = CheckGap(gap_threshold=GAP_THRESHOLD, y_layer_threshold=30)
        gap_pairs = gap_checker.check(centroids)  # 回傳 [(p1, p2, d), ...]

        # print(f"gap : {len(gap_pairs)}")
        if gap_pairs:
            # 原程式：若成對的空隙數量為 3 視為有空隙（6 顆積木情境）
            IS_GAP = len(gap_pairs) // 2 == 3
            if MODE_SIDE == 0:  # 階梯模式遇到空隙降為 1 分
                SCORE = 1
        else:
            if MODE_SIDE == 1:  # 金字塔模式沒空隙降為 1 分
                SCORE = 1

    # 分層並做模式判定
    grouper = LayerGrouping(layer_ratio=0.2)
    layers = grouper.group_by_y(centroids, boxes=boxes)

    if MODE_SIDE == 0: # 階梯模式
        stair_checker = StairChecker()
        result, _ = stair_checker.check(layers)
        if not result:
            SCORE = 0
    elif MODE_SIDE == 1: # 金字塔模式
        # 估計單塊寬度
        block_width = (
            (np.mean([b[2] - b[0] for b in boxes]) // 2) if len(boxes) > 0 else 0
        )
        pyramid_checker = PyramidCheck()
        is_pyramid, _ = pyramid_checker.check_pyramid(layers, block_width, IS_GAP)
        if not is_pyramid:
            SCORE = 0

    return SCORE

# ====================================================================
# === 主程式執行區塊
# ====================================================================

if __name__ == "__main__":
    
    # --- 圖片路徑設定 ---
    # 請根據您的實際情況修改這些路徑
    SIDE_IMG_PATH = r'Pyramid_side.jpg'
    TOP_IMG_PATH = r'top_2.jpg'
    MODEL_PATH = r"toybrick.pt"

    # --- 載入模型 ---
    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"錯誤：載入 YOLO 模型失敗 (路徑: {MODEL_PATH})。請確保檔案存在。")
        sys.exit(1)


    # --- 1. 執行側視圖分析 (main_side.py 的邏輯) ---
    score_side = -1
    try:
        score_side = analyze_image_side(SIDE_IMG_PATH, yolo_model)
        print(f"側視圖 ({SIDE_IMG_PATH}) 得分: {score_side}")
    except ValueError as e:
        print(f"側視圖分析失敗: {e}")
    except Exception as e:
        print(f"側視圖分析時發生錯誤: {e}")


    # --- 2. 執行俯視圖分析 (main_top.py 的邏輯) ---
    score_top = -1
    try:
        frame_top = cv2.imread(TOP_IMG_PATH)
        if frame_top is None:
            raise ValueError("讀取俯視圖失敗")
            
        initial_score = 2
        analyzed_frame, summary, score_top = analyze_image_top(frame_top, yolo_model, initial_score)
        print(f"俯視圖 ({TOP_IMG_PATH}) 檢測結果: {summary}")
        print(f"俯視圖得分: {score_top}")

        # 顯示結果 (可選，但為了排版，將此註解)
        # cv2.imshow("TOP Detection Result", analyzed_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except ValueError as e:
        print(f"俯視圖分析失敗: {e}")
    except Exception as e:
        print(f"俯視圖分析時發生錯誤: {e}")


    # --- 3. 輸出最低得分 ---
    
    valid_scores = [s for s in [score_side, score_top] if s != -1]
    
    if not valid_scores:
        print("\n總結：兩項分析皆失敗或未執行，無法計算最低得分。")
        final_score = -1
    else:
        final_score = min(valid_scores)
        print(f"\n最終最低得分：{final_score}")