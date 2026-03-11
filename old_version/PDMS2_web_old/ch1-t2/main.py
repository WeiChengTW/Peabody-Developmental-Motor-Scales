import cv2
from ultralytics import YOLO
import numpy as np
from check_gap import CheckGap
from MaskAnalyzer import MaskAnalyzer
from StairChecker import StairChecker
from PyramidChecker import PyramidCheck
from LayerGrouping import LayerGrouping
import sys
import os
import json

# ===== 參數 =====
MODE = 1  # 0 = 階梯, 1 = 金字塔(一定要有空隙)
MODEL_PATH = r"ch1-t2\model\toybrick.pt"
CONF = 0.8
GAP_THRESHOLD_RATIO = 0.366  # 用平均 bbox 寬度 * 這個比例


def main(IMG_PATH):

    # ===== 讀圖 =====
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise ValueError(f"讀不到圖片：{IMG_PATH}")

    # ===== YOLO 偵測 =====
    model = YOLO(MODEL_PATH)
    results = model.predict(source=frame, conf=CONF, verbose=False)
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

        print(f"gap : {len(gap_pairs)}")
        if gap_pairs:
            # 原程式：若成對的空隙數量為 3 視為有空隙（6 顆積木情境）
            IS_GAP = len(gap_pairs) // 2 == 3
            if MODE == 0:  # 階梯模式遇到空隙降為 1 分
                SCORE = 1
        else:
            if MODE == 1:  # 金字塔模式沒空隙降為 1 分
                SCORE = 1

    # 分層並做模式判定
    grouper = LayerGrouping(layer_ratio=0.2)
    layers = grouper.group_by_y(centroids, boxes=boxes)

    if MODE == 0:
        stair_checker = StairChecker()
        result, _ = stair_checker.check(layers)
        if not result:
            SCORE = 0
    elif MODE == 1:
        # 估計單塊寬度（與原碼一致）
        block_width = (
            (np.mean([b[2] - b[0] for b in boxes]) // 2) if len(boxes) > 0 else 0
        )
        pyramid_checker = PyramidCheck()
        is_pyramid, _ = pyramid_checker.check_pyramid(layers, block_width, IS_GAP)
        if not is_pyramid:
            SCORE = 0

    return SCORE


if __name__ == "__main__":
    # IMG_PATH = r'1.jpg'
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = rf"kid\{uid}\{img_id}.jpg"
    score = main(image_path)
    print(f"score : {score}")
    result_file = "result.json"
    try:
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = {}
    except (json.JSONDecodeError, FileNotFoundError):
        results = {}

    # 確保 uid 存在於結果中
    if uid not in results:
        results[uid] = {}

    # 更新對應 uid 的關卡分數
    results[uid][img_id] = score

    # 儲存到 result.json
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"結果已儲存到 {result_file} - 用戶 {uid} 的關卡 {img_id} 分數: {score}")
