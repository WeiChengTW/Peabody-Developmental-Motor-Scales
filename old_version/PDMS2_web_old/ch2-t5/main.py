import cv2
import os
from cut_paper import crop_paper
from exceed_correct import detect_horizontal_lines
from draw_range_correct import analyze_paint  # 或 analyze_image
import json
import sys

if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        image_path = rf"kid\{uid}\{img_id}.jpg"
    # img = 1
    # in_path = os.path.join("image", f"{img}.jpg")  # ✅ 跨平台路徑

    out_path = rf"ch2-t5\new\{img_id}.jpg"

    # 1) 裁切
    warped = crop_paper(image_path, output_path=out_path, show_debug=False)
    if warped is None:
        # 若你的 crop_paper 只存檔不回傳，可改成：warped = cv2.imread(out_path)
        warped = cv2.imread(out_path)
        if warped is None:
            raise RuntimeError(f"裁切後讀不到影像：{out_path}")

    # 2) 找兩條線（函式已改為回傳 (y_top, y_bot)）
    y_top, y_bot = detect_horizontal_lines(warped, show_debug=False)
    if y_top is None or y_bot is None:
        raise RuntimeError("偵測不到兩條主線")

    # 3) 分析塗色 + 超出
    result = analyze_paint(warped, int(y_top), int(y_bot), show_windows=True)
    # print("得分：", result["score"])
    # print("說明：", result["rule"])
    score = result["score"]
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
