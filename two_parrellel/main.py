import cv2
import os
from cut_paper import crop_paper
from exceed_correct import detect_horizontal_lines
from draw_range_correct import analyze_paint  # 或 analyze_image

if __name__ == "__main__":
    img = 1
    in_path  = os.path.join("image", f"{img}.jpg")  # ✅ 跨平台路徑
    out_dir  = "new"
    out_path = os.path.join(out_dir, f"new{img}.jpg")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 裁切
    warped = crop_paper(in_path, output_path=out_path, show_debug=False)
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
    print("得分：", result["score"])
    print("說明：", result["rule"])
