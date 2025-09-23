import cv2
import os
from get_pixel_per_cm import crop_paper
from exceed_correct import detect_horizontal_lines
from draw_range_correct import analyze_paint  # 或 analyze_image

if __name__ == "__main__":
    img = 1
    in_path  = rf"image\{img}.jpg"
    out_dir  = "new"
    out_path = os.path.join(out_dir, f"new{img}.jpg")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 裁切
    warped = crop_paper(in_path, output_path=out_path, show_debug=False)

    # 2) 找兩條線（直接用記憶體影像也可以）
    lines = detect_horizontal_lines(warped, show_debug=False)
    if len(lines) < 2:
        raise RuntimeError("偵測不到兩條主線")
    y_top, y_bot = min(lines), max(lines)

    # 3) 分析塗色 + 超出
    result = analyze_paint(warped, y_top, y_bot, show_windows=True)
    print("得分：", result["score"])
    print("說明：", result["rule"])
