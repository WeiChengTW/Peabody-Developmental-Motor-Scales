# main.py — 單線分析（單圖+單一 px_cm.json）
import os
import json
import cv2
from final import find_baseline_and_show_all
import sys
from pathlib import Path

CROP_FOLDER = r"ch2-t4\new"
BASE_DIR = Path(__file__).resolve().parent
PXCM_JSON = BASE_DIR.parent / "px2cm.json"


def return_score(score):
    sys.exit(int(score))


def main():
    # Step 1) 讀比例
    if not os.path.exists(PXCM_JSON):
        raise FileNotFoundError(f"找不到 {PXCM_JSON}")
    with open(PXCM_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pixel_per_cm = float(obj.get("pixel_per_cm", 0.0))
    if pixel_per_cm <= 0:
        raise ValueError(f"{PXCM_JSON} 內容無效: {obj}")

    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        image_path = rf"kid\{uid}\{img_id}.jpg"
    # Step 2) 指定一張圖（改這裡）
    # img_num = 1
    # img_path = os.path.join(CROP_FOLDER, f"new{img_num}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到 {image_path}")

    # Step 3) 執行分析
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取影像：{image_path}")

    print(
        f"\n=== 處理 {os.path.basename(image_path)} | pixel_per_cm={pixel_per_cm:.4f} ==="
    )
    score = find_baseline_and_show_all(img, pixel_per_cm)
    # print(f"得分：{score}")
    return_score(score)


if __name__ == "__main__":
    main()
