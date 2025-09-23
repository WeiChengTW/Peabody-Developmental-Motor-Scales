# main.py — 單線分析（單圖+單一 px_cm.json）
import os
import json
import cv2
from final import find_baseline_and_show_all

CROP_FOLDER = "new"
PXCM_JSON   = "px_cm.json"

def main():
    # Step 1) 讀比例
    if not os.path.exists(PXCM_JSON):
        raise FileNotFoundError(f"找不到 {PXCM_JSON}")
    with open(PXCM_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pixel_per_cm = float(obj.get("pixel_per_cm", 0.0))
    if pixel_per_cm <= 0:
        raise ValueError(f"{PXCM_JSON} 內容無效: {obj}")

    # Step 2) 指定一張圖（改這裡）
    img_num = 1
    img_path = os.path.join(CROP_FOLDER, f"new{img_num}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到 {img_path}")

    # Step 3) 執行分析
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"無法讀取影像：{img_path}")

    print(f"\n=== 處理 {os.path.basename(img_path)} | pixel_per_cm={pixel_per_cm:.4f} ===")
    find_baseline_and_show_all(img, pixel_per_cm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
