# main.py — ONE_PARALLEL：裁切+像素/公分 → 單線分析（合併版）
import os
import glob
import json
import argparse
import cv2

# 你的模組
from get_pixel_per_cm import batch_crop_and_measure, CROP_FOLDER, ORIG_FOLDER, PIXEL_MAP_JSON
from final import find_baseline_and_show_all

def run_analysis(crop_dir: str, ppcm_map_path: str, show_windows: bool = True):
    """讀取 new/ 與 pixel_per_cm_map.json，逐張呼叫 find_baseline_and_show_all"""
    if not os.path.exists(ppcm_map_path):
        raise FileNotFoundError(f"找不到 {ppcm_map_path}，請先執行裁切與量測步驟。")

    with open(ppcm_map_path, "r", encoding="utf-8") as f:
        ratio_map = json.load(f)  # {"new1.jpg": 12.34, ...}

    images = sorted(
        glob.glob(os.path.join(crop_dir, "*.jpg")) +
        glob.glob(os.path.join(crop_dir, "*.jpeg")) +
        glob.glob(os.path.join(crop_dir, "*.png"))
    )
    if not images:
        raise ValueError(f"資料夾 {crop_dir} 沒有裁切後的圖片。")

    for path in images:
        fname = os.path.basename(path)
        ppcm = float(ratio_map.get(fname, 0.0))
        if ppcm <= 0:
            print(f"⚠️ {fname} 沒有對應的 pixel_per_cm，略過")
            continue

        print(f"\n=== 處理 {fname} | pixel_per_cm={ppcm:.4f} ===")
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 無法讀取：{path}")
            continue

        find_baseline_and_show_all(img, ppcm)
        if show_windows:
            cv2.waitKey(0); cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="一條水平線分析：裁切+像素/公分 → 基準線偏離分析"
    )
    parser.add_argument("--images", default=ORIG_FOLDER, help="原始圖資料夾（預設：images）")
    parser.add_argument("--out",    default=CROP_FOLDER, help="裁切輸出資料夾（預設：new）")
    parser.add_argument("--ppcm_json", default=PIXEL_MAP_JSON, help="像素/公分對照JSON（預設：pixel_per_cm_map.json）")
    parser.add_argument("--skip_crop", action="store_true", help="略過裁切/量測步驟（直接使用既有 new/ 與 JSON）")
    parser.add_argument("--no_show", action="store_true", help="不開視窗（只列印結果）")
    args = parser.parse_args()

    # Step 1) 裁切 + 量測（可跳過）
    if not args.skip_crop:
        print("[Step 1] 批次裁切 + 估算 pixel_per_cm ...")
        os.makedirs(args.out, exist_ok=True)
        batch_crop_and_measure(src_dir=args.images, out_dir=args.out, save_map=args.ppcm_json)
    else:
        print("[Step 1] 跳過裁切/量測，直接使用現有檔案。")

    # Step 2) 分析
    print("[Step 2] 逐張分析 ...")
    run_analysis(crop_dir=args.out, ppcm_map_path=args.ppcm_json, show_windows=(not args.no_show))

if __name__ == "__main__":
    main()
