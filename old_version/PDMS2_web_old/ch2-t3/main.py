# -*- coding: utf-8 -*-
import os, glob
from Analyze_graphics import Analyze_graphics
from scorec import CrossScorer  # 你的 CrossScorer
import sys
import json


def main():
    # === 原圖（只處理這一張）===
    # image_path = r"kid\cgu\ch2-t3.jpg"
    # uid = "cgu"
    # img_id = "ch2-t3"
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = rf"kid\{uid}\{img_id}.jpg"

    # === 已知比例 ===
    # CM_PER_PIXEL = 1 / 40.47139447721568  # 例：1 像素 ≈ 0.038 cm
    json_path = "px2cm.json"
    if json_path is not None:
        with open(json_path, "r") as f:
            data = json.load(f)
            pixel_per_cm = data.get("pixel_per_cm", 19.597376925845985)
    CM_PER_PIXEL = 1 / pixel_per_cm
    # === 評分參數 ===
    ANGLE_MIN, ANGLE_MAX = 70.0, 110.0
    MAX_SPREAD_CM = 0.6

    print("\n== Analyze_graphics 裁切 ==\n")
    segmenter = Analyze_graphics()
    out = segmenter.infer_and_draw(image_path)

    # ---- 規範化成候選清單 ----
    if out is None:
        raise FileNotFoundError("沒有可評分的圖形 patch")
    elif isinstance(out, str):
        patch_files = [out]
    elif isinstance(out, (list, tuple, set)):
        patch_files = list(out)
    elif isinstance(out, dict):
        patch_files = out.get("paths", []) or out.get("save_paths", [])
        if isinstance(patch_files, str):
            patch_files = [patch_files]
    else:
        patch_files = []

    # 展開資料夾、只留圖檔
    IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    all_files = []
    for f in patch_files:
        f = os.path.abspath(os.path.normpath(f))
        if os.path.isdir(f):
            for pat in IMAGE_EXTS:
                all_files.extend(glob.glob(os.path.join(f, pat)))
        elif os.path.isfile(f):
            all_files.append(f)

    # 過濾掉 _binary/_skeleton
    all_files = [
        f
        for f in all_files
        if "_binary" not in os.path.basename(f).lower()
        and "_skeleton" not in os.path.basename(f).lower()
    ]

    if not all_files:
        raise FileNotFoundError("沒有找到可評分的圖檔")

    # ✅ 只取第一張，不跑迴圈
    patch_file = all_files[0]
    print("本次要評分的檔案：", patch_file)

    scorer = CrossScorer(
        cm_per_pixel=CM_PER_PIXEL,
        angle_min=ANGLE_MIN,
        angle_max=ANGLE_MAX,
        max_spread_cm=MAX_SPREAD_CM,
        out_dir="ch2-t3\\output",
        output_jpg_quality=95,
    )

    res, bin_path, skel_path, vis_path = scorer.score_image(patch_file)

    # 直接印一次結果
    print(f"\n== 評分結果 ==")
    print(f"檔案: {patch_file}")
    print(f"分數: {res['score']}  |  理由: {res['reason']}")
    print(f"角度: {res['theta_deg']}")
    print(f"Arms(px): {res['arms_px']}")
    if res["arms_cm"] is not None:
        print(f"Arms(cm): {res['arms_cm']}")
        print(f"Spread(cm): {res['spread_cm']}")
    else:
        print(f"Spread(px): {res['spread_px']}")
    print(f"輸出：{vis_path}")
    score = res["score"]
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


if __name__ == "__main__":
    main()
