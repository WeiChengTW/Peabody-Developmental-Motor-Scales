import cv2
import numpy as np
import glob
import os
from get_cm1 import auto_crop_paper, auto_crop_wood_board
from correct import analyze_image
import sys
import json

if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = rf"kid\{uid}\{img_id}.jpg"
    image_path = rf"kid\cgu\ch2-t6.jpg"
    uid = "cgu"
    img_id = "ch2-t6"
    output_folder = "new"

    os.makedirs(output_folder, exist_ok=True)

    # img=6
    # img_path=rf"image\{img}.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取：{image_path}")
        # continue
    try:
        crop_board = auto_crop_wood_board(image, debug=False)
        clean_paper = auto_crop_paper(crop_board, trim=12, debug=False)
        out_path = os.path.join(output_folder, f"ch2-t6/new/{img_id}.jpg")
        cv2.imwrite(out_path, clean_paper)
        print(f"{image_path} → {out_path}")
    except Exception as e:
        print(f"{image_path} 發生錯誤：{e}")

    result = analyze_image(out_path, dot_distance_cm=10.0)
    print("得分：", result["score"])
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
