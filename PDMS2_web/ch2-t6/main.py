import cv2
import numpy as np
import glob
import os
from pathlib import Path
from get_cm1 import auto_crop_paper, auto_crop_wood_board
from correct import analyze_image
import sys
import json


def return_score(score):
    sys.exit(int(score))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        # image_path = Path(rf"kid\{uid}\{img_id}.jpg")
        image_path = os.path.join('kid', uid, f'{img_id}.jpg')

    # === 以這支 .py 所在資料夾為基準，避免工作目錄不同造成找不到檔案 ===
    # BASE = Path(__file__).resolve().parent
    # img = 1  # 你要處理的圖片編號

    # === 乾淨的輸入/輸出路徑（不要把中途資料夾夾到檔名裡）===
    # image_path = rf"kid\{uid}\{img_id}.jpg"        # e.g. .../ch2-t6/image/6.jpg
    # output_dir = Path(rf"PDMS2_web\ch2-t6\new")  # e.g. .../ch2-t6/new/
    output_dir = os.path.join('PDMS2_web', 'ch2-t6', 'new')
    # output_dir.mkdir(exist_ok=True)
    # out_path = output_dir / f"new{img_id}.jpg"  # e.g. .../ch2-t6/new/new6.jpg
    out_path = os.path.join(output_dir, f'new{img_id}.jpg')
    # === 讀圖 ===
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"無法讀取：{image_path}")
        return_score(0)  # 或者 sys.exit(1)

    # === 木板偵測：失敗就用原圖（你說「不一定要偵測木板」）===
    try:
        board = auto_crop_wood_board(image, debug=False)
    except Exception:
        print(f"⚠️ {image_path.name} 木板偵測失敗 → 改用原圖裁紙")
        board = image

    # === 裁紙（透視校正）===
    try:
        paper = auto_crop_paper(board, trim=12, debug=False)
    except Exception as e:
        print(f"裁紙失敗：{e}")
        return_score(0)

    # === 先寫檔，成功才做 analyze ===
    ok = cv2.imwrite(out_path, paper)
    if not ok:
        print(f"⚠️ 影像儲存失敗：{out_path}")
        return_score(0)

    print(f"{image_path.name} → {out_path.name}")

    # === 呼叫分析（傳「絕對路徑」最穩）===
    try:
        score, result_img = analyze_image(str(out_path), dot_distance_cm=10.0)
        # result_path = rf"kid\{uid}\{img_id}_result.jpg"
        result_path = os.path.join('kid', uid, f"{img_id}_result.jpg")
        print("得分：", score)
        return_score(score)
    except Exception as e:
        print(f"分析失敗：{e}")
        return_score(0)
