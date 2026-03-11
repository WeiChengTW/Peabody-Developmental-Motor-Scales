# main.py (修正版：支援中文路徑、正確存檔)
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import os
from pathlib import Path

# 引用模組
from get_cm1 import auto_crop_paper, auto_crop_wood_board
from correct import analyze_image

# 設定根目錄
ROOT = Path(__file__).resolve().parent.parent

def main():
    # 1. 接收參數
    if len(sys.argv) < 3:
        print("Usage: python main.py [uid] [task_id]")
        sys.exit(0)

    uid = sys.argv[1]
    img_id = sys.argv[2] 

    # 來源: kid/UID/Ch2-t6.jpg
    src_path = ROOT / "kid" / uid / f"{img_id}.jpg"
    # 結果: kid/UID/Ch2-t6_result.jpg
    res_path = ROOT / "kid" / uid / f"{img_id}_result.jpg"

    print(f"[-] Processing: {src_path}")

    if not src_path.exists():
        print(f"[Error] 檔案不存在: {src_path}")
        sys.exit(0)

    # 2. 讀取圖片 (支援中文路徑)
    img = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("[Error] 圖片讀取失敗")
        sys.exit(0)

    # 3. 影像前處理 (木板偵測 + 裁紙)
    try:
        board = auto_crop_wood_board(img, debug=False)
        paper = auto_crop_paper(board, trim=12, debug=False)
    except Exception as e:
        print(f"[Warning] 裁切失敗，使用原圖: {e}")
        paper = img

    # 4. 分析連線
    try:
        score, result_img = analyze_image(paper, dot_distance_cm=10.0)
    except Exception as e:
        print(f"[Error] 分析失敗: {e}")
        score = 0
        result_img = paper

    # 5. 儲存結果圖
    try:
        is_success, buffer = cv2.imencode(".jpg", result_img)
        if is_success:
            with open(str(res_path), "wb") as f:
                f.write(buffer)
            print(f"[Success] 結果圖已儲存: {res_path}")
        else:
            print("[Error] 圖片編碼失敗")
    except Exception as e:
        print(f"[Error] 寫入檔案失敗: {e}")

    # 6. 回傳分數
    sys.exit(score)

if __name__ == "__main__":
    main()