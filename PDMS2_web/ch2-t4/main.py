# main.py — 單線分析（kid\<uid>\<img_id>.jpg → 儲存到 kid\<uid>\<img_id>\）
import os
import json
import cv2
import sys
from final import find_baseline_and_show_all

PXCM_JSON = "px2cm.json"


def return_score(score: int):
    """與 ch2-t5 一樣，用 exit code 回傳分數"""
    sys.exit(int(score))


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def main():
    # 1) 讀比例檔
    if not os.path.exists(PXCM_JSON):
        raise FileNotFoundError(f"找不到比例檔：{PXCM_JSON}")
    with open(PXCM_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pixel_per_cm = float(obj.get("pixel_per_cm", 0.0))
    if pixel_per_cm <= 0:
        raise ValueError(f"{PXCM_JSON} 內的 pixel_per_cm 非正值：{obj}")

    # 2) 解析參數與輸入影像路徑（與 ch2-t5 相同介面）
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = rf"kid\{uid}\{img_id}.jpg"
    else:
        # 後備：避免未給參數時變數未定義（開發測試用）
        uid = "devuser"
        img_id = "ch2-t4"
        image_path = rf"kid\{uid}\{img_id}.jpg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    # 3) 讀影像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"圖片讀取失敗：{image_path}")

    # 4) 分析（盡量相容：優先嘗試取回 (score, result_img)）
    result_img = None
    try:
        score, result_img = find_baseline_and_show_all(img, pixel_per_cm)
    except TypeError:
        # 舊版只回傳 score
        score = find_baseline_and_show_all(img, pixel_per_cm)

    # 5) 準備輸出路徑：存到 kid\<uid>\<img_id>\ 下
    out_dir = rf"kid\{uid}\{img_id}"
    ensure_dir(out_dir)
    out_name = f"{img_id}_result.jpg"
    out_path = os.path.join(out_dir, out_name)

    # 6) 寫出結果圖（若無 result_img，至少把原圖存一份）
    to_save = result_img if result_img is not None else img
    ok = cv2.imwrite(out_path, to_save)
    # 顯示結果圖（可選）
    # cv2.imshow("Result", to_save)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if not ok:
        print(f"⚠️ 影像儲存失敗：{out_path}", file=sys.stderr)

    # 7) 與 ch2-t5 一樣：stdout 印出「圖檔名」，exit code 回傳分數
    #    這裡依你的要求只印檔名（不含路徑），若要相對路徑可改印：{uid}\\{img_id}\\{out_name}
    sys.stdout.write(out_name + "\n")
    return_score(score)


if __name__ == "__main__":
    main()
