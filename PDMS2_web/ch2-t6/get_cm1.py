import cv2
import numpy as np
import glob
import os
import re


def auto_crop_wood_board(image, debug=False, allow_fallback=True):
    """
    嘗試只留下木板區域；若偵測失敗且 allow_fallback=True 則回傳原圖，避免整段流程中斷。
    """
    h, w = image.shape[:2]

    # （1）木板顏色 HSV 範圍：可視現場光線微調
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_wood = np.array([5, 10, 50])  # 放寬一些
    upper_wood = np.array([45, 160, 255])

    mask = cv2.inRange(hsv, lower_wood, upper_wood)

    # （2）形態學：閉運算填洞、再做一次開運算去雜點
    k_close = np.ones((25, 25), np.uint8)
    k_open = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # （3）找最大輪廓，並做最小面積比過濾，避免選到小雜訊
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if allow_fallback:
            if debug:
                print("⚠️ 木板未偵測到，改回傳原圖")
            return image
        raise ValueError("找不到木板區域")

    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < 0.1 * (h * w):  # 面積過小視為失敗（閾值可調）
        if allow_fallback:
            if debug:
                print("⚠️ 木板面積太小，改回傳原圖")
            return image
        raise ValueError("找不到木板區域（面積過小）")

    x, y, ww, hh = cv2.boundingRect(biggest)

    # （4）四周留一點 buffer，避免剛好切到邊緣
    x = max(x - 10, 0)
    y = max(y - 10, 0)
    ww = min(ww + 20, w - x)
    hh = min(hh + 20, h - y)

    crop_board = image[y : y + hh, x : x + ww]

    if debug:
        # debug 可視化（選擇性）
        vis = image.copy()
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        # cv2.imshow("Wood Board BBox", vis); cv2.waitKey(0); cv2.destroyAllWindows()

    return crop_board


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_crop_paper(image, trim=10, debug=True):
    """
    偵測紙張四邊，做透視校正，只留下紙
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("沒找到紙張邊緣")
    max_cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(max_cnt, 0.02 * cv2.arcLength(max_cnt, True), True)
    if len(approx) < 4:
        raise ValueError("找不到 4 個角點")
    pts = approx.reshape(-1, 2)
    if len(pts) > 4:
        hull = cv2.convexHull(approx)
        pts = hull.reshape(-1, 2)
        if len(pts) > 4:
            rect = order_points(pts)
        else:
            rect = order_points(pts[:4])
    else:
        rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 可選 trim
    if warped.shape[0] > 2 * trim and warped.shape[1] > 2 * trim:
        cropped = warped[trim:-trim, trim:-trim]
        cleaned = cv2.resize(cropped, (maxWidth, maxHeight))
    else:
        cleaned = warped.copy()
    # if debug:
    #     cv2.imshow("Paper Only (Final Crop)", cleaned)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    return cleaned


# ===== 多檔案批次處理 =====
if __name__ == "__main__":
    import os, glob, re

    # 是否嘗試先裁木板；預設 False（直接拿原圖做紙張裁切）
    USE_BOARD = False

    BASE = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(BASE, "image")
    output_folder = os.path.join(BASE, "new")
    os.makedirs(output_folder, exist_ok=True)

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(input_folder, p)))

    # 去重（避免大小寫/重覆收集）
    seen, unique_paths = set(), []
    for p in img_paths:
        ap = os.path.abspath(p)
        key = os.path.normcase(ap)
        if key not in seen:
            seen.add(key)
            unique_paths.append(ap)
    img_paths = unique_paths

    if not img_paths:
        print(f"⚠️ 在 {input_folder} 找不到影像（支援 jpg/jpeg/png）")
        raise SystemExit(0)

    # 依檔名中的數字排序（沒數字放後）
    def sort_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"\d+", name)
        num = int(m.group()) if m else float("inf")
        return (num, name.lower())

    img_paths = sorted(img_paths, key=sort_key)

    for i, img_path in enumerate(img_paths, start=1):
        image = cv2.imread(img_path)
        if image is None:
            print(f"無法讀取：{os.path.basename(img_path)}")
            continue

        # 先決定輸出檔名（優先用原檔名中的數字）
        stem = os.path.splitext(os.path.basename(img_path))[0]
        m = re.search(r"\d+", stem)
        out_name = f"new{int(m.group())}.jpg" if m else f"new{i}.jpg"
        out_path = os.path.join(output_folder, out_name)

        try:
            # ➤ 不一定要偵測木板：預設直接用原圖
            crop_src = image
            if USE_BOARD:
                try:
                    crop_src = auto_crop_wood_board(image, debug=False)
                except Exception as _:
                    # 木板偵測失敗就退回用原圖
                    crop_src = image

            clean_paper = auto_crop_paper(crop_src, trim=12, debug=False)

            ok = cv2.imwrite(out_path, clean_paper)
            if ok:
                print(f"{os.path.basename(img_path)} → {out_name}")
            else:
                print(f"⚠️ 影像儲存失敗（cv2.imwrite 回傳 False）：{out_name}")
        except Exception as e:
            print(f"{os.path.basename(img_path)} 發生錯誤：{e}")
