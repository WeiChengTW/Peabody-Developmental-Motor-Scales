import cv2
import numpy as np
import glob
import os


def auto_crop_wood_board(image, debug=True):
    """
    只留下木板區域（排除外圍雜訊地墊），然後回傳裁切後畫面
    """
    h, w = image.shape[:2]
    # 木板顏色 HSV 範圍（依實際狀況可調整）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_wood = np.array([10, 20, 80])
    upper_wood = np.array([40, 120, 255])
    mask = cv2.inRange(hsv, lower_wood, upper_wood)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
    # 找最大輪廓（即木板區）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("找不到木板區域")
    biggest = max(contours, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(biggest)
    # 四周留 buffer
    x, y, ww, hh = (
        max(x - 10, 0),
        max(y - 10, 0),
        min(ww + 20, w - x),
        min(hh + 20, h - y),
    )
    crop_board = image[y : y + hh, x : x + ww]
    # if debug:
    #     cv2.imshow("Wood Board Only", crop_board)
    #     cv2.waitKey(0)
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
    input_folder = "image"
    output_folder = "new"

    os.makedirs(output_folder, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

    for idx, img_path in enumerate(img_paths, start=1):
        image = cv2.imread(img_path)
        if image is None:
            print(f"無法讀取：{img_path}")
            continue
        try:
            crop_board = auto_crop_wood_board(image, debug=False)
            clean_paper = auto_crop_paper(crop_board, trim=12, debug=False)
            out_path = os.path.join(output_folder, f"new{idx}.jpg")
            cv2.imwrite(out_path, clean_paper)
            print(f"{img_path} → {out_path}")
        except Exception as e:
            print(f"{img_path} 發生錯誤：{e}")
