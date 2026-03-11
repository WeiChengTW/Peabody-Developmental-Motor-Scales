# get_cm1.py
import cv2
import numpy as np

def auto_crop_wood_board(image, debug=False, allow_fallback=True):
    """
    嘗試只留下木板區域；若偵測失敗且 allow_fallback=True 則回傳原圖。
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 木板顏色 HSV 範圍
    lower_wood = np.array([5, 10, 50])
    upper_wood = np.array([45, 160, 255])
    mask = cv2.inRange(hsv, lower_wood, upper_wood)

    k_close = np.ones((25, 25), np.uint8)
    k_open = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if allow_fallback: return image
        raise ValueError("找不到木板區域")

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.1 * (h * w):
        if allow_fallback: return image
        raise ValueError("木板面積過小")

    x, y, ww, hh = cv2.boundingRect(biggest)
    x = max(x - 10, 0)
    y = max(y - 10, 0)
    ww = min(ww + 20, w - x)
    hh = min(hh + 20, h - y)

    return image[y : y + hh, x : x + ww]

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
        return image # 找不到就回傳原圖

    max_cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(max_cnt, 0.02 * cv2.arcLength(max_cnt, True), True)
    
    if len(approx) < 4:
        return image # 找不到四邊形就回傳原圖

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

    dst = np.array([
        [0, 0], 
        [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight - 1], 
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if warped.shape[0] > 2 * trim and warped.shape[1] > 2 * trim:
        return warped[trim:-trim, trim:-trim]
    
    return warped