import cv2
import numpy as np
import glob
import os

ori_area = 21576


def detect_paper_area(image_path):
    # 支援直接傳入影像陣列
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖片: {image_path}")
            return [], None, []
    else:
        img = image_path.copy()
    # 先裁切中央區域
    h, w = img.shape[:2]
    crop_size = int(min(h, w) * 0.8)
    cx, cy = w // 2, h // 2
    x1 = max(cx - crop_size // 2, 0)
    y1 = max(cy - crop_size // 2, 0)
    x2 = min(cx + crop_size // 2, w)
    y2 = min(cy + crop_size // 2, h)
    img = img[y1:y2, x1:x2]
    # 放大1.5倍
    img = cv2.resize(img, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = edges.shape
    img_center = (w // 2, h // 2)

    # 直接計算所有物件的總面積
    total_area = 0
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        total_area += area
        valid_contours.append(cnt)

    # 畫出所有物件輪廓
    img_contour = img.copy()
    if valid_contours:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, valid_contours, -1, 255, -1)
        color_layer = np.zeros_like(img)
        color_layer[:, :] = (0, 0, 255)
        alpha = 0.4
        img_contour = np.where(
            mask[..., None] == 255,
            (img * (1 - alpha) + color_layer * alpha).astype(np.uint8),
            img,
        )
        cv2.drawContours(img_contour, valid_contours, -1, (0, 0, 255), 2)

    print(f"畫面中物件總面積: {total_area:.2f}")
    return total_area, img_contour, valid_contours


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟相機")
        exit()
    print("按下 ENTER 鍵偵測並輸出當下畫面物件總面積，ESC 離開")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取相機畫面")
            break
        # 先顯示畫面
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == 13:  # ENTER 鍵
            area, img_contour, _ = detect_paper_area(frame)
            if img_contour is not None:
                cv2.imshow("detect_paper_area", img_contour)
                cv2.waitKey(0)
                cv2.destroyWindow("detect_paper_area")
        elif key == 27:  # ESC 鍵
            break
    cap.release()
    cv2.destroyAllWindows()
