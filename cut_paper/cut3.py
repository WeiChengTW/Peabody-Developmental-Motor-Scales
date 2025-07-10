import cv2
import numpy as np

# 啟用相機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟相機")
    exit()

scope = 0.75  # 裁切比例
org_area = None
print("請將紙張放在鏡頭前，按 ENTER 鍵偵測並輸出面積，ESC 設定基準面積，Q 離開...")
while True:
    ret, img = cap.read()
    if not ret:
        print("無法讀取影像")
        break

    H, W, _ = img.shape
    crop_w, crop_h = int(W * scope), int(H * scope)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # 裁切並放大
    cropped = img[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
    show_img = zoomed.copy()

    # 轉灰階
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)

    # 邊緣偵測
    edges = cv2.Canny(gray, 50, 100)

    # 找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 畫輪廓
    contour_img = zoomed.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    # === 從畫好綠色線的圖中找面積 ===
    # 先轉 HSV，抓出綠線遮罩
    hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 找綠色輪廓
    green_contours, _ = cv2.findContours(
        mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 計算所有綠色輪廓包圍起來的面積
    green_area_total = sum(cv2.contourArea(cnt) for cnt in green_contours)

    # 塗滿區域（藍色顯示計算區域）
    filled_mask = np.zeros_like(zoomed)
    cv2.drawContours(
        filled_mask, green_contours, -1, (0, 111, 255), thickness=cv2.FILLED
    )
    highlighted = cv2.addWeighted(contour_img, 1.0, filled_mask, 0.4, 0)
    cv2.putText(
        highlighted,
        f"Area: {green_area_total:.0f} px",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # cv2.imshow("Canny", edges)
    cv2.imshow("Contours", highlighted)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # ENTER 鍵
        if org_area is None:
            print("請先按 ESC 設定基準面積！")
        else:
            print(f"\n偵測面積：{green_area_total:.2f} 像素")
            result_area = org_area - green_area_total
            if result_area > 0:
                print(f"往內剪了 {result_area:.2f} 像素")
            else:
                print(f"往外剪了：{abs(result_area):.2f} 像素")
            print(f"比例：{(abs(result_area)/org_area)*100:.2f}%")
    elif key == 27:  # ESC 鍵
        org_area = green_area_total
        print(f"已設定基準面積：{org_area:.2f} 像素")
    elif key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()
