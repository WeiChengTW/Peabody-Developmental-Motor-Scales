import cv2
import numpy as np


# 讀取圖片
img = cv2.imread("img19.png")

# 轉為灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用 Canny 邊緣偵測
edges = cv2.Canny(gray, 150, 300)

# 根據 Canny 邊緣偵測結果尋找輪廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 假設最大輪廓為白紙
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # 在原圖上畫出白紙邊框
    result = img.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 只在白紙區域內尋找直線
    roi_edges = edges[y : y + h, x : x + w]
    lines = cv2.HoughLinesP(
        roi_edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10
    )

    # 標示出兩條線的端點
    if lines is not None and len(lines) >= 2:
        # 只取前兩條線
        for i in range(2):
            l = lines[i][0]
            pt1 = (l[0] + x, l[1] + y)
            pt2 = (l[2] + x, l[3] + y)
            cv2.line(result, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(result, pt1, 5, (255, 0, 0), -1)
            cv2.circle(result, pt2, 5, (255, 0, 0), -1)
    else:
        print("找不到足夠的線段")

    # 縮小顯示的視窗
    small_result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("White Paper Border and Lines", small_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("找不到白紙邊框")
