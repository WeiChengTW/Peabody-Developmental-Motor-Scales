import cv2
import numpy as np
from skimage.morphology import skeletonize


def detect_notch_on_hollowline(img_path, scale=4):
    # 1. 讀圖與二值化
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)  # 0/1

    # 2. 閉運算補縫，讓骨架只留一條線
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. 骨架化
    skeleton = skeletonize(closed).astype(np.uint8)

    # 4. 找骨架端點
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x]:
                nb = np.sum(skeleton[y - 1 : y + 2, x - 1 : x + 2]) - 1
                if nb == 1:
                    endpoints.append((x, y))
    endpoints = list({(x, y) for (x, y) in endpoints})

    # 5. 畫圖
    out = cv2.cvtColor((skeleton * 255), cv2.COLOR_GRAY2BGR)
    for idx, pt in enumerate(endpoints):
        color = (0, 0, 255) if idx == 0 else (255, 0, 0)
        cv2.circle(out, pt, 7, color, -1)
    out_big = cv2.resize(
        out,
        (out.shape[1] * scale, out.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow("Skeleton Notch Detection (Hollowline)", out_big)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. 回報
    print(f"骨架端點數: {len(endpoints)}")
    print("端點座標:", endpoints)
    if len(endpoints) >= 2:
        print("【偵測結果：有缺口！】")
    else:
        print("【偵測結果：無缺口】")


# 用法 73
detect_notch_on_hollowline(r"result/Square/141_1_Square_004_conf0.84.jpg")
