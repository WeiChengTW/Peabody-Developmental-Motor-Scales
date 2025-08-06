import cv2
import numpy as np
import os

def is_non_straight(img_path, threshold=0.98, debug=True, scale=3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"無法讀取: {img_path}")
        return False
    # 二值化
    binary = (img > 127).astype(np.uint8) * 255

    # 霍夫線變換
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=30, minLineLength=0.6*min(binary.shape), maxLineGap=10)
    if lines is None or len(lines) < 2:
        # 沒找到明顯線段，或數量太少，直接判非直線
        result = True
    else:
        # 判斷有沒有「橫直」兩條長直線
        angles = []
        lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            length = np.sqrt(dx**2 + dy**2)
            angles.append(angle)
            lengths.append(length)

        # 統計有沒有接近水平與垂直的長線
        horizontal = [l for a, l in zip(angles, lengths) if abs(a) < 20 or abs(a) > 160]
        vertical = [l for a, l in zip(angles, lengths) if 70 < abs(a) < 110]
        img_w, img_h = binary.shape[1], binary.shape[0]
        # 水平線必須佔寬度，直線必須佔高度的比例
        hor_ratio = max(horizontal)/img_w if horizontal else 0
        ver_ratio = max(vertical)/img_h if vertical else 0

        # 只要有一個比值低於 threshold（線太短或彎），就判「非直線」
        result = (hor_ratio < threshold) or (ver_ratio < threshold)

    if debug:
        print(f"{os.path.basename(img_path)}: 水平直線佔比={hor_ratio:.2f}, 垂直直線佔比={ver_ratio:.2f}")
        print("判斷：", "非直線" if result else "直線")
        color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(color, (x1, y1), (x2, y2), (0, 0, 255), 2)
        color = cv2.resize(color, (color.shape[1]*scale, color.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Straightness", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result

# ====== 支援 jpg, jpeg, png 副檔名 ======
src_folder = "test"
extensions = ('.jpg', '.jpeg', '.png')
for fname in os.listdir(src_folder):
    if not fname.lower().endswith(extensions):
        continue
    img_path = os.path.join(src_folder, fname)
    is_non_straight(img_path, threshold=0.98, debug=True, scale=3)
