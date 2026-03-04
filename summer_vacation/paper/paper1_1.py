# 用 OpenCV 讀取圖片並用 Canny 邊緣偵測描繪輪廓
import cv2
import glob
import os


def draw_edges_on_images(image_folder, pattern="img*.png"):
    # 取得所有符合條件的圖片檔案
    image_paths = sorted(glob.glob(os.path.join(image_folder, pattern)))
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            continue
        # 使用 Canny 邊緣偵測
        edges = cv2.Canny(img, 50, 200)

        # 輪廓合併：線條太粗時，將靠近的輪廓合併為一條
        import numpy as np

        def merge_close_contours(contours, threshold=10):
            merged = []
            used = [False] * len(contours)
            for i, cnt1 in enumerate(contours):
                if used[i]:
                    continue
                group = [cnt1]
                used[i] = True
                for j, cnt2 in enumerate(contours):
                    if i == j or used[j]:
                        continue
                    # 判斷兩個輪廓是否有點距離小於 threshold
                    min_dist = np.min(
                        [np.linalg.norm(p1[0] - p2[0]) for p1 in cnt1 for p2 in cnt2]
                    )
                    if min_dist < threshold:
                        group.append(cnt2)
                        used[j] = True
                merged_cnt = np.vstack(group)
                merged.append(merged_cnt)
            return merged

        # 膨脹邊緣，讓線條連成一體（更靈敏，kernel縮小，膨脹次數減少）
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print(f"找不到輪廓: {img_path}")
            continue
        merged_contours = merge_close_contours(contours, threshold=5)
        # 找最大面積的合併輪廓
        max_contour = max(merged_contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
        print(f"{os.path.basename(img_path)} 最大輪廓面積: {max_area}")

        # 用彩色顯示最大合併輪廓
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [max_contour], -1, (0, 255, 0), 2)

        # --- Black Lines (edges in zone) ---
        # 只顯示最大輪廓區域內的邊緣（模仿 paper2 的黑線遮罩）
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        black_lines = cv2.bitwise_and(edges, edges, mask=mask)
        # 形態學開運算去雜點，再膨脹
        kernel2 = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(black_lines, cv2.MORPH_OPEN, kernel2, iterations=1)
        opened = cv2.dilate(opened, kernel2, iterations=1)
        _, opened_strict = cv2.threshold(opened, 32, 255, cv2.THRESH_BINARY)

        # 顯示結果
        cv2.imshow("Black Lines (edges in zone)", opened_strict)
        cv2.imshow("Black Lines (raw mask)", black_lines)
        cv2.imshow("Max Contour + Edges", img_color)
        cv2.imshow("Edges", edges)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 按 Q 或 q 直接中斷所有顯示
        if key in [ord("q"), ord("Q")]:
            break


if __name__ == "__main__":
    floder = "."
    # path = "img18.png"
    path = "img*.png"
    draw_edges_on_images(floder, path)
