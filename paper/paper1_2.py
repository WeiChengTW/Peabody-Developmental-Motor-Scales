# 用 OpenCV 讀取圖片並用 Canny 邊緣偵測描繪輪廓
import cv2
import glob
import os
import numpy as np
from PIXEL_TO_CM import get_cm_per_pixel  # 假設你有這個轉換函式


def draw_edges_on_images(image_folder, pattern="img*.png"):
    # 取得所有符合條件的圖片檔案
    image_paths = sorted(glob.glob(os.path.join(image_folder, pattern)))
    for img_path in image_paths:
        # cm_per_pixel, _, _ = get_cm_per_pixel(img_path, cm_length=16, show=False)
        cm_per_pixel = 0.015694
        print(f"像素與公分比例: 1 px = {cm_per_pixel:.2f} cm")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            continue
        # 使用 Canny 邊緣偵測
        edges = cv2.Canny(img, 80, 200)

        # 輪廓合併：線條太粗時，將靠近的輪廓合併為一條

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
        # 取得最大輪廓邊框遮罩
        contour_border = np.zeros_like(black_lines)
        cv2.drawContours(
            contour_border, [max_contour], -1, 255, 5
        )  # 2為邊框寬度，可調整
        # 扣除邊框
        black_lines_wo_border = cv2.subtract(black_lines, contour_border)
        # 形態學開運算去雜點，再膨脹
        kernel2 = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(
            black_lines_wo_border, cv2.MORPH_OPEN, kernel2, iterations=1
        )
        opened = cv2.dilate(opened, kernel2, iterations=1)
        _, opened_strict = cv2.threshold(opened, 2, 255, cv2.THRESH_BINARY)

        # 顯示結果
        # cv2.imshow("Black Lines (edges in zone)", opened_strict)
        # cv2.imshow("Black Lines (raw mask)", black_lines)
        cv2.imshow("Black Lines (no border)", black_lines_wo_border)

        # --- 在 Black Lines (raw mask) 內找出線條（Hough Transform） ---
        lines = cv2.HoughLinesP(
            black_lines_wo_border,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=10,
        )
        line_img = img_color.copy()
        lines_only = np.zeros_like(img_color)
        if lines is not None:
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                # 畫紅線（疊在原圖）
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 畫紅線（只線條圖）
                cv2.line(lines_only, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 計算中點
                mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                # 計算中點到輪廓的最短距離
                dist = cv2.pointPolygonTest(max_contour, (mx, my), True)
                # 找出中點到輪廓的最近點
                min_dist = float("inf")
                nearest_point = (mx, my)
                for pt in max_contour:
                    px, py = pt[0]
                    d = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
                    if d < min_dist:
                        min_dist = d
                        nearest_point = (px, py)
                # 可視化中點
                cv2.circle(line_img, (mx, my), 3, (255, 0, 0), -1)
                cv2.circle(lines_only, (mx, my), 3, (255, 0, 0), -1)
                # 畫中點到輪廓的距離線
                cv2.line(lines_only, (mx, my), nearest_point, (0, 255, 255), 1)
                # 標示線的idx
                cv2.putText(
                    lines_only,
                    str(idx),
                    (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                print(f"Line {idx}: dist={dist:.2f}, {dist*cm_per_pixel:.4f} cm")
        cv2.imshow("Lines in Mask", line_img)
        cv2.imshow("Max Contour + Edges", img_color)
        cv2.imshow("Lines Only", lines_only)
        # cv2.imshow("Edges", edges)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 按 Q 或 q 直接中斷所有顯示
        if key in [ord("q"), ord("Q")]:
            break


if __name__ == "__main__":
    folder = ""
    # path = "paper\img5.png"
    path = "img*.png"
    draw_edges_on_images(folder, path)
