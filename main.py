import cv2
import glob
import os
import numpy as np
import sys
import json

# 找最大四邊形輪廓
def get_largest_quadrilateral_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    max_quad, max_area = None, 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area, max_quad = area, approx
    return max_quad

# 合併輪廓
def merge_close_contours(contours, threshold=10):
    merged, used, centers = [], [False]*len(contours), []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
        else:
            centers.append((0, 0))
    for i, cnt1 in enumerate(contours):
        if used[i]: continue
        group = [cnt1]; used[i] = True
        for j, cnt2 in enumerate(contours):
            if i == j or used[j]: continue
            if np.linalg.norm(np.array(centers[i]) - np.array(centers[j])) < threshold:
                group.append(cnt2); used[j] = True
        merged.append(np.vstack(group))
    return merged

# 角點排序（環狀）
def sort_corners(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:,1]-center[1], pts[:,0]-center[0])
    return [tuple(p) for p in pts[np.argsort(ang)]]

# 主流程
def analyze_image(img_path):
    cm_per_pixel = 0.02068
    actual_length_cm = 7.5

    img_color_raw = cv2.imread(img_path)
    if img_color_raw is None:
        raise ValueError("無法讀取圖片")

    max_contour = get_largest_quadrilateral_contour(img_color_raw)

    if max_contour is None:
        hsv = cv2.cvtColor(img_color_raw, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 80, 255]))
        edges_white = cv2.Canny(white_mask, 80, 200)
        dilated_white = cv2.dilate(edges_white, np.ones((5, 5), np.uint8), iterations=1)
        contours_white, _ = cv2.findContours(dilated_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_white = [cnt for cnt in contours_white if cv2.contourArea(cnt) > 500]
        if not contours_white:
            return 0
        merged_white = merge_close_contours(contours_white, threshold=5)
        max_contour = max(merged_white, key=cv2.contourArea)
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.05 * peri, True)
        if len(approx) != 4:
            return 0
    else:
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.05 * peri, True)
        if len(approx) != 4:
            return 0

    # 計算短邊
    corners = [np.array(p) for p in sort_corners([pt[0] for pt in approx])]
    edge_pts = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]

    edges_info = []
    for p1, p2 in edge_pts:
        length_px = np.linalg.norm(p1 - p2)
        edges_info.append((p1, p2, length_px * cm_per_pixel))

    edges_info.sort(key=lambda x: x[2])
    short_edges = edges_info[:2]

    s1, s2 = short_edges[0][2], short_edges[1][2]
    d1, d2 = abs(s1 - actual_length_cm), abs(s2 - actual_length_cm)

    if d1 <= 0.3 and d2 <= 0.3:
        score = 2
    elif d1 <= 1.2 and d2 <= 1.2:
        score = 1
    else:
        score = 0

    return score

if __name__ == "__main__":
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = rf"kid\{uid}\{img_id}.jpg"

        score = analyze_image(image_path)
        print(f"score : {score}")

        result_file = "result.json"
        try:
            if os.path.exists(result_file):
                with open(result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            else:
                results = {}
        except (json.JSONDecodeError, FileNotFoundError):
            results = {}

        if uid not in results:
            results[uid] = {}

        results[uid][img_id] = score

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"結果已儲存到 {result_file} - 用戶 {uid} 的關卡 {img_id} 分數: {score}")
