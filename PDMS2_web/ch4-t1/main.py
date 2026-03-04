import cv2
import os
import numpy as np
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def return_score(score):
    print("\n[提示] 檢視圖片後，按任意鍵關閉視窗並結束程式...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(int(score))

# 用 Canny 邊緣檢測找最大四邊形輪廓
def get_largest_quadrilateral_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊降噪，避免 Canny 抓到雜訊邊緣
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自動計算 Canny 雙閾值（Otsu 方法）
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low  = otsu_thresh * 0.5
    high = otsu_thresh

    edges = cv2.Canny(blurred, low, high)

    # 膨脹讓邊緣更連續，有助於輪廓封閉
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    img_area = h * w
    max_quad, max_area = None, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 面積下限 1000px²，上限 90% 圖面積（排除圖片邊框）
        if 1000 < area < (img_area * 0.9):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

            if len(approx) == 4 and area > max_area:
                max_area, max_quad = area, approx

    return max_quad

# 合併輪廓
def merge_close_contours(contours, threshold=10):
    merged, used, centers = [], [False] * len(contours), []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            centers.append((0, 0))
    for i, cnt1 in enumerate(contours):
        if used[i]: continue
        group = [cnt1]
        used[i] = True
        for j, cnt2 in enumerate(contours):
            if i == j or used[j]: continue
            if np.linalg.norm(np.array(centers[i]) - np.array(centers[j])) < threshold:
                group.append(cnt2)
                used[j] = True
        merged.append(np.vstack(group))
    return merged

# 角點排序（依角度順序）
def sort_corners(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return [tuple(p) for p in pts[np.argsort(ang)]]

# 主流程
def analyze_image(img_path, uid="default", img_id="test"):
    result_dir = f"kid/{uid}"
    os.makedirs(result_dir, exist_ok=True)

    # 讀取單位換算
    try:
        json_path = BASE_DIR.parent / "px2cm.json"
        with open(json_path, "r") as f:
            data = json.load(f)
            px2cm = data["pixel_per_cm"]
    except:
        px2cm = 47.4416628993705
    cm_per_pixel = 1 / px2cm
    actual_length_cm = 7.5

    img_raw = cv2.imread(img_path)
    if img_raw is None:
        print(f"無法讀取圖片: {img_path}")
        return 0

    # 調整對比度與亮度
    alpha = 1.3
    beta = 3
    img_color = cv2.convertScaleAbs(img_raw, alpha=alpha, beta=beta)

    # 預覽增強後圖片
    cv2.namedWindow('Enhanced Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Enhanced Image', img_color)

    # --- 純 Canny 邊緣檢測 ---
    max_contour = get_largest_quadrilateral_contour(img_color)

    score = 0
    result_img = img_color.copy()
    edges_info = []

    if max_contour is not None:
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.05 * peri, True)

        if len(approx) == 4:
            # 繪製偵測到的邊界
            cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)

            # 計算四條邊長
            corners = [np.array(p) for p in sort_corners([pt[0] for pt in approx])]
            for i in range(4):
                p1, p2 = corners[i], corners[(i + 1) % 4]
                length_cm = np.linalg.norm(p1 - p2) * cm_per_pixel
                edges_info.append((p1, p2, length_cm))

            # 取最短兩邊計分
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

            # 標註每條邊長
            for (p1, p2, length_cm) in edges_info:
                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(result_img, f"{length_cm:.1f}cm", tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 文字資訊面板
    info_text = [
        f"Score: {score}",
        f"Scale: {px2cm:.2f} px/cm",
        "Short Sides Check:"
    ]
    if len(edges_info) >= 2:
        info_text.append(f"S1: {edges_info[0][2]:.1f}cm, S2: {edges_info[1][2]:.1f}cm")

    for i, text in enumerate(info_text):
        cv2.putText(result_img, text, (20, 40 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 顯示分析結果
    cv2.namedWindow('Analysis Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Analysis Result', result_img)

    # 儲存結果
    result_path = rf"kid\{uid}\{img_id}_result.jpg"
    cv2.imwrite(result_path, result_img)

    return score


if __name__ == "__main__":
    if len(sys.argv) > 2:
        u_id = sys.argv[1]
        i_id = sys.argv[2]
        target_path = rf"kid\{u_id}\{i_id}.jpg"

        final_score = analyze_image(target_path, u_id, i_id)
        print(f"Score: {final_score}")
        return_score(final_score)
    else:
        # 預設本地測試
        test_path = r"PDMS2_web\ch4-t1\ch4-t1.jpg"
        if os.path.exists(test_path):
            final_score = analyze_image(test_path)
            print(f"Local Test Score: {final_score}")
            return_score(final_score)
        else:
            print("請提供參數或確認測試路徑存在。")