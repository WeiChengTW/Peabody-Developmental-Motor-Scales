import cv2
import numpy as np
import os

# 讓 result 資料夾固定建在這支 .py 檔同一層，不受執行時的工作目錄影響
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

def analyze_image(
    img_path,
    dot_distance_cm=10.0,
    show_windows=False,
    max_widths=(500, 500, 500, 800),
):
    """
    兩點連線任務的穩健版分析：專注在兩點之間的細長帶狀區域，做連通與偏差判定。
    回傳仍為 (score, img_disp)，不改你的介面。
    """
    # === 小工具 ===
    def detect_dots(gray):
        H, W = gray.shape
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        _, bwA = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bwA = cv2.morphologyEx(bwA, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        bwA = cv2.morphologyEx(bwA, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

        def pick_from_binary(bw, area_min=100, area_max=9000, circ_th=0.58, ar_th=1.5):
            dots_local = []
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (area_min <= area <= area_max):
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circularity = 4 * np.pi * area / (peri**2)
                if circularity < circ_th:
                    continue
                (w1, h1) = cv2.minAreaRect(cnt)[1]
                if min(w1, h1) == 0:
                    continue
                ar = max(w1, h1) / min(w1, h1)
                if ar > ar_th:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dots_local.append((cx, cy, area))
            # ✦ 這裡加強：>2 顆時取最遠兩點，避免選到彼此很近且非真正端點的圓點
            if len(dots_local) > 2:
                pts = np.array([(x, y) for x, y, _ in dots_local], dtype=np.float32)
                dmax, pair = -1, (0, 1)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d > dmax:
                            dmax = d
                            pair = (i, j)
                a, b = dots_local[pair[0]], dots_local[pair[1]]
                return sorted([a, b], key=lambda d: -d[2])
            return sorted(dots_local, key=lambda d: -d[2])[:2]

        dots = pick_from_binary(bwA)
        if len(dots) >= 2:
            return dots, bwA

        inv = cv2.bitwise_not(gray)
        bwB = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 5)
        bwB = cv2.morphologyEx(bwB, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        bwB = cv2.morphologyEx(bwB, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        dots = pick_from_binary(bwB, area_min=80, area_max=12000, circ_th=0.52, ar_th=1.6)
        if len(dots) >= 2:
            return dots, bwB

        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=max(40, W // 6),
                                   param1=100, param2=14, minRadius=6, maxRadius=40)
        bwC = np.zeros_like(gray, dtype=np.uint8)
        dots = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            # ✦ 同樣取最遠兩顆，避免誤抓
            if len(circles) > 2:
                best = None
                dmax = -1
                for i in range(len(circles)):
                    for j in range(i + 1, len(circles)):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        d = np.hypot(x1 - x2, y1 - y2)
                        if d > dmax:
                            dmax = d
                            best = (circles[i], circles[j])
                circles = np.array(best)
            else:
                circles = sorted(circles, key=lambda c: -c[2])[:2]
            for x, y, r in circles:
                cv2.circle(bwC, (x, y), r, 255, -1)
                dots.append((int(x), int(y), int(np.pi * r * r)))
            if len(dots) >= 2:
                return dots, bwC

        return [], (bwC if np.count_nonzero(bwC) > 0 else bwB if np.count_nonzero(bwB) > 0 else bwA)

    def point_line_dist(px, py, x1, y1, x2, y2):
        A = np.array([x2 - x1, y2 - y1], dtype=float)
        B = np.array([px - x1, py - y1], dtype=float)
        L = np.linalg.norm(A)
        if L < 1e-6:
            return np.linalg.norm(B), (x1, y1), 0.0
        t_raw = float(np.dot(A, B) / (L * L))
        proj = np.array([x1, y1], dtype=float) + t_raw * A
        cross = abs(A[0] * B[1] - A[1] * B[0])
        d_perp = cross / L
        return d_perp, tuple(proj.astype(int)), t_raw

    # ✦ 新增：建立「兩點之間窄帶」遮罩，只在這條帶內看線
    def build_band_mask(shape, p1, p2, half_width_px):
        H, W = shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        x1, y1 = p1
        x2, y2 = p2
        length = int(max(1, np.hypot(x2 - x1, y2 - y1)))
        # 旋轉矩形四點
        dx, dy = (x2 - x1), (y2 - y1)
        if length == 0:
            return mask
        nx, ny = -dy / length, dx / length  # 法向量
        # 四角點（以 p1/p2 為中心向法向量偏移）
        p1a = (int(x1 + nx * half_width_px), int(y1 + ny * half_width_px))
        p1b = (int(x1 - nx * half_width_px), int(y1 - ny * half_width_px))
        p2a = (int(x2 + nx * half_width_px), int(y2 + ny * half_width_px))
        p2b = (int(x2 - nx * half_width_px), int(y2 - ny * half_width_px))
        poly = np.array([p1a, p2a, p2b, p1b], dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask

    def score_by_rule(connectable, deviation_cm):
        if not connectable:
            return 0, "Can't connect"
        if deviation_cm <= 0.65:
            return 2, "Connect and offset <= 0.65 cm"
        if deviation_cm <= 1.25:
            return 1, "Connect offset between 0.65 and 1.25 cm"
        return 0, "Connect but offset > 1.25 cm"

    # === 讀圖 ===
    img = cv2.imread(img_path)
    if img is None:
        return 0, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # -- 黑點偵測 + 除錯視圖（保留） --
    dots, dot_bw_debug = detect_dots(gray)
    dot_points = [(x, y) for x, y, _ in dots]

    # Dot Mask（保留）
    dot_mask_clean = np.zeros_like(gray, dtype=np.uint8)
    for x, y in dot_points:
        cv2.circle(dot_mask_clean, (x, y), 13, 255, -1)

    # === 黑點不足兩顆：直接回報 ===
    if len(dot_points) != 2:
        img_disp = img_color.copy()
        cv2.putText(img_disp, f"Only {len(dot_points)} dots found",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return 0, img_disp

    # === 比例尺（兩點距離） ===
    (x1, y1), (x2, y2) = dot_points
    dist_pixel = float(np.hypot(x2 - x1, y2 - y1))
    pixel_per_cm = dist_pixel / float(dot_distance_cm)
    pixel_per_cm = max(1e-6, pixel_per_cm)  # 保底

    # === 建立「兩點間窄帶」 ===
    # 半寬度設為 ~0.35 cm 等效像素（可微調：線條粗一點就拉大）
    band_halfwidth_px = int(max(4, round(0.35 * pixel_per_cm)))
    band_mask = build_band_mask(gray.shape, (x1, y1), (x2, y2), band_halfwidth_px)

    # === 線條粗取：先做自適應二值化，再只取帶內像素 ===
    rough = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 15)
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    rough = cv2.bitwise_and(rough, band_mask)

    # === 帶內小幅「補橋」：閉運算（bridge 大小跟 px/cm 走） ===
    bridge = int(max(1, round(0.20 * pixel_per_cm)))  # 約 0.2 cm
    bridge = bridge + (1 - bridge % 2)                # 保證奇數核
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge))
    line_mask_raw = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel_bridge, iterations=1)

    # === 端點焊接（讓 touch 更寬容） ===
    touch_r_px = int(max(4, round(0.25 * pixel_per_cm)))  # 約 0.25 cm
    tmp = line_mask_raw.copy()
    cv2.circle(tmp, (x1, y1), touch_r_px, 255, -1)
    cv2.circle(tmp, (x2, y2), touch_r_px, 255, -1)
    _, labels = cv2.connectedComponents(tmp)
    id1 = labels[y1, x1]
    id2 = labels[y2, x2]
    touch1 = id1 != 0
    touch2 = id2 != 0
    same_component = (id1 != 0) and (id1 == id2)
    connectable = touch1 and touch2 and same_component

    # === 偏差（只看帶內線像素，且忽略靠近端點的 10% 區段） ===
    max_dist = 0.0
    max_curve_pt, proj_pt = None, None
    A = np.array([x2 - x1, y2 - y1], dtype=float)
    L = np.linalg.norm(A)
    if L > 1e-6 and np.count_nonzero(line_mask_raw) > 0:
        ys, xs = np.where(line_mask_raw > 0)
        A2 = L * L
        for px, py in zip(xs, ys):
            B = np.array([px - x1, py - y1], dtype=float)
            t = float(np.dot(A, B) / A2)
            # 忽略太靠近兩端（端點附近容易膨脹）
            if t <= 0.10 or t >= 0.90:
                continue
            cross = abs(A[0] * B[1] - A[1] * B[0])
            d_perp = cross / L
            if d_perp > max_dist:
                max_dist = d_perp
                proj = np.array([x1, y1], dtype=float) + t * A
                proj_pt = tuple(proj.astype(int))
                max_curve_pt = (int(px), int(py))

    deviation_cm = (max_dist / pixel_per_cm) if max_dist > 0 else float("inf")

    # === 評分 ===
    if not connectable:
        score, reason = 0, "Can't connect"
        deviation_to_show = None
    else:
        score, reason = score_by_rule(True, deviation_cm)
        deviation_to_show = deviation_cm

    # === 視覺化 ===
    img_disp = img_color.copy()
    # dots + 基準線
    for idx, (x, y) in enumerate(dot_points):
        cv2.circle(img_disp, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(img_disp, f"dot{idx+1}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.line(img_disp, dot_points[0], dot_points[1], (0, 0, 255), 2)

    # 觸點狀態
    for idx, (x, y) in enumerate(dot_points):
        ok = touch1 if idx == 0 else touch2
        c = (255, 0, 0) if ok else (0, 0, 255)
        cv2.circle(img_disp, (x, y), max(10, touch_r_px), c, 2)
        cv2.putText(img_disp, f"touch{idx+1}:{'Yes' if ok else 'No'}",
                    (x + 12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    # 偏差線
    if (deviation_to_show is not None) and max_curve_pt and proj_pt and np.isfinite(deviation_to_show):
        cv2.circle(img_disp, max_curve_pt, 7, (0, 255, 255), -1)
        cv2.circle(img_disp, proj_pt, 7, (0, 255, 255), -1)
        cv2.line(img_disp, max_curve_pt, proj_pt, (0, 255, 255), 2)
        mid_x = int((max_curve_pt[0] + proj_pt[0]) / 2)
        mid_y = int((max_curve_pt[1] + proj_pt[1]) / 2)
        cv2.putText(img_disp, f"{deviation_to_show:.2f} cm",
                    (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:
        cv2.putText(img_disp, "deviation: No (not connected)",
                    (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 訊息欄
    score_color = (0, 180, 0) if score == 2 else ((0, 165, 255) if score == 1 else (0, 0, 255))
    cv2.putText(img_disp, f"pixel/cm: {pixel_per_cm:.2f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(img_disp, f"same_component: {'Yes' if same_component else 'No'}",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 0, 255), 2)
    cv2.putText(img_disp, f"Score: {score} ({reason})", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)

    # === 輸出結果影像（檔名/資料夾維持你的規則） ===
    os.makedirs(RESULT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    dev_str2 = "{:.2f}".format(deviation_to_show) if (deviation_to_show is not None and np.isfinite(deviation_to_show)) else "N_A"
    out_name = f"{base}_score{score}_dev{dev_str2.replace('.', '_')}cm.jpg"
    out_path = os.path.join(RESULT_DIR, out_name)
    if img_disp.dtype != np.uint8:
        img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(out_path, img_disp)
    if ok:
        print(f"✅ 已儲存結果影像：{out_path}")
    else:
        print(f"❌ 儲存失敗：{out_path}")

    return score, img_disp

# ===== 使用範例 =====
if __name__ == "__main__":
    result = analyze_image(
        r"PDMS2_web\ch2-t6\new\new6.jpg", dot_distance_cm=10.0, show_windows=True
    )
    print("得分：", result["score"])
