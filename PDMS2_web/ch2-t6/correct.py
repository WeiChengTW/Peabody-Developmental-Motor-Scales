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
    max_widths=(
        500,
        500,
        500,
        800,
    ),  # (Dot Mask, Black Line Mask, All Dot Contours, Black Dot Detection)
):
    """
    讀取單張影像並完成：黑點偵測、線條擷取、是否可連線、最大垂直偏差、評分與視覺化顯示。
    參數
    ----
    img_path : str
        影像路徑
    dot_distance_cm : float
        兩黑點實際距離（公分），預設 10 cm
    show_windows : bool
        是否顯示四個除錯視窗
    max_widths : tuple(int, int, int, int)
        四個視窗各自的最大寬度

    回傳
    ----
    result : dict
        {
          "score": int,
          "reason": str,
          "deviation_cm": float or None,
          "pixel_per_cm": float,
          "touch1": bool, "touch2": bool, "same_component": bool,
          "dots": [(x1,y1),(x2,y2)] or [],
          "img_path": str
        }
    """

    # === 內部工具 ===
    def detect_dots(gray):
        H, W = gray.shape
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

        _, bwA = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bwA = cv2.morphologyEx(
            bwA, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
        )
        bwA = cv2.morphologyEx(
            bwA, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
        )

        def pick_from_binary(bw, area_min=100, area_max=9000, circ_th=0.58, ar_th=1.5):
            dots_local = []
            contours, _ = cv2.findContours(
                bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
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
            return sorted(dots_local, key=lambda d: -d[2])[:2]

        dots = pick_from_binary(bwA)
        if len(dots) >= 2:
            return dots, bwA

        inv = cv2.bitwise_not(gray)
        bwB = cv2.adaptiveThreshold(
            inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5
        )
        bwB = cv2.morphologyEx(
            bwB, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        bwB = cv2.morphologyEx(
            bwB, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )

        dots = pick_from_binary(
            bwB, area_min=80, area_max=12000, circ_th=0.52, ar_th=1.6
        )
        if len(dots) >= 2:
            return dots, bwB

        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(40, W // 6),
            param1=100,
            param2=14,
            minRadius=6,
            maxRadius=40,
        )
        bwC = np.zeros_like(gray, dtype=np.uint8)
        dots = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            circles = sorted(circles, key=lambda c: -c[2])[:2]
            for x, y, r in circles:
                cv2.circle(bwC, (x, y), r, 255, -1)
                dots.append((int(x), int(y), int(np.pi * r * r)))
            if len(dots) >= 2:
                return dots, bwC

        return [], (
            bwC
            if np.count_nonzero(bwC) > 0
            else bwB if np.count_nonzero(bwB) > 0 else bwA
        )

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

    def touch_and_connect(line_mask_raw, p1, p2, touch_r=6, bridge_dilate=3):
        def touching(p):
            m = np.zeros_like(line_mask_raw)
            cv2.circle(m, p, touch_r, 255, -1)
            return cv2.countNonZero(cv2.bitwise_and(line_mask_raw, m)) > 0

        t1 = touching(p1)
        t2 = touching(p2)
        tmp = cv2.dilate(
            line_mask_raw,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (bridge_dilate, bridge_dilate)
            ),
            1,
        )
        cv2.circle(tmp, p1, touch_r, 255, -1)
        cv2.circle(tmp, p2, touch_r, 255, -1)
        _, labels = cv2.connectedComponents(tmp)
        id1 = labels[p1[1], p1[0]]
        id2 = labels[p2[1], p2[0]]
        conn = (id1 != 0) and (id1 == id2)
        return t1, t2, conn

    def score_by_rule(connectable, deviation_cm):
        if not connectable:
            return 0, "Can't connect"
        if deviation_cm <= 0.65:
            return 2, "Connect and offset <= 0.65 cm"
        if deviation_cm <= 1.25:
            return 1, "Connect offset between 0.65 and 1.25 cm"
        return 0, "Connect but offset > 1.25 cm"

    def show_scaled(title, img, max_width=500):
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, img.shape[1], img.shape[0])
        # cv2.imshow(title, img)

    # === 讀圖 ===
    img = cv2.imread(img_path)
    if img is None:
        return {
            "score": 0,
            "reason": "Image read failed",
            "deviation_cm": None,
            "pixel_per_cm": 0.0,
            "touch1": False,
            "touch2": False,
            "same_component": False,
            "dots": [],
            "img_path": img_path,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # -- 黑點偵測 + 除錯視圖（保留 All Dot Contours） --
    dots, dot_bw_debug = detect_dots(gray)
    dot_points = [(x, y) for x, y, _ in dots]
    contour_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours_dbg, _ = cv2.findContours(
        dot_bw_debug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    for i, c in enumerate(contours_dbg):
        cv2.drawContours(contour_viz, [c], -1, colors[i % len(colors)], 2)

    # -- Dot Mask（保留） --
    dot_mask_clean = np.zeros_like(gray, dtype=np.uint8)
    for x, y in dot_points:
        cv2.circle(dot_mask_clean, (x, y), 13, 255, -1)

    # -- 線條 mask（保留） --
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    adaptive_mask = cv2.morphologyEx(
        adaptive_mask, cv2.MORPH_OPEN, kernel2, iterations=1
    )
    adaptive_mask = cv2.dilate(adaptive_mask, kernel2, iterations=1)

    contours2, _ = cv2.findContours(
        adaptive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = adaptive_mask.shape
    possible_curves = []
    for cnt in contours2:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if y > h // 2:  # 你的原條件
            continue
        if cv2.contourArea(cnt) < 50:
            continue
        possible_curves.append(cnt)

    # -- 同步建立 raw / clean 兩份遮罩 --
    black_line_mask_clean = np.zeros_like(adaptive_mask)  # 顯示用
    black_line_mask_raw = np.zeros_like(adaptive_mask)  # 連通判斷用
    max_curve = max(possible_curves, key=cv2.contourArea) if possible_curves else None
    if max_curve is not None:
        cv2.drawContours(black_line_mask_raw, [max_curve], -1, 255, -1)
        cv2.drawContours(black_line_mask_clean, [max_curve], -1, 255, -1)
        contours_dot, _ = cv2.findContours(
            dot_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(black_line_mask_clean, contours_dot, -1, 0, -1)
        for x, y in dot_points:
            cv2.circle(black_line_mask_clean, (x, y), 20, 0, -1)

    # === 若黑點不足 2 個：顯示三窗並結束 ===
    if len(dot_points) != 2:
        # if show_windows:
        # show_scaled("Dot Mask", dot_mask_clean, max_widths[0])
        # show_scaled("Black Line Mask", black_line_mask_clean, max_widths[1])
        # show_scaled("All Dot Contours", contour_viz, max_widths[2])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return {
            "score": 0,
            "reason": f"Only {len(dot_points)} dots found",
            "deviation_cm": None,
            "pixel_per_cm": 0.0,
            "touch1": False,
            "touch2": False,
            "same_component": False,
            "dots": dot_points,
            "img_path": img_path,
        }

    # === 比例尺（兩點 = dot_distance_cm 公分） ===
    (x1, y1), (x2, y2) = dot_points
    dist_pixel = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    pixel_per_cm = dist_pixel / float(dot_distance_cm)
    if pixel_per_cm < 15 or pixel_per_cm > 100:
        # 可視需要印出警告；這裡只回傳數值
        pass

    # === 偏差距離（曲線至端點直線最大垂距） ===
    max_dist = 0.0
    max_curve_pt, proj_pt = None, None
    valid_found = False

    if max_curve is not None:
        for pt in max_curve.reshape(-1, 2):
            d_perp, proj, t_raw = point_line_dist(pt[0], pt[1], x1, y1, x2, y2)
            if 0.0 < t_raw < 1.0:
                valid_found = True
                if d_perp > max_dist:
                    max_dist = d_perp
                    max_curve_pt = (int(pt[0]), int(pt[1]))
                    proj_pt = proj
        if not valid_found:
            for pt in max_curve.reshape(-1, 2):
                A = np.array([x2 - x1, y2 - y1], dtype=float)
                B = np.array([pt[0] - x1, pt[1] - y1], dtype=float)
                L = np.linalg.norm(A)
                if L < 1e-6:
                    continue
                t = np.clip(np.dot(A, B) / (L * L), 0.0, 1.0)
                proj = np.array([x1, y1], dtype=float) + t * A
                d = np.linalg.norm(np.array([pt[0], pt[1]], dtype=float) - proj)
                if d > max_dist:
                    max_dist = d
                    max_curve_pt = (int(pt[0]), int(pt[1]))
                    proj_pt = tuple(proj.astype(int))

    deviation_cm = (
        (max_dist / pixel_per_cm)
        if (max_dist > 0 and pixel_per_cm > 1e-6)
        else float("inf")
    )

    # === 是否兩點皆接到線 + 是否同一條線 ===
    p1, p2 = (x1, y1), (x2, y2)
    touch1, touch2, same_component = touch_and_connect(
        black_line_mask_raw, p1, p2, touch_r=6, bridge_dilate=3
    )
    connectable = touch1 and touch2 and same_component

    # === 評分 ===
    if not connectable:
        score, reason = 0, "Can't connect"
        deviation_to_show = None
    else:
        score, reason = score_by_rule(True, deviation_cm)
        deviation_to_show = deviation_cm

    # === 視覺化主圖 ===
    img_disp = img_color.copy()
    for idx, (x, y) in enumerate(dot_points):
        cv2.circle(img_disp, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            img_disp,
            f"dot{idx+1}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
    cv2.line(img_disp, dot_points[0], dot_points[1], (0, 0, 255), 2)

    for idx, (x, y) in enumerate(dot_points):
        ok = touch1 if idx == 0 else touch2
        c = (255, 0, 0) if ok else (0, 0, 255)
        cv2.circle(img_disp, (x, y), 10, c, 2)
        cv2.putText(
            img_disp,
            f"touch{idx+1}:{'Yes' if ok else 'No'}",
            (x + 12, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            c,
            2,
        )

    if (
        (deviation_to_show is not None)
        and max_curve_pt
        and proj_pt
        and np.isfinite(deviation_to_show)
    ):
        cv2.circle(img_disp, max_curve_pt, 7, (0, 255, 255), -1)
        cv2.circle(img_disp, proj_pt, 7, (0, 255, 255), -1)
        cv2.line(img_disp, max_curve_pt, proj_pt, (0, 255, 255), 2)
        mid_x = int((max_curve_pt[0] + proj_pt[0]) / 2)
        mid_y = int((max_curve_pt[1] + proj_pt[1]) / 2)
        cv2.putText(
            img_disp,
            f"{deviation_to_show:.2f} cm",
            (mid_x + 10, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
    else:
        cv2.putText(
            img_disp,
            "deviation: No (not connected)",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    score_color = (
        (0, 180, 0) if score == 2 else ((0, 165, 255) if score == 1 else (0, 0, 255))
    )
    cv2.putText(
        img_disp,
        f"pixel/cm: {pixel_per_cm:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        img_disp,
        f"same_component: {'Yes' if same_component else 'No'}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (250, 0, 255),
        2,
    )
    cv2.putText(
        img_disp,
        f"Score: {score} ({reason})",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        score_color,
        2,
    )

    # === 另存主視窗影像到 result/（穩健版） ===
    os.makedirs(RESULT_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(img_path))[0]
    dev_str2 = (
        "{:.2f}".format(deviation_to_show)
        if (deviation_to_show is not None and np.isfinite(deviation_to_show))
        else "N_A"
    )
    out_name = f"{base}_score{score}_dev{dev_str2.replace('.', '_')}cm.jpg"
    out_path = os.path.join(RESULT_DIR, out_name)

    # 確保圖像型別正確
    if not isinstance(img_disp, np.ndarray) or img_disp.size == 0:
        print("❌ img_disp 無效，無法儲存")
    else:
        if img_disp.dtype != np.uint8:
            img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)
        # cv2.imshow("result",img_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 用 imencode 轉成 JPEG，再用 tofile 寫入，避免 Windows/路徑問題
        # === 用 cv2.imwrite() 儲存結果影像 ===
        ok = cv2.imwrite(out_path, img_disp)
        if ok:
            print(f"✅ 已儲存結果影像：{out_path}")
        else:
            print(f"❌ 儲存失敗：{out_path}")

    # === 顯示四窗（可關閉） ===
    # if show_windows:
    #     show_scaled("Dot Mask", dot_mask_clean, max_widths[0])
    #     show_scaled("Black Line Mask", black_line_mask_clean, max_widths[1])
    #     show_scaled("All Dot Contours", contour_viz, max_widths[2])
    #     show_scaled("Black Dot Detection", img_disp, max_widths[3])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    dev_str = (
        "{:.3f}".format(deviation_to_show)
        if (deviation_to_show is not None and np.isfinite(deviation_to_show))
        else "N/A"
    )
    print(
        f"{os.path.basename(img_path)} → score={score}, deviation={dev_str} cm, "
        f"touch=({touch1},{touch2}), same={same_component}"
    )
    return score, img_disp
    # return {
    #     "score": score,
    #     # "reason": reason,
    #     # "deviation_cm": deviation_to_show,
    #     # "pixel_per_cm": float(pixel_per_cm),
    #     # "touch1": bool(touch1),
    #     # "touch2": bool(touch2),
    #     # "same_component": bool(same_component),
    #     # "dots": (
    #     #     [
    #     #         (int(dot_points[0][0]), int(dot_points[0][1])),
    #     #         (int(dot_points[1][0]), int(dot_points[1][1])),
    #     #     ]
    #     #     if len(dot_points) == 2
    #     #     else []
    #     # ),
    #     "img_path": img_path,
    # }


# ===== 使用範例 =====
if __name__ == "__main__":
    result = analyze_image(
        r"PDMS2_web\ch2-t6\new\new6.jpg", dot_distance_cm=10.0, show_windows=True
    )
    print("得分：", result["score"])
