# analyze_main.py
import cv2
import numpy as np
import json
import glob
import os

CROP_FOLDER = os.path.join("PDMS2_web", "ch2-t4", "new")
PXCM_JSON = "px2cm.json"  # 單一比例檔，內容：{"pixel_per_cm": 40.47...}
SHOW_SCALE = 0.7
MARGIN = 40


def show(title, img, scale=SHOW_SCALE):
    h, w = img.shape[:2]
    # cv2.imshow(title, cv2.resize(img, (int(w * scale), int(h * scale))))


def preprocess_blackhat(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    )
    _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(bw) < 50:
        _, bw = cv2.threshold(bh, 25, 255, cv2.THRESH_BINARY)

    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)), 1
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)

    h, w = bw.shape
    bw[:MARGIN, :] = 0
    bw[-MARGIN:, :] = 0
    bw[:, :MARGIN] = 0
    bw[:, -MARGIN:] = 0

    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 3))
    )
    return bw, horiz


def extract_trace_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 25, 20])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([165, 25, 20])
    upper2 = np.array([180, 255, 255])
    mask_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2)
    )

    b, g, r = cv2.split(img_bgr)
    r16, g16, b16 = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
    dom = (r16 - g16 > 18) & (r16 - b16 > 18) & (r > 45)
    mask_dom = dom.astype(np.uint8) * 255

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1]
    mask_cr = (Cr > 135).astype(np.uint8) * 255

    trace = cv2.bitwise_or(mask_hsv, mask_dom)
    trace = cv2.bitwise_or(trace, mask_cr)
    trace = cv2.morphologyEx(trace, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), 1)
    trace = cv2.morphologyEx(trace, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
    trace = cv2.dilate(trace, np.ones((3, 3), np.uint8), 1)
    return trace


def fallback_baseline_from_projection(bw, prefer_band_px=6):
    h, w = bw.shape
    proj = bw.sum(axis=1).astype(np.float32)
    proj = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 9), 0).ravel()
    y0 = int(np.argmax(proj))
    band_px = prefer_band_px
    y1 = max(0, y0 - band_px)
    y2 = min(h - 1, y0 + band_px)

    band = np.zeros_like(bw)
    band[y1 : y2 + 1, :] = 255
    line_pixels = cv2.bitwise_and(bw, band)
    line_pixels = cv2.morphologyEx(
        line_pixels, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(line_pixels, 8)
    if num <= 1:
        return (0, y0, w - 1, y0)

    areas = stats[1:, cv2.CC_STAT_AREA]
    target = 1 + int(np.argmax(areas)) if len(areas) > 0 else 0
    comp_mask = (labels == target).astype(np.uint8) * 255

    ys, xs = np.where(comp_mask > 0)
    if len(xs) < 2:
        return (0, y0, w - 1, y0)

    pts = np.float32(np.column_stack([xs, ys]))
    vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, cx, cy = float(vx[0]), float(vy[0]), float(cx[0]), float(cy[0])

    t = (xs - cx) * vx + (ys - cy) * vy
    tL, tR = t.min(), t.max()
    xL = int(round(cx + vx * tL))
    yL = int(round(cy + vy * tL))
    xR = int(round(cx + vx * tR))
    yR = int(round(cy + vy * tR))
    xL = np.clip(xL, 0, w - 1)
    xR = np.clip(xR, 0, w - 1)
    yL = np.clip(yL, 0, h - 1)
    yR = np.clip(yR, 0, h - 1)
    return (xL, yL, xR, yR)


def find_baseline_and_show_all(img_bgr, pixel_per_cm):
    h, w = img_bgr.shape[:2]
    bw, horiz = preprocess_blackhat(img_bgr)

    edges = cv2.Canny(horiz, 30, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=w // 6, maxLineGap=30
    )
    if lines is None:
        edges2 = cv2.Canny(bw, 30, 150)
        lines = cv2.HoughLinesP(
            edges2, 1, np.pi / 180, threshold=60, minLineLength=w // 6, maxLineGap=30
        )

    all_on_img = img_bgr.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(all_on_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if lines is not None:

        def near_border(x1, y1, x2, y2, m=MARGIN):
            return (
                min(x1, x2) < m
                or max(x1, x2) > w - 1 - m
                or min(y1, y2) < m
                or max(y1, y2) > h - 1 - m
            )

        best, best_score = None, -1e9
        for x1, y1, x2, y2 in lines[:, 0]:
            length = np.hypot(x2 - x1, y2 - y1)
            tilt = abs(y2 - y1)
            xmid = 0.5 * (x1 + x2)
            center_pen = abs(xmid - w / 2)
            border_pen = 200 if near_border(x1, y1, x2, y2) else 0
            score = length - 8.0 * tilt - 0.8 * center_pen - border_pen
            if score > best_score:
                best_score = score
                best = (int(x1), int(y1), int(x2), int(y2))
        bx1, by1, bx2, by2 = best
        A = by1 - by2
        B = bx2 - bx1
        C = bx1 * by2 - bx2 * by1
        den = (A * A + B * B) ** 0.5 + 1e-6

        band_px = 6
        yy, xx = np.indices((h, w))
        dist = np.abs(A * xx + B * yy + C) / den
        band_mask = (dist <= band_px).astype(np.uint8) * 255

        line_pixels = cv2.bitwise_and(bw, band_mask)
        k_bridge = cv2.getStructuringElement(
            cv2.MORPH_RECT, (51, 1) if abs(B) >= abs(A) else (1, 51)
        )
        line_pixels = cv2.morphologyEx(line_pixels, cv2.MORPH_CLOSE, k_bridge, 1)
        line_pixels = cv2.morphologyEx(
            line_pixels, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1
        )

        num, labels, stats, _ = cv2.connectedComponentsWithStats(line_pixels, 8)
        if num <= 1:
            xL, yL, xR, yR = fallback_baseline_from_projection(bw)
        else:

            def safe_lab_at(x, y):
                x = np.clip(int(x), 0, w - 1)
                y = np.clip(int(y), 0, h - 1)
                return labels[y, x]

            lab1, lab2 = safe_lab_at(bx1, by1), safe_lab_at(bx2, by2)
            target = (
                lab1
                if lab1 != 0
                else (lab2 if lab2 != 0 else np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
            )
            comp_mask = (labels == target).astype(np.uint8) * 255

            ysc, xsc = np.where(comp_mask > 0)
            if len(xsc) < 2:
                xL, yL, xR, yR = fallback_baseline_from_projection(bw)
            else:
                pts = np.float32(np.column_stack([xsc, ysc]))
                vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, cx, cy = float(vx[0]), float(vy[0]), float(cx[0]), float(cy[0])
                t = (xsc - cx) * vx + (ysc - cy) * vy
                tL, tR = t.min(), t.max()
                xL, yL = int(round(cx + vx * tL)), int(round(cy + vy * tL))
                xR, yR = int(round(cx + vx * tR)), int(round(cy + vy * tR))

    else:
        xL, yL, xR, yR = fallback_baseline_from_projection(bw)

    # 畫基準線粗遮罩
    base_only = np.zeros((h, w), np.uint8)
    cv2.line(base_only, (xL, yL), (xR, yR), 255, 5)
    base_only = cv2.dilate(base_only, np.ones((1, 9), np.uint8), 1)

    # 抽出手畫曲線區域
    trace = extract_trace_mask(img_bgr)
    A = yL - yR
    B = xR - xL
    C = xL * yR - xR * yL
    den = (A * A + B * B) ** 0.5 + 1e-6

    TH_CM = 1.2
    TH_PX = TH_CM * pixel_per_cm

    yy, xx = np.indices(trace.shape)
    dist_from_line = np.abs(A * xx + B * yy + C) / den

    band_limit_px = int(max(TH_PX * 3.0, 50))
    trace[dist_from_line > band_limit_px] = 0

    baseline_thick = cv2.dilate(base_only, np.ones((1, 19), np.uint8), 1)
    curve_from_bw = cv2.bitwise_and(bw, cv2.bitwise_not(baseline_thick))
    curve_from_bw[dist_from_line > band_limit_px] = 0
    curve_from_bw = cv2.morphologyEx(
        curve_from_bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1
    )
    curve_from_bw = cv2.morphologyEx(
        curve_from_bw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3)),
        1,
    )
    trace = cv2.bitwise_or(trace, curve_from_bw)

    EDGE_STRIP = 35
    trace[:, :EDGE_STRIP] = 0
    trace[:, -EDGE_STRIP:] = 0

    # 限定於基準線附近的一段
    num, labels, stats, _ = cv2.connectedComponentsWithStats(trace, 8)
    keep = np.zeros_like(trace)
    x_min = max(0, min(xL, xR) - 25)
    x_max = min(trace.shape[1] - 1, max(xL, xR) + 25)
    for i in range(1, num):
        x, y, w_, h_, area = stats[i]
        touches_border = (
            x == 0
            or y == 0
            or x + w_ >= trace.shape[1] - 1
            or y + h_ >= trace.shape[0] - 1
        )
        if touches_border:
            continue
        cx = x + w_ / 2.0
        if cx < x_min or cx > x_max:
            continue
        keep[labels == i] = 255
    trace = keep

    # 最遠偏差
    ys, xs = np.where(trace > 0)
    if len(xs) > 0:
        dist_all = dist_from_line[ys, xs]
        over_mask = dist_all > TH_PX
        if np.any(over_mask):
            idx_pool = np.flatnonzero(over_mask)
            pick = idx_pool[np.argmax(dist_all[over_mask])]
        else:
            pick = int(np.argmax(dist_all))
        far_x, far_y = int(xs[pick]), int(ys[pick])
        far_dev_px = float(dist_all[pick])
        far_dev_cm = far_dev_px / pixel_per_cm
        print(
            f"最遠點: ({far_x}, {far_y}) 偏差 {far_dev_cm:.3f} cm （{far_dev_px:.1f} px）"
        )
    else:
        print("最遠點: 無（未找到 trace 像素）")
        far_dev_cm = 0.0

    # 依符號距離區分在上/下方
    ys, xs = np.where(trace > 0)
    signed = (A * xs + B * ys + C) / den if len(xs) > 0 else np.array([])
    over_pos = np.zeros_like(trace, np.uint8)
    over_neg = np.zeros_like(trace, np.uint8)

    if len(xs) > 0:
        # 門檻帶的遞進（hysteresis）
        delta_px = int(max(1, round(0.25 * pixel_per_cm)))
        strong_pos = np.zeros_like(trace, np.uint8)
        strong_neg = np.zeros_like(trace, np.uint8)
        weak_pos = np.zeros_like(trace, np.uint8)
        weak_neg = np.zeros_like(trace, np.uint8)

        strong_pos[ys[signed > TH_PX], xs[signed > TH_PX]] = 255
        strong_neg[ys[signed < -TH_PX], xs[signed < -TH_PX]] = 255
        weak_pos[ys[signed > TH_PX - delta_px], xs[signed > TH_PX - delta_px]] = 255
        weak_neg[ys[signed < -TH_PX + delta_px], xs[signed < -TH_PX + delta_px]] = 255

        def hysteresis_keep(weak, strong):
            num, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
            if num <= 1:
                return np.zeros_like(weak)
            out = np.zeros_like(weak)
            for i in range(1, num):
                comp = labels == i
                if (strong[comp] > 0).any():
                    out[comp] = 255
            return out

        over_pos = hysteresis_keep(weak_pos, strong_pos)
        over_neg = hysteresis_keep(weak_neg, strong_neg)

        # 基準線方向量測向量（供橋接）
        L = np.hypot(xR - xL, yR - yL) + 1e-6
        vx = (xR - xL) / L
        vy = (yR - yL) / L
        cx0 = 0.5 * (xL + xR)
        cy0 = 0.5 * (yL + yR)

        def bridge_along_baseline(mask, cx, cy, vx, vy, gap_px):
            h, w = mask.shape
            ang = np.degrees(np.arctan2(vy, vx))
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
            if gap_px % 2 == 0:
                gap_px += 1
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_px, 3))
            rot = cv2.morphologyEx(rot, cv2.MORPH_CLOSE, k, 1)
            Minv = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
            return cv2.warpAffine(rot, Minv, (w, h), flags=cv2.INTER_NEAREST)

        merge_gap_px = int(max(TH_PX * 1.2, pixel_per_cm * 0.45))
        over_pos = bridge_along_baseline(over_pos, cx0, cy0, vx, vy, merge_gap_px)
        over_neg = bridge_along_baseline(over_neg, cx0, cy0, vx, vy, merge_gap_px)

    over = cv2.bitwise_or(over_pos, over_neg)

    # === NEW: 任何偏離(>= ε cm) 的統計用遮罩（在 over 之後做） ===
    EPS_CM = 0.4
    EPS_PX = int(max(1, round(EPS_CM * pixel_per_cm)))

    any_pos = np.zeros_like(trace, np.uint8)
    any_neg = np.zeros_like(trace, np.uint8)
    if len(xs) > 0:
        any_pos[ys[signed > EPS_PX], xs[signed > EPS_PX]] = 255
        any_neg[ys[signed < -EPS_PX], xs[signed < -EPS_PX]] = 255
        any_pos = bridge_along_baseline(any_pos, cx0, cy0, vx, vy, merge_gap_px)
        any_neg = bridge_along_baseline(any_neg, cx0, cy0, vx, vy, merge_gap_px)

    any_dev_mask = cv2.bitwise_or(any_pos, any_neg)  # 輕微+嚴重
    mild_dev_mask = cv2.bitwise_and(any_dev_mask, cv2.bitwise_not(over))  # 僅輕微

    # === 著色：綠(在線/極小)、黃(輕微)、紅(>1.2cm) ===
    overlay_dev = img_bgr.copy()
    overlay_dev[trace > 0] = (0, 255, 0)
    overlay_dev[mild_dev_mask > 0] = (0, 255, 255)
    overlay_dev[over > 0] = (0, 0, 255)
    cv2.line(overlay_dev, (xL, yL), (xR, yR), (255, 0, 0), 3)
    show("Overlay (blue baseline + green trace + protrusions)", overlay_dev)

    # 區塊計數與評分
    def count_blocks(bin_img, min_area_px):
        num, _, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
        areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([])
        return int(np.sum(areas >= min_area_px))

    min_len_cm = 0.4
    min_area_px = int(max(20, pixel_per_cm * min_len_cm))

    dev_count = count_blocks(any_dev_mask, min_area_px)  # 任何偏離
    over_count = count_blocks(over, min_area_px)  # 嚴重偏離

    print(f"偏離次數（任何偏離）：{dev_count} 次")
    print(f"嚴重偏離（> {TH_CM:.1f} cm）：{over_count} 次")

    # 評分（最大偏差 >=1.2cm 直接 0 分；否則依偏離次數）
    if far_dev_cm >= 1.2:
        score = 0
        reason = "最大偏差 ≥ 1.2 cm"
    else:
        if dev_count <= 2:
            score = 2
        elif dev_count <= 4:  # 2~4 次 → 1 分（若要改：把這行改成 3 <= dev_count <= 4）
            score = 1
        else:
            score = 0
        reason = f"偏離次數={dev_count}"

    print(f"最大偏差: {far_dev_cm:.3f} cm")
    print(f"得分: {score} 分（{reason}）")
    return score


# 取代整個 main()：只跑單張 + 從 px_cm.json 讀比例
def main():
    # 1) 讀單一比例 px_cm.json
    if not os.path.exists(PXCM_JSON):
        raise FileNotFoundError(
            f'找不到 {PXCM_JSON}，請先準備好比例檔（例如：{{"pixel_per_cm": 40.47}}）'
        )
    with open(PXCM_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pixel_per_cm = float(obj.get("pixel_per_cm", 0.0))
    if pixel_per_cm <= 0:
        raise ValueError(f"{PXCM_JSON} 內 pixel_per_cm 無效：{pixel_per_cm}")

    # 2) 只跑一張圖（改這個編號即可）
    img_num = 1
    img_path = os.path.join(CROP_FOLDER, f"new{img_num}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到裁切圖：{os.path.abspath(img_path)}")

    # 3) 讀圖並分析
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"無法讀取影像：{img_path}")

    print(
        f"\n=== 處理 {os.path.basename(img_path)} | pixel_per_cm={pixel_per_cm:.4f} ==="
    )
    find_baseline_and_show_all(img, pixel_per_cm)

    # 4) 視窗控制
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
