import cv2
import numpy as np

def refine_y_center(bw, y0, cover_frac=0.25, search=12):
    h, w = bw.shape
    y0 = int(np.clip(y0, 0, h - 1))
    y_lo = max(0, y0 - search)
    y_hi = min(h - 1, y0 + search)
    row_sum = (bw > 0).sum(axis=1)
    idx = np.where(row_sum[y_lo:y_hi + 1] > int(cover_frac * w))[0]
    if idx.size == 0:
        return y0
    y_top = y_lo + idx[0]
    y_bot = y_lo + idx[-1]
    return (y_top + y_bot) // 2

def group_lines_1d(ylist, gap=8):
    if not ylist:
        return []
    ylist = sorted(ylist)
    grouped = []
    group = [ylist[0]]
    for y in ylist[1:]:
        if abs(y - group[-1]) <= gap:
            group.append(y)
        else:
            grouped.append(group)
            group = [y]
    grouped.append(group)
    return [int(np.median(g)) for g in grouped]

def detect_horizontal_lines(image, show_debug=False):
    """
    輸入影像 (numpy array 或影像路徑)，回傳找到的水平線 y 座標列表
    - image: 可給路徑字串或已讀取的影像 (ndarray)
    - show_debug: 是否顯示偵測過程
    """
    # 若傳入的是路徑就讀圖
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    if img is None:
        return []

    h, w = img.shape[:2]

    # 1) BlackHat 強化暗線條
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    )
    # Otsu + 保底門檻
    _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(bw) < 0.003 * h * w:
        _, bw = cv2.threshold(bh, 25, 255, cv2.THRESH_BINARY)

    # 2) 移除紅色塗線（若有）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 40], np.uint8);   upper1 = np.array([10, 255, 255], np.uint8)
    lower2 = np.array([170, 70, 40], np.uint8); upper2 = np.array([180, 255, 255], np.uint8)
    mask_red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    bw[mask_red > 0] = 0

    # 3) 收斂範圍
    MARGIN_X = max(10, int(0.03 * w))
    MARGIN_Y = max(10, int(0.03 * h))
    bw[:, :MARGIN_X] = 0
    bw[:, w - MARGIN_X:] = 0
    bw[:MARGIN_Y, :] = 0
    bw[h - MARGIN_Y:, :] = 0

    # 4) 水平閉運算：補斷線
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(31, w // 20), 3))
    bw_close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 5) 列投影法
    row_sum = (bw_close > 0).sum(axis=1).astype(np.int32)
    ker = np.ones(5, dtype=np.float32) / 5
    row_smooth = np.convolve(row_sum, ker, mode='same')

    cover_thr = int(0.12 * w)
    cand_rows = np.where(row_smooth > cover_thr)[0].tolist()
    cand_rows = group_lines_1d(cand_rows, gap=8)

    def longest_run(row_bin: np.ndarray) -> int:
        r = (row_bin > 0).astype(np.uint8)
        diff = np.diff(np.concatenate(([0], r, [0])))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        return int((ends - starts).max()) if starts.size else 0

    good_rows = []
    for y in cand_rows:
        y = int(np.clip(y, 0, h - 1))
        band_row = bw_close[max(0, y - 1):min(h, y + 2), :].max(axis=0)
        if longest_run(band_row) >= int(0.35 * w):
            good_rows.append(y)

    # Hough 輔助
    if len(good_rows) < 2:
        edges = cv2.Canny(bw_close, 50, 120, apertureSize=3)
        minLineLength = int(0.20 * w)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40,
            minLineLength=minLineLength, maxLineGap=25
        )
        ylist = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if abs(y2 - y1) <= 8:
                    ylist.append((y1 + y2) // 2)
        ylist = group_lines_1d(ylist, gap=8)
        good_rows = sorted(set(good_rows + ylist))

    # 取上下兩條
    if len(good_rows) >= 2:
        ylist_grouped = group_lines_1d(good_rows, gap=8)
        center_lines = [ylist_grouped[0], ylist_grouped[-1]]
    else:
        center_lines = good_rows

    # 細緻化
    center_lines = [
        refine_y_center(bw_close, y, cover_frac=0.12, search=18) for y in center_lines
    ]

    if show_debug:
        dbg = cv2.cvtColor(bw_close, cv2.COLOR_GRAY2BGR)
        for y in center_lines:
            cv2.line(dbg, (0, int(y)), (w - 1, int(y)), (0, 255, 255), 2)
        cv2.imshow("Detected lines", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return center_lines


# 測試區：單獨執行這個檔案，會去 new/ 找圖並輸出 main_lines.json
if __name__ == "__main__":
    import glob, os, json
    new_folder = "new"
    os.makedirs(new_folder, exist_ok=True)

    image_files = sorted(
        glob.glob(os.path.join(new_folder, "*.jpg")) +
        glob.glob(os.path.join(new_folder, "*.png"))
    )

    all_lines = {}
    for img_path in image_files:
        base = os.path.basename(img_path)
        center_lines = detect_horizontal_lines(img_path, show_debug=False)
        if len(center_lines) >= 2:
            y_top, y_bot = int(min(center_lines)), int(max(center_lines))
        elif len(center_lines) == 1:
            y_top = y_bot = int(center_lines[0])
        else:
            y_top = y_bot = None
        all_lines[base] = {"y_top": y_top, "y_bot": y_bot}

    with open("main_lines.json", "w", encoding="utf-8") as f:
        json.dump(all_lines, f, ensure_ascii=False, indent=2)
    print("✅ 已輸出 main_lines.json")
