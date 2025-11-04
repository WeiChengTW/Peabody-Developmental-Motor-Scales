import cv2
import numpy as np
import json
import os
from pathlib import Path

# === 參數（可微調）===
MERGE_GAP_PX = 16  # 合併相鄰突出片段的最小水平距離（12~20 都可試）
MIN_BLOB_AREA = 60  # 最小面積門檻；原本 20 太小容易數到雜點

# 改動：統一用檔案所在資料夾為路徑基準
BASE = Path(__file__).resolve().parent

def calc_mainline_paint_ratio(bw, y_top, y_bot, shrink=6, min_keep_frac=0.5):
    """
    只用來回傳 (x_start, x_end)。為了相容你主程式，第一個回傳值 ratio_dummy 固定 0.0。
    - bw: 黑線專用二值圖（bw_lines）
    - y_top, y_bot: 兩條主線的 y
    - shrink: 垂直向內縮，避免把線身厚度/留白納入
    - min_keep_frac: 左右修剪後至少保留的寬度比例，避免修過頭
    回傳: (0.0, x_start, x_end)
    """
    h, w = bw.shape[:2]

    # 垂直縮內（與主程式邏輯一致，但只用於判斷 x 範圍）
    y0 = max(0, min(h - 1, min(y_top, y_bot) + shrink))
    y1 = max(0, min(h - 1, max(y_top, y_bot) - shrink))
    if y1 < y0:
        y0, y1 = min(y_top, y_bot), max(y_top, y_bot)

    # 先在 y0 / y1 各 ±5 列找左右邊界
    def lr_from_band(band):
        ys, xs = np.where(band > 0)
        if xs.size == 0:
            return 0, w - 1  # 找不到就先退回整寬，後面再修剪
        return int(xs.min()), int(xs.max())

    top_band = bw[max(0, y0 - 5) : min(h, y0 + 6), :]
    bot_band = bw[max(0, y1 - 5) : min(h, y1 + 6), :]
    xl_t, xr_t = lr_from_band(top_band)
    xl_b, xr_b = lr_from_band(bot_band)

    x_start = max(xl_t, xl_b)
    x_end = min(xr_t, xr_b)
    if x_end < x_start:
        x_start, x_end = 0, w - 1

    # 再用帶狀區列最大值修剪左右沒線的空白（避免分母灌水）
    band = bw[y0 : y1 + 1, x_start : x_end + 1]
    if band.size > 0:
        col_has_line = band.max(axis=0) > 0
        xs = np.where(col_has_line)[0]
        if xs.size > 0:
            nx0 = x_start + int(xs[0])
            nx1 = x_start + int(xs[-1])
            if (nx1 - nx0) >= int(min_keep_frac * (x_end - x_start)):
                x_start, x_end = nx0, nx1

    return 0.0, int(x_start), int(x_end)


def analyze_paint(image, y_top, y_bot, show_windows=False):
    """
    分析單張圖的「兩條水平線間塗色」是否達標與超出次數。

    參數：
      - image: 影像 (ndarray) 或 檔案路徑 (str)
      - y_top, y_bot: 上下兩條主線的 y 座標 (int)
      - show_windows: 是否顯示視窗（預覽 debug）

    回傳 dict：
      {
        "ratio": float,             # 塗色佔比 (0~1)
        "protrude_count": int,      # 超出主線區域的「突出片段」數
        "score": int,               # 0/1/2 分
        "rule": str,                # 文字說明
        "x_start": int, "x_end": int,
        "y_in0": int, "y_in1": int
      }
    """
    # 允許傳入路徑或 ndarray
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"圖片讀取失敗：{image}")
    else:
        img = image
        if img is None or not hasattr(img, "shape"):
            raise ValueError("image 需為有效的 ndarray 或檔案路徑字串")

    # ========== 3A. 黑線專用二值圖（for 紫線/x 範圍） ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 只處理主線附近的水平帶
    pad = 24
    y0 = max(0, y_top - pad)
    y1 = min(gray.shape[0], y_bot + pad)
    band = gray[y0:y1, :].copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    band_eq = clahe.apply(band)
    bh = cv2.morphologyEx(
        band_eq, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    )
    band_bw = cv2.adaptiveThreshold(
        bh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    band_bw = cv2.morphologyEx(band_bw, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 做紅筆遮罩（HSV + a 通道）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 35, 35], np.uint8)
    upper1 = np.array([10, 255, 255], np.uint8)
    lower2 = np.array([170, 35, 35], np.uint8)
    upper2 = np.array([180, 255, 255], np.uint8)
    mask_hsv = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_ch = cv2.GaussianBlur(lab[:, :, 1], (5, 5), 0)
    _, mask_a = cv2.threshold(a_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_red_full = cv2.bitwise_and(mask_hsv, mask_a)

    # 移除紅筆在黑線圖上的影響
    band_bw[mask_red_full[y0:y1, :] > 0] = 0
    band_bw = cv2.dilate(band_bw, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)), 1)

    bw_lines = np.zeros_like(gray, dtype=np.uint8)
    bw_lines[y0:y1, :] = band_bw

    # ========== 3B. 筆劃遮罩（紅色 OR 深色鉛筆） ==========

    # 3B-1) 紅色（保留你原本的做法）
    mask_red = mask_red_full.copy()
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
    mask_red = cv2.dilate(mask_red, np.ones((3, 3), np.uint8), 1)

    # 3B-2) 深色鉛筆線（灰階黑線）
    #   做自適應對比＋黑帽凸顯「比背景更暗的細線」，再 Otsu 閾值
    g_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    bh2  = cv2.morphologyEx(
        g_eq, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    _, mask_dark = cv2.threshold(bh2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)

    #   移除主水平線本體（避免把主線當成塗色）
    mainline_thick = cv2.dilate(bw_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)), 1)
    mask_dark[mainline_thick > 0] = 0

    #   限縮到畫面中間亮度偏低的像素（排除淺噪聲，可視情況開/關）
    #   這行可選：若背景很乾淨可先註解
    low_sat_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] < 80
    mask_dark = cv2.bitwise_and(mask_dark, (low_sat_gray.astype(np.uint8) * 255))

    # 3B-3) 最終「筆劃」遮罩＝紅色 OR 深色線
    mask_paint = cv2.bitwise_or(mask_red, mask_dark)


    # 用黑線圖算 x_start, x_end
    _, x_start, x_end = calc_mainline_paint_ratio(bw_lines, y_top, y_bot)

    # ===== A) 垂直縮內（避免把線身厚度/留白吃進分母）=====
    shrink = 6  # 可調 4~10，線越粗縮越多
    y_in0 = max(0, y_top + shrink)
    y_in1 = min(bw_lines.shape[0] - 1, y_bot - shrink)
    if y_in1 < y_in0:  # 防呆
        y_in0, y_in1 = y_top, y_bot

    # ===== B) 用黑線投影修剪左右空白（避免左右沒塗到也算分母）=====
    band2 = bw_lines[y_in0 : y_in1 + 1, x_start : x_end + 1]
    col_has_line = band2.max(axis=0) > 0  # 哪些直欄真的有主線
    xs = np.where(col_has_line)[0]
    if xs.size > 0:
        new_x_start = x_start + int(xs[0])
        new_x_end = x_start + int(xs[-1])
        # 只在合理的情況下收縮（避免誤砍太多）
        if new_x_end - new_x_start >= int(0.5 * (x_end - x_start)):
            x_start, x_end = new_x_start, new_x_end

    # ===== C) 計算突出（x 範圍內、離邊界安全距離）=====
    ys, xs = np.where(mask_paint > 0)
    mask_in_xrange = (xs >= x_start) & (xs <= x_end)
    h, w = mask_red.shape
    margin = 20
    mask_not_near_edge = (
        (xs > margin) & (xs < w - margin) & (ys > margin) & (ys < h - margin)
    )

    tol = 2
    over_mask = (
        ((ys < (y_top - tol)) | (ys > (y_bot + tol)))
        & mask_in_xrange
        & mask_not_near_edge
    )
    over_ys = ys[over_mask]
    over_xs = xs[over_mask]

    over_img = np.zeros_like(mask_red)
    over_img[over_ys, over_xs] = 255
    over_img = cv2.morphologyEx(
        over_img,
        cv2.MORPH_CLOSE,
        np.ones((5, MERGE_GAP_PX), np.uint8),  # 加大水平核合併近距離片段
        iterations=1,
    )
    contours, _ = cv2.findContours(over_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    protrude_count = sum(cv2.contourArea(c) > MIN_BLOB_AREA for c in contours)

    # ===== D) 計算塗色佔比（在縮內區塊）=====
    roi_red = mask_paint[y_in0 : y_in1 + 1, x_start : x_end + 1].copy()
    # 收斂就好，避免過度膨脹偏離筆劃
    roi_red = cv2.morphologyEx(roi_red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    roi_red = cv2.dilate(roi_red, np.ones((5, 3), np.uint8), 1)

    total_area = int(roi_red.size)
    n_white = int(cv2.countNonZero(roi_red))
    ratio = (n_white / total_area) if total_area else 0.0

    # ===== E) 評分規則 =====
    RATIO_MIN = 0.70
    RATIO_TOL = 0.01  # 1% 容忍：避免 74.9%/75.1% 抖動

    def grade(paint_ratio, protrusion_count):
        meets_cover = (paint_ratio + RATIO_TOL) >= RATIO_MIN
        if meets_cover:
            if protrusion_count <= 2:
                return 2, "達到 >= 70%，且超線 <= 2 次 → 2 分"
            elif protrusion_count <= 4:
                return 1, "達到 >= 70%，但超線 3~4 次 → 1 分"
            else:
                return 0, "達到 >= 70%，但超線 >4 次 → 0 分"
        else:
            if protrusion_count <= 2:
                return 1, "未達 70%，雖超線 <= 2 次，但降級 → 1 分"
            elif protrusion_count <= 4:
                return 0, "未達 70%，且超線 3~4 次 → 0 分"
            else:
                return 0, "未達 70%，且超線 >4 次 → 0 分"

    score, rule_msg = grade(ratio, protrude_count)

        # === Draw Area 圖建立與結果輸出 ===
    # 突出標記 + 紫線
    img_out = img.copy()
    img_out[over_ys, over_xs] = (0, 0, 255)
    for y in (y_in0, y_in1):
        cv2.line(img_out, (x_start, y), (x_end, y), (255, 0, 255), 2)

    # Draw Area 領域（主視覺圖）
    img_mask = img.copy()
    roi_color = img_mask[y_in0 : y_in1 + 1, x_start : x_end + 1]
    roi_color[roi_red > 0] = (0, 255, 0)
    cv2.rectangle(img_mask, (x_start, y_in0), (x_end, y_in1), (0, 255, 255), 2)

    def _pick_font():
        for p in [
            r"C:\Windows\Fonts\msjh.ttc",
            r"C:\Windows\Fonts\msjhbd.ttc",
            r"C:\Windows\Fonts\mingliu.ttc",
            r"C:\Windows\Fonts\Arial.ttf",
        ]:
            if os.path.exists(p):
                return p
        return None

    def draw_text_cn(bgr, text, topleft=(20, 35), font_size=28,
                     color=(255,255,255), bg=(0,0,0), pad=8):
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_path = _pick_font()
            if not font_path:
                raise RuntimeError
            img_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(font_path, font_size)
            x, y = topleft
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle((bbox[0]-pad, bbox[1]-pad,
                            bbox[2]+pad, bbox[3]+pad), fill=bg)
            draw.text((x, y), text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(bgr, text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            return bgr

    # 組合輸出文字內容
    base_name = Path(image).name if isinstance(image, (str, os.PathLike)) else "result.jpg"
    summary = (
        f"{base_name} → 佔比: {ratio:.2%}, 突出: {protrude_count}, 分數: {score}\n"
        f"{rule_msg}"
    )

    # 疊字到 Draw Area 圖上
    img_mask = draw_text_cn(img_mask, summary)

    # === 儲存到 result/ ===
    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / base_name

    ok, buf = cv2.imencode(".jpg", img_mask)
    if ok:
        buf.tofile(str(out_path))
        print(f"✅ 已輸出 {out_path}")
    else:
        print(f"⚠️ 儲存失敗：{out_path}")

    # === 不顯示視窗 ===
    # （若你想要臨時預覽，把下面三行解開即可）
    cv2.imshow("Draw Area", img_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return score,img_mask
    # return {
    #     "ratio": ratio,
    #     "protrude_count": protrude_count,
    #     "score": score,
    #     "rule": rule_msg,
    #     "x_start": int(x_start),
    #     "x_end": int(x_end),
    #     "y_in0": int(y_in0),
    #     "y_in1": int(y_in1),
    # }


# 兼容舊命名
analyze_image = analyze_paint



# ============ 測試區（單張）=========== #
def main():
    """
    單張跑法：
      1) 先準備 new/new{img}.jpg
      2) 自動偵測兩條水平線 y_top, y_bot
      3) 呼叫 analyze_paint 做評分
    """
    img = 3  # ← 改成你要測的編號（對應 new/new{img}.jpg）
    # 改動：用 BASE / "new" / ...
    img_path = BASE / "new" / f"new{img}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    # 單張自動抓 y_top, y_bot（不再用 JSON）
    from exceed_correct import detect_horizontal_lines

    y_top, y_bot = detect_horizontal_lines(str(img_path), show_debug=False)
    if y_top is None or y_bot is None:
        raise RuntimeError("偵測不到兩條主線")

    # 分析
    result = analyze_paint(str(img_path), int(y_top), int(y_bot), show_windows=True)
    print(
        f"{os.path.basename(img_path)} → 佔比: {result['ratio']:.2%}, "
        f"突出: {result['protrude_count']}, 分數: {result['score']} | {result['rule']}"
    )


if __name__ == "__main__":
    main()
