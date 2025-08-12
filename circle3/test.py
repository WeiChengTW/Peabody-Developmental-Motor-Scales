"""判斷圖形是圓還是橢圓的批次腳本

流程:
1. 讀取 img 目錄所有影像
2. 前處理 (灰階 -> 模糊 -> OTSU 閾值)；雙向(黑底白形/白底黑形)取較佳者
3. 擷取最大輪廓
4. 計算:
   - 外接矩形長寬比 (w/h)
   - 橢圓擬合主軸比 (major/minor)
   - 圓形度 circularity = 4πA/P²
5. 依多種指標綜合判斷:
   - axis_ratio = major/minor (或 w/h 輔助)
   - 若 axis_ratio < ratio_thresh 且 circularity > circ_thresh -> Circle
   - 否則 Ellipse
6. 輸出結果於終端並把標記後圖片存到 output

可調參數: ratio_thresh, circ_thresh。
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError as e:  # 讓使用者知道需要安裝套件
    print(
        "缺少必要套件: opencv-python 或 numpy，請先安裝後再執行。\n"
        "可執行: pip install -r requirements.txt"
    )
    raise


IMG_DIR = "img"
OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class ShapeMetrics:
    classification: str
    axis_ratio: float
    bbox_ratio: float
    circularity: float
    area: float
    perimeter: float
    major_axis: float
    minor_axis: float
    ellipse_angle: Optional[float]
    ellipse_area_fit: Optional[float]
    hough_circle: bool = False  # 是否經由 Hough 補救成圓


def preprocess(img_bgr):
    # 灰階 + 高斯模糊
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 自適應二值化
    # th1 = cv2.adaptiveThreshold(
    #     blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    th2 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # cv2.imshow(
    #     "Threshold 1",
    #     cv2.resize(th1, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
    # )
    # 如需觀察閾值效果可解除註解顯示
    # cv2.imshow(
    #     "Threshold 2",
    #     cv2.resize(th2, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
    # )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 侵蝕操作 (可選, 視圖形特性而定)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # th1 = cv2.erode(th1, kernel_erode, iterations=1)
    th2 = cv2.erode(th2, kernel_erode, iterations=1)

    # 選擇連通區最大輪廓面積較大的那個版本
    def largest_area(th):
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0
        return max(cv2.contourArea(c) for c in cnts)

    # 簡單形態學平滑 (避免破洞)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph


def repair_mask(bin_img: np.ndarray) -> np.ndarray:
    """修補有缺口 / 斷線輪廓。
    步驟:
    1. 膨脹 + 閉運算: 橋接小裂縫
    2. 合併所有輪廓 (填滿)
    3. 孔洞填補: floodFill 外背景後取反
    """
    work = bin_img.copy()
    h, w = work.shape
    k = max(3, int(min(h, w) * 0.02))
    if k % 2 == 0:
        k += 1
    k_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    work = cv2.dilate(work, k_ell, iterations=1)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k_ell, iterations=1)
    cnts, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged = np.zeros_like(work)
    cv2.drawContours(merged, cnts, -1, 255, thickness=cv2.FILLED)
    flood = merged.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    flood[flood != 128] = 255
    flood[flood == 128] = 0
    return flood


def find_main_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    # 過濾太小的
    cnts = [c for c in cnts if cv2.contourArea(c) > 50]
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def classify_contour(
    contour,
    ratio_thresh=1.15,
    circ_thresh=0.75,
    area_fit_tol=0.20,
) -> ShapeMetrics:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * math.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(contour)
    bbox_ratio = max(w, h) / (min(w, h) + 1e-6)

    major_axis = minor_axis = ellipse_angle = None
    ellipse_area_fit = None
    axis_ratio = bbox_ratio
    if len(contour) >= 5:  # fitEllipse 要求 >=5 點
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
        # MA: 長軸, ma: 短軸
        major_axis, minor_axis = max(MA, ma), min(MA, ma)
        axis_ratio = major_axis / (minor_axis + 1e-6)
        ellipse_angle = angle
        ellipse_area = math.pi * major_axis * minor_axis / 4.0
        if ellipse_area > 0:
            ellipse_area_fit = area / ellipse_area
    else:
        major_axis = float(max(w, h))
        minor_axis = float(min(w, h))

    # 綜合判斷: 軸比接近 1 且圓形度高 -> 圓
    classification = "Ellipse"
    if axis_ratio < ratio_thresh:
        cond_circ = circularity >= circ_thresh
        cond_fit = (
            ellipse_area_fit is not None and abs(1 - ellipse_area_fit) <= area_fit_tol
        )
        if cond_circ or cond_fit:
            classification = "Circle"

    return ShapeMetrics(
        classification,
        axis_ratio,
        bbox_ratio,
        circularity,
        area,
        perimeter,
        float(major_axis),
        float(minor_axis),
        ellipse_angle,
        ellipse_area_fit,
    )


def annotate_and_save(img, contour, metrics: ShapeMetrics, out_path: str):
    vis = img.copy()
    color = (0, 255, 0) if metrics.classification == "Circle" else (0, 165, 255)
    cv2.drawContours(vis, [contour], -1, color, 2)
    # 畫擬合橢圓
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(vis, ellipse, (255, 0, 0), 1)
    fit_txt = (
        f" fit={metrics.ellipse_area_fit:.2f}"
        if metrics.ellipse_area_fit is not None
        else ""
    )
    text = (
        f"{metrics.classification} r={metrics.axis_ratio:.3f} circ={metrics.circularity:.3f}"
        f" a={metrics.major_axis:.1f}/{metrics.minor_axis:.1f} bbox={metrics.bbox_ratio:.3f}{fit_txt}"
    )
    cv2.putText(
        vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2, cv2.LINE_AA
    )
    cv2.imwrite(out_path, vis)


def process_image(
    path: str,
    ratio_thresh=1.15,
    circ_thresh=0.75,
    area_fit_tol=0.20,
    repair: bool = False,
    use_hough: bool = False,
    circle_by_hough_only: bool = False,
) -> Optional[ShapeMetrics]:
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] 無法讀取: {path}")
        return None
    bin_img = preprocess(img)
    if repair:
        bin_img = repair_mask(bin_img)
    contour = find_main_contour(bin_img)
    if contour is None:
        print(f"[WARN] 找不到形狀: {os.path.basename(path)}")
        return None
    metrics = classify_contour(contour, ratio_thresh, circ_thresh, area_fit_tol)

    # 若使用 Hough 且仍為 Ellipse，嘗試用 HoughCircles 補救破碎圓
    if use_hough and metrics.classification == "Ellipse":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        rows = gray.shape[0]
        min_r = int(min(gray.shape[:2]) * 0.15)
        max_r = int(min(gray.shape[:2]) * 0.60)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=rows / 3,
            param1=120,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None and metrics.axis_ratio < 1.3:
            metrics.classification = "Circle"
            metrics.hough_circle = True

    # 特殊模式: 僅當 Hough 偵測到才算 Circle，其餘一律視為 Ellipse
    if circle_by_hough_only and not metrics.hough_circle:
        metrics.classification = "Ellipse"
    out_name = os.path.splitext(os.path.basename(path))[0] + "_annotated.jpg"
    annotate_and_save(img, contour, metrics, os.path.join(OUT_DIR, out_name))
    return metrics


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Circle / Ellipse 分類")
    parser.add_argument(
        "--ratio", type=float, default=1.15, help="軸比閾值 (預設: 1.15)"
    )
    parser.add_argument(
        "--circ", type=float, default=0.75, help="圓形度閾值 (預設: 0.75)"
    )
    parser.add_argument(
        "--fit_tol",
        type=float,
        default=0.20,
        help="擬合面積允收差 |1-fit| (預設: 0.20)",
    )
    parser.add_argument("--repair", action="store_true", help="啟用輪廓修補")
    # 預設啟用 Hough，可用 --no-hough 關閉
    parser.add_argument(
        "--hough", dest="hough", action="store_true", help="啟用 Hough 輔助 (預設: 開)"
    )
    parser.add_argument(
        "--no-hough", dest="hough", action="store_false", help="停用 Hough 輔助"
    )
    # 預設啟用 only-by-hough，可用 --no-circle_by_hough_only 關閉
    parser.add_argument(
        "--circle_by_hough_only",
        dest="circle_by_hough_only",
        action="store_true",
        help="只有 Hough 偵測才標 Circle (預設: 開)",
    )
    parser.add_argument(
        "--no-circle_by_hough_only",
        dest="circle_by_hough_only",
        action="store_false",
        help="允許幾何規則直接判 Circle",
    )
    parser.set_defaults(hough=True, circle_by_hough_only=True)
    return parser.parse_args()


def main():
    args = parse_args()
    files = [
        f
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    files.sort()
    if not files:
        print("img 資料夾沒有影像。")
        return
    print("檔名, 判斷, 軸比(ellipse), 外接框比, 圓形度, 面積, 周長, 擬合面積比, Hough")
    for f in files:
        full = os.path.join(IMG_DIR, f)
        metrics = process_image(
            full,
            ratio_thresh=args.ratio,
            circ_thresh=args.circ,
            area_fit_tol=args.fit_tol,
            repair=args.repair,
            use_hough=args.hough,
            circle_by_hough_only=args.circle_by_hough_only,
        )
        if metrics:
            fit_val = (
                f"{metrics.ellipse_area_fit:.3f}"
                if metrics.ellipse_area_fit is not None
                else "-"
            )
            hough_flag = "Y" if metrics.hough_circle else "-"
            print(
                f"{f}, {metrics.classification}, {metrics.axis_ratio:.4f}, {metrics.bbox_ratio:.4f}, "
                f"{metrics.circularity:.4f}, {metrics.area:.1f}, {metrics.perimeter:.2f}, {fit_val}, {hough_flag}"
            )

    print(f"完成。標記後影像已輸出到 {OUT_DIR} 目錄。")


if __name__ == "__main__":
    main()
