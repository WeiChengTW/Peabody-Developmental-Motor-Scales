# crop_measure.py
import cv2
import numpy as np
import json
import glob
import os

ORIG_FOLDER = "images"   # 原始圖片
CROP_FOLDER = "new"      # 裁切輸出
PIXEL_MAP_JSON = "pixel_per_cm_map.json"

os.makedirs(CROP_FOLDER, exist_ok=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]        # tl
    rect[2] = pts[np.argmax(s)]        # br
    rect[1] = pts[np.argmin(diff)]     # tr
    rect[3] = pts[np.argmax(diff)]     # bl
    return rect

def detect_quad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        return approx.reshape(4,2).astype(np.float32)
    # 後援：最小外接矩形
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def warp_a4(image, quad):
    quad = order_points(quad)
    (tl, tr, br, bl) = quad
    w1 = np.linalg.norm(tr - tl); w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl); h2 = np.linalg.norm(br - tr)
    width_px  = int(round(max(w1, w2)))
    height_px = int(round(max(h1, h2)))
    dst = np.array([[0,0],[width_px-1,0],[width_px-1,height_px-1],[0,height_px-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (width_px, height_px))
    return warped, width_px, height_px

def measure_pixel_per_cm(width_px, height_px, long_cm=29.7, short_cm=21.0):
    # 以較長邊對應 29.7 cm 為基準
    if width_px < height_px:
        width_px, height_px = height_px, width_px
    return float(width_px / long_cm)

def batch_crop_and_measure(src_dir=ORIG_FOLDER, out_dir=CROP_FOLDER, save_map=PIXEL_MAP_JSON):
    # 1) 擴充分檔名 + 印出找到的檔案
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for p in patterns:
        image_files += glob.glob(os.path.join(src_dir, p))
    image_files = sorted(image_files)

    print(f"[INFO] CWD = {os.getcwd()}")
    print(f"[INFO] 將從資料夾：{os.path.abspath(src_dir)} 讀圖")
    print(f"[INFO] 找到 {len(image_files)} 張：")
    for fp in image_files:
        print(" -", os.path.abspath(fp))

    if not image_files:
        raise ValueError(f"資料夾 {src_dir} 沒有找到圖片（請確認執行時的工作目錄與副檔名）")

    ratio_map = {}
    counter = 1  

    for path in image_files:
        base = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 無法讀取：{path}")
            continue

        # 2) 先試 A4 偵測；失敗就用「整張」當作裁切（保證會輸出）
        quad = detect_quad(img)
        if quad is not None and len(quad) == 4:
            warped, wpx, hpx = warp_a4(img, quad)
        else:
            print(f"⚠️ {base} 偵測不到四邊形，啟用後援：直接使用整張影像")
            warped = img.copy()
            hpx, wpx = warped.shape[:2]

        # 3) 計算像素/公分：以較長邊對應 A4 29.7 cm
        if wpx < hpx:
            wpx, hpx = hpx, wpx
        ppcm = float(wpx / 29.7)

        # 4) 存檔（new2.jpg, new3.jpg...）
        out_filename = f"new{counter}.jpg"
        out_path = os.path.join(out_dir, out_filename)
        ok = cv2.imwrite(out_path, warped)
        if not ok:
            print(f"❌ 寫檔失敗：{out_path}")
            continue

        ratio_map[out_filename] = ppcm
        print(f"✅ {out_filename} → 輸出完成，pixel_per_cm={ppcm:.4f}")
        counter += 1

    with open(save_map, "w", encoding="utf-8") as f:
        json.dump(ratio_map, f, ensure_ascii=False, indent=2)
    print(f"📄 已寫入 {save_map} ，共 {len(ratio_map)} 筆")
    return ratio_map

if __name__ == "__main__":
    batch_crop_and_measure()
