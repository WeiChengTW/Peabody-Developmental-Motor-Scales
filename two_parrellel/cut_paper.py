import cv2
import numpy as np
import os
import glob

def crop_paper(image_path, output_path=None, show_debug=False):
    """
    偵測並裁切白紙區域
    - image_path: 輸入圖片路徑
    - output_path: 輸出裁切圖片路徑（可選）
    - show_debug: 是否顯示偵測過程

    回傳: warped (裁切後的影像)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"圖片讀取失敗：{image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    for idx, cnt in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        print(f"輪廓 {idx}: 頂點數 = {len(approx)}，面積 = {cv2.contourArea(cnt)}")
        if len(approx) == 4:  # 找到四邊形
            approx_paper = approx
            found = True
            break

    if not found:
        raise ValueError("無法偵測白紙四邊形輪廓")

    pts = approx_paper.reshape(4, 2)
    pts = sorted(pts, key=lambda p: p[0])
    left = sorted(pts[0:2], key=lambda p: p[1])
    right = sorted(pts[2:4], key=lambda p: p[1])
    tl, bl = left
    tr, br = right

    dst_width = int(np.linalg.norm(tr - tl))
    dst_height = int(np.linalg.norm(bl - tl))
    dst_pts = np.array([
        [0, 0], [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1], [0, dst_height - 1]
    ], dtype="float32")
    src_pts = np.array([tl, tr, br, bl], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (dst_width, dst_height))

    # 輸出檔案（如果有指定）
    if output_path is not None:
        cv2.imwrite(output_path, warped)

    # 偵錯顯示
    if show_debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, [approx_paper], -1, (0, 0, 255), 3)
        cv2.imshow("Detected Paper Contour", debug_img)
        cv2.imshow("Cropped Paper", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped


# 測試區：單獨執行這個檔案時，會批次處理 image/ 資料夾
if __name__ == "__main__":
    image_folder = "image"
    output_folder = "new"
    os.makedirs(output_folder, exist_ok=True)

    # 指定要處理的單張圖片（例如 2.jpg）
    img_num = 1
    img_path = os.path.join(image_folder, f"{img_num}.jpg")
    output_path = os.path.join(output_folder, f"new{img_num}.jpg")

    try:
        crop_paper(img_path, output_path, show_debug=False)
        print(f"{img_path} → 已裁切儲存到 {output_path}")
    except Exception as e:
        print(f"{img_path} → {e}")

