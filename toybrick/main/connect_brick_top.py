import cv2
import numpy as np
from ultralytics import YOLO

# ================== YOLO 模型 ==================
model = YOLO(r'toybrick.pt')
CONF = 0.5

# ================== YOLO 偵測方塊 & 取得 mask ==================
def detect_blocks_mask(frame, CONF=0.5):
    results = model.predict(source=frame, conf=CONF, verbose=False)
    boxes, masks = [], []

    for r in results:
        if r.boxes is None:
            continue
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == 0:  # 方塊類別
                boxes.append((x1, y1, x2, y2))
                if r.masks is not None:
                    mask = r.masks.data.cpu().numpy()[i]
                    masks.append(mask)
    return boxes, masks, results

# ================== 遮掉方塊 ==================
def remove_blocks_with_mask(binary, masks, extra_px=10):
    h, w = binary.shape
    for mask in masks:
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # 膨脹 mask，增加遮擋範圍
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_px*2, extra_px*2))
        mask_dilated = cv2.dilate((mask_resized > 0).astype(np.uint8), kernel)
        binary[mask_dilated > 0] = 0
    return binary

# ================== 骨架化 ==================
def extract_line_skeleton(binary):
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = np.copy(binary)
    while True:
        open_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
        temp2 = cv2.subtract(temp, open_img)
        eroded = cv2.erode(temp, element)
        skeleton = cv2.bitwise_or(skeleton, temp2)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton

# ================== 判斷是否在骨架線點附近 ==================
def is_mask_near_skeleton(mask, skeleton, tol=5):
    """
    mask: 二值 mask (0/1 或 0/255)
    skeleton: 骨架二值圖
    tol: 搜尋半徑
    """
    mask_resized = cv2.resize(mask, (skeleton.shape[1], skeleton.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_resized > 0)
    h, w = skeleton.shape
    for x, y in zip(xs, ys):
        x0 = max(0, x - tol);  x1 = min(w, x + tol + 1)
        y0 = max(0, y - tol);  y1 = min(h, y + tol + 1)
        if np.any(skeleton[y0:y1, x0:x1] > 0):
            return True
    return False

# ================== 封裝：讀取 img_path 並回傳 score ==================
def score_from_image(img_path, conf=CONF):
    """
    輸入：img_path（圖片路徑）
    輸出：score（int）→ 4 個正確=2 分；3 個正確=1 分；其他=0 分
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"讀取圖片失敗：{img_path}")

    display_frame = img.copy()

    # 灰階 + 模糊 + Otsu 二值化
    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 閉運算去雜點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # YOLO 偵測方塊 & 取得 mask
    boxes, masks, _ = detect_blocks_mask(display_frame, CONF=conf)

    # 遮掉方塊
    if masks:
        binary = remove_blocks_with_mask(binary, masks)

    # 骨架化
    skeleton = extract_line_skeleton(binary)

    # 檢查每個方塊是否靠近骨架
    correct_num = 0
    for mask in masks:
        if is_mask_near_skeleton(mask, skeleton, tol=10):
            correct_num += 1

    # 依規則計分
    if correct_num == 4:
        score = 2
    elif correct_num == 3:
        score = 1
    else:
        score = 0

    return score, correct_num

# ======= 範例用法（不需要可刪） =======
if __name__ == "__main__":
    test_img = r"c_b_2.jpg"  # 讀取圖片
    s, num = score_from_image(test_img)
    print("score =", s)
    print("num = ", num)
