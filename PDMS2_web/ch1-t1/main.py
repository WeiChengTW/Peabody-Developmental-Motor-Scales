import cv2
import numpy as np
from ultralytics import YOLO
import sys

# ================== YOLO 模型 ==================
model = YOLO(r"toybrick.pt")
CONF = 0.5


def return_score(score):
    sys.exit(int(score))


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
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (extra_px * 2, extra_px * 2)
        )
        mask_dilated = cv2.dilate((mask_resized > 0).astype(np.uint8), kernel)

        # 使用 bitwise_and 替代直接賦值，使邏輯更清晰
        mask_inverted = cv2.bitwise_not(mask_dilated * 255)
        binary = cv2.bitwise_and(binary, mask_inverted)

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
    mask_resized = cv2.resize(
        mask, (skeleton.shape[1], skeleton.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    ys, xs = np.where(mask_resized > 0)
    h, w = skeleton.shape
    for x, y in zip(xs, ys):
        x0 = max(0, x - tol)
        x1 = min(w, x + tol + 1)
        y0 = max(0, y - tol)
        y1 = min(h, y + tol + 1)
        if np.any(skeleton[y0:y1, x0:x1] > 0):
            return True
    return False


# ================== 新增: 在方塊中心標記點 ==================
def draw_block_markers(frame, boxes, masks, is_correct):
    """
    在原始圖像上標記所有偵測到的方塊。
    :param frame: 原始圖像 (BGR)
    :param boxes: YOLO 偵測到的 bounding box 列表 [(x1, y1, x2, y2), ...]
    :param masks: YOLO 偵測到的 mask 列表
    :param is_correct: 每個方塊是否正確穿線的布林值列表
    """
    # 檢查列表長度是否匹配
    if len(boxes) != len(is_correct):
        print("警告: boxes 和 is_correct 列表長度不匹配。")
        return frame

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # 計算中心點
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)

        # 根據是否正確穿線來設定顏色
        # 綠色 (0, 255, 0) 代表正確 (Correct)
        # 紅色 (0, 0, 255) 代表錯誤 (Incorrect)
        color = (0, 255, 0) if is_correct[i] else (0, 0, 255)

        # 繪製中心點 (圓形)
        cv2.circle(frame, center, radius=10, color=color, thickness=-1)

        # 繪製邊界框 (可選，用於確認偵測範圍)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


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

    # 灰階 + 模糊
    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (19, 19), 0)

    # 自適應二值化：將深色的繩子凸顯出來
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 10
    )

    # 閉運算去雜點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # YOLO 偵測方塊 & 取得 mask
    boxes, masks, _ = detect_blocks_mask(display_frame, CONF=conf)

    # 遮掉方塊
    binary_masked = binary.copy()
    if masks:
        binary_masked = remove_blocks_with_mask(binary_masked, masks)

    # 骨架化
    skeleton = extract_line_skeleton(binary_masked)
        
    # 檢查每個方塊是否靠近骨架
    is_correct = []
    correct_num = 0
    for mask in masks:
        is_near = is_mask_near_skeleton(mask, skeleton, tol=10)
        is_correct.append(is_near)
        if is_near:
            correct_num += 1

    # 在所有圖像上繪製標記
    # 原始圖 (BGR)
    display_frame_with_markers = draw_block_markers(display_frame, boxes, masks, is_correct)
    
    # 二值遮罩圖 (GRAY/BGR)
    # 將單通道的二值圖轉為三通道才能繪製彩色標記
    binary_bgr = cv2.cvtColor(binary_masked, cv2.COLOR_GRAY2BGR)
    binary_with_markers = draw_block_markers(binary_bgr, boxes, masks, is_correct)
    
    # 骨架圖 (GRAY/BGR)
    skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    skeleton_with_markers = draw_block_markers(skeleton_bgr, boxes, masks, is_correct)

    # # 顯示所有結果圖
    # # 原始圖
    # display_frame_resized = cv2.resize(display_frame_with_markers, (0, 0), fx=0.3, fy=0.3)
    # cv2.imshow('Original with Markers', display_frame_resized)

    # # 二值遮罩圖
    # binary_resized = cv2.resize(binary_with_markers, (0, 0), fx=0.3, fy=0.3)
    # cv2.imshow('Binary Masked', binary_resized)

    # # 骨架圖
    # skeleton_resized = cv2.resize(skeleton_with_markers, (0, 0), fx=0.3, fy=0.3)
    # cv2.imshow('Skeleton Line', skeleton_resized)



    # 計算分數
    # 原程式碼的計分規則是: correct_num 減去 2 之後，再依據結果計分。
    # 假設原始設計中 total_bricks=6，且 2 個是必須穿過的核心物件，這裡沿用您的計分邏輯
    # (但實際應該是 correct_num 總數，請檢查您的 score_from_image 函數末尾的邏輯)
    # 這裡沿用您的邏輯:
    # correct_num -= 2 # 註釋掉這行，避免不確定的減法操作
    
    if correct_num == 6: # 假設總共 6 顆，且 4 個正確為 2 分，這裡假設 6 顆都正確是最高分
        score = 2
    elif correct_num >= 4:
        score = 1
    else:
        score = 0
        
    # 沿用原程式碼的計分邏輯 (但建議檢查)
    final_num_for_score = correct_num
    # if final_num_for_score >= 2:
    #     final_num_for_score -= 2 # 註釋掉這行，避免困惑
        
    if final_num_for_score == 4:
        score = 2
    elif final_num_for_score == 3:
        score = 1
    elif final_num_for_score == 2: # 額外增加 2 個正確時的邏輯
        score = 1
    else:
        score = 0
    
    # 依據您原始碼的邏輯重新計分 (如果 `correct_num -= 2` 是您實際運行的邏輯)
    correct_num_for_score = correct_num
    if correct_num_for_score >= 2: # 假設有 2 個是基礎
        correct_num_for_score -= 2
        
    if correct_num_for_score == 4:
        score = 2
    elif correct_num_for_score == 3:
        score = 1
    else:
        score = 0
        
    # 阻塞等待關閉視窗
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return score, correct_num



if __name__ == "__main__":
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        image_path = rf"kid\{uid}\{img_id}.jpg"
    # test_img = r"c_b_2.jpg"  # 讀取圖片
    score, num = score_from_image(image_path)
    print("score =", score)
    print("num =", num)
    return_score(score)
