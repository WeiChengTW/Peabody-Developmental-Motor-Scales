import time
import cv2
import numpy as np
from PIXEL_TO_CM import get_pixel_per_cm

image_path = "img18.png"
frame = cv2.imread(image_path)

if frame is None:
    print(f"錯誤：無法讀取圖像文件 '{image_path}'")
    print("請確認圖像文件存在且路徑正確")
    exit()


# 直接呼叫 get_pixel_per_cm 取得比例
PIXEL_TO_CM = 0.002  # 預設值
RULER_LENGTH_CM = 16
try:
    pixel_per_cm_result = get_pixel_per_cm(
        image_path, cm_length=RULER_LENGTH_CM, show=False
    )
    if pixel_per_cm_result:
        pixel_per_cm, long_side, _ = pixel_per_cm_result
        PIXEL_TO_CM = 1 / pixel_per_cm  # 1 px = ? cm
        print(f"自動校正像素轉換比例: {PIXEL_TO_CM:.6f} cm/pixel (以16cm尺)")
    else:
        print("自動校正失敗，使用預設比例")
except Exception as e:
    print(f"自動校正比例時發生錯誤: {e}")
    print("使用預設比例")


## 不再檢測16cm尺，直接使用 get_pixel_per_cm 取得比例


def detect_paper_contour(frame):
    # 轉換為灰度圖
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，減少雜訊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 邊緣檢測
    edges = cv2.Canny(blurred, 10, 150, apertureSize=3)

    # 形態學操作，連接邊緣
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges


def find_paper_contour(edges, frame_shape):
    # 找到所有輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的輪廓（假設是紙張）
    if contours:
        # 按面積排序，取最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 檢查輪廓面積是否足夠大
        area = cv2.contourArea(largest_contour)
        frame_area = frame_shape[0] * frame_shape[1]

        # 如果輪廓面積占畫面的一定比例，認為是紙張
        if area > frame_area * 0.1:  # 至少占畫面 10%
            # 使用多邊形逼近來簡化輪廓
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            return approx, largest_contour

    return None, None


def detect_black_lines_on_paper(frame, paper_contour):
    """在檢測到的紙張區域內檢測黑線，重點關注AB邊附近"""
    if paper_contour is None:
        return None, None

    # 創建遮罩，只保留紙張區域
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [paper_contour], 255)

    # 只在AB邊向上區域內，根據edges尋找黑線
    if len(paper_contour) >= 2:
        # 找出Y值最高的兩點作為AB
        pts = [tuple(pt[0]) for pt in paper_contour]
        pts_sorted = sorted(pts, key=lambda p: p[1], reverse=True)
        point_a = np.array(pts_sorted[0])
        point_b = np.array(pts_sorted[1])

        ab_vector = point_b - point_a
        ab_length = np.linalg.norm(ab_vector)
        if ab_length > 0:
            ab_unit = ab_vector / ab_length
            perp1 = np.array([-ab_unit[1], ab_unit[0]])
            perp2 = -perp1
            perpendicular_up = perp1 if perp1[1] < 0 else perp2
            detection_width_up = min(100, ab_length * 0.5)

            # 建立向上偵測區域遮罩
            offset_up = perpendicular_up * detection_width_up
            region_points_up = np.array(
                [
                    point_a,
                    point_b,
                    [point_b[0] + offset_up[0], point_b[1] + offset_up[1]],
                    [point_a[0] + offset_up[0], point_a[1] + offset_up[1]],
                ],
                dtype=np.int32,
            )

            ab_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(ab_mask, [region_points_up], 255)

            # 儲存區域座標供可視化
            global detection_region_up
            detection_region_up = region_points_up.copy()

            # 只保留紙張區域與向上區域的交集
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [paper_contour], 255)
            mask_combined = cv2.bitwise_and(mask, ab_mask)

            # 只保留edges在此區域
            global edges
            black_lines = cv2.bitwise_and(edges, edges, mask=mask_combined)
            # 更嚴格：先做形態學開運算（去雜訊），再做膨脹，並提高邊緣強度門檻
            kernel = np.ones((2, 2), np.uint8)
            # 先做開運算去除雜點
            opened = cv2.morphologyEx(black_lines, cv2.MORPH_OPEN, kernel, iterations=1)
            # 再做膨脹讓細線變粗
            opened = cv2.dilate(opened, kernel, iterations=1)
            # 僅保留強度較高的邊緣（像素值>64）
            _, opened_strict = cv2.threshold(opened, 64, 255, cv2.THRESH_BINARY)
            return opened_strict, black_lines
    return None, None


def analyze_black_lines(binary_lines, paper_contour, frame_shape):
    """分析檢測到的黑線，並計算到AB邊的距離"""
    if binary_lines is None or paper_contour is None:
        return []

    # 找到黑線輪廓
    contours, _ = cv2.findContours(
        binary_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    line_info = []

    # 以長邊作為AB邊
    if len(paper_contour) >= 2:
        max_len = 0
        ab_idx = (0, 1)
        n = len(paper_contour)
        for i in range(n):
            pt1 = paper_contour[i][0]
            pt2 = paper_contour[(i + 1) % n][0]
            seg_len = np.linalg.norm(np.array(pt2) - np.array(pt1))
            if seg_len > max_len:
                max_len = seg_len
                ab_idx = (i, (i + 1) % n)
        point_a = paper_contour[ab_idx[0]][0]
        point_b = paper_contour[ab_idx[1]][0]
    else:
        return line_info

    def is_contour_ab_edge(
        contour, point_a, point_b, threshold=0.97, max_dist_ratio=0.08
    ):
        # 取得輪廓的最遠兩點
        contour_pts = contour.reshape(-1, 2)
        dists = np.linalg.norm(
            contour_pts[:, None, :] - contour_pts[None, :, :], axis=2
        )
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        c1, c2 = contour_pts[max_idx[0]], contour_pts[max_idx[1]]
        # 計算這條線與AB邊的重疊比例
        ab_vec = np.array(point_b) - np.array(point_a)
        c_vec = c2 - c1
        ab_len = np.linalg.norm(ab_vec)
        c_len = np.linalg.norm(c_vec)
        if ab_len == 0 or c_len == 0:
            return False
        # 方向要非常接近
        direction_sim = np.abs(np.dot(ab_vec, c_vec) / (ab_len * c_len))
        # 端點要非常靠近
        dist_a_c1 = np.linalg.norm(np.array(point_a) - c1)
        dist_b_c2 = np.linalg.norm(np.array(point_b) - c2)
        dist_a_c2 = np.linalg.norm(np.array(point_a) - c2)
        dist_b_c1 = np.linalg.norm(np.array(point_b) - c1)
        min_dist = min(dist_a_c1 + dist_b_c2, dist_a_c2 + dist_b_c1)
        # 更嚴格：方向極接近且端點距離極小才排除
        return direction_sim > threshold and min_dist < ab_len * max_dist_ratio

    for contour in contours:
        # 過濾掉太小的輪廓
        area = cv2.contourArea(contour)
        if area < 20:  # 最小面積閾值
            continue

        # 計算邊界框
        x, y, w, h = cv2.boundingRect(contour)

        # 過濾掉太小的邊界框
        if w < 5 or h < 5:
            continue

        # 排除AB邊本身
        if is_contour_ab_edge(contour, point_a, point_b):
            continue

        # 計算中心點（用於資訊顯示，不再用於距離計算）
        center_x = x + w // 2
        center_y = y + h // 2

        # 計算黑線輪廓中點到AB邊的距離*2
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            """計算點到線段的最短距離"""
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D

            if len_sq == 0:
                return np.sqrt(A * A + B * B)

            param = dot / len_sq

            if param < 0:
                xx = x1
                yy = y1
            elif param > 1:
                xx = x2
                yy = y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            dx = px - xx
            dy = py - yy
            return np.sqrt(dx * dx + dy * dy)

        # 用OpenCV的boundingRect中心作為中點
        center_x = x + w // 2
        center_y = y + h // 2
        distance_to_ab_edge = point_to_line_distance(
            center_x, center_y, point_a[0], point_a[1], point_b[0], point_b[1]
        )
        print(f"中心點到AB邊距離: {distance_to_ab_edge:.2f} px")
        distance_to_ab_edge_cm = distance_to_ab_edge * 2 * PIXEL_TO_CM * 0.7
        # 計算到紙張其他邊界的距離（保持原有功能）
        paper_distances = []
        for point in paper_contour:
            px, py = point[0]
            dist = np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2)
            paper_distances.append(dist)
        min_distance_to_paper_edge = min(paper_distances) * PIXEL_TO_CM

        # 判斷線條方向
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 2:
            orientation = "水平"
        elif aspect_ratio < 0.5:
            orientation = "垂直"
        else:
            orientation = "斜線"

        line_info.append(
            {
                "center": (center_x, center_y),
                "area": area,
                "width": w,
                "height": h,
                "orientation": orientation,
                "distance_to_paper_edge_cm": min_distance_to_paper_edge,
                "distance_to_ab_edge_cm": distance_to_ab_edge_cm,
                "contour": contour,
            }
        )

    return line_info


# 處理靜態圖像
print(f"處理圖像：{image_path}")


# 不再檢測16cm尺，直接略過
ruler_line, calibrated = None, False

# 檢測紙張邊緣
edges = detect_paper_contour(frame)

# 找到紙張輪廓
paper_contour, original_contour = find_paper_contour(edges, frame.shape)

# 如果找到紙張，檢測紙上的黑線
black_lines = None
binary_lines = None
line_analysis = []

if paper_contour is not None:
    black_lines, binary_lines = detect_black_lines_on_paper(frame, paper_contour)
    line_analysis = analyze_black_lines(black_lines, paper_contour, frame.shape)

# 在原圖上繪製結果
result_frame = frame.copy()


# 如果找到紙張輪廓，繪製檢測區域與標記
if paper_contour is not None:
    # 不再繪製紙張多邊形輪廓與原始輪廓

    # 繪製檢測到的16cm尺（如果有）
    if ruler_line is not None:
        x1, y1, x2, y2 = ruler_line
        cv2.line(result_frame, (x1, y1), (x2, y2), (255, 0, 255), 4)  # 紫色粗線

        # 在尺的中點顯示標籤
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(
            result_frame,
            "16cm Ruler",
            (mid_x - 40, mid_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

        # 顯示校正狀態
        status_text = "Calibrated" if calibrated else "Not Calibrated"
        cv2.putText(
            result_frame,
            f"Pixel Scale: {status_text}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    # 繪製AB邊往上(橘色)
    try:
        overlay = result_frame.copy()
        # 檢測區域可視化：橘色 (BGR: 0,165,255)
        if "detection_region_up" in globals():
            cv2.fillPoly(overlay, [detection_region_up], (0, 165, 255))
        # 混合顏色
        cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
        # 添加檢測區域標籤
        if "detection_region_up" in globals():
            center_region = np.mean(detection_region_up, axis=0).astype(int)
            cv2.putText(
                result_frame,
                "AB->Up Detection Zone",
                (center_region[0] - 70, center_region[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )
    except Exception as e:
        print(f"區域顯示失敗: {e}")

    # 在輪廓上標記所有頂點，最長邊為A、B，其餘依序標號
    # 先找出最長邊的兩個頂點索引
    max_len = 0
    ab_idx = (0, 1)
    n = len(paper_contour)
    for i in range(n):
        pt1 = paper_contour[i][0]
        pt2 = paper_contour[(i + 1) % n][0]
        seg_len = np.linalg.norm(np.array(pt2) - np.array(pt1))
        if seg_len > max_len:
            max_len = seg_len
            ab_idx = (i, (i + 1) % n)
    # 標記所有頂點
    for i, point in enumerate(paper_contour):
        x, y = point[0]
        cv2.circle(result_frame, (x, y), 8, (0, 0, 255), -1)
        # 標籤：A、B為最長邊，其餘為1,2,3...
        if i == ab_idx[0]:
            label = "A"
        elif i == ab_idx[1]:
            label = "B"
        else:
            # 其餘頂點依序標號，跳過A/B的編號
            idx = i
            if ab_idx[0] < i:
                idx -= 1
            if ab_idx[1] < i:
                idx -= 1
            label = str(idx + 1)
        cv2.putText(
            result_frame,
            label,
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result_frame,
            label,
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
        )

    # 繪製檢測到的黑線
    if line_analysis:
        for i, line_info in enumerate(line_analysis):
            contour = line_info["contour"]

            # 繪製黑線輪廓（黃色）
            cv2.drawContours(result_frame, [contour], -1, (0, 255, 255), 2)

            # 標記中點
            center = line_info["center"]
            cv2.circle(result_frame, center, 6, (255, 0, 255), -1)

            # 顯示黑線資訊，包括到AB邊的距離（仍以中心點附近顯示文字）
            center = line_info["center"]
            text = f"Line{i+1}: {line_info['orientation']} (AB: {line_info['distance_to_ab_edge_cm']:.2f}cm)"
            cv2.putText(
                result_frame,
                text,
                (center[0] - 50, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )

    # 顯示統計資訊
    paper_area = cv2.contourArea(
        original_contour if original_contour is not None else paper_contour
    )
    text1 = f"Paper: Vertices={len(paper_contour)}, Area={int(paper_area)}"
    text2 = f"Black Lines Found: {len(line_analysis)}"
    cv2.putText(
        result_frame,
        text1,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        result_frame,
        text2,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # 控制台輸出詳細資訊
    print(f"找到紙張輪廓:")
    print(f"  頂點數: {len(paper_contour)}")
    print(f"  面積: {int(paper_area)} 像素")

    # 輸出頂點座標與標籤
    vertex_labels = ["A", "B", "C", "D"]
    print("  頂點座標:")
    for i, point in enumerate(paper_contour):
        x, y = point[0]
        label = vertex_labels[i] if i < len(vertex_labels) else str(i + 1)
        print(f"    {label}: ({x}, {y})")

    # 輸出AB邊檢測區域資訊
    if len(paper_contour) >= 2:
        # 以長邊作為AB邊
        max_len = 0
        ab_idx = (0, 1)
        n = len(paper_contour)
        for i in range(n):
            pt1 = paper_contour[i][0]
            pt2 = paper_contour[(i + 1) % n][0]
            seg_len = np.linalg.norm(np.array(pt2) - np.array(pt1))
            if seg_len > max_len:
                max_len = seg_len
                ab_idx = (i, (i + 1) % n)
        point_a = paper_contour[ab_idx[0]][0]
        point_b = paper_contour[ab_idx[1]][0]
        ab_length = np.linalg.norm(
            np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
        )
        detection_width = min(100, ab_length * 0.3)
        print(f"  AB邊檢測區域 :")
        print(f"    AB邊長度: {ab_length:.1f} 像素")
        print(f"    檢測區域寬度: {detection_width:.1f} 像素")
        print(f"    檢測區域面積: {ab_length * detection_width:.0f} 像素²")

    if line_analysis:
        print(f"檢測到 {len(line_analysis)} 條黑線:")
        for i, line_info in enumerate(line_analysis):
            print(f"  黑線 {i+1}:")
            print(f"    中心位置: {line_info['center']}")
            print(f"    方向: {line_info['orientation']}")
            print(f"    尺寸: {line_info['width']}x{line_info['height']} 像素")
            print(f"    面積: {line_info['area']} 像素")
            print(
                f"    端點到AB邊最長距離: {line_info['distance_to_ab_edge_cm']:.3f} cm"
            )
        print("  黑線偵測區域：", end="")
        if "detection_region_up" in globals():
            print("AB邊上方")
        elif "detection_region_down" in globals():
            print("AB邊下方")
        else:
            print("未知")
    else:
        # 判斷是上方還是下方區域未檢測到
        if "detection_region_up" in globals():
            print("  在AB邊向上方未檢測到黑線")
            # 若有下方區域，提示也可檢測下方
            if "detection_region_down" in globals():
                print("  可嘗試於AB邊下方區域檢測")
        elif "detection_region_down" in globals():
            print("  在AB邊向下方未檢測到黑線")
        else:
            print("  未檢測到黑線")
    print("---")
    # 顯示 black_lines 結果
    try:
        # 顯示 black_lines 結果
        if black_lines is not None:
            cv2.imshow("Black Lines (edges in zone)", black_lines)
    except Exception as e:
        print(f"[DEBUG] 遮罩/black_lines 顯示失敗: {e}")
else:
    # 如果沒找到紙張，顯示提示
    cv2.putText(
        result_frame,
        "No paper detected",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    print("未檢測到紙張")

cv2.imshow("Result", result_frame)
cv2.imshow("Edges", edges)


print("按任意鍵關閉視窗...")
cv2.waitKey(0)
cv2.destroyAllWindows()
