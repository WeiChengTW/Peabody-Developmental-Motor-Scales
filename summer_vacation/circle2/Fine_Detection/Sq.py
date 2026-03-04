import cv2
import numpy as np
import math


def calculate_angle(p1, p2, p3):
    """計算三點形成的角度"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免數值誤差
    angle = math.degrees(math.acos(cos_angle))
    return angle


def is_non_right_angle_parallelogram(contour):
    """判斷輪廓是否為非直角平行四邊形（包括菱形）"""
    # 近似輪廓為多邊形
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 必須是四邊形
    if len(approx) != 4:
        return False, "不是四邊形"

    # 計算四個頂點
    points = approx.reshape(4, 2)

    # 計算四條邊的長度
    sides = []
    for i in range(4):
        p1, p2 = points[i], points[(i + 1) % 4]
        side_length = np.linalg.norm(p2 - p1)
        sides.append(side_length)

    # 檢查對邊是否相等（容許10%誤差）
    # 對邊1: sides[0] vs sides[2], 對邊2: sides[1] vs sides[3]
    opposite_sides_equal = True
    side1_diff = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
    side2_diff = abs(sides[1] - sides[3]) / max(sides[1], sides[3])

    if side1_diff > 0.2 or side2_diff > 0.2:
        opposite_sides_equal = False

    # 計算四個內角
    angles = []
    for i in range(4):
        p1 = points[(i - 1) % 4]
        p2 = points[i]
        p3 = points[(i + 1) % 4]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)

    # 檢查是否所有角度都接近90度（容許5度誤差）
    right_angle_count = 0
    for angle in angles:
        if abs(angle - 90) <= 10:
            right_angle_count += 1

    # 建立詳細的檢查結果訊息
    side_lengths_str = [f"{side:.1f}" for side in sides]
    angles_str = [f"{angle:.1f}°" for angle in angles]

    # 邊長檢查結果
    if not opposite_sides_equal:
        edge_result = f"對邊不相等 [{', '.join(side_lengths_str)}]"
    else:
        # 判斷是否為正方形（四邊相等）
        mean_side = np.mean(sides)
        is_square = all(abs(side - mean_side) / mean_side <= 0.1 for side in sides)

        if is_square:
            edge_result = f"四邊相等 [{', '.join(side_lengths_str)}]"
        else:
            edge_result = f"對邊相等 [{', '.join(side_lengths_str)}]"

    # 角度檢查結果
    if right_angle_count == 4:
        angle_result = f"四個直角 [{', '.join(angles_str)}]"
    else:
        angle_result = f"非直角 [{', '.join(angles_str)}]"

    # 綜合判斷
    if not opposite_sides_equal:
        return False, f"{edge_result} | {angle_result} → 不是平行四邊形"
    elif right_angle_count == 4:
        # 檢查是否為正方形
        mean_side = np.mean(sides)
        is_square = all(abs(side - mean_side) / mean_side <= 0.1 for side in sides)

        if is_square:
            # 檢查是否為標準方向的正方形（邊平行於座標軸）
            # 計算邊的方向向量
            edge_vectors = []
            for i in range(4):
                p1, p2 = points[i], points[(i + 1) % 4]
                vec = p2 - p1
                edge_vectors.append(vec)

            # 檢查是否有邊平行於X軸或Y軸（容許5度誤差）
            is_axis_aligned = False
            for vec in edge_vectors:
                # 計算與X軸的夾角
                angle_with_x = abs(math.degrees(math.atan2(vec[1], vec[0])))
                # 標準化角度到0-90度範圍
                angle_with_x = min(
                    angle_with_x, 180 - angle_with_x, abs(angle_with_x - 90)
                )
                if angle_with_x <= 5:  # 平行於座標軸
                    is_axis_aligned = True
                    break

            if is_axis_aligned:
                return False, f"{edge_result} | {angle_result} → 是直角正方形"
            else:
                return False, f"{edge_result} | {angle_result} → 是直角菱形"
        else:
            return False, f"{edge_result} | {angle_result} → 是矩形"
    else:
        # 檢查是否為菱形
        mean_side = np.mean(sides)
        is_square = all(abs(side - mean_side) / mean_side <= 0.1 for side in sides)

        if is_square:
            return True, f"{edge_result} | {angle_result} → 是非直角菱形"
        else:
            return True, f"{edge_result} | {angle_result} → 是非直角平行四邊形"


def detect_non_right_angle_parallelograms(image_path):
    """檢測圖片中的非直角平行四邊形"""
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print("無法讀取圖片")
        return

    # 轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 邊緣檢測
    edges = cv2.Canny(gray, 50, 150)

    # 尋找輪廓 - 只檢測外輪廓
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 檢測結果
    non_right_parallelograms = []
    all_shapes = []

    # 對輪廓按面積排序，優先處理較大的輪廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, contour in enumerate(contours):
        # 過濾太小的輪廓
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        is_non_right, message = is_non_right_angle_parallelogram(contour)
        all_shapes.append((contour, message))

        if is_non_right:
            non_right_parallelograms.append((contour, message))
            # 在圖片上標記非直角平行四邊形
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

            # 標記輪廓編號和面積
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    img,
                    f"Non-Right Parallelogram {len(non_right_parallelograms)} (Area:{int(area)})",
                    (cx - 90, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

    # 顯示結果
    cv2.imshow("Non-Right Angle Parallelograms Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 輸出檢測結果
    print(f"\n=== 圖形檢測結果 (僅外輪廓) ===")
    print(f"總共檢測到 {len(all_shapes)} 個四邊形:")

    for i, (contour, message) in enumerate(all_shapes, 1):
        area = cv2.contourArea(contour)
        print(f"圖形 {i} (面積:{int(area)}): {message}")

    if non_right_parallelograms:
        print(f"\n其中有 {len(non_right_parallelograms)} 個非直角平行四邊形:")
        for i, (contour, message) in enumerate(non_right_parallelograms, 1):
            print(f"非直角平行四邊形 {i}: {message}")
    else:
        print("\n未檢測到非直角平行四邊形")


# 使用範例
if __name__ == "__main__":
    # 替換為您的圖片路徑
    image_path = "result/Square/2.png"
    detect_non_right_angle_parallelograms(image_path)

    # 從圖片中檢測輪廓進行測試
    test_img = cv2.imread(image_path)
    if test_img is not None:
        # 轉換為灰度圖
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        # 邊緣檢測
        test_edges = cv2.Canny(test_gray, 50, 150)

        # 尋找輪廓 - 只檢測外輪廓
        test_contours, test_hierarchy = cv2.findContours(
            test_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 測試第一個面積足夠大的輪廓
        # 按面積排序，選擇最大的輪廓
        test_contours = sorted(test_contours, key=cv2.contourArea, reverse=True)

        for contour in test_contours:
            area = cv2.contourArea(contour)
            if area >= 1000:
                is_non_right, message = is_non_right_angle_parallelogram(contour)
                print(f"測試輪廓結果 (面積:{int(area)}): {message}")
                break
        else:
            print("未找到面積足夠大的輪廓進行測試")
    else:
        print("無法讀取測試圖片")
