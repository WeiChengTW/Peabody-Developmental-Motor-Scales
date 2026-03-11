import cv2
import numpy as np
import math
import os


def calculate_angle(p1, p2, p3):
    """計算三點形成的角度"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免數值誤差
    angle = math.degrees(math.acos(cos_angle))
    return angle


def merge_close_points(points, min_distance=20, angle_threshold=30):
    """合併距離太近的點，保留轉折較大的點"""
    if len(points) <= 1:
        return points

    merged_points = []
    used = [False] * len(points)
    n = len(points)

    for i in range(n):
        if used[i]:
            continue

        # 找到所有與當前點距離小於min_distance的點
        close_indices = [i]
        used[i] = True

        for j in range(i + 1, n):
            if used[j]:
                continue
            distance = np.linalg.norm(points[i] - points[j])
            if distance < min_distance:
                close_indices.append(j)
                used[j] = True

        if len(close_indices) == 1:
            merged_points.append(points[i])
        else:
            # 保留轉折最大的點
            max_angle = -1
            max_idx = close_indices[0]
            for idx in close_indices:
                prev_pt = points[(idx - 1) % n]
                curr_pt = points[idx]
                next_pt = points[(idx + 1) % n]
                angle = calculate_angle(prev_pt, curr_pt, next_pt)
                # 轉折大 = 與180度差距大
                turn = abs(angle - 180)
                if turn > max_angle:
                    max_angle = turn
                    max_idx = idx
            # 只有當最大轉折超過閾值才保留，否則取平均
            if max_angle >= angle_threshold:
                merged_points.append(points[max_idx])
            else:
                merged_point = np.mean(points[close_indices], axis=0)
                merged_points.append(merged_point)

    return np.array(merged_points)


def filter_significant_corners(points, min_angle_change=20):
    """過濾掉角度變化太小的點（忽略邊線上的小彎曲）"""
    if len(points) < 3:
        return points

    filtered_points = []
    n = len(points)

    for i in range(n):
        # 取前一個點、當前點、下一個點
        prev_pt = points[(i - 1) % n]
        curr_pt = points[i]
        next_pt = points[(i + 1) % n]

        # 計算角度變化
        angle = calculate_angle(prev_pt, curr_pt, next_pt)

        # 過濾掉接近0度或接近180度的點（幾乎是直線）
        # 保留角度在 min_angle_change 到 (180 - min_angle_change) 範圍內的點
        if min_angle_change < angle < (180 - min_angle_change):
            filtered_points.append(curr_pt)

    return np.array(filtered_points) if len(filtered_points) > 0 else points


def detect_non_right_angle_parallelograms(image_path):
    """檢測圖片中的非直角平行四邊形"""
    # 讀取圖片
    is_right = None
    img = cv2.imread(image_path)
    if img is None:
        print("無法讀取圖片")
        return
    # 擴大邊框 100 像素
    border_size = 100
    img = cv2.copyMakeBorder(
        img,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
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

        epsilon_n = 0.02  # 降低基礎epsilon值，讓初始檢測更敏感
        epsilon_r = 0.01
        # 統一的頂點檢測邏輯
        epsilon = epsilon_n * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果檢測到的頂點太多，嘗試用更大的epsilon值
        if len(approx) > 8:
            epsilon = (epsilon_n + epsilon_r) * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果還是太多頂點，再增加epsilon值
        if len(approx) > 6:
            epsilon = (epsilon_n + epsilon_r * 2) * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

        # 合併距離太近的頂點
        if len(approx) > 0:
            points = approx.reshape(len(approx), 2)
            merged_points = merge_close_points(points, min_distance=25)

            # 過濾掉角度變化太小的點（忽略邊線上的小彎曲）
            if len(merged_points) >= 3:
                filtered_points = filter_significant_corners(
                    merged_points, min_angle_change=20
                )
                if len(filtered_points) >= 3:
                    merged_points = filtered_points

            # 如果合併後點數有變化，重新構建approx
            if len(merged_points) != len(points):
                approx = merged_points.reshape(-1, 1, 2).astype(np.int32)

        # 標記所有檢測到的頂點
        if len(approx) >= 3:
            points = approx.reshape(len(approx), 2)
            # 用小紅點標示頂點
            for j, point in enumerate(points):
                cv2.circle(
                    img, tuple(point.astype(int)), 5, (0, 0, 255), -1
                )  # 紅色實心圓點
                # 標記頂點編號
                cv2.putText(
                    img,
                    f"{j+1}",
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),  # 黃色文字
                    1,
                )
                # 顯示所有頂點座標
            n = len(points)
            is_right = True
            for j, point in enumerate(points):

                prev_pt = points[(j - 1) % n]
                curr_pt = points[j]
                next_pt = points[(j + 1) % n]
                angle = calculate_angle(prev_pt, curr_pt, next_pt)
                if not (80 < angle < 100):
                    is_right = False

                # print(
                #     f"頂點 {j+1}: ({int(point[0])}, {int(point[1])}), 角度: {angle:.1f}°"
                # )

        # 檢查是否為非直角平行四邊形（只對四邊形進行檢查）
        is_non_right = False
        message = ""

        if len(approx) != 4:
            message = f"不是四邊形 (檢測到{len(approx)}個頂點)"
        else:
            # 計算四個頂點
            points = approx.reshape(4, 2)

            # 計算四條邊的長度
            sides = []
            for k in range(4):
                p1, p2 = points[k], points[(k + 1) % 4]
                side_length = np.linalg.norm(p2 - p1)
                sides.append(side_length)

            # 檢查對邊是否相等（容許20%誤差）
            opposite_sides_equal = True
            side1_diff = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
            side2_diff = abs(sides[1] - sides[3]) / max(sides[1], sides[3])

            if side1_diff > 0.2 or side2_diff > 0.2:
                opposite_sides_equal = False

            # 計算四個內角
            angles = []
            for k in range(4):
                p1 = points[(k - 1) % 4]
                p2 = points[k]
                p3 = points[(k + 1) % 4]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)

            # 檢查是否所有角度都接近90度（容許10度誤差）
            right_angle_count = 0
            for angle in angles:
                if abs(angle - 90) <= 10:
                    right_angle_count += 1

            # 檢查對角是否相等（平行四邊形的特性）
            opposite_angles_equal = True
            angle1_diff = abs(angles[0] - angles[2])
            angle2_diff = abs(angles[1] - angles[3])

            # 容許5度誤差
            if angle1_diff > 5 or angle2_diff > 5:
                opposite_angles_equal = False

            # 建立詳細的檢查結果訊息
            side_lengths_str = [f"{side:.1f}" for side in sides]
            angles_str = [f"{angle:.1f}°" for angle in angles]

            # 邊長檢查結果
            if not opposite_sides_equal:
                edge_result = f"對邊不相等 [{', '.join(side_lengths_str)}]"
            else:
                # 判斷是否為正方形（四邊相等）
                mean_side = np.mean(sides)
                is_square = all(
                    abs(side - mean_side) / mean_side <= 0.1 for side in sides
                )

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
                message = f"{edge_result} | {angle_result} → 四邊形（對邊不相等）"
            elif not opposite_angles_equal:
                message = f"{edge_result} | {angle_result} → 四邊形（對角不相等：{angle1_diff:.1f}°, {angle2_diff:.1f}°）"
            elif right_angle_count == 4:
                # 檢查是否為正方形
                mean_side = np.mean(sides)
                is_square = all(
                    abs(side - mean_side) / mean_side <= 0.1 for side in sides
                )

                if is_square:
                    # 檢查是否為標準方向的正方形（邊平行於座標軸）
                    edge_vectors = []
                    for k in range(4):
                        p1, p2 = points[k], points[(k + 1) % 4]
                        vec = p2 - p1
                        edge_vectors.append(vec)

                    # 檢查是否有邊平行於X軸或Y軸（容許5度誤差）
                    is_axis_aligned = False
                    for vec in edge_vectors:
                        angle_with_x = abs(math.degrees(math.atan2(vec[1], vec[0])))
                        angle_with_x = min(
                            angle_with_x, 180 - angle_with_x, abs(angle_with_x - 90)
                        )
                        if angle_with_x <= 5:
                            is_axis_aligned = True
                            break

                    if is_axis_aligned:
                        message = f"{edge_result} | {angle_result} → 是直角正方形"
                    else:
                        message = f"{edge_result} | {angle_result} → 是直角菱形"
                else:
                    message = f"{edge_result} | {angle_result} → 是矩形"
            else:
                # 檢查是否為菱形
                mean_side = np.mean(sides)
                is_square = all(
                    abs(side - mean_side) / mean_side <= 0.1 for side in sides
                )

                if is_square:
                    is_non_right = True
                    message = f"{edge_result} | {angle_result} → 是非直角菱形"
                else:
                    is_non_right = True
                    message = f"{edge_result} | {angle_result} → 是非直角平行四邊形"

        all_shapes.append((contour, message))

        if is_non_right:
            non_right_parallelograms.append((contour, message))
            # 在圖片上標記非直角平行四邊形
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

            # 標記輪廓編號和面積
            M = cv2.moments(contour)
            # if M["m00"] != 0:
            #     cx = int(M["m10"] / M["m00"])
            #     cy = int(M["m01"] / M["m00"])
            #     cv2.putText(
            #         img,
            #         f"Non-Right Parallelogram {len(non_right_parallelograms)} (Area:{int(area)})",
            #         (cx - 90, cy),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.4,
            #         (0, 255, 255),  # 黃色文字
            #         1,
            #     )
        else:
            # 標記其他形狀
            if len(approx) == 4:
                cv2.drawContours(img, [contour], -1, (255, 0, 0), 1)
                shape_name = "四邊形"
            elif len(approx) == 3:
                cv2.drawContours(img, [contour], -1, (255, 0, 255), 1)
                shape_name = "三角形"
            elif len(approx) == 5:
                cv2.drawContours(img, [contour], -1, (0, 165, 255), 1)
                shape_name = "五邊形"
            elif len(approx) == 6:
                cv2.drawContours(img, [contour], -1, (255, 255, 0), 1)
                shape_name = "六邊形"
            else:
                cv2.drawContours(img, [contour], -1, (128, 128, 128), 1)
                shape_name = f"{len(approx)}邊形"

            # 標記輪廓編號和面積
            M = cv2.moments(contour)
            # if M["m00"] != 0:
            #     cx = int(M["m10"] / M["m00"])
            #     cy = int(M["m01"] / M["m00"])
            #     cv2.putText(
            #         img,
            #         f"{shape_name} {i+1} (Area:{int(area)})",
            #         (cx - 50, cy + 15),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.3,
            #         (0, 255, 255),  # 黃色文字
            #         1,
            #     )

    # 輸出檢測結果
    # print(f"\n=== 圖形檢測結果 (僅外輪廓) ===")
    # print(f"總共檢測到 {len(all_shapes)} 個形狀:")

    # for i, (contour, message) in enumerate(all_shapes, 1):
    #     area = cv2.contourArea(contour)
    #     print(f"圖形 {i} (面積:{int(area)}): {message}")

    # if non_right_parallelograms:
    #     print(f"\n其中有 {len(non_right_parallelograms)} 個非直角平行四邊形:")
    #     for i, (contour, message) in enumerate(non_right_parallelograms, 1):
    #         print(f"非直角平行四邊形 {i}: {message}")
    # else:
    #     print("\n未檢測到非直角平行四邊形")
    # 只保留檔名中的 "97_1"
    base_name = os.path.basename(image_path)
    prefix = (
        base_name.split("_")[0] + "_" + base_name.split("_")[1]
        if "_" in base_name
        else base_name
    )
    print(f"{prefix}", end=", ")
    # 檢查所有輪廓的對邊是否等長
    for i, (contour, message) in enumerate(all_shapes, 1):
        if "對邊相等" in message or "四邊相等" in message:
            print(f"對邊等長", end=", ")
        else:
            print(f"對邊不等長", end=", ")
    if is_right:
        print("直角")
    elif is_right is None:
        print("無法判斷直角")
    else:
        print("非直角")
        # 顯示結果
    resized_img = cv2.resize(img, (300, 300))
    cv2.imshow("Non-Right Angle Parallelograms Detection", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用範例
if __name__ == "__main__":

    folder = r"result\Square"
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(folder, filename)
            detect_non_right_angle_parallelograms(image_path)
