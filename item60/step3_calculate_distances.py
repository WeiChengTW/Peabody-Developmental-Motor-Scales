import cv2
import numpy as np
import os
from step1_paper_contour import detect_paper_contour
from step2_aruco_quarter_a4 import detect_aruco_and_draw_quarter_a4


def point_to_line_distance(point, line_start, line_end):
    """計算點到線段的距離"""
    # 線段的向量
    line_vec = line_end - line_start
    # 點到線段起點的向量
    point_vec = point - line_start

    # 線段長度的平方
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.linalg.norm(point_vec)

    # 計算投影參數
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))

    # 最近點
    closest_point = line_start + t * line_vec

    # 距離
    distance = np.linalg.norm(point - closest_point)
    return distance, closest_point


def point_to_contour_distance(point, contour):
    """計算點到輪廓的最短距離"""
    min_distance = float("inf")
    closest_point = None

    # 遍歷輪廓的所有線段
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]

        distance, closest = point_to_line_distance(point, p1, p2)
        if distance < min_distance:
            min_distance = distance
            closest_point = closest

    return min_distance, closest_point


def rectangle_to_contour_distance(rect_corners, contour):
    """計算矩形邊線到輪廓的距離 - 找出最長距離"""
    distances = []
    lines = []

    # 矩形的四條邊
    for i in range(4):
        p1 = rect_corners[i]
        p2 = rect_corners[(i + 1) % 4]

        # 在這條邊上取多個點
        num_points = 20  # 減少採樣點數提高性能
        edge_distances = []
        edge_points = []
        closest_points = []

        for j in range(num_points + 1):
            t = j / num_points
            point = p1 + t * (p2 - p1)
            distance, closest = point_to_contour_distance(point, contour)
            edge_distances.append(distance)
            edge_points.append(point)
            closest_points.append(closest)

        # 找到這條邊上的最長距離（而不是最短距離）
        max_idx = np.argmax(edge_distances)
        max_distance = edge_distances[max_idx]
        best_point = edge_points[max_idx]
        best_closest = closest_points[max_idx]

        distances.append(max_distance)
        lines.append(
            {
                "edge_start": p1,
                "edge_end": p2,
                "closest_point_on_edge": best_point,
                "closest_point_on_contour": best_closest,
                "distance": max_distance,
            }
        )

    return distances, lines


def calculate_distances(image_path, output_path=None):
    """
    計算綠框線與藍色輪廓的距離(橘線) 以及綠框線四個角點垂直藍色輪廓的距離(紫線)
    """
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return None

    result_image = image.copy()

    # 步驟1：檢測紙張輪廓
    paper_contour, _ = detect_paper_contour(image_path)
    if paper_contour is None:
        print("無法檢測到紙張輪廓")
        return None

    # 簡化輪廓以提高性能
    epsilon = 0.005 * cv2.arcLength(paper_contour, True)
    simplified_contour = cv2.approxPolyDP(paper_contour, epsilon, True)
    print(f"輪廓簡化: {len(paper_contour)} -> {len(simplified_contour)} 點")

    # 步驟2：檢測ArUco並繪製1/4 A4矩形
    quarter_a4_corners, pixels_per_cm, _ = detect_aruco_and_draw_quarter_a4(image_path)
    if quarter_a4_corners is None:
        print("無法檢測到ArUco標記")
        return None

    # 織製紙張輪廓（藍色） - 使用原始輪廓維持視覺效果
    cv2.drawContours(result_image, [paper_contour], -1, (255, 0, 0), 3)

    # 繪製1/4 A4矩形（綠色）
    cv2.polylines(result_image, [quarter_a4_corners], True, (0, 255, 0), 3)

    # 計算綠框線到藍色輪廓的距離（橘線） - 使用簡化輪廓
    edge_distances, edge_lines = rectangle_to_contour_distance(
        quarter_a4_corners, simplified_contour
    )

    # 繪製橘線（邊線距離）
    for line_info in edge_lines:
        pt1 = tuple(line_info["closest_point_on_edge"].astype(int))
        pt2 = tuple(line_info["closest_point_on_contour"].astype(int))
        cv2.line(result_image, pt1, pt2, (0, 165, 255), 2)  # 橘色線

        # 在中點標註距離
        mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        distance_cm = (
            line_info["distance"] / pixels_per_cm
            if pixels_per_cm
            else line_info["distance"]
        )
        cv2.putText(
            result_image,
            f"{distance_cm:.1f}cm",
            mid_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )

    # 計算綠框線四個角點垂直到藍色輪廓的距離（紫線） - 使用簡化輪廓
    corner_distances = []
    for i, corner in enumerate(quarter_a4_corners):
        distance, closest_point = point_to_contour_distance(corner, simplified_contour)
        corner_distances.append(distance)

        # 繪製紫線（角點距離）
        pt1 = tuple(corner)
        pt2 = tuple(closest_point.astype(int))
        cv2.line(result_image, pt1, pt2, (255, 0, 255), 2)  # 紫色線

        # 標註距離
        mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        distance_cm = distance / pixels_per_cm if pixels_per_cm else distance
        cv2.putText(
            result_image,
            f"{distance_cm:.1f}cm",
            mid_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
        )

        # 標記角點
        cv2.circle(result_image, pt1, 8, (255, 0, 255), -1)

    # 在左下角顯示距離數據
    info_y = result_image.shape[0] - 200
    font_scale = 0.8
    font_thickness = 2

    # 顯示橘色線（邊線最長距離）數據
    edge_distances_cm = (
        [d / pixels_per_cm for d in edge_distances] if pixels_per_cm else edge_distances
    )
    cv2.putText(
        result_image,
        f"Orange Lines (Max Edge Distance):",
        (10, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 165, 255),
        font_thickness,
    )

    for i, dist in enumerate(edge_distances_cm):
        cv2.putText(
            result_image,
            f"  Edge {i+1}: {dist:.2f}cm",
            (10, info_y + 30 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            1,
        )

    # 顯示紫色線（角點距離）數據
    corner_distances_cm = (
        [d / pixels_per_cm for d in corner_distances]
        if pixels_per_cm
        else corner_distances
    )
    purple_start_y = info_y + 30 + len(edge_distances_cm) * 25 + 20
    cv2.putText(
        result_image,
        f"Purple Lines (Corner Distance):",
        (10, purple_start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 255),
        font_thickness,
    )

    for i, dist in enumerate(corner_distances_cm):
        cv2.putText(
            result_image,
            f"  Corner {i+1}: {dist:.2f}cm",
            (10, purple_start_y + 30 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            1,
        )

    # 添加圖例（移到右下角）
    legend_y = result_image.shape[0] - 130
    cv2.putText(
        result_image,
        "Blue: Paper Contour",
        (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        result_image,
        "Green: 1/4 A4 Rectangle",
        (10, legend_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        result_image,
        "Orange: Max Edge Distance",
        (10, legend_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
    )
    cv2.putText(
        result_image,
        "Purple: Corner Distance",
        (10, legend_y + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    print(f"邊線最大距離 (cm): {[d/pixels_per_cm for d in edge_distances]}")
    print(f"角點距離 (cm): {[d/pixels_per_cm for d in corner_distances]}")

    # 保存結果圖片
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"結果已保存到: {output_path}")

    return result_image, edge_distances, corner_distances


def main():
    """測試函數"""
    input_dir = "img"
    output_dir = "result"

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 處理所有圖片
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"step3_{filename}")

            print(f"\n處理圖片: {filename}")
            result = calculate_distances(input_path, output_path)

            if result is not None:
                print(f"✓ 成功計算距離")
            else:
                print("✗ 距離計算失敗")


if __name__ == "__main__":
    main()
