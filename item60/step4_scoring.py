import cv2
import numpy as np
import os
from step1_paper_contour import detect_paper_contour
from step2_aruco_quarter_a4 import detect_aruco_and_draw_quarter_a4
from step3_calculate_distances import calculate_distances


def count_aruco_markers(image_path):
    """
    計算圖片中ArUco標記的數量
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0

    # 初始化ArUco字典和檢測器
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:
        try:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                image, aruco_dict, parameters=aruco_params
            )
        except AttributeError:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)

    return len(corners) if corners else 0


def calculate_cutting_score(image_path, output_path=None):
    """
    根據剪紙評分標準計算分數
    2分：沿著線剪完，且剪完的紙邊緣與原來的線間距<1.2cm
    1分：同上，但間距大於1.2cm
    0分：小朋友只是動動剪刀而沒剪下去（如果檢測到一張圖裡有兩個ArUco）
    """
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return None, None

    result_image = image.copy()

    # 檢查ArUco標記數量
    aruco_count = count_aruco_markers(image_path)
    print(f"檢測到 {aruco_count} 個ArUco標記")

    # 評分邏輯
    score = 0
    score_reason = ""

    if aruco_count >= 2:
        # 0分：檢測到兩個或更多ArUco標記（沒有剪下去）
        score = 0
        score_reason = "檢測到多個ArUco標記，判斷為未完成剪紙"

        # 在圖片左邊中間標註
        img_height = result_image.shape[0]
        cv2.putText(
            result_image,
            f"Score: 0/2",
            (50, img_height // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            result_image,
            f"Multiple ArUco detected",
            (50, img_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    elif aruco_count == 1:
        # 有一個ArUco標記，進行距離分析
        distance_result = calculate_distances(image_path)

        if distance_result is not None:
            result_image, edge_distances, corner_distances = distance_result

            # 獲取像素比例尺
            _, pixels_per_cm, _ = detect_aruco_and_draw_quarter_a4(image_path)

            if pixels_per_cm:
                # 轉換為厘米
                edge_distances_cm = [d / pixels_per_cm for d in edge_distances]
                corner_distances_cm = [d / pixels_per_cm for d in corner_distances]

                # 計算所有距離的最大值
                max_edge_distance = max(edge_distances_cm)
                max_corner_distance = max(corner_distances_cm)
                max_distance = max(max_edge_distance, max_corner_distance)

                print(f"最大邊線距離: {max_edge_distance:.2f}cm")
                print(f"最大角點距離: {max_corner_distance:.2f}cm")
                print(f"總體最大距離: {max_distance:.2f}cm")

                # 評分標準
                if max_distance < 1.2:
                    score = 2
                    score_reason = (
                        f"剪紙完成且精確，最大間距 {max_distance:.2f}cm < 1.2cm"
                    )
                    score_color = (0, 255, 0)  # 綠色
                else:
                    score = 1
                    score_reason = (
                        f"剪紙完成但不夠精確，最大間距 {max_distance:.2f}cm > 1.2cm"
                    )
                    score_color = (0, 165, 255)  # 橘色

                # 在圖片左邊中間標註分數
                img_height = result_image.shape[0]
                cv2.putText(
                    result_image,
                    f"Score: {score}/2",
                    (50, img_height // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    score_color,
                    4,
                )
                cv2.putText(
                    result_image,
                    f"Max Distance: {max_distance:.2f}cm",
                    (50, img_height // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    score_color,
                    2,
                )
                cv2.putText(
                    result_image,
                    f"Threshold: 1.2cm",
                    (50, img_height // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (128, 128, 128),
                    2,
                )

                # 添加評分說明
                if score == 2:
                    cv2.putText(
                        result_image,
                        "Excellent!",
                        (50, img_height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        score_color,
                        2,
                    )
                elif score == 1:
                    cv2.putText(
                        result_image,
                        "Good, needs improvement",
                        (50, img_height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        score_color,
                        2,
                    )
            else:
                score = 0
                score_reason = "無法獲取比例尺"
        else:
            score = 0
            score_reason = "無法進行距離分析"

    else:
        # 0個ArUco標記
        score = 0
        score_reason = "未檢測到ArUco標記"
        img_height = result_image.shape[0]
        cv2.putText(
            result_image,
            f"Score: 0/2",
            (50, img_height // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            result_image,
            f"No ArUco detected",
            (50, img_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    print(f"最終評分: {score}/2")
    print(f"評分原因: {score_reason}")

    # 保存結果圖片
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"評分結果已保存到: {output_path}")

    return score, score_reason, result_image


def main():
    """主程序：為所有圖片進行評分"""
    input_dir = "img"
    output_dir = "result"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("剪紙評分程序")
    print("評分標準:")
    print("2分：沿著線剪完，且剪完的紙邊緣與原來的線間距<1.2cm")
    print("1分：同上，但間距大於1.2cm")
    print("0分：小朋友只是動動剪刀而沒剪下去（檢測到兩個ArUco）")
    print("=" * 60)

    results = []
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for i, filename in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 評分圖片: {filename}")

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"scored_{filename}")

        score, reason, result_img = calculate_cutting_score(input_path, output_path)

        results.append({"filename": filename, "score": score, "reason": reason})

    # 輸出總結
    print("\n" + "=" * 60)
    print("評分總結")
    print("=" * 60)

    total_score = 0
    for result in results:
        print(f"圖片: {result['filename']}")
        print(f"  評分: {result['score']}/2")
        print(f"  原因: {result['reason']}")
        total_score += result["score"]

    average_score = total_score / len(results) if results else 0
    print(f"\n總分: {total_score}/{len(results)*2}")
    print(f"平均分: {average_score:.2f}/2")

    # 分數分布
    score_counts = {0: 0, 1: 0, 2: 0}
    for result in results:
        if result["score"] in score_counts:
            score_counts[result["score"]] += 1

    print(f"\n分數分布:")
    print(f"0分: {score_counts[0]} 張圖片")
    print(f"1分: {score_counts[1]} 張圖片")
    print(f"2分: {score_counts[2]} 張圖片")


if __name__ == "__main__":
    main()
