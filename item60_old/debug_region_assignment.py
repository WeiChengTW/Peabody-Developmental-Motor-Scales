"""
調試版本：檢查矩形分區和距離計算問題
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from detect_aruco_and_draw_quarter_a4 import ArUcoQuarterA4Detector
from original_paper_detector import OriginalPaperDetector


def debug_region_assignment(image_path):
    """
    調試矩形分區和距離計算問題
    """
    # 初始化檢測器
    aruco_detector = ArUcoQuarterA4Detector()
    paper_detector = OriginalPaperDetector()

    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'='*60}")
    print(f"調試圖片: {base_name}")
    print(f"{'='*60}")

    # 步驟1：ArUco偵測和矩形資訊
    print("\n🔍 第1步：ArUco偵測和四分之一A4矩形計算")

    # 偵測ArUco標記
    corners, ids, rejected = aruco_detector.detect_aruco_markers(image)

    if ids is None:
        print("❌ 未偵測到ArUco標記")
        return

    # 獲取長方形資訊
    temp_image, detection_results = aruco_detector.draw_quarter_a4_rectangles(
        image.copy(), corners, ids
    )

    rectangles_info = []
    scale_info = None
    for i, result in enumerate(detection_results):
        corner_data = corners[i]
        rectangle_corners, current_scale_info = (
            aruco_detector.calculate_quarter_a4_rectangle(
                corner_data, result["marker_id"]
            )
        )

        rectangles_info.append(
            {
                "marker_id": result["marker_id"],
                "corners": rectangle_corners,
                "aruco_corners": corner_data,
            }
        )

        if scale_info is None:
            scale_info = current_scale_info

    print(f"偵測到 {len(rectangles_info)} 個ArUco標記:")
    for i, rect_info in enumerate(rectangles_info):
        marker_id = rect_info["marker_id"]
        corners = rect_info["corners"]
        center_x = np.mean([corner[0] for corner in corners])
        center_y = np.mean([corner[1] for corner in corners])
        print(f"  標記 ID{marker_id}: 中心位置 ({center_x:.1f}, {center_y:.1f})")

    # 步驟2：檢查分區邏輯
    print(f"\n🔍 第2步：檢查左右分區邏輯")
    left_mask, right_mask, left_rectangles, right_rectangles = (
        paper_detector.create_region_masks(image.shape, rectangles_info)
    )

    print(f"左側區域分配到的標記:")
    for rect_info in left_rectangles:
        marker_id = rect_info["marker_id"]
        corners = rect_info["corners"]
        center_x = np.mean([corner[0] for corner in corners])
        print(f"  ID{marker_id}: 中心x座標 {center_x:.1f}")

    print(f"右側區域分配到的標記:")
    for rect_info in right_rectangles:
        marker_id = rect_info["marker_id"]
        corners = rect_info["corners"]
        center_x = np.mean([corner[0] for corner in corners])
        print(f"  ID{marker_id}: 中心x座標 {center_x:.1f}")

    # 步驟3：檢查輪廓偵測
    print(f"\n🔍 第3步：檢查左右輪廓偵測")

    if left_rectangles:
        left_contours = paper_detector.detect_paper_contours(
            image, left_mask, filter_center_line=True
        )
        print(f"左側偵測到 {len(left_contours)} 個紙張輪廓")

        # 輸出每個輪廓的範圍
        for i, contour in enumerate(left_contours):
            x, y, w, h = cv2.boundingRect(contour)
            print(f"  左側輪廓 {i}: 範圍 x={x}-{x+w}, y={y}-{y+h}")

    if right_rectangles:
        right_contours = paper_detector.detect_paper_contours(
            image, right_mask, filter_center_line=True
        )
        print(f"右側偵測到 {len(right_contours)} 個紙張輪廓")

        # 輸出每個輪廓的範圍
        for i, contour in enumerate(right_contours):
            x, y, w, h = cv2.boundingRect(contour)
            print(f"  右側輪廓 {i}: 範圍 x={x}-{x+w}, y={y}-{y+h}")

    # 步驟4：檢查距離計算
    print(f"\n🔍 第4步：檢查距離計算")

    if left_rectangles and "left_contours" in locals() and left_contours:
        print(f"\n左側區域距離計算:")
        for rect_info in left_rectangles:
            marker_id = rect_info["marker_id"]
            print(f"\n  處理左側標記 ID{marker_id}:")

            distance_result = paper_detector.calculate_rectangle_distance_no_draw(
                rect_info, left_contours, f"左側ID{marker_id}", scale_info
            )

            if distance_result:
                print(f"    最佳距離結果: {distance_result}")
                if "distance" in distance_result:
                    print(
                        f"    最佳距離: {distance_result['distance']:.2f} (類型: {distance_result.get('type', '未知')})"
                    )
                    print(f"    參考點: {distance_result.get('point', '未知')}")
                    print(f"    對應輪廓: {distance_result.get('contour_idx', '未知')}")
                else:
                    print(f"    錯誤：距離結果格式異常")

    if right_rectangles and "right_contours" in locals() and right_contours:
        print(f"\n右側區域距離計算:")
        for rect_info in right_rectangles:
            marker_id = rect_info["marker_id"]
            print(f"\n  處理右側標記 ID{marker_id}:")

            distance_result = paper_detector.calculate_rectangle_distance_no_draw(
                rect_info, right_contours, f"右側ID{marker_id}", scale_info
            )

            if distance_result:
                print(f"    最佳距離結果: {distance_result}")
                if "distance" in distance_result:
                    print(
                        f"    最佳距離: {distance_result['distance']:.2f} (類型: {distance_result.get('type', '未知')})"
                    )
                    print(f"    參考點: {distance_result.get('point', '未知')}")
                    print(f"    對應輪廓: {distance_result.get('contour_idx', '未知')}")
                else:
                    print(f"    錯誤：距離結果格式異常")

    print(f"\n{'='*60}")
    print("調試完成")


if __name__ == "__main__":
    # 測試第一張圖片
    image_path = "img/1.jpg"
    if os.path.exists(image_path):
        debug_region_assignment(image_path)
    else:
        print(f"找不到圖片: {image_path}")
        # 列出可用的圖片
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(f"image/{ext}"))

        if image_files:
            print("可用的圖片:")
            for img in image_files[:5]:  # 只顯示前5個
                print(f"  {img}")
            print(f"使用第一個圖片進行測試...")
            debug_region_assignment(image_files[0])
