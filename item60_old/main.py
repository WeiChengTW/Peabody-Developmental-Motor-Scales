"""
整合 ArUco 偵測與紙張輪廓分析的主程式
結合 detect_aruco_and_draw_quarter_a4.py 和紙張輪廓偵測功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from detect_aruco_and_draw_quarter_a4 import ArUcoQuarterA4Detector
from original_paper_detector import OriginalPaperDetector


class IntegratedAnalyzer:
    """
    整合分析器：ArUco 偵測 + 紙張輪廓分析
    """

    def __init__(self):
        self.aruco_detector = ArUcoQuarterA4Detector()
        self.paper_detector = OriginalPaperDetector()

    def process_single_image(self, image_path, save_result=True, show_result=True):
        """
        處理單張圖片：分步驟處理並保存每一步的結果

        Args:
            image_path: 圖片路徑
            save_result: 是否保存結果
            show_result: 是否顯示結果

        Returns:
            final_image: 最終結果圖像
            analysis_results: 分析結果
        """
        # 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")

        print(f"\n{'='*60}")
        print(f"處理圖片: {os.path.basename(image_path)}")
        print(f"圖片尺寸: {image.shape[1]} x {image.shape[0]} 像素")
        print(f"{'='*60}")

        # 準備保存路徑
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 第0步：原始圖片
        print("\n===== 第0步：原始圖片 =====")
        step0_image = image.copy()
        if save_result:
            self.save_step_image(step0_image, base_name, 0, "原始圖片")

        # ArUco 偵測（不繪製長方形，避免影響紙張輪廓偵測）
        print("\n===== ArUco 偵測階段 =====")
        corners, ids, rejected = self.aruco_detector.detect_aruco_markers(image)

        if ids is None:
            print("未偵測到 ArUco 標記，無法繼續分析")
            return image, {}

        # 獲取 ArUco 結果但不繪製到圖像上
        temp_image, detection_results = self.aruco_detector.draw_quarter_a4_rectangles(
            image.copy(), corners, ids
        )  # 準備長方形資訊供紙張輪廓分析使用
        rectangles_info = []
        for result in detection_results:
            # 重新計算長方形角點（因為 draw_quarter_a4_rectangles 沒有返回角點）
            corner_data = corners[len(rectangles_info)]  # 對應的 ArUco 角點
            rectangle_corners, scale_info = (
                self.aruco_detector.calculate_quarter_a4_rectangle(
                    corner_data, result["marker_id"]
                )
            )

            rectangles_info.append(
                {
                    "corners": rectangle_corners,
                    "marker_id": result["marker_id"],
                    "scale_info": result,
                }
            )

        # 第二步：在原圖上進行紙張輪廓偵測和距離計算
        print("\n2. 紙張輪廓偵測和距離計算...")
        result_image_with_contours, distance_results = (
            self.paper_detector.process_image_with_rectangles(image, rectangles_info)
        )

        # 第三步：在已有紙張輪廓的圖像上繪製ArUco長方形（綠色）
        print("\n3. 繪製ArUco長方形...")
        final_image = self.draw_aruco_rectangles_on_result(
            result_image_with_contours, rectangles_info
        )

        # 第四步：找出最長距離
        print("\n4. 分析最長距離...")
        longest_distance = self.paper_detector.find_longest_distance(distance_results)

        # 在圖上特別標註最長距離
        if longest_distance:
            # 根據距離類型確定標記點
            if longest_distance["type"] == "edge_to_box":
                point = longest_distance["details"]["edge_point"]
            else:  # corner_to_paper
                point = longest_distance["details"]["corner_point"]

            if point:
                # 繪製更大的標記
                cv2.circle(final_image, point, 12, (255, 0, 255), 3)  # 紫色圓圈
                cv2.circle(final_image, point, 15, (255, 0, 255), 2)  # 外圈

                # 添加最長距離標籤
                distance_type_text = (
                    "邊緣" if longest_distance["type"] == "edge_to_box" else "角點"
                )
                text = f"MAX-{distance_type_text}: {longest_distance['distance']:.1f}px"
                text_pos = (point[0] - 30, point[1] - 25)
                cv2.putText(
                    final_image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

        # 整合分析結果
        analysis_results = {
            "aruco_results": detection_results,
            "distance_results": distance_results,
            "longest_distance": longest_distance,
            "rectangles_info": rectangles_info,
        }

        # 顯示結果
        if show_result:
            self.show_step_by_step_results(image_path, base_name)

        print(f"\n✅ {base_name} 所有步驟處理完成！")
        return final_image, analysis_results

    def draw_aruco_rectangles_on_result(self, image, rectangles_info):
        """
        在已有紙張輪廓的圖像上繪製ArUco長方形（綠色）
        """
        result_image = image.copy()

        for rect_info in rectangles_info:
            corners = rect_info["corners"]
            marker_id = rect_info["marker_id"]

            # 繪製綠色長方形
            cv2.polylines(result_image, [corners], True, (0, 255, 0), 2)

            # 標註標記ID
            center = np.mean(corners, axis=0).astype(int)
            cv2.putText(
                result_image,
                f"ID{marker_id}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return result_image

    def save_step_image(self, image, base_name, step_num, step_name):
        """
        保存每一步的圖像結果
        """
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        filename = f"{base_name}_step{step_num}_{step_name}.jpg"
        filepath = os.path.join(result_dir, filename)
        cv2.imwrite(filepath, image)
        print(f"💾 步驟{step_num}圖像已保存: {filename}")

    def step1_detect_paper_contours(self, image, rectangles_info):
        """
        第1步：偵測並繪製紙張輪廓（藍色線條）
        """
        result_image = image.copy()

        # 創建左右分區遮罩
        left_mask, right_mask, left_rectangles, right_rectangles = (
            self.paper_detector.create_region_masks(image.shape, rectangles_info)
        )

        # 繪製分區線
        if left_rectangles and right_rectangles:
            # 找到分界點
            left_max_x = 0
            right_min_x = image.shape[1]

            for rect in left_rectangles:
                corners = rect["corners"]
                max_x = np.max([corner[0] for corner in corners])
                left_max_x = max(left_max_x, max_x)

            for rect in right_rectangles:
                corners = rect["corners"]
                min_x = np.min([corner[0] for corner in corners])
                right_min_x = min(right_min_x, min_x)

            # 繪製分區線
            division_x = (left_max_x + right_min_x) // 2
            cv2.line(
                result_image,
                (division_x, 0),
                (division_x, image.shape[0]),
                (128, 128, 128),
                1,
            )

        distance_results = []

        # 處理左側區域
        if left_rectangles:
            print(f"左側區域處理 ({len(left_rectangles)}個標記)...")
            left_contours = self.paper_detector.detect_paper_contours(image, left_mask)
            if left_contours:
                print(f"左側偵測到 {len(left_contours)} 個紙張輪廓")
                # 繪製左側紙張輪廓 (藍色)
                cv2.drawContours(result_image, left_contours, -1, (255, 0, 0), 2)

                # 計算距離但不繪製距離線
                for rect_info in left_rectangles:
                    distance_result = (
                        self.paper_detector.calculate_rectangle_distance_no_draw(
                            rect_info, left_contours, "左側"
                        )
                    )
                    if distance_result:
                        distance_results.append(distance_result)

        # 處理右側區域
        if right_rectangles:
            print(f"右側區域處理 ({len(right_rectangles)}個標記)...")
            right_contours = self.paper_detector.detect_paper_contours(
                image, right_mask
            )
            if right_contours:
                print(f"右側偵測到 {len(right_contours)} 個紙張輪廓")
                # 繪製右側紙張輪廓 (藍色)
                cv2.drawContours(result_image, right_contours, -1, (255, 0, 0), 2)

                # 計算距離但不繪製距離線
                for rect_info in right_rectangles:
                    distance_result = (
                        self.paper_detector.calculate_rectangle_distance_no_draw(
                            rect_info, right_contours, "右側"
                        )
                    )
                    if distance_result:
                        distance_results.append(distance_result)

        return result_image, distance_results

    def step2_draw_aruco_rectangles(self, image, rectangles_info):
        """
        第2步：繪製ArUco長方形（綠色線條）
        """
        result_image = image.copy()

        for rect_info in rectangles_info:
            corners = rect_info["corners"]
            marker_id = rect_info["marker_id"]

            # 繪製綠色長方形
            cv2.polylines(result_image, [corners], True, (0, 255, 0), 2)

            # 標註標記ID
            center = np.mean(corners, axis=0).astype(int)
            cv2.putText(
                result_image,
                f"ID{marker_id}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return result_image

    def step3_draw_distance_lines(self, image, distance_results, rectangles_info):
        """
        第3步：繪製距離線（橘色邊緣距離、紫色角點距離）
        """
        result_image = image.copy()

        # 繪製所有距離線
        for result in distance_results:
            if result.get("edge_to_box") and result.get("corner_to_paper"):
                self.paper_detector.draw_distance_annotations(
                    result_image,
                    result["edge_to_box"],
                    result["corner_to_paper"],
                    result["marker_id"],
                    result["region"],
                )

        # 找出並標記最長距離
        longest_distance = self.paper_detector.find_longest_distance(distance_results)
        if longest_distance:
            # 根據距離類型確定標記點
            if longest_distance["type"] == "edge_to_box":
                point = longest_distance["details"]["edge_point"]
            else:  # corner_to_paper
                point = longest_distance["details"]["corner_point"]

            if point:
                # 繪製更大的標記
                cv2.circle(result_image, point, 12, (255, 255, 255), 3)  # 白色外圈
                cv2.circle(result_image, point, 8, (0, 0, 0), 2)  # 黑色內圈

                # 添加最長距離標籤
                distance_type_text = (
                    "邊緣" if longest_distance["type"] == "edge_to_box" else "角點"
                )
                text = f"MAX-{distance_type_text}: {longest_distance['distance']:.1f}px"
                text_pos = (point[0] - 50, point[1] - 30)

                # 白色背景黑色文字
                cv2.putText(
                    result_image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    3,  # 白色粗體背景
                )
                cv2.putText(
                    result_image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,  # 黑色文字
                )

        return result_image

    def show_step_by_step_results(self, image_path, base_name):
        """
        顯示分步驟的處理結果
        """
        result_dir = "result"
        step_files = []

        # 查找所有步驟圖片
        for i in range(4):  # 0-3步驟
            pattern = f"{base_name}_step{i}_*.jpg"
            import glob

            matches = glob.glob(os.path.join(result_dir, pattern))
            if matches:
                step_files.append(matches[0])

        if step_files:
            plt.figure(figsize=(20, 5))

            for i, filepath in enumerate(step_files):
                img = cv2.imread(filepath)
                if img is not None:
                    plt.subplot(1, len(step_files), i + 1)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    step_name = (
                        os.path.basename(filepath)
                        .replace(f"{base_name}_step{i}_", "")
                        .replace(".jpg", "")
                    )
                    plt.title(f"步驟{i}: {step_name}")
                    plt.axis("off")

            plt.tight_layout()
            plt.show()
            print(f"📊 已顯示 {base_name} 的分步驟處理結果")

    def save_results(self, original_path, result_image, analysis_results):
        """
        保存分析結果
        """
        # 確保結果目錄存在
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        base_name = os.path.splitext(os.path.basename(original_path))[0]

        # 保存圖像結果
        image_output_path = os.path.join(
            result_dir, f"{base_name}_integrated_analysis.jpg"
        )
        cv2.imwrite(image_output_path, result_image)
        print(f"\n圖像結果已保存: {image_output_path}")

    def show_results(self, original_image, result_image, analysis_results):
        """
        顯示分析結果
        """
        plt.figure(figsize=(16, 8))

        # 原圖
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("原始圖像")
        plt.axis("off")

        # 結果圖
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

        # 標題包含最長距離資訊
        title = "ArUco 偵測 + 改進紙張輪廓分析"
        if analysis_results["longest_distance"]:
            longest = analysis_results["longest_distance"]
            distance_type_text = "邊緣" if longest["type"] == "edge_to_box" else "角點"
            title += f'\\n最長距離({distance_type_text}): {longest["distance"]:.1f}px (ID{longest["marker_id"]})'

        plt.title(title)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def process_directory(self, directory_path):
        """
        批次處理目錄中的所有圖片
        """
        if not os.path.exists(directory_path):
            print(f"目錄不存在: {directory_path}")
            return

        # 支援的圖片格式
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = [
            f
            for f in os.listdir(directory_path)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]

        if not image_files:
            print(f"目錄中沒有找到圖片檔案: {directory_path}")
            return

        print(f"\\n開始批次處理 {len(image_files)} 個圖片檔案...")

        all_results = []
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(directory_path, filename)
            print(f"\\n[{i}/{len(image_files)}] 處理: {filename}")

            try:
                result_image, analysis_results = self.process_single_image(
                    image_path, save_result=True, show_result=False
                )
                all_results.append({"filename": filename, "analysis": analysis_results})
                print(f"✅ {filename} 處理完成")

            except Exception as e:
                print(f"❌ {filename} 處理失敗: {e}")


def main():
    """
    主程式入口
    """
    print("改進分析程式: ArUco 偵測 + 原本精確紙張輪廓分析")
    print("功能: 偵測 ArUco 標記，繪製 1/4 A4 長方形，使用原本精確輪廓偵測")
    print("新功能: 1.保留原本精確輪廓偵測 2.左右分區分組 3.雙種距離計算")
    print("=" * 60)

    analyzer = IntegratedAnalyzer()

    # 檢查輸入目錄
    img_dir = "img"
    if os.path.exists(img_dir):
        print(f"發現圖片目錄: {img_dir}")
        analyzer.process_directory(img_dir)
    else:
        print(f"圖片目錄不存在: {img_dir}")
        print("請將待處理的圖片放入 img/ 目錄")
        return

    print(f"\\n{'='*60}")
    print("處理完成！")
    print("📊 結果保存在 result/ 目錄:")
    print("   - *_integrated_analysis.jpg (圖像結果)")
    print("🎆 新功能: 原本精確輪廓 + 左右分區 + 雙距離計算")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
