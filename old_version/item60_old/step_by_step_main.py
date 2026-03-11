"""
分步驟處理程式：ArUco 偵測與紙張輪廓分析
每一步都會保存圖像結果，便於觀察處理過程
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from detect_aruco_and_draw_quarter_a4 import ArUcoQuarterA4Detector
from original_paper_detector import OriginalPaperDetector


class StepByStepAnalyzer:
    """
    分步驟分析器：每一步都保存結果圖像
    """

    def __init__(self):
        self.aruco_detector = ArUcoQuarterA4Detector()
        self.paper_detector = OriginalPaperDetector()

    def process_single_image(self, image_path, show_result=True):
        """
        分步驟處理單張圖片

        Steps:
        0. 原始圖片
        1. 藍線畫出左右紙張輪廓 (避免中央分隔線影響)
        2. 綠線畫出ArUco長方形
        3. 橘色邊緣距離線、紫色角點距離線
        """
        # 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'='*60}")
        print(f"分步驟處理圖片: {base_name}")
        print(f"圖片尺寸: {image.shape[1]} x {image.shape[0]} 像素")
        print(f"{'='*60}")

        # 確保結果目錄存在
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # ===== 第0步：保存原始圖片 =====
        print("\n🔵 第0步：原始圖片")
        step0_image = image.copy()
        self.save_step_image(step0_image, base_name, 0, "原始圖片")

        # ===== ArUco 偵測準備工作 =====
        print("\n📍 ArUco 偵測...")
        corners, ids, rejected = self.aruco_detector.detect_aruco_markers(image)

        if ids is None:
            print("❌ 未偵測到 ArUco 標記，無法繼續分析")
            return image, {}

        # 獲取長方形資訊（不繪製到圖像上）
        temp_image, detection_results = self.aruco_detector.draw_quarter_a4_rectangles(
            image.copy(), corners, ids
        )

        rectangles_info = []
        for result in detection_results:
            corner_data = corners[len(rectangles_info)]
            rectangle_corners, current_scale_info = (
                self.aruco_detector.calculate_quarter_a4_rectangle(
                    corner_data, result["marker_id"]
                )
            )
            rectangles_info.append(
                {
                    "corners": rectangle_corners,
                    "marker_id": result["marker_id"],
                    "scale_info": current_scale_info,  # 使用正確的比例尺資訊
                }
            )

        # ===== 第1步：藍線畫出紙張輪廓 =====
        print("\n🔵 第1步：藍線畫出左右紙張輪廓（過濾中央分隔線）")
        step1_image, distance_results = self.step1_detect_paper_contours(
            image, rectangles_info
        )
        self.save_step_image(step1_image, base_name, 1, "藍線紙張輪廓")

        # ===== 第2步：綠線畫出ArUco長方形 =====
        print("\n🟢 第2步：綠線畫出ArUco長方形")
        step2_image = self.step2_draw_aruco_rectangles(step1_image, rectangles_info)
        self.save_step_image(step2_image, base_name, 2, "綠線ArUco長方形")

        # ===== 第3步：畫出距離線 =====
        print("\n🟠 第3步：橘色邊緣距離線、🟣 紫色角點距離線")
        step3_image = self.step3_draw_distance_lines(
            step2_image, distance_results, rectangles_info
        )
        self.save_step_image(step3_image, base_name, 3, "最終結果_距離線標註")

        # 分析結果
        longest_distance = self.paper_detector.find_longest_distance(distance_results)

        analysis_results = {
            "aruco_results": detection_results,
            "distance_results": distance_results,
            "longest_distance": longest_distance,
            "rectangles_info": rectangles_info,
        }

        # 顯示結果
        if show_result:
            self.show_step_by_step_results(base_name)

        print(f"\n✅ {base_name} 所有步驟處理完成！")
        return step3_image, analysis_results

    def save_step_image(self, image, base_name, step_num, step_name):
        """
        保存每一步的圖像結果
        """
        # 使用絕對路徑
        result_dir = os.path.abspath("result")

        # 將中文步驟名稱映射為英文
        step_name_mapping = {
            "原始圖片": "original",
            "藍線紙張輪廓": "blue_contours",
            "綠線ArUco長方形": "green_rectangles",
            "最終結果_距離線標註": "final_with_distances",
        }

        english_step_name = step_name_mapping.get(
            step_name, step_name.replace(" ", "_")
        )
        filename = f"{base_name}_step{step_num}_{english_step_name}.jpg"
        filepath = os.path.join(result_dir, filename)

        # 確保目錄存在
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print(f"📁 創建目錄: {result_dir}")

        # 保存圖片並檢查結果
        success = cv2.imwrite(filepath, image)
        if success:
            print(f"💾 已保存: {filename}")
            print(f"   完整路徑: {filepath}")
            # 驗證檔案是否真的存在
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"   檔案大小: {file_size} bytes")
            else:
                print(f"   ❌ 警告: 檔案不存在！")
        else:
            print(f"❌ 保存失敗: {filename}")
            print(f"   嘗試路徑: {filepath}")

    def step1_detect_paper_contours(self, image, rectangles_info):
        """
        第1步：偵測並繪製紙張輪廓（藍色線條）
        過濾中央分隔線，基於ArUco標記連接的紙張區域進行左右分區
        """
        result_image = image.copy()

        # 🔍 第一步：檢查是否有剪切證據
        has_cutting_evidence, cutting_analysis = (
            self.paper_detector.detect_cutting_evidence(image, rectangles_info)
        )

        print(f"剪切檢測結果: {cutting_analysis['reason']}")
        if not has_cutting_evidence:
            print("⚠️  警告: 未偵測到剪切證據，將直接評為0分")

        # 創建智能的左右分區遮罩
        left_mask, right_mask, left_rectangles, right_rectangles = (
            self.paper_detector.create_region_masks(image.shape, rectangles_info)
        )

        # 不繪製分區線，避免干擾視覺效果
        # 分區邏輯已經在create_region_masks中實現，不需要視覺標示

        distance_results = []

        # 處理左側區域
        if left_rectangles:
            left_contours = self.paper_detector.detect_paper_contours(
                image, left_mask, filter_center_line=True
            )
            if left_contours:
                # 繪製左側紙張輪廓 (藍色)
                cv2.drawContours(result_image, left_contours, -1, (255, 0, 0), 2)

                # 計算距離但不繪製距離線
                for rect_info in left_rectangles:
                    # 獲取比例尺資訊
                    scale_info = None
                    if "scale_info" in rect_info:
                        scale_info = rect_info["scale_info"]

                    distance_result = (
                        self.paper_detector.calculate_rectangle_distance_no_draw(
                            rect_info,
                            left_contours,
                            "左側",
                            scale_info,
                            cutting_analysis,
                        )
                    )
                    if distance_result:
                        distance_results.append(distance_result)

        # 處理右側區域
        if right_rectangles:
            right_contours = self.paper_detector.detect_paper_contours(
                image, right_mask, filter_center_line=True
            )
            if right_contours:
                # 繪製右側紙張輪廓 (藍色)
                cv2.drawContours(result_image, right_contours, -1, (255, 0, 0), 2)

                # 計算距離但不繪製距離線
                for rect_info in right_rectangles:
                    # 獲取比例尺資訊
                    scale_info = None
                    if "scale_info" in rect_info:
                        scale_info = rect_info["scale_info"]

                    distance_result = (
                        self.paper_detector.calculate_rectangle_distance_no_draw(
                            rect_info,
                            right_contours,
                            "右側",
                            scale_info,
                            cutting_analysis,
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

        # 找出並特別標記最長距離
        longest_distance = self.paper_detector.find_longest_distance(distance_results)
        if longest_distance:
            # 根據距離類型確定標記點
            if longest_distance["type"] == "edge_to_box":
                point = longest_distance["details"]["edge_point"]
            else:  # corner_to_paper
                point = longest_distance["details"]["corner_point"]

            if point:
                # 繪製醒目的最長距離標記
                cv2.circle(result_image, point, 15, (255, 255, 255), 3)  # 白色外圈
                cv2.circle(result_image, point, 12, (0, 0, 0), 2)  # 黑色內圈
                cv2.circle(result_image, point, 8, (0, 255, 255), -1)  # 黃色填充

                # 添加最長距離標籤
                distance_type_text = (
                    "Edge" if longest_distance["type"] == "edge_to_box" else "Corner"
                )
                # 轉換為公分顯示（估算比例尺 1px ≈ 0.2mm）
                distance_cm = longest_distance["distance"] / 50.0  # 簡化轉換
                text = f"MAX-{distance_type_text}: {distance_cm:.1f}cm"
                text_pos = (point[0] - 60, point[1] - 35)

                # 白色背景黑色文字
                cv2.putText(
                    result_image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    4,  # 白色粗體背景
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

        # 在左上角顯示評分
        if distance_results:
            # 取得最高分數作為整體評分
            max_score = max([result.get("score", -1) for result in distance_results])

            # 選擇評分顏色
            if max_score == 2:
                score_color = (0, 255, 0)  # 綠色 - 優秀
                score_bg_color = (0, 128, 0)
            elif max_score == 1:
                score_color = (0, 165, 255)  # 橘色 - 良好
                score_bg_color = (0, 100, 200)
            elif max_score == 0:
                score_color = (0, 0, 255)  # 紅色 - 需要改進
                score_bg_color = (0, 0, 128)
            else:
                score_color = (128, 128, 128)  # 灰色 - 無法評分
                score_bg_color = (64, 64, 64)

            # 評分文字
            score_text = f"Score: {max_score}" if max_score >= 0 else "Score: N/A"

            # 左上角位置
            score_pos = (20, 50)

            # 繪製背景矩形
            cv2.rectangle(result_image, (10, 15), (200, 65), score_bg_color, -1)
            cv2.rectangle(result_image, (10, 15), (200, 65), (255, 255, 255), 2)

            # 繪製評分文字
            cv2.putText(
                result_image,
                score_text,
                score_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                score_color,
                3,
            )

        return result_image

    def show_step_by_step_results(self, base_name):
        """
        顯示分步驟的處理結果
        """
        # 查找所有步驟圖片
        step_files = []
        for i in range(4):  # 0-3步驟
            pattern = f"result/{base_name}_step{i}_*.jpg"
            matches = glob.glob(pattern)
            if matches:
                step_files.append(matches[0])

        if step_files:
            plt.figure(figsize=(20, 5))

            step_names = [
                "原始圖片",
                "藍線紙張輪廓",
                "綠線ArUco長方形",
                "最終結果_距離線標註",
            ]

            for i, filepath in enumerate(step_files):
                img = cv2.imread(filepath)
                if img is not None:
                    plt.subplot(1, len(step_files), i + 1)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(
                        f"步驟{i}: {step_names[i] if i < len(step_names) else '未知步驟'}"
                    )
                    plt.axis("off")

            plt.tight_layout()
            plt.show()
            print(f"📊 已顯示 {base_name} 的分步驟處理結果")

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

        print(f"\n開始批次分步驟處理 {len(image_files)} 個圖片檔案...")

        all_results = []
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(directory_path, filename)
            print(f"\n[{i}/{len(image_files)}] 處理: {filename}")

            try:
                result_image, analysis_results = self.process_single_image(
                    image_path, show_result=False
                )
                all_results.append({"filename": filename, "analysis": analysis_results})
                print(f"✅ {filename} 處理完成")

            except Exception as e:
                print(f"❌ {filename} 處理失敗: {e}")

        return all_results


def main():
    """
    主程式入口
    """
    print("分步驟分析程式: ArUco 偵測 + 紙張輪廓分析")
    print("功能: 分步驟處理並保存每一步的結果")
    print("步驟: 1.藍線紙張輪廓 → 2.綠線ArUco長方形 → 3.橘紫距離線")
    print("=" * 60)

    analyzer = StepByStepAnalyzer()

    # 檢查輸入目錄
    img_dir = "img"
    if os.path.exists(img_dir):
        print(f"發現圖片目錄: {img_dir}")
        analyzer.process_directory(img_dir)
    else:
        print(f"圖片目錄不存在: {img_dir}")
        print("請將待處理的圖片放入 img/ 目錄")
        return

    print(f"\n{'='*60}")
    print("處理完成！")
    print("📊 結果保存在 result/ 目錄:")
    print("   - *_step0_原始圖片.jpg")
    print("   - *_step1_藍線紙張輪廓.jpg")
    print("   - *_step2_綠線ArUco長方形.jpg")
    print("   - *_step3_最終結果_距離線標註.jpg")
    print("🎆 特色: 智能過濾中央分隔線 + ArUco連接區域分組")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
