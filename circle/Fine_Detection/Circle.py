import cv2
import numpy as np
import os
import math
from typing import List, Tuple, Dict


class CircleGapDetector:
    """圓形缺口檢測器"""

    def __init__(self):
        self.min_contour_area = 50  # 降低最小輪廓面積，適應線條圖形
        self.circularity_threshold = 0.3  # 進一步放寬圓形度閾值，適應線條圓形
        self.gap_angle_threshold = 20  # 調整缺口角度閾值

    def find_circle_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """尋找圓形輪廓"""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circle_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            # 計算圓形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)

            if circularity > self.circularity_threshold:
                circle_contours.append(contour)

        return circle_contours

    def analyze_circle_completeness(
        self, contour: np.ndarray, image_shape: Tuple[int, int]
    ) -> Dict:
        """分析圓形完整性"""
        # 擬合圓形
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        center = (int(center_x), int(center_y))
        radius = int(radius) - 10

        # 創建理想圓形遮罩
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        # 創建實際輪廓遮罩
        contour_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour], 255)

        # 計算覆蓋率
        ideal_area = math.pi * radius * radius
        actual_area = cv2.contourArea(contour)
        coverage_ratio = actual_area / ideal_area if ideal_area > 0 else 0

        # 檢測缺口
        gap_info = self.detect_gaps(contour, center, radius)

        return {
            "center": center,
            "radius": radius,
            "coverage_ratio": coverage_ratio,
            "has_gap": gap_info["has_gap"],
            "gap_angles": gap_info["gap_angles"],
            "gap_count": gap_info["gap_count"],
            "max_gap_angle": gap_info["max_gap_angle"],
        }

    def detect_gaps(
        self, contour: np.ndarray, center: Tuple[int, int], radius: int
    ) -> Dict:
        """檢測圓形缺口 - 針對線條圓形優化"""
        # 將輪廓點轉換為極坐標
        angles = []
        for point in contour.reshape(-1, 2):
            x, y = point[0] - center[0], point[1] - center[1]
            angle = math.atan2(y, x) * 180 / math.pi
            if angle < 0:
                angle += 360
            angles.append(angle)

        if len(angles) < 5:  # 如果點太少，認為沒有缺口
            return {
                "has_gap": False,
                "gap_angles": [],
                "gap_count": 0,
                "max_gap_angle": 0,
            }

        # 排序角度
        angles.sort()

        # 檢測角度間隙
        gaps = []
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            angle_diff = angles[next_i] - angles[i]
            if angle_diff < 0:
                angle_diff += 360

            if angle_diff > self.gap_angle_threshold:
                gaps.append(angle_diff)

        # 檢查首尾連接
        if len(angles) > 1:
            first_last_gap = angles[0] + 360 - angles[-1]
            if first_last_gap > self.gap_angle_threshold:
                gaps.append(first_last_gap)

        has_gap = len(gaps) > 0
        max_gap = max(gaps) if gaps else 0

        return {
            "has_gap": has_gap,
            "gap_angles": gaps,
            "gap_count": len(gaps),
            "max_gap_angle": max_gap,
        }

    def visualize_results(
        self, image: np.ndarray, analysis_results: List[Dict], filename: str = ""
    ) -> np.ndarray:
        """視覺化檢測結果"""
        result_image = image.copy()

        for i, result in enumerate(analysis_results):
            center = result["center"]
            radius = result["radius"]
            has_gap = result["has_gap"]

            # 繪製圓形
            color = (
                (0, 0, 255) if has_gap else (0, 255, 0)
            )  # 紅色表示有缺口，綠色表示完整
            cv2.circle(result_image, center, radius, color, 2)
            cv2.circle(result_image, center, 3, color, -1)

            # 添加文字標籤
            gap_text = f"Gap: {'Yes' if has_gap else 'No'}"
            coverage_text = f"Coverage: {result['coverage_ratio']:.2f}"

            text_y = center[1] - radius - 10
            cv2.putText(
                result_image,
                gap_text,
                (center[0] - 50, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            cv2.putText(
                result_image,
                coverage_text,
                (center[0] - 50, text_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            if has_gap:
                max_gap = result["max_gap_angle"]
                gap_count = result["gap_count"]
                cv2.putText(
                    result_image,
                    f"Max gap: {max_gap:.1f}°",
                    (center[0] - 50, text_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                cv2.putText(
                    result_image,
                    f"Gaps: {gap_count}",
                    (center[0] - 50, text_y - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        # 顯示結果
        window_name = f"Circle Gap Analysis - {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 創建可調整大小的視窗
        cv2.resizeWindow(window_name, 500, 500)  # 設定視窗大小為 800x600
        cv2.imshow(window_name, result_image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        return result_image

    def process_image(self, image_path: str) -> List[Dict]:
        """處理單張圖片"""
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            return []

        # 顯示二值化結果以供調試
        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Binary Image", 400, 400)
        cv2.imshow("Binary Image", image)
        cv2.waitKey(0)  # 顯示1秒
        cv2.destroyWindow("Binary Image")

        # 尋找圓形輪廓
        circle_contours = self.find_circle_contours(image)

        if not circle_contours:
            print(f"在 {image_path} 中未找到圓形")
            return []

        # 分析每個圓形
        results = []
        for contour in circle_contours:
            analysis = self.analyze_circle_completeness(contour, image.shape)
            results.append(analysis)

        # 顯示結果
        filename = os.path.basename(image_path)
        self.visualize_results(image, results, filename)

        return results

    def process_directory(self, input_dir: str) -> Dict[str, List[Dict]]:
        """處理整個資料夾"""
        all_results = {}

        # 支援的圖片格式
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(input_dir, filename)
                print(f"處理圖片: {filename}")

                results = self.process_image(image_path)
                all_results[filename] = results

                # 印出結果
                if results:
                    for i, result in enumerate(results):
                        gap_status = "有缺口" if result["has_gap"] else "完整"
                        coverage = result["coverage_ratio"]
                        print(f"  圓形 {i+1}: {gap_status}, 覆蓋率: {coverage:.2f}")
                        if result["has_gap"]:
                            print(f"    最大缺口角度: {result['max_gap_angle']:.1f}°")
                            print(f"    缺口數量: {result['gap_count']}")
                else:
                    print(f"  未檢測到圓形")
                print()

        return all_results


def main():
    """主函數"""
    # 初始化檢測器
    detector = CircleGapDetector()

    # 設定路徑
    input_dir = r"c:\Users\chang\Downloads\circle\result\Circle"

    print("開始檢測圓形缺口...")
    print("按下任意鍵查看下一張圖片，ESC鍵退出")
    print("=" * 50)

    # 處理所有圖片
    results = detector.process_directory(input_dir)

    # 統計結果
    total_images = len(results)
    images_with_gaps = 0
    total_circles = 0
    circles_with_gaps = 0

    for filename, image_results in results.items():
        if image_results:
            total_circles += len(image_results)
            has_gap_in_image = any(result["has_gap"] for result in image_results)
            if has_gap_in_image:
                images_with_gaps += 1
            circles_with_gaps += sum(1 for result in image_results if result["has_gap"])

    print("=" * 50)
    print("檢測結果統計:")
    print(f"總圖片數: {total_images}")
    print(f"含有缺口圓形的圖片數: {images_with_gaps}")
    print(f"總圓形數: {total_circles}")
    print(f"有缺口的圓形數: {circles_with_gaps}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
