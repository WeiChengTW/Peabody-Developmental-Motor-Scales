import cv2
import numpy as np
import os
from typing import List, Tuple


class PartialCutDetector:
    """
    專門檢測評分1的情況：只剪到紙的1/4或更少
    """

    def __init__(self):
        self.max_cut_ratio = 0.25  # 最大剪切比例（1/4）
        self.min_cut_length = 20  # 最小剪切長度（像素）

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """預處理圖像"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def find_paper_contour(self, binary_image: np.ndarray) -> np.ndarray:
        """找到紙張的輪廓"""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            return max(contours, key=cv2.contourArea)
        return None

    def detect_cut_lines(self, binary_image: np.ndarray) -> List[Tuple]:
        """檢測剪切線"""
        # 使用邊緣檢測
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)

        # 使用霍夫變換檢測直線
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=self.min_cut_length,
            maxLineGap=10,
        )

        cut_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > self.min_cut_length:
                    cut_lines.append(((x1, y1), (x2, y2), length))

        return cut_lines

    def calculate_cut_depth_ratio(
        self, cut_lines: List[Tuple], paper_contour: np.ndarray
    ) -> float:
        """計算剪切深度比例"""
        if not cut_lines or paper_contour is None:
            return 0.0

        # 獲取紙張的邊界矩形
        x, y, w, h = cv2.boundingRect(paper_contour)
        paper_max_dimension = max(w, h)

        max_cut_ratio = 0.0
        for start_point, end_point, length in cut_lines:
            # 計算剪切深度相對於紙張最大尺寸的比例
            cut_ratio = length / paper_max_dimension
            max_cut_ratio = max(max_cut_ratio, cut_ratio)

        return max_cut_ratio

    def check_partial_cut_in_edge(
        self, cut_lines: List[Tuple], paper_contour: np.ndarray
    ) -> bool:
        """檢查剪切是否從邊緣開始（部分剪切的特徵）"""
        if not cut_lines or paper_contour is None:
            return False

        # 獲取紙張邊界
        x, y, w, h = cv2.boundingRect(paper_contour)
        edge_threshold = 10  # 距離邊緣的閾值

        for start_point, end_point, _ in cut_lines:
            # 檢查剪切線的任一端點是否靠近紙張邊緣
            for point in [start_point, end_point]:
                px, py = point
                # 檢查是否靠近任何邊緣
                near_left = abs(px - x) < edge_threshold
                near_right = abs(px - (x + w)) < edge_threshold
                near_top = abs(py - y) < edge_threshold
                near_bottom = abs(py - (y + h)) < edge_threshold

                if near_left or near_right or near_top or near_bottom:
                    return True

        return False

    def detect_partial_cut(self, image_path: str) -> bool:
        """
        檢測是否為部分剪切（評分1）
        返回True表示符合評分1的條件
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False

            # 預處理
            binary = self.preprocess_image(image)

            # 找到紙張輪廓
            paper_contour = self.find_paper_contour(binary)
            if paper_contour is None:
                return False

            # 檢測剪切線
            cut_lines = self.detect_cut_lines(binary)
            if not cut_lines:
                return False

            # 計算剪切深度比例
            cut_depth_ratio = self.calculate_cut_depth_ratio(cut_lines, paper_contour)

            # 檢查是否從邊緣開始剪切
            is_edge_cut = self.check_partial_cut_in_edge(cut_lines, paper_contour)

            # 判斷是否為部分剪切
            is_partial_cut = (
                0.05 < cut_depth_ratio <= self.max_cut_ratio
            ) and is_edge_cut

            if is_partial_cut:
                print(f"✓ 檢測到部分剪切：{image_path}")
                print(f"  - 剪切深度比例: {cut_depth_ratio:.3f}")
                print(f"  - 檢測到的剪切線數量: {len(cut_lines)}")
                print(f"  - 從邊緣開始剪切: {is_edge_cut}")
                return True
            else:
                print(f"✗ 未檢測到部分剪切：{image_path}")
                print(f"  - 剪切深度比例: {cut_depth_ratio:.3f}")
                return False

        except Exception as e:
            print(f"處理圖像時發生錯誤: {e}")
            return False

    def visualize_detection(self, image_path: str, output_path: str = None):
        """視覺化檢測結果"""
        try:
            image = cv2.imread(image_path)
            binary = self.preprocess_image(image)
            paper_contour = self.find_paper_contour(binary)
            cut_lines = self.detect_cut_lines(binary)

            result_image = image.copy()

            # 繪製紙張輪廓
            if paper_contour is not None:
                cv2.drawContours(result_image, [paper_contour], -1, (0, 255, 0), 2)

            # 繪製剪切線
            for start_point, end_point, length in cut_lines:
                cv2.line(result_image, start_point, end_point, (0, 0, 255), 3)
                # 標記剪切線長度
                mid_point = (
                    (start_point[0] + end_point[0]) // 2,
                    (start_point[1] + end_point[1]) // 2,
                )
                cv2.putText(
                    result_image,
                    f"{length:.0f}px",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )

            # 添加檢測結果
            is_partial = self.detect_partial_cut(image_path)
            cut_depth_ratio = self.calculate_cut_depth_ratio(cut_lines, paper_contour)

            status_text = (
                f"Score 1: Partial Cut (Depth: {cut_depth_ratio:.2f})"
                if is_partial
                else "Not Partial Cut"
            )
            color = (0, 255, 0) if is_partial else (0, 0, 255)
            cv2.putText(
                result_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # 顯示剪切線統計
            cv2.putText(
                result_image,
                f"Cut lines: {len(cut_lines)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if output_path:
                cv2.imwrite(output_path, result_image)
            else:
                cv2.imshow("Partial Cut Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"視覺化過程中發生錯誤: {e}")


def main():
    """示範用法"""
    detector = PartialCutDetector()

    # 測試單張圖像
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        result = detector.detect_partial_cut(test_image)
        print(f"部分剪切檢測結果: {result}")
        detector.visualize_detection(test_image, "partial_cut_result.jpg")

    # 批量測試
    image_folder = "images"
    if os.path.exists(image_folder):
        print("\n批量檢測結果:")
        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(image_folder, filename)
                result = detector.detect_partial_cut(image_path)
                print(f"{filename}: {'評分1' if result else '非評分1'}")


if __name__ == "__main__":
    main()
