import cv2
import numpy as np
import os
from typing import List, Tuple


class NoActualCutDetector:
    """
    專門檢測評分0的情況：只動動剪刀未剪下去
    """

    def __init__(self):
        self.scissors_template_paths = []  # 剪刀模板路径列表
        self.motion_threshold = 0.02  # 動作檢測閾值
        self.min_cut_evidence = 5  # 最小剪切證據像素數

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """預處理圖像"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return blurred

    def detect_scissors_presence(self, image: np.ndarray) -> bool:
        """檢測圖像中是否有剪刀存在"""
        gray = self.preprocess_image(image)

        # 使用邊緣檢測來尋找剪刀的輪廓特徵
        edges = cv2.Canny(gray, 50, 150)

        # 尋找具有剪刀特徵的輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        scissors_indicators = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area > 100:  # 過濾小輪廓
                # 檢查輪廓的複雜度（剪刀通常有複雜的形狀）
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)

                if hull_area > 0:
                    solidity = area / hull_area
                    # 剪刀的實心度通常較低（因為有孔洞和複雜形狀）
                    if 0.3 < solidity < 0.8:
                        scissors_indicators += 1

        return scissors_indicators > 0

    def detect_paper_integrity(self, image: np.ndarray) -> bool:
        """檢測紙張是否保持完整（沒有明顯的剪切痕跡）"""
        gray = self.preprocess_image(image)

        # 二值化以突出紙張
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 找到紙張輪廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        # 獲取最大輪廓（假設是紙張）
        paper_contour = max(contours, key=cv2.contourArea)

        # 檢查輪廓的平滑度
        epsilon = 0.02 * cv2.arcLength(paper_contour, True)
        approx = cv2.approxPolyDP(paper_contour, epsilon, True)

        # 完整的紙張應該有相對簡單的輪廓（矩形狀）
        # 如果有很多頂點，可能表示有剪切痕跡
        return len(approx) <= 6  # 允許一些誤差

    def check_minimal_disturbance(self, image: np.ndarray) -> bool:
        """檢查是否只有最小程度的干擾（動剪刀但沒有實際剪切）"""
        gray = self.preprocess_image(image)

        # 使用形態學操作檢測細微的變化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # 計算開運算前後的差異
        diff = cv2.absdiff(gray, opened)

        # 統計顯著變化的像素數量
        significant_changes = np.sum(diff > 10)
        total_pixels = diff.shape[0] * diff.shape[1]

        # 如果顯著變化很少，可能只是動了剪刀但沒有實際剪切
        change_ratio = significant_changes / total_pixels

        return change_ratio < self.motion_threshold

    def detect_no_actual_cut(self, image_path: str) -> bool:
        """
        檢測是否為無實際剪切（評分0）
        返回True表示符合評分0的條件
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False

            # 檢測剪刀存在
            has_scissors = self.detect_scissors_presence(image)

            # 檢測紙張完整性
            paper_intact = self.detect_paper_integrity(image)

            # 檢測是否只有最小干擾
            minimal_disturbance = self.check_minimal_disturbance(image)

            # 評分0的條件：可能有剪刀動作但紙張基本完整
            is_no_cut = paper_intact and minimal_disturbance

            if is_no_cut:
                print(f"✓ 檢測到無實際剪切：{image_path}")
                print(f"  - 檢測到剪刀: {has_scissors}")
                print(f"  - 紙張完整: {paper_intact}")
                print(f"  - 最小干擾: {minimal_disturbance}")
                return True
            else:
                print(f"✗ 檢測到有實際剪切：{image_path}")
                print(f"  - 紙張完整: {paper_intact}")
                print(f"  - 最小干擾: {minimal_disturbance}")
                return False

        except Exception as e:
            print(f"處理圖像時發生錯誤: {e}")
            return False

    def analyze_image_features(self, image: np.ndarray) -> dict:
        """分析圖像特徵以輔助判斷"""
        gray = self.preprocess_image(image)

        # 邊緣密度分析
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 輪廓複雜度分析
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total_contour_length = sum(cv2.arcLength(contour, True) for contour in contours)

        # 紋理變化分析
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            "edge_density": edge_density,
            "total_contour_length": total_contour_length,
            "texture_variance": laplacian_var,
            "num_contours": len(contours),
        }

    def visualize_detection(self, image_path: str, output_path: str = None):
        """視覺化檢測結果"""
        try:
            image = cv2.imread(image_path)
            gray = self.preprocess_image(image)

            result_image = image.copy()

            # 繪製邊緣以顯示紙張狀況
            edges = cv2.Canny(gray, 50, 150)
            edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result_image = cv2.addWeighted(result_image, 0.7, edge_colored, 0.3, 0)

            # 找到並繪製紙張輪廓
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                paper_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(result_image, [paper_contour], -1, (0, 255, 0), 2)

            # 添加檢測結果
            is_no_cut = self.detect_no_actual_cut(image_path)
            features = self.analyze_image_features(image)

            status_text = "Score 0: No Actual Cut" if is_no_cut else "Has Actual Cut"
            color = (0, 255, 0) if is_no_cut else (0, 0, 255)
            cv2.putText(
                result_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # 顯示分析特徵
            cv2.putText(
                result_image,
                f"Edge Density: {features['edge_density']:.4f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                result_image,
                f"Contours: {features['num_contours']}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                result_image,
                f"Texture Var: {features['texture_variance']:.2f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if output_path:
                cv2.imwrite(output_path, result_image)
            else:
                cv2.imshow("No Cut Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"視覺化過程中發生錯誤: {e}")


def main():
    """示範用法"""
    detector = NoActualCutDetector()

    # 測試單張圖像
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        result = detector.detect_no_actual_cut(test_image)
        print(f"無實際剪切檢測結果: {result}")
        detector.visualize_detection(test_image, "no_cut_result.jpg")

    # 批量測試
    image_folder = "images"
    if os.path.exists(image_folder):
        print("\n批量檢測結果:")
        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(image_folder, filename)
                result = detector.detect_no_actual_cut(image_path)
                print(f"{filename}: {'評分0' if result else '非評分0'}")


if __name__ == "__main__":
    main()
