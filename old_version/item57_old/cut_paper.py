import cv2
import numpy as np
import os
from typing import Tuple, List, Dict


class PaperCuttingAnalyzer:
    def __init__(self):
        """初始化剪紙分析器"""
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        預處理圖像
        """
        # 轉為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 二值化
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def find_paper_contour(self, binary_image: np.ndarray) -> np.ndarray:
        """
        找到紙張的輪廓
        """
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 找到最大的輪廓（假設是紙張）
        if contours:
            paper_contour = max(contours, key=cv2.contourArea)
            return paper_contour
        return None

    def detect_cuts(
        self, binary_image: np.ndarray, paper_contour: np.ndarray
    ) -> List[Dict]:
        """
        檢測剪切線
        """
        # 使用形態學操作來檢測細線（剪切線）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # 檢測邊緣
        edges = cv2.Canny(opened, 50, 150)

        # 使用霍夫變換檢測直線
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
        )

        cuts = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                cuts.append({"start": (x1, y1), "end": (x2, y2), "length": length})

        return cuts

    def calculate_paper_dimensions(
        self, paper_contour: np.ndarray
    ) -> Tuple[float, float]:
        """
        計算紙張的寬度和高度
        """
        if paper_contour is None:
            return 0, 0

        # 獲取邊界矩形
        x, y, w, h = cv2.boundingRect(paper_contour)
        return w, h

    def analyze_cut_depth(
        self, cuts: List[Dict], paper_width: float, paper_height: float
    ) -> float:
        """
        分析剪切深度比例
        """
        if not cuts or paper_width == 0:
            return 0.0

        max_cut_depth = 0
        for cut in cuts:
            # 計算剪切線相對於紙張的深度
            cut_depth = min(cut["length"] / paper_width, cut["length"] / paper_height)
            max_cut_depth = max(max_cut_depth, cut_depth)

        return max_cut_depth

    def check_paper_separation(self, binary_image: np.ndarray) -> bool:
        """
        檢查紙張是否被完全分離成兩部分
        """
        # 檢測連通區域
        num_labels, labels = cv2.connectedComponents(binary_image)

        # 如果有兩個主要的連通區域（除了背景），可能表示紙張被分成兩部分
        unique_labels, counts = np.unique(labels, return_counts=True)

        # 過濾掉小的區域
        large_regions = 0
        total_pixels = binary_image.shape[0] * binary_image.shape[1]

        for i, count in enumerate(counts):
            if unique_labels[i] != 0 and count > total_pixels * 0.1:  # 至少佔10%的面積
                large_regions += 1

        return large_regions >= 2

    def score_cutting(self, image_path: str) -> int:
        """
        主要評分函數
        返回：
        2 - 把紙剪成均分的2等份
        1 - 只剪到紙的1/4或更少
        0 - 只動動剪刀未剪下去
        """
        try:
            # 讀取圖像
            image = cv2.imread(image_path)
            if image is None:
                print(f"無法讀取圖像: {image_path}")
                return 0

            # 預處理
            binary = self.preprocess_image(image)

            # 找到紙張輪廓
            paper_contour = self.find_paper_contour(binary)
            if paper_contour is None:
                print("未能檢測到紙張")
                return 0

            # 計算紙張尺寸
            paper_width, paper_height = self.calculate_paper_dimensions(paper_contour)

            # 檢查是否完全分離
            is_separated = self.check_paper_separation(binary)
            if is_separated:
                return 2  # 完全剪成兩部分

            # 檢測剪切線
            cuts = self.detect_cuts(binary, paper_contour)

            if not cuts:
                return 0  # 沒有檢測到剪切

            # 分析剪切深度
            cut_depth_ratio = self.analyze_cut_depth(cuts, paper_width, paper_height)

            # 根據剪切深度評分
            if cut_depth_ratio > 0.4:  # 剪切深度超過40%
                return 2
            elif cut_depth_ratio > 0.1:  # 剪切深度在10%-40%之間
                return 1
            else:
                return 0  # 剪切深度很小或沒有剪切

        except Exception as e:
            print(f"處理圖像時發生錯誤: {e}")
            return 0

    def batch_analyze(self, image_folder: str) -> Dict[str, int]:
        """
        批量分析資料夾中的所有圖像
        """
        results = {}

        if not os.path.exists(image_folder):
            print(f"資料夾不存在: {image_folder}")
            return results

        # 支援的圖像格式
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(image_folder, filename)
                score = self.score_cutting(image_path)
                results[filename] = score
                print(f"{filename}: 分數 = {score}")

        return results

    def visualize_analysis(self, image_path: str, output_path: str = None):
        """
        視覺化分析結果
        """
        try:
            # 讀取原始圖像
            image = cv2.imread(image_path)
            if image is None:
                return

            # 預處理
            binary = self.preprocess_image(image)

            # 找到紙張輪廓
            paper_contour = self.find_paper_contour(binary)

            # 在原圖上繪製結果
            result_image = image.copy()

            if paper_contour is not None:
                # 繪製紙張輪廓
                cv2.drawContours(result_image, [paper_contour], -1, (0, 255, 0), 2)

                # 檢測和繪製剪切線
                cuts = self.detect_cuts(binary, paper_contour)
                for cut in cuts:
                    cv2.line(result_image, cut["start"], cut["end"], (0, 0, 255), 2)

            # 添加評分
            score = self.score_cutting(image_path)
            score_text = f"Score: {score}"
            cv2.putText(
                result_image,
                score_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            # 顯示或保存結果
            if output_path:
                cv2.imwrite(output_path, result_image)
                print(f"結果已保存到: {output_path}")
            else:
                cv2.imshow("Analysis Result", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"視覺化過程中發生錯誤: {e}")


def main():
    """
    主函數 - 示範用法
    """
    analyzer = PaperCuttingAnalyzer()

    # 單張圖像分析
    image_path = "test_image.jpg"  # 替換為實際的圖像路徑
    if os.path.exists(image_path):
        score = analyzer.score_cutting(image_path)
        print(f"剪紙評分: {score}")

        # 視覺化結果
        analyzer.visualize_analysis(image_path, "result.jpg")

    # 批量分析
    folder_path = "images"  # 替換為包含圖像的資料夾路徑
    if os.path.exists(folder_path):
        results = analyzer.batch_analyze(folder_path)
        print("\n批量分析結果:")
        for filename, score in results.items():
            print(f"{filename}: {score}")


if __name__ == "__main__":
    main()
