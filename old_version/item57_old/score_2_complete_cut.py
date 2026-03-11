import cv2
import numpy as np
import os


class CompleteCutDetector:
    """
    專門檢測評分2的情況：把紙剪成均分的2等份
    """

    def __init__(self):
        self.min_separation_ratio = 0.8  # 最小分離比例
        self.min_area_ratio = 0.1  # 每部分最小面積比例

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """預處理圖像"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def find_separated_parts(self, binary_image: np.ndarray) -> list:
        """找到分離的紙張部分"""
        # 使用連通區域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image
        )

        parts = []
        total_area = binary_image.shape[0] * binary_image.shape[1]

        for i in range(1, num_labels):  # 跳過背景（標籤0）
            area = stats[i, cv2.CC_STAT_AREA]
            # 只考慮面積足夠大的區域
            if area > total_area * self.min_area_ratio:
                parts.append(
                    {
                        "label": i,
                        "area": area,
                        "centroid": centroids[i],
                        "bbox": stats[i],
                    }
                )

        return parts

    def check_equal_division(self, parts: list) -> bool:
        """檢查是否為均等分割"""
        if len(parts) != 2:
            return False

        # 檢查兩部分的面積是否相近
        area1, area2 = parts[0]["area"], parts[1]["area"]
        area_ratio = min(area1, area2) / max(area1, area2)

        # 面積比例應該接近1（均等分割）
        return area_ratio > 0.7  # 允許30%的誤差

    def detect_complete_cut(self, image_path: str) -> bool:
        """
        檢測是否為完全剪切（評分2）
        返回True表示符合評分2的條件
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False

            # 預處理
            binary = self.preprocess_image(image)

            # 找到分離的部分
            parts = self.find_separated_parts(binary)

            # 檢查是否為均等分割
            is_equal_division = self.check_equal_division(parts)

            if is_equal_division:
                print(f"✓ 檢測到完全剪切：{image_path}")
                print(f"  - 分離部分數量: {len(parts)}")
                print(
                    f"  - 面積比例: {min(parts[0]['area'], parts[1]['area']) / max(parts[0]['area'], parts[1]['area']):.2f}"
                )
                return True
            else:
                print(f"✗ 未檢測到完全剪切：{image_path}")
                return False

        except Exception as e:
            print(f"處理圖像時發生錯誤: {e}")
            return False

    def visualize_detection(self, image_path: str, output_path: str = None):
        """視覺化檢測結果"""
        try:
            image = cv2.imread(image_path)
            binary = self.preprocess_image(image)
            parts = self.find_separated_parts(binary)

            result_image = image.copy()

            # 為每個部分繪製不同顏色的邊界
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            for i, part in enumerate(parts):
                color = colors[i % len(colors)]
                mask = (cv2.connectedComponents(binary)[1] == part["label"]).astype(
                    np.uint8
                ) * 255
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result_image, contours, -1, color, 3)

                # 標記重心
                center = tuple(map(int, part["centroid"]))
                cv2.circle(result_image, center, 5, color, -1)
                cv2.putText(
                    result_image,
                    f"Part {i+1}",
                    (center[0] - 30, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # 添加檢測結果
            is_complete = self.check_equal_division(parts)
            status_text = "Score 2: Complete Cut" if is_complete else "Not Complete Cut"
            color = (0, 255, 0) if is_complete else (0, 0, 255)
            cv2.putText(
                result_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            if output_path:
                cv2.imwrite(output_path, result_image)
            else:
                cv2.imshow("Complete Cut Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"視覺化過程中發生錯誤: {e}")


def main():
    """示範用法"""
    detector = CompleteCutDetector()

    # 測試單張圖像
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        result = detector.detect_complete_cut(test_image)
        print(f"完全剪切檢測結果: {result}")
        detector.visualize_detection(test_image, "complete_cut_result.jpg")

    # 批量測試
    image_folder = "images"
    if os.path.exists(image_folder):
        print("\n批量檢測結果:")
        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(image_folder, filename)
                result = detector.detect_complete_cut(image_path)
                print(f"{filename}: {'評分2' if result else '非評分2'}")


if __name__ == "__main__":
    main()
