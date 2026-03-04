import cv2
import numpy as np
import json
import os
from find_point import PointDetector
from PaperDetector_edge import PaperDetector_edges
from px2cm import get_pixel_per_cm_from_a4


class ShapeEdgeDistanceCalculator:
    def __init__(self, image_path, json_path="px2cm.json"):
        self.image_path = image_path
        self.json_path = json_path
        self.original = None
        self.pixel_per_cm = None
        self.edges = None
        self.edge_points = []
        self.given_points = []
        self.min_distance = float("inf")
        self.max_distance = 0
        self.min_point = None
        self.max_point = None
        self.result_image = None

    def load_image_and_config(self):
        """載入圖像和像素比例配置"""
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            print(f"無法載入圖像: {self.image_path}")
            return False

        # 讀取像素比例
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.pixel_per_cm = data.get("pixel_per_cm", 40.47)
                    print(
                        f"從 {self.json_path} 讀取像素每公分比例: {self.pixel_per_cm:.2f}"
                    )
            except Exception as e:
                print(f"讀取 {self.json_path} 失敗: {e}")
                self.pixel_per_cm = 40.47
        else:
            self.pixel_per_cm = 40.47
            print(f"使用預設像素每公分比例: {self.pixel_per_cm:.2f}")

        return True

    def detect_shape_edges(self, show_debug=False):
        """使用邊緣檢測來偵測中間圖形的邊緣"""
        if self.original is None:
            print("請先載入圖像")
            return False

        # 轉換為灰度圖像
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny邊緣檢測 - 調整參數以更好地檢測中間圖形
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

        # 形態學操作來連接斷開的邊緣
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找出輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 過濾輪廓，找到中間的主要圖形
        # 根據面積和位置來判斷哪個是中間的圖形
        height, width = self.original.shape[:2]
        center_x, center_y = width // 2, height // 2

        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # 過濾太小的輪廓
            if area < 1000:
                continue

            # 計算輪廓的中心點
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 計算到圖像中心的距離
                distance_to_center = np.sqrt(
                    (cx - center_x) ** 2 + (cy - center_y) ** 2
                )

                # 分數越高越好 (面積大，且靠近中心)
                score = area / (1 + distance_to_center * 0.01)

                if score > best_score:
                    best_score = score
                    best_contour = contour

        if best_contour is not None:
            # 提取邊緣點
            self.edge_points = best_contour.reshape(-1, 2)
            print(f"檢測到 {len(self.edge_points)} 個邊緣點")

            if show_debug:
                # 顯示檢測結果
                debug_image = self.original.copy()
                cv2.drawContours(debug_image, [best_contour], -1, (0, 255, 0), 2)

                # 標記輪廓中心
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(debug_image, (cx, cy), 5, (255, 0, 0), -1)

                # 縮放顯示
                height, width = debug_image.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    debug_image = cv2.resize(debug_image, (new_width, new_height))

                cv2.imshow("debug_image", debug_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return True
        else:
            print("未檢測到明顯的中間圖形")
            return False

    def set_given_points(self, points):
        """設置給定的點"""
        self.given_points = points
        print(f"設置了 {len(points)} 個給定點, {points}")

    def calculate_distances(self):
        """計算邊緣點與給定點之間的最短和最長距離"""
        if len(self.edge_points) == 0 or len(self.given_points) == 0:
            print("請先檢測邊緣點和設置給定點")
            return False

        self.min_distance = float("inf")
        self.max_distance = 0

        for given_point in self.given_points:
            gx, gy = given_point

            for edge_point in self.edge_points:
                ex, ey = edge_point

                # 計算歐氏距離
                distance_pixels = np.sqrt((gx - ex) ** 2 + (gy - ey) ** 2)
                distance_cm = distance_pixels / self.pixel_per_cm

                # 更新最短距離
                if distance_cm < self.min_distance:
                    self.min_distance = distance_cm
                    self.min_point = (ex, ey)

                # 更新最長距離
                if distance_cm > self.max_distance:
                    self.max_distance = distance_cm
                    self.max_point = (ex, ey)

        print(f"最短距離: {self.min_distance:.2f} 公分")
        print(f"最長距離: {self.max_distance:.2f} 公分")

        return True

    def visualize_results(self, save_path=r"result/distance_analysis_result.jpg"):
        """視覺化結果"""
        if (
            self.original is None
            or len(self.edge_points) == 0
            or len(self.given_points) == 0
        ):
            print("請先完成邊緣檢測和距離計算")
            return

        result_image = self.original.copy()

        # 畫出所有邊緣點
        for point in self.edge_points:
            cv2.circle(result_image, tuple(point), 1, (0, 255, 0), -1)

        # 畫出給定點
        for point in self.given_points:
            cv2.circle(result_image, tuple(point), 8, (255, 0, 0), -1)

        # 標記最短距離點和最長距離點
        if self.min_point is not None:
            cv2.circle(result_image, tuple(self.min_point), 5, (0, 0, 255), -1)
            # 畫線到最近的給定點
            closest_given = min(
                self.given_points,
                key=lambda p: np.sqrt(
                    (p[0] - self.min_point[0]) ** 2 + (p[1] - self.min_point[1]) ** 2
                ),
            )
            cv2.line(
                result_image,
                tuple(self.min_point),
                tuple(closest_given),
                (0, 0, 255),
                2,
            )

        if self.max_point is not None:
            cv2.circle(result_image, tuple(self.max_point), 5, (255, 0, 255), -1)
            # 畫線到最遠的給定點
            farthest_given = max(
                self.given_points,
                key=lambda p: np.sqrt(
                    (p[0] - self.max_point[0]) ** 2 + (p[1] - self.max_point[1]) ** 2
                ),
            )
            cv2.line(
                result_image,
                tuple(self.max_point),
                tuple(farthest_given),
                (255, 0, 255),
                2,
            )

        # 添加文字標註
        cv2.putText(
            result_image,
            f"Min: {self.min_distance:.2f}cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            result_image,
            f"Max: {self.max_distance:.2f}cm",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )
        Correct_length_1 = 4.1
        Correct_length_2 = 4.1
        print(
            f"Correct_length_1: {Correct_length_1}, Correct_length_2: {Correct_length_2}"
        )
        kid = max(
            abs(self.min_distance - Correct_length_1),
            abs(self.max_distance - Correct_length_2),
        )
        score = None
        if kid < 0.6:
            score = 2
        elif kid < 1.2:
            score = 1
        else:
            score = 0
        cv2.putText(
            result_image,
            f"Score: {score}, {kid:.2f}cm",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        self.result_image = result_image

        # 顯示結果
        height, width = result_image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(result_image, (new_width, new_height))
        else:
            display_image = result_image

        cv2.imshow("display_image", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存結果
        if save_path:
            name = self.image_path.split("\\")[-1].split(".")[0]
            save_path = rf"result/{name}_distance_analysis_result.jpg"
            cv2.imwrite(save_path, result_image)
            print(f"結果已保存至: {save_path}")


def main():
    """主函數 - 整合所有功能"""
    # 設置參數
    img_num = 1
    image_path = rf"img\{img_num}.jpg"

    # 步驟1: 檢測紙張並提取區域
    print("步驟1: 檢測紙張區域...")
    _, json_path = get_pixel_per_cm_from_a4(rf"a4/a4_2.jpg", show_debug=False)
    detector = PaperDetector_edges(image_path)
    detector.detect_paper_by_color()

    if detector.original is None:
        print("無法載入圖像")
        return

    detector.show_results()
    region = detector.extract_paper_region()

    if region is None:
        print("無法提取紙張區域")
        return

    # 調整大小
    height, width = region.shape[:2]
    if width > 600:
        scale = 600 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        region = cv2.resize(region, (new_width, new_height))

    detector_path = detector.save_results()

    # 步驟2: 檢測點
    print("步驟2: 檢測點...")
    point_detector = PointDetector(detector_path, image_path)
    point_detector.load_image(json_path=json_path)
    points = point_detector.detect_points()

    if len(points) != 1:
        print(f"檢測到 {len(points)} 個點，需要恰好1個點")
        return

    point_detector.draw_squares_at_points(square_size_cm=8)
    point_detector.save_result()
    given_points = point_detector.points

    # 步驟3: 進行邊緣檢測和距離計算
    print("步驟3: 進行邊緣檢測和距離分析...")
    calculator = ShapeEdgeDistanceCalculator(detector_path, json_path)

    if not calculator.load_image_and_config():
        print("無法載入圖像或配置")
        return

    if not calculator.detect_shape_edges(show_debug=True):
        print("無法檢測形狀邊緣")
        return

    calculator.set_given_points(given_points)

    if not calculator.calculate_distances():
        print("無法計算距離")
        return

    # 視覺化和保存結果

    calculator.visualize_results()

    print(f"\n=== 分析完成 ===")
    print(f"圖像: {image_path}")
    print(f"給定點數量: {len(given_points)}")
    print(f"邊緣點數量: {len(calculator.edge_points)}")
    print(f"最短距離: {calculator.min_distance:.2f} 公分")
    print(f"最長距離: {calculator.max_distance:.2f} 公分")


if __name__ == "__main__":
    main()
