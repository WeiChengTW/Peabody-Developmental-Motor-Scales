import cv2
import numpy as np
from PaperDetector_edge import PaperDetector_edges
from px2cm import get_pixel_per_cm_from_a4
import os
import json


class PointDetector:
    def __init__(self, image_path, original_image_path=None):
        self.image_path = image_path
        self.original_image_path = original_image_path or image_path  # 用於計算像素比例
        self.original = None
        self.processed = None
        self.points = []
        self.result_with_squares = None
        self.pixel_per_cm = None
        self.eroded = None

    def load_image(self, json_path="px2cm.json"):
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            print("無法載入圖像")
            return False

        # 優先使用 px2cm.json 中的像素比例
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "pixel_per_cm" in data:
                        self.pixel_per_cm = data["pixel_per_cm"]
                        print(
                            f"從 px2cm.json 讀取像素每公分比例: {self.pixel_per_cm:.2f}"
                        )
                    else:
                        print("px2cm.json 中沒有 pixel_per_cm 欄位")
                        self.pixel_per_cm = 40.47  # 預設值
            except Exception as e:
                print(f"讀取 px2cm.json 失敗: {e}")
                self.pixel_per_cm = 40.47  # 預設值
        else:
            print("沒有找到 px2cm.json，將嘗試自動計算像素比例")
            try:
                self.pixel_per_cm = get_pixel_per_cm_from_a4(self.original_image_path)
                print(f"自動計算像素每公分比例: {self.pixel_per_cm:.2f}")
            except Exception as e:
                print(f"無法計算像素比例: {e}")
                self.pixel_per_cm = 40.47  # 使用預設值，約40.47像素/公分

        return True

    def detect_points(self):
        """
        偵測圖像中的點
        只使用霍夫圓檢測方法
        """
        if self.original is None:
            print("請先載入圖像")
            return []

        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # 只使用霍夫圓檢測
        points = self._detect_circles(gray)

        self.points = points
        return points

    def _detect_circles(self, gray):
        """霍夫圓檢測"""

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # 自適應二值化
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        # cv2.imshow("binary", binary)
        # 膨脹操作，讓圓點更明顯
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(binary, kernel, iterations=2)
        # cv2.imshow("dilated", dilate)
        # 侵蝕操作，去除雜訊
        kernel = np.ones((7, 7), np.uint8)
        eroded = cv2.erode(dilate, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(eroded, kernel, iterations=1)
        self.eroded = eroded
        # 霍夫圓檢測 - 調整參數以檢測更多小圓點
        circles = cv2.HoughCircles(
            eroded,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,  # 進一步減少最小距離
            param1=20,  # 降低高閾值
            param2=7,  # 大幅降低累加器閾值
            minRadius=1,  # 最小半徑設為1
            maxRadius=10,  # 最大半徑設為10
        )

        points = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"霍夫圓檢測找到 {len(circles)} 個圓")

            # 取得圖像中心和尺寸
            height, width = gray.shape
            center_x, center_y = width // 2, height // 2

            # 定義中心區域的範圍（可調整這個比例）
            center_region_ratio = 0.3  # 只檢測中心30%區域的點
            region_width = int(width * center_region_ratio)
            region_height = int(height * center_region_ratio)

            # 計算中心區域的邊界
            left = center_x - region_width // 2
            right = center_x + region_width // 2
            top = center_y - region_height // 2
            bottom = center_y + region_height // 2

            print(f"圖像尺寸: {width} x {height}, 中心: ({center_x}, {center_y})")
            print(f"檢測區域: x=[{left}, {right}], y=[{top}, {bottom}]")

            # 創建一個顯示圖像來可視化中心區域和檢測結果
            display_image = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)

            # 畫出中心區域邊界 (綠色矩形)
            cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)

            # 標記圖像中心點 (藍色十字)
            cv2.drawMarker(
                display_image,
                (center_x, center_y),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            for idx, (x, y, r) in enumerate(circles):
                # 檢查半徑和位置條件
                if 1 <= r <= 5:
                    # 檢查點是否在中心區域內
                    if left <= x <= right and top <= y <= bottom:
                        # 計算到中心的距離
                        distance_to_center = np.sqrt(
                            (x - center_x) ** 2 + (y - center_y) ** 2
                        )
                        points.append((x, y))
                        print(
                            f"檢測到第{idx+1}號圓: 中心({x}, {y}), 半徑{r}, 到中心距離: {distance_to_center:.1f}"
                        )
                        # 在檢測區域圖像上標記檢測到的點 (紅色圓)
                        cv2.circle(display_image, (x, y), r, (0, 0, 255), 2)
                        cv2.putText(
                            display_image,
                            f"P{idx+1}",
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                    else:
                        print(f"過濾邊緣點: ({x}, {y}), 半徑{r} - 不在中心區域")
                        # 在檢測區域圖像上標記被過濾的點 (黃色圓)
                        cv2.circle(display_image, (x, y), r, (0, 255, 255), 1)

            # 添加說明文字
            cv2.putText(
                display_image,
                f"Center Region ({center_region_ratio*100}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_image,
                f"Valid Points: {len(points)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_image,
                "Green: Detection Area",
                (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                display_image,
                "Red: Valid Points",
                (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                display_image,
                "Yellow: Filtered Points",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            # 顯示檢測區域圖像
            height_display, width_display = display_image.shape[:2]
            if width_display > 800:
                scale = 800 / width_display
                new_width = int(width_display * scale)
                new_height = int(height_display * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))

            cv2.imshow("Detection Region & Results", display_image)

        # cv2.imshow("eroded", eroded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return points

    def draw_squares_at_points(self, square_size_cm=8):
        """在檢測到的點周圍畫方塊

        Args:
            square_size_cm: 方塊大小（公分）
        """
        if self.original is None or len(self.points) == 0:
            print("沒有檢測到點或圖像未載入")
            return None

        if self.pixel_per_cm is None:
            print("無法取得像素比例，使用預設值")
            self.pixel_per_cm = 30

        # 將公分轉換為像素
        square_size_pixels = int(square_size_cm * self.pixel_per_cm)
        # print(f"方塊大小: {square_size_cm}cm = {square_size_pixels} 像素")

        self.result_with_squares = self.original.copy()

        for i, point in enumerate(self.points):
            x, y = point
            half_size = square_size_pixels // 2

            # 計算方塊的四個角點
            top_left = (x - half_size, y - half_size)
            bottom_right = (x + half_size, y + half_size)

            # # 畫方塊（綠色外框）
            # cv2.rectangle(
            #     self.result_with_squares, top_left, bottom_right, (0, 255, 0), 3
            # )

            # 在中心畫一個小圓點標記檢測到的點（紅色）
            cv2.circle(self.result_with_squares, (x, y), 5, (0, 0, 255), -1)

            # 添加標籤顯示點的編號
            cv2.putText(
                self.result_with_squares,
                f"P{i+1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        return self.result_with_squares

    def show_results(self):
        """顯示結果"""
        if self.original is None:
            print("沒有載入圖像")
            return

        # 調整顯示大小
        def resize_for_display(img, max_width=800):
            height, width = img.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                return cv2.resize(img, (new_width, new_height))
            return img

        original_display = resize_for_display(self.original)
        # cv2.imshow("original_display", original_display)

        if self.result_with_squares is not None:
            result_display = resize_for_display(self.result_with_squares)
            # cv2.imshow("Points with Squares", result_display)
            print(f"檢測到 {len(self.points)} 個點")
            # if self.pixel_per_cm:
            #     print(
            #         f"每個方塊大小: 8cm x 8cm ({int(8 * self.pixel_per_cm)} x {int(8 * self.pixel_per_cm)} 像素)"
            #     )

        if self.eroded is not None:
            eroded_display = resize_for_display(self.eroded)
            cv2.imshow("Eroded Image", eroded_display)
        print("按任意鍵關閉視窗...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_result(self, output_path=None):
        """儲存結果"""
        if self.result_with_squares is None:
            print("沒有結果可儲存")
            return

        if output_path is None:
            name = self.image_path.split("\\")[-1].split(".")[0].split("_")[0]
            output_path = f"points\\{name}_points.jpg"

        cv2.imwrite(output_path, self.result_with_squares)
        print(f"結果已儲存為 '{output_path}'")

        cv2.imwrite(rf"eroded/{name}_eroded.jpg", self.eroded)
        return output_path


if __name__ == "__main__":
    img = 1
    image_path = rf"img\{img}.jpg"
    detector_path = rf"extracted\{img}_extracted_paper.jpg"
    json_path = rf"px2cm.json"

    # 接下來檢測點
    point_detector = PointDetector(detector_path, image_path)
    point_detector.load_image(json_path=json_path)  # 使用 px2cm.json 中的像素比例
    points = point_detector.detect_points()
    if len(points) > 0:
        point_detector.draw_squares_at_points(square_size_cm=8)  # 畫出8cm的方塊
        point_detector.show_results()
        point_detector.save_result()
    else:
        cv2.imshow("eroded", point_detector.eroded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("未檢測到任何點")
