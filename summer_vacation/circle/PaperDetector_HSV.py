import cv2
import numpy as np


class PaperDetector_HSV:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = None
        self.result = None
        self.contour = None
        self.paper_region = None

    def detect_paper_by_color(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print("無法讀取圖像檔案")
            return None, None, None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median_value = np.median(gray)
        print(f"圖像亮度中位數: {median_value}")

        threshold_range = 30
        lower_threshold = max(0, median_value - threshold_range)
        upper_threshold = min(255, median_value + threshold_range)
        if median_value < 128:
            lower_threshold = max(0, median_value - 5)  # 允許比中位數稍暗的區域
            upper_threshold = 255
        print(f"紙張亮度範圍: {lower_threshold} - {upper_threshold}")

        mask = cv2.inRange(gray, lower_threshold, upper_threshold)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        paper_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    rectangularity = area / hull_area
                    if area > max_area and rectangularity > 0.7:
                        max_area = area
                        paper_contour = contour

        if paper_contour is not None:
            epsilon = 0.02 * cv2.arcLength(paper_contour, True)
            paper_contour = cv2.approxPolyDP(paper_contour, epsilon, True)

        result_image = image.copy()
        if paper_contour is not None:
            cv2.drawContours(result_image, [paper_contour], -1, (0, 0, 255), 3)
            print(f"檢測到紙張區域，面積: {max_area}")
            print(f"紙張輪廓點數: {len(paper_contour)}")
        else:
            print("未檢測到明顯的紙張區域")

        # cv2.imshow("mask", mask)
        self.original = image
        self.result = result_image
        self.contour = paper_contour
        return image, result_image, paper_contour

    def analyze_paper_color(self):
        if self.original is None:
            print("尚未載入圖像")
            return
        image = self.original
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("\n=== 紙張顏色分析 ===")
        print(
            f"灰度 - 中位數: {np.median(gray):.1f}, 平均: {np.mean(gray):.1f}, 標準差: {np.std(gray):.1f}"
        )
        h, s, v = cv2.split(hsv)
        # print(f"HSV H - 中位數: {np.median(h):.1f}")
        # print(f"HSV S - 中位數: {np.median(s):.1f}")
        # print(f"HSV V - 中位數: {np.median(v):.1f}")
        l, a, b = cv2.split(lab)
        # print(f"LAB L - 中位數: {np.median(l):.1f}")
        # print(f"LAB A - 中位數: {np.median(a):.1f}")
        # print(f"LAB B - 中位數: {np.median(b):.1f}")

    def show_results(self):
        if self.original is None or self.result is None:
            print("尚未有檢測結果")
            return
        original = self.original
        result = self.result
        height, width = original.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            original_resized = cv2.resize(original, (new_width, new_height))
            result_resized = cv2.resize(result, (new_width, new_height))
        else:
            original_resized = original
            result_resized = result
        cv2.imshow("original_resized", original_resized)
        cv2.imshow("result_resized", result_resized)
        print("按任意鍵關閉視窗...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_paper_region(self):
        if self.original is None or self.contour is None:
            print("無法提取紙張區域")
            return None
        image = self.original
        contour = self.contour
        if len(contour) != 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            contour = np.int0(box)
        points = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        width_a = np.linalg.norm(rect[0] - rect[1])
        width_b = np.linalg.norm(rect[2] - rect[3])
        max_width = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(rect[0] - rect[3])
        height_b = np.linalg.norm(rect[1] - rect[2])
        max_height = max(int(height_a), int(height_b))
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        self.paper_region = warped
        return warped

    def save_results(self):
        # if self.result is not None:
        #     cv2.imwrite(f"{self.image_path}detected_paper.jpg", self.result)
        name = self.image_path.split("/")[-1].split(".")[0]
        if self.paper_region is not None:
            cv2.imwrite(f"{name}_extracted_paper.jpg", self.paper_region)
        print(f"結果已儲存為 '{name}_extracted_paper.jpg'")
        return f"{name}_extracted_paper.jpg"


if __name__ == "__main__":
    image_path = r"test_img\\img6.jpg"
    detector = PaperDetector_HSV(image_path)
    detector.detect_paper_by_color()
    if detector.original is not None:
        detector.analyze_paper_color()
        detector.show_results()
        region = detector.extract_paper_region()
        if region is not None:
            height, width = region.shape[:2]
            if width > 600:
                scale = 600 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                region = cv2.resize(region, (new_width, new_height))
            # cv2.imshow("提取的紙張區域", region)
            # print("按任意鍵關閉視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            detector.save_results()
