import cv2
import numpy as np


class PaperDetector_edges:
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

        # 轉換為灰度圖像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny邊緣檢測
        edges = cv2.Canny(blurred, 20, 100)

        # 形態學操作來連接斷開的邊緣
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

        # 縮小顯示邊緣圖像視窗
        height, width = edges.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            edges_resized = cv2.resize(edges, (new_width, new_height))
        else:
            edges_resized = edges
        # cv2.imshow("edges", edges_resized)
        # 尋找輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 找到最大面積的輪廓作為紙張
        paper_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # 設定最小面積閾值，過濾掉太小的輪廓
            if area > 1000:
                if area > max_area:
                    max_area = area
                    paper_contour = contour

        if paper_contour is not None:
            # 多邊形近似，簡化輪廓
            epsilon = 0.02 * cv2.arcLength(paper_contour, True)
            paper_contour = cv2.approxPolyDP(paper_contour, epsilon, True)

        result_image = image.copy()
        # if paper_contour is not None:
        #     cv2.drawContours(result_image, [paper_contour], -1, (0, 0, 255), 3)
        #     print(f"檢測到紙張區域，面積: {max_area}")
        #     print(f"紙張輪廓點數: {len(paper_contour)}")
        # else:
        #     print("未檢測到明顯的紙張區域")

        # 可選：顯示邊緣檢測結果
        # 縮小顯示邊緣圖像視窗
        height, width = edges.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            edges_resized = cv2.resize(edges, (new_width, new_height))
        else:
            edges_resized = edges
        cv2.imshow("edges", edges_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.original = image
        self.result = result_image
        self.contour = paper_contour
        name = self.image_path.split("\\")[-1].split(".")[0]

        result_path = f"edges\{name}_edges_paper.jpg"
        cv2.imwrite(result_path, edges)
        print(f"結果已儲存為 '{result_path}'")

        return result_path


if __name__ == "__main__":
    image_path = r"img\4.jpg"
    detector = PaperDetector_edges(image_path)
    detector.detect_paper_by_color()
