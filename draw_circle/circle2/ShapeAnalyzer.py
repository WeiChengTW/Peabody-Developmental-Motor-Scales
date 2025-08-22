import cv2
import numpy as np
import math
import os


class ShapeAnalyzer:
    def __init__(self, image_path="extracted_paper.jpg"):
        self.image_path = image_path
        self.img = None
        self.result_img = None

    def process(self, debug=False):

        self.img = cv2.imread(self.image_path)
        # cv2.imshow("original", self.img)
        if self.img is None:
            print("無法讀取圖片，請檢查路徑是否正確")
            return
        self.result_img = self.img.copy()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # cv2.imshow("blurred", blurred)
        # 使用自適應二值化取代 Canny 邊緣檢測
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2
        )

        # cv2.imshow("adaptive_thresh", adaptive_thresh)

        # 反轉圖像，讓物體變成白色，背景變成黑色
        thresh = cv2.bitwise_not(adaptive_thresh)

        # 形態學操作：先連接斷線，再去除雜訊
        # 使用閉運算連接斷開的線條
        kernel_close = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 再用開運算去除小雜訊
        kernel_open = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # 最後再用閉運算確保圖形完整性
        # kernel_final = np.ones((5, 5), np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_final, iterations=1)
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 根據輸入圖片名稱動態生成二值化圖像的保存路徑
        base_name = os.path.basename(self.image_path)
        img_prefix = base_name.split("_")[0]
        thresh_path = f"thresh/{img_prefix}_thresh.jpg"
        cv2.imwrite(thresh_path, thresh)
        if debug:
            cv2.imshow("original", self.img)
            cv2.imshow("adaptive_thresh", adaptive_thresh)
            cv2.imshow("thresh", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return thresh_path, thresh, self.result_img, adaptive_thresh


# test_img\img3_extracted_paper.jpg
if __name__ == "__main__":
    img = "S__10280963.jpg"
    detector = ShapeAnalyzer(image_path=f"{img}")
    thresh_img_path, thresh, result_img, adaptive_thresh = detector.process(debug=True)
    print(f"二值化圖像已保存到: {thresh_img_path}")
