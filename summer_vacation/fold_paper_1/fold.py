import cv2
import numpy as np
import os


class AdvancedFoldDetector:
    def __init__(self):
        """進階摺痕檢測器 - 包含多種檢測方法"""
        pass

    def preprocess_image(self, image):
        """預處理圖像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊去除噪音
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        return blurred

    def adaptive_threshold(self, image_path, max_value=255, block_size=11, C=2):
        """自適應閾值檢測摺痕"""
        # 載入圖像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法載入圖像: {image_path}")
        # enhanced_image = enhance_image(image, contrast=1.5, saturation=1.3)
        # cv2.imshow("Enhanced Image", enhanced_image)
        processed = self.preprocess_image(image)
        cv2.imshow("Processed", processed)
        adaptive = cv2.adaptiveThreshold(
            processed,
            max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )
        cv2.imshow("Adaptive", adaptive)
        # 溫和的線條連接 - 避免線條消失
        # 只用很小的核心進行輕微的閉運算
        kernel_small = np.ones((2, 2), np.uint8)
        adaptive = cv2.morphologyEx(
            adaptive, cv2.MORPH_CLOSE, kernel_small, iterations=1
        )

        # 或者嘗試用極小的線性核心
        kernel_h = np.ones((1, 3), np.uint8)
        kernel_v = np.ones((3, 1), np.uint8)

        adaptive_h = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_h, iterations=1)
        adaptive_v = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        adaptive = cv2.bitwise_and(adaptive, cv2.bitwise_or(adaptive_h, adaptive_v))

        # 膨脹與侵蝕
        kernel = np.ones((2, 2), np.uint8)
        adaptive = cv2.dilate(adaptive, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        adaptive = cv2.dilate(adaptive, kernel, iterations=1)
        adaptive = cv2.erode(adaptive, kernel, iterations=1)

        name = os.path.basename(image_path).split("_")[0].split(".")[0]
        # 反轉顏色以突出暗線
        folder = "adaptive"
        # 確保資料夾存在
        os.makedirs(folder, exist_ok=True)
        adaptive = cv2.bitwise_not(adaptive)
        path = os.path.join(folder, f"{name}_adaptive.jpg")
        cv2.imwrite(path, adaptive)
        print(f"圖片已儲存至: {path}")
        cv2.imshow("Adaptive Threshold", adaptive)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return adaptive


def enhance_image(image, contrast=1.5, saturation=1.3):
    """
    增強圖片的對比度和飽和度

    Args:
        image: 輸入圖片
        contrast: 對比度倍數 (1.0為原始值)
        saturation: 飽和度倍數 (1.0為原始值)
    """
    # 調整對比度 (對比度 = alpha * 原圖 + beta)
    enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    # 轉換到HSV色彩空間以調整飽和度
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)

    # 調整飽和度 (S通道)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # 轉換回BGR
    hsv = hsv.astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced


if __name__ == "__main__":

    detector = AdvancedFoldDetector()

    # 處理第一個找到的圖像
    for i in [10]:
        image_path = rf"extracted\{i}_extracted_paper.jpg"
        print(f"正在處理圖像: {image_path}")
        detector.adaptive_threshold(image_path)
