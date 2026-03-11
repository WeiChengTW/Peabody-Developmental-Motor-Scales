import cv2
import numpy as np
import os


def detect_paper_contour(image_path, output_path=None):
    """
    檢測紙張輪廓並用藍線畫出
    """
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return None, None

    # 複製原圖用於繪製結果
    result_image = image.copy()

    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用雙邊濾波保持邊緣清晰的同時去噪
    filtered = cv2.bilateralFilter(gray, 11, 80, 80)

    # 自適應閾值處理
    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Canny邊緣檢測，使用較低的閾值獲取更多細節
    edges = cv2.Canny(filtered, 30, 80)

    # 形態學操作，連接斷開的邊緣但保持輪廓精度
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

    # 尋找輪廓，使用CHAIN_APPROX_NONE保持所有輪廓點
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 找到最大的輪廓（假設為紙張）
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 使用精確的輪廓，不進行多邊形近似
        paper_contour = largest_contour

        # 在結果圖像上畫藍色輪廓線（精確輪廓）
        cv2.drawContours(
            result_image, [paper_contour], -1, (255, 0, 0), 3
        )  # 藍色，線寬3

        print(f"檢測到紙張輪廓，包含 {len(paper_contour)} 個輪廓點")

        # 保存結果圖片
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"結果已保存到: {output_path}")

        return paper_contour, result_image
    else:
        print("未檢測到紙張輪廓")
        return None, result_image


def main():
    """測試函數"""
    input_dir = "img"
    output_dir = "result"

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 處理所有圖片
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"step1_{filename}")

            print(f"\n處理圖片: {filename}")
            contour, result = detect_paper_contour(input_path, output_path)

            if contour is not None:
                print(f"✓ 成功檢測紙張輪廓")
            else:
                print("✗ 未能檢測到紙張輪廓")


if __name__ == "__main__":
    main()
