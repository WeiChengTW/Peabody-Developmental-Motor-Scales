import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def calculate_line_lengths(image_path):
    """
    計算圖像中線段的像素長度
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print("無法讀取圖像文件")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 獲取圖像尺寸
    height, width = image.shape
    print(f"圖像尺寸: {width} x {height}")

    # 定義邊框區域（距離邊緣的像素數）
    border_margin = 30  # 可以調整這個值

    # 方法1: 使用輪廓檢測計算長度
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\n=== 方法1: 輪廓檢測結果 ===")
    print(f"檢測到 {len(contours)} 個輪廓")

    # 過濾掉邊框輪廓
    filtered_contours = []
    contour_lengths = []
    for i, contour in enumerate(contours):
        # 檢查輪廓是否靠近邊框
        x, y, w, h = cv2.boundingRect(contour)

        # 如果輪廓的邊界矩形靠近圖像邊緣，跳過它
        if (
            x < border_margin
            or y < border_margin
            or x + w > width - border_margin
            or y + h > height - border_margin
        ):
            print(f"輪廓 {i+1} 靠近邊框，已過濾")
            continue

        # 計算輪廓的弧長
        length = cv2.arcLength(contour, False)
        contour_lengths.append(length)
        filtered_contours.append(contour)
        print(f"輪廓 {len(filtered_contours)} 長度: {length:.2f} 像素")

    print(f"過濾後剩餘 {len(filtered_contours)} 個輪廓")

    # 方法2: 使用Hough線段檢測
    print(f"\n=== 方法2: Hough線段檢測結果 ===")

    # 邊緣檢測
    edges = cv2.Canny(image, 70, 150, apertureSize=3)
    # 形態學侵蝕操作
    erode_kernel = np.ones((1, 1), np.uint8)
    eroded = cv2.erode(edges, erode_kernel, iterations=1)
    # cv2.imshow("Eroded Edges", eroded)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Hough線段檢測
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=45, maxLineGap=7
    )

    # 過濾線段：排除邊框線段和過短線段
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 檢查線段是否靠近邊框
            if (
                x1 < border_margin
                or y1 < border_margin
                or x2 < border_margin
                or y2 < border_margin
                or x1 > width - border_margin
                or y1 > height - border_margin
                or x2 > width - border_margin
                or y2 > height - border_margin
            ):
                continue  # 跳過靠近邊框的線段

            # 計算線段長度
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # 只保留長度大於某個閾值的線段
            if length > 100:  # 過濾掉太短的線段
                filtered_lines.append(line)

    print(f"過濾後剩餘 {len(filtered_lines)} 條線段")
    hough_lengths = []
    if filtered_lines:
        print(f"檢測到 {len(filtered_lines)} 條線段")
        for i, line in enumerate(filtered_lines):
            x1, y1, x2, y2 = line[0]
            # 計算歐幾里得距離
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            hough_lengths.append(length)
            print(
                f"線段 {i+1}: 起點({x1},{y1}) -> 終點({x2},{y2}), 長度: {length:.2f} 像素"
            )
    else:
        print("未檢測到線段")

    # 方法3: 骨架化方法計算像素數量
    print(f"\n=== 方法3: 骨架化像素計數 ===")

    # 形態學操作獲得線條骨架
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)

    # 細化處理
    skeleton = cv2.ximgproc.thinning(skeleton) if hasattr(cv2, "ximgproc") else skeleton

    # 計算白色像素總數
    white_pixels = np.sum(skeleton == 255)
    print(f"總白色像素數: {white_pixels}")

    # 視覺化結果
    plt.figure(figsize=(15, 5))

    # 原始圖像
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("原始圖像 edges")
    plt.axis("off")

    # 檢測結果可視化
    plt.subplot(1, 2, 2)
    result_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # 繪製輪廓
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)

    # 繪製Hough線段（使用過濾後的線段）
    if filtered_lines:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(result_image)
    plt.title("檢測結果 (綠色:輪廓, 紅色:Hough線段) edges")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 總結結果
    print(f"\n=== 總結 ===")
    if contour_lengths:
        print(f"輪廓方法平均長度: {np.mean(contour_lengths):.2f} 像素")
        print(f"輪廓長度列表: {[f'{l:.2f}' for l in sorted(contour_lengths)]}")

    if hough_lengths:
        print(f"Hough方法平均長度: {np.mean(hough_lengths):.2f} 像素")
        print(f"Hough長度列表: {[f'{l:.2f}' for l in sorted(hough_lengths)]}")

    return {
        "contour_lengths": contour_lengths,
        "hough_lengths": hough_lengths,
        "total_pixels": white_pixels,
    }


# 使用方法
if __name__ == "__main__":
    # 替換為您的圖像路徑
    image_path = r"adaptive\7_adaptive.jpg"

    try:
        results = calculate_line_lengths(image_path)

        # 如果需要保存結果到文件
        with open("line_lengths_results.txt", "w", encoding="utf-8") as f:
            f.write("線段長度分析結果\n")
            f.write("================\n\n")

            if results["contour_lengths"]:
                f.write("輪廓檢測結果:\n")
                for i, length in enumerate(results["contour_lengths"]):
                    f.write(f"線段 {i+1}: {length:.2f} 像素\n")
                f.write(f"平均長度: {np.mean(results['contour_lengths']):.2f} 像素\n\n")

            if results["hough_lengths"]:
                f.write("Hough變換檢測結果:\n")
                for i, length in enumerate(results["hough_lengths"]):
                    f.write(f"線段 {i+1}: {length:.2f} 像素\n")
                f.write(f"平均長度: {np.mean(results['hough_lengths']):.2f} 像素\n\n")

            f.write(f"總白色像素數: {results['total_pixels']}\n")

        print("\n結果已保存到 line_lengths_results.txt")

    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
