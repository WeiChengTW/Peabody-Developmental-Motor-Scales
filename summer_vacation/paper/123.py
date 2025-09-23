import cv2
import numpy as np
from PIXEL_TO_CM import get_pixel_per_cm

IMG_PATH = "img5.png"  # 請替換為您的圖像路徑


def get_paper_bottom_from_contour(contour):
    """從白紙輪廓計算實際的下緣位置"""
    # 找到輪廓中y座標最大的點（最下方的點）
    max_y = 0
    for point in contour:
        y = point[0][1]
        if y > max_y:
            max_y = y
    return max_y


def find_largest_contour(image):
    """找到畫面中最大面積的物體（白紙）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化處理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 找到所有輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # 找到面積最大的輪廓（白紙）
    largest_contour = max(contours, key=cv2.contourArea)

    # 獲取邊界矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 計算白紙的實際下緣位置（基於輪廓而非矩形）
    paper_bottom = get_paper_bottom_from_contour(largest_contour)

    return largest_contour, (x, y, w, h), paper_bottom


def detect_lines(image, paper_bbox):
    """檢測圖像中的線條"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 只在白紙區域內檢測
    x, y, w, h = paper_bbox
    roi = gray[y : y + h, x : x + w]

    # Canny邊緣檢測
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)

    # 霍夫線變換檢測線條
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
    )

    if lines is None:
        return []

    # 將線條座標轉換回原圖座標系
    adjusted_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        adjusted_lines.append([[x1 + x, y1 + y, x2 + x, y2 + y]])

    return adjusted_lines


def classify_lines(lines, paper_bbox, paper_bottom):
    """分類線條：水平線（可能是紙張邊緣）和其他線條（黑線）"""
    x, y, w, h = paper_bbox
    paper_top = y
    paper_left = x
    paper_right = x + w

    horizontal_lines = []
    black_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 計算線條角度
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        # 計算線條中點
        mid_y = (y1 + y2) / 2
        mid_x = (x1 + x2) / 2

        # 計算線條長度
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 判斷是否為白紙邊緣線條
        is_paper_edge = False

        # 檢查是否為水平邊緣線（上緣或下緣）
        if abs(angle) < 15 or abs(angle) > 165:  # 水平線
            # 檢查是否接近紙張上緣
            if abs(mid_y - paper_top) < 15:
                is_paper_edge = True
            # 檢查是否為很長的線條且接近下緣（可能是紙張邊緣）
            elif abs(mid_y - paper_bottom) < 10 and line_length > w * 0.7:
                is_paper_edge = True
            # 檢查線條是否很長且橫跨大部分紙張寬度（可能是紙張邊緣）
            elif line_length > w * 0.8:
                is_paper_edge = True

        # 檢查是否為垂直邊緣線（左緣或右緣）
        elif abs(angle - 90) < 15 or abs(angle + 90) < 15:  # 垂直線
            # 檢查是否接近紙張左緣或右緣
            if abs(mid_x - paper_left) < 15 or abs(mid_x - paper_right) < 15:
                is_paper_edge = True
            # 檢查線條是否很長（可能是紙張邊緣）
            elif line_length > h * 0.7:
                is_paper_edge = True

        # 分類線條
        if is_paper_edge:
            horizontal_lines.append(line)
        else:
            # 所有其他線條都視為黑線，包括接近下緣的短線條
            black_lines.append(line)

    return horizontal_lines, black_lines


def calculate_distance_to_paper_bottom(black_lines, paper_bbox, paper_bottom):
    """計算黑線到白紙下緣的像素距離"""
    if not black_lines:
        return None

    distances = []

    for line in black_lines:
        x1, y1, x2, y2 = line[0]

        # 計算線條中點
        mid_y = (y1 + y2) / 2

        # 計算到紙張實際下緣的距離
        distance = abs(paper_bottom - mid_y)
        distances.append(distance)

    # 返回最近的距離
    return min(distances) if distances else None


def draw_results(image, paper_contour, paper_bbox, black_lines, distance, paper_bottom):
    """在圖像上繪製檢測結果"""
    result_image = image.copy()

    # 繪製白紙輪廓
    cv2.drawContours(result_image, [paper_contour], -1, (0, 255, 0), 2)

    # 繪製紙張實際下緣線（基於輪廓）
    x, y, w, h = paper_bbox
    cv2.line(result_image, (x, paper_bottom), (x + w, paper_bottom), (0, 0, 255), 3)

    # 繪製檢測到的黑線
    for line in black_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # 顯示距離信息
    if distance is not None:
        pixel_per_cm_result = get_pixel_per_cm(IMG_PATH, cm_length=16, show=False)
        if pixel_per_cm_result and len(pixel_per_cm_result) > 0:
            pixel_per_cm = pixel_per_cm_result[0]  # 取第一個元素作為 pixel_per_cm
            distance_cm = distance / pixel_per_cm
        else:
            distance_cm = 0
        cv2.putText(
            result_image,
            f"Distance: {distance:.1f} pixels ({distance_cm:.2f} cm)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
    print(f"1 pxel = {1/pixel_per_cm:.2f} cm")  # 顯示像素與公分的比例
    return result_image


def main():
    """主函數"""
    # 讀取圖像
    image_path = IMG_PATH
    image = cv2.imread(image_path)

    if image is None:
        print("無法讀取圖像，請檢查檔案路徑")
        return

    # 找到最大面積物體（白紙）
    paper_contour, paper_bbox, paper_bottom = find_largest_contour(image)

    if paper_contour is None:
        print("未找到白紙")
        return

    print(f"白紙邊界框: {paper_bbox}")
    print(f"白紙實際下緣位置: {paper_bottom}")

    # 檢測線條
    lines = detect_lines(image, paper_bbox)
    print(f"檢測到 {len(lines)} 條線")

    # 分類線條
    horizontal_lines, black_lines = classify_lines(lines, paper_bbox, paper_bottom)
    print(f"水平線: {len(horizontal_lines)} 條")
    print(f"黑線: {len(black_lines)} 條")

    # 計算距離
    distance = calculate_distance_to_paper_bottom(black_lines, paper_bbox, paper_bottom)

    if distance is not None:
        # print(f"黑線到白紙下緣的距離: {distance:.1f} 像素")
        print(f"黑線到白紙下緣的距離: {distance / 16:.2f} cm")  # 假設16cm為參考長度
    else:
        print("未找到黑線")

    # 繪製結果
    result_image = draw_results(
        image, paper_contour, paper_bbox, black_lines, distance, paper_bottom
    )

    # 顯示結果
    cv2.imshow("Detection Result", result_image)

    # 等待按鍵
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return distance


if __name__ == "__main__":
    distance = main()
