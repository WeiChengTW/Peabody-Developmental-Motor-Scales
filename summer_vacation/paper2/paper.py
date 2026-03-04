import cv2
import os
import numpy as np
from pathlib import Path


def analyze_single_image(image_path):
    """
    分析單張圖片中的物體輪廓

    Args:
        image_path (str): 圖片檔案路徑
    """
    print(f"\n正在分析圖片: {os.path.basename(image_path)}")
    print("-" * 50)

    # 直接檢測物體輪廓並標示長邊
    detect_object_contours(image_path)


def detect_object_contours(image_path):
    """
    檢測圖片中物體的輪廓並標示長邊

    Args:
        image_path (str): 圖片檔案路徑
    """
    try:
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print("無法讀取圖片來檢測輪廓")
            return

        # 創建圖片副本用於繪製
        image_with_contours = image.copy()

        # 轉換為HSV色彩空間來檢測白色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定義白色的HSV範圍
        # 白色在HSV中的特徵：低飽和度和高明度
        lower_white = np.array([0, 0, 0])  # 下限：任何色相，低飽和度，高明度
        upper_white = np.array([255, 30, 255])  # 上限：任何色相，較低飽和度，最高明度

        # 創建白色遮罩
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 形態學操作來去除噪點和填充孔洞
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # 邊緣檢測（在白色遮罩上）
        edges = cv2.Canny(white_mask, 50, 150)

        # 顯示白色遮罩用於調試
        cv2.imshow("White Mask", white_mask)
        # 顯示白色遮罩用於調試
        cv2.imshow("White Mask", white_mask)
        cv2.imshow("Edges", edges)

        # 形態學操作來連接邊緣
        kernel_edge = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge)

        # 尋找輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"找到 {len(contours)} 個輪廓")

        # 過濾小輪廓，專注於較大的白色紙面
        min_area = 2000  # 提高最小面積閾值，因為紙面通常較大
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        print(f"有效輪廓數量: {len(valid_contours)}")

        # 如果還是沒找到，降低閾值但仍保持合理大小
        if not valid_contours and len(contours) > 0:
            min_area = 1000  # 適度降低閾值
            valid_contours = [
                cnt for cnt in contours if cv2.contourArea(cnt) > min_area
            ]
            print(f"降低閾值後有效輪廓數量: {len(valid_contours)}")

        # 進一步篩選：檢查輪廓是否主要在白色區域內
        if valid_contours:
            filtered_contours = []
            for contour in valid_contours:
                # 創建輪廓遮罩
                contour_mask = np.zeros(white_mask.shape, np.uint8)
                cv2.fillPoly(contour_mask, [contour], 255)

                # 計算輪廓區域內白色像素的比例
                overlap = cv2.bitwise_and(white_mask, contour_mask)
                white_ratio = np.sum(overlap > 0) / np.sum(contour_mask > 0)

                # 如果白色比例超過閾值，則認為是白色紙面
                if white_ratio > 0.7:  # 70%以上是白色
                    filtered_contours.append(contour)
                    print(f"輪廓白色比例: {white_ratio:.2f}")

            valid_contours = filtered_contours
            print(f"篩選後的白色輪廓數量: {len(valid_contours)}")

        if not valid_contours:
            print("未找到有效的物體輪廓")
            print("顯示原圖以供檢查...")
            show_result_image(image, f"Original Image - {os.path.basename(image_path)}")
            return

        # 分析每個輪廓
        for i, contour in enumerate(valid_contours):
            # 計算輪廓的最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # 獲取矩形的寬度和高度
            width = rect[1][0]
            height = rect[1][1]

            # 計算長邊和短邊
            long_side = max(width, height)
            short_side = min(width, height)

            # 以長邊作為比例尺，長邊 = 15 公分
            scale_factor = 15.0 / long_side  # 公分/像素
            short_side_cm = short_side * scale_factor

            print(f"\n物體 {i+1}:")
            print(f"  輪廓面積: {cv2.contourArea(contour):.1f} 平方像素")
            print(f"  外接矩形寬度: {width:.1f} 像素")
            print(f"  外接矩形高度: {height:.1f} 像素")
            print(f"  長邊: {long_side:.1f} 像素 (= 15.0 公分)")
            print(f"  短邊: {short_side:.1f} 像素 (= {short_side_cm:.2f} 公分)")
            print(f"  長寬比: {long_side/short_side:.2f}")
            print(f"  比例尺: {scale_factor:.6f} 公分/像素")

            # 繪製輪廓
            cv2.drawContours(
                image_with_contours, [contour], -1, (0, 255, 0), 2
            )  # 綠色輪廓

            # 繪製最小外接矩形
            cv2.drawContours(image_with_contours, [box], -1, (255, 0, 0), 2)  # 藍色矩形

            # 計算並繪製短邊
            draw_object_short_side(image_with_contours, rect, i + 1, scale_factor)

        # 顯示圖片
        show_result_image(
            image_with_contours,
            f"Object Measurement - {os.path.basename(image_path)}",
        )

    except Exception as e:
        print(f"檢測物體輪廓時發生錯誤: {e}")


def draw_object_short_side(image, rect, object_num, scale_factor):
    """
    在物體上繪製短邊紅線

    Args:
        image: OpenCV圖片對象
        rect: 最小外接矩形 (center, (width, height), angle)
        object_num (int): 物體編號
        scale_factor (float): 比例尺 (公分/像素)
    """
    try:
        center, (width, height), angle = rect

        # 計算短邊
        short_side = min(width, height)
        short_side_cm = short_side * scale_factor
        is_width_shorter = width <= height

        # 轉換角度到弧度
        angle_rad = np.radians(angle)

        # 如果高度是短邊，需要調整角度
        if not is_width_shorter:
            angle_rad += np.pi / 2

        # 計算短邊線的端點
        half_length = short_side / 2
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        start_point = (
            int(center[0] - half_length * cos_angle),
            int(center[1] - half_length * sin_angle),
        )
        end_point = (
            int(center[0] + half_length * cos_angle),
            int(center[1] + half_length * sin_angle),
        )

        # 繪製短邊紅線
        cv2.line(image, start_point, end_point, (0, 0, 255), 3)  # 紅色粗線

        # 在兩端添加圓點標記
        cv2.circle(image, start_point, 5, (0, 0, 255), -1)
        cv2.circle(image, end_point, 5, (0, 0, 255), -1)

        # 添加文字標籤（顯示像素和公分）
        text = f"Obj{object_num}: {short_side:.1f}px ({short_side_cm:.2f}cm)"
        text_position = (int(center[0] + 10), int(center[1] - 10))

        # 添加文字背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]

        cv2.rectangle(
            image,
            (text_position[0] - 2, text_position[1] - text_size[1] - 2),
            (text_position[0] + text_size[0] + 2, text_position[1] + 2),
            (255, 255, 255),
            -1,
        )  # 白色背景

        # 添加文字
        cv2.putText(
            image, text, text_position, font, font_scale, (0, 0, 255), text_thickness
        )

        print(
            f"  已繪製物體 {object_num} 的短邊紅線: {short_side:.1f} 像素 ({short_side_cm:.2f} 公分)"
        )

    except Exception as e:
        print(f"繪製物體短邊時發生錯誤: {e}")


def draw_long_side_line(image_path, dimensions):
    """
    在圖片上畫出紅線表示長邊

    Args:
        image_path (str): 圖片檔案路徑
        dimensions (dict): 圖片尺寸資訊
    """
    try:
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print("無法讀取圖片來繪製長邊線")
            return

        # 創建圖片副本用於繪製
        image_with_line = image.copy()

        width = dimensions["width"]
        height = dimensions["height"]

        # 設定線條參數
        line_color = (0, 0, 255)  # 紅色 (BGR格式)
        line_thickness = 5

        # 根據長邊方向畫線
        if width >= height:
            # 寬度是長邊，畫水平線
            start_point = (0, height // 2)
            end_point = (width, height // 2)
            print(f"在圖片中央畫水平紅線 (長邊: {width} 像素)")
        else:
            # 高度是長邊，畫垂直線
            start_point = (width // 2, 0)
            end_point = (width // 2, height)
            print(f"在圖片中央畫垂直紅線 (長邊: {height} 像素)")

        # 畫線
        cv2.line(image_with_line, start_point, end_point, line_color, line_thickness)

        # 在線的兩端加上箭頭標記
        arrow_size = 20
        if width >= height:
            # 水平線的箭頭
            cv2.arrowedLine(
                image_with_line,
                (arrow_size, height // 2),
                (0, height // 2),
                line_color,
                line_thickness,
            )
            cv2.arrowedLine(
                image_with_line,
                (width - arrow_size, height // 2),
                (width, height // 2),
                line_color,
                line_thickness,
            )
        else:
            # 垂直線的箭頭
            cv2.arrowedLine(
                image_with_line,
                (width // 2, arrow_size),
                (width // 2, 0),
                line_color,
                line_thickness,
            )
            cv2.arrowedLine(
                image_with_line,
                (width // 2, height - arrow_size),
                (width // 2, height),
                line_color,
                line_thickness,
            )

        # 在圖片上添加文字標註
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_color = (0, 0, 255)  # 紅色
        text_thickness = 2

        # 準備文字內容
        long_side_text = f"Long Side: {dimensions['long_side']} px"

        # 計算文字位置
        text_size = cv2.getTextSize(long_side_text, font, font_scale, text_thickness)[0]
        text_x = 10
        text_y = 30

        # 添加文字背景
        cv2.rectangle(
            image_with_line,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (255, 255, 255),
            -1,
        )  # 白色背景

        # 添加文字
        cv2.putText(
            image_with_line,
            long_side_text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            text_thickness,
        )
        # 顯示圖片 (可選)
        show_result_image(
            image_with_line, f"Long Side Analysis - {os.path.basename(image_path)}"
        )

    except Exception as e:
        print(f"繪製長邊線時發生錯誤: {e}")


def show_result_image(image, window_name):
    """
    顯示結果圖片

    Args:
        image: OpenCV圖片對象
        window_name (str): 視窗名稱
    """
    try:
        # 計算顯示尺寸 (如果圖片太大則縮放)
        height, width = image.shape[:2]
        max_display_size = 800

        if max(width, height) > max_display_size:
            scale = max_display_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
        else:
            image_resized = image

        cv2.imshow(window_name, image_resized)
        print(f"顯示圖片視窗: {window_name}")
        print("按任意鍵關閉圖片視窗並繼續...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("圖片視窗已關閉\n")

    except Exception as e:
        print(f"顯示圖片時發生錯誤: {e}")


def analyze_folder(folder_path):
    """
    分析資料夾中的所有圖片

    Args:
        folder_path (str): 資料夾路徑
    """
    # 支援的圖片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    folder = Path(folder_path)

    if not folder.exists():
        print(f"資料夾不存在: {folder_path}")
        return

    # 獲取所有圖片檔案
    image_files = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"在 {folder_path} 中找不到圖片檔案")
        return

    print(f"找到 {len(image_files)} 張圖片")

    # 分析每張圖片中的物體輪廓
    for image_file in sorted(image_files):

        analyze_single_image(str(image_file))


def main():
    """
    主程式
    """
    print("物體輪廓短邊測量工具 (長邊=15公分比例尺)")
    print("=" * 60)

    # 設定測試圖片資料夾路徑
    test_folder = r"paper2\test_img"

    # 檢查是否存在測試資料夾
    if os.path.exists(test_folder):
        print(f"分析資料夾: {test_folder}")
        analyze_folder(test_folder)
    else:
        # 如果沒有測試資料夾，可以分析單張圖片
        print("找不到 test_img 資料夾")
        print("請將圖片放入 test_img 資料夾中，或修改程式碼指定圖片路徑")

        # 示例：分析單張圖片
        # image_path = "your_image.jpg"  # 替換為您的圖片路徑
        # if os.path.exists(image_path):
        #     analyze_single_image(image_path)


if __name__ == "__main__":
    main()
