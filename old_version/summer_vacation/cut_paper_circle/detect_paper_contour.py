import cv2
import numpy as np
import os


def detect_paper_contour(
    image_path, output_path=None, paper_width_cm=21.0, paper_height_cm=29.7
):
    """
    偵測圖片中紙張的輪廓並計算公分/像素比例

    Args:
        image_path (str): 輸入圖片路徑
        output_path (str, optional): 輸出圖片路徑，如果不指定則顯示結果
        paper_width_cm (float): 紙張實際寬度（公分），預設為A4紙寬度21.0cm
        paper_height_cm (float): 紙張實際高度（公分），預設為A4紙高度29.7cm

    Returns:
        tuple: (original_image, contours, largest_contour, cm_per_pixel_x, cm_per_pixel_y)
    """

    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"無法讀取圖片: {image_path}")

    # 複製原始圖片用於繪製
    original = image.copy()

    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 應用高斯模糊以減少雜訊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 邊緣偵測
    edges = cv2.Canny(blurred, 50, 150)

    # 進行形態學操作以連接斷裂的邊緣
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 尋找輪廓
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 過濾輪廓：找到面積最大的輪廓（假設是紙張）
    if len(contours) == 0:
        print("未找到任何輪廓")
        return original, [], None

    # 依照面積排序輪廓
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # 找到最大的四邊形輪廓（紙張通常是矩形）
    largest_contour = None
    for contour in contours_sorted:
        # 計算輪廓周長
        perimeter = cv2.arcLength(contour, True)
        # 近似多邊形
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 如果輪廓有4個點且面積足夠大，很可能是紙張
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > 1000:  # 面積閾值可調整
            largest_contour = approx
            break

    # 如果沒有找到四邊形，就使用面積最大的輪廓
    if largest_contour is None:
        largest_contour = contours_sorted[0]

    # 在原始圖片上繪製輪廓
    result_image = original.copy()
    cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 3)

    # 如果輪廓是四邊形，標記角點
    if len(largest_contour) == 4:
        for point in largest_contour:
            cv2.circle(result_image, tuple(point[0]), 8, (255, 0, 0), -1)

    # 使用 cv2.imshow 顯示結果
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"結果已保存至: {output_path}")

    # 顯示原始圖片
    cv2.imshow("原始圖片", original)

    # 顯示邊緣偵測結果
    cv2.imshow("邊緣偵測", edges)

    # 顯示輪廓偵測結果
    cv2.imshow("紙張輪廓偵測", result_image)

    print("按任意鍵關閉視窗...")
    cv2.waitKey(0)  # 等待按鍵
    cv2.destroyAllWindows()  # 關閉所有視窗

    # 輸出輪廓資訊
    print(f"找到 {len(contours)} 個輪廓")
    print(f"最大輪廓面積: {cv2.contourArea(largest_contour):.2f} 像素")
    print(f"最大輪廓周長: {cv2.arcLength(largest_contour, True):.2f} 像素")

    # 計算公分/像素比例
    cm_per_pixel_x = None
    cm_per_pixel_y = None

    if len(largest_contour) == 4:
        print("偵測到四邊形紙張輪廓")
        print("角點座標:")
        for i, point in enumerate(largest_contour):
            print(f"  角點 {i+1}: ({point[0][0]}, {point[0][1]})")

        # 計算紙張在圖片中的像素尺寸
        # 重新排序角點：左上、右上、右下、左下
        points = largest_contour.reshape(4, 2)
        # 計算質心
        center = np.mean(points, axis=0)

        # 根據相對於質心的位置排序角點
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])

        sorted_points = sorted(points, key=angle_from_center)

        # 重新排列為：左上、右上、右下、左下
        # 先按y座標排序，再按x座標排序
        sorted_by_y = sorted(sorted_points, key=lambda p: p[1])
        top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])  # 上方兩點按x排序
        bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])  # 下方兩點按x排序

        top_left, top_right = top_points
        bottom_left, bottom_right = bottom_points

        # 計算寬度和高度（像素）
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        # 取平均值
        avg_width_pixels = (width_top + width_bottom) / 2
        avg_height_pixels = (height_left + height_right) / 2

        # 計算公分/像素比例
        cm_per_pixel_x = paper_width_cm / avg_width_pixels
        cm_per_pixel_y = paper_height_cm / avg_height_pixels

        print(f"\n=== 尺寸計算結果 ===")
        print(f"紙張實際尺寸: {paper_width_cm} x {paper_height_cm} 公分")
        print(f"紙張像素尺寸: {avg_width_pixels:.2f} x {avg_height_pixels:.2f} 像素")
        print(f"X方向比例: {cm_per_pixel_x:.6f} 公分/像素")
        print(f"Y方向比例: {cm_per_pixel_y:.6f} 公分/像素")
        print(f"平均比例: {(cm_per_pixel_x + cm_per_pixel_y) / 2:.6f} 公分/像素")
        print(f"解析度: {1/cm_per_pixel_x:.2f} x {1/cm_per_pixel_y:.2f} 像素/公分")

        # 在結果圖片上標記角點和尺寸
        cv2.putText(
            result_image,
            f"Top-Left",
            tuple(top_left.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            result_image,
            f"Top-Right",
            tuple(top_right.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            result_image,
            f"Bottom-Left",
            tuple(bottom_left.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            result_image,
            f"Bottom-Right",
            tuple(bottom_right.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # 在圖片上顯示比例資訊
        text_y = 30
        cv2.putText(
            result_image,
            f"cm/pixel: {cm_per_pixel_x:.6f} x {cm_per_pixel_y:.6f}",
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            result_image,
            f"Size: {avg_width_pixels:.1f} x {avg_height_pixels:.1f} px",
            (10, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    else:
        print(f"偵測到 {len(largest_contour)} 個點的輪廓")
        print("警告: 無法精確計算公分/像素比例，因為輪廓不是四邊形")

        # 嘗試用邊界框估算
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"邊界框尺寸: {w} x {h} 像素")

        # 假設較長的邊是高度，較短的邊是寬度
        if w > h:
            cm_per_pixel_x = paper_height_cm / w  # 較長邊對應高度
            cm_per_pixel_y = paper_width_cm / h  # 較短邊對應寬度
        else:
            cm_per_pixel_x = paper_width_cm / w  # 較短邊對應寬度
            cm_per_pixel_y = paper_height_cm / h  # 較長邊對應高度

        print(f"估算比例 - X方向: {cm_per_pixel_x:.6f} 公分/像素")
        print(f"估算比例 - Y方向: {cm_per_pixel_y:.6f} 公分/像素")

    return original, contours, largest_contour, cm_per_pixel_x, cm_per_pixel_y


def process_all_images_in_folder(folder_path):
    """
    處理資料夾中的所有圖片

    Args:
        folder_path (str): 資料夾路徑
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"\n處理圖片: {filename}")
            print("-" * 50)

            try:
                detect_paper_contour(image_path)
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤: {e}")


if __name__ == "__main__":
    # 設定當前工作目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # # 處理資料夾中的所有圖片
    # print("偵測資料夾中所有圖片的紙張輪廓...")
    # process_all_images_in_folder(current_dir)

    # 或者可以單獨處理特定圖片
    image_path = "1.jpg"  # 替換為您要處理的圖片檔名
    if os.path.exists(os.path.join(current_dir, image_path)):
        print(f"\n單獨處理圖片: {image_path}")
        # 可以指定紙張尺寸，預設為A4 (21.0 x 29.7 cm)
        # 如果是其他尺寸，請修改 paper_width_cm 和 paper_height_cm 參數
        result = detect_paper_contour(
            os.path.join(current_dir, image_path),
            paper_width_cm=21.0,  # A4紙寬度
            paper_height_cm=29.7,  # A4紙高度
        )

        if len(result) >= 5:
            original, contours, largest_contour, cm_per_pixel_x, cm_per_pixel_y = result
            if cm_per_pixel_x and cm_per_pixel_y:
                print(f"\n=== 可用於後續計算的比例 ===")
                print(f"X方向比例: {cm_per_pixel_x}")
                print(f"Y方向比例: {cm_per_pixel_y}")
    else:
        print(f"找不到圖片檔案: {image_path}")
        print("請確認檔案名稱和路徑是否正確")
