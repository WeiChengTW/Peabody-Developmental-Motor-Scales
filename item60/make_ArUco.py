import cv2
import numpy as np
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from PIL import Image
import io


def create_aruco_marker(marker_id, marker_size=100):
    """
    創建ArUco標記
    """
    try:
        # 嘗試新版本的OpenCV API (4.x)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        try:
            # 嘗試舊版本的OpenCV API (3.x)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        except AttributeError:
            # 如果都不行，使用最新版本的API
            aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50)

    try:
        # 嘗試新版本的drawMarker方法
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    except AttributeError:
        # 使用舊版本的drawMarker方法
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)

    # 轉換為3通道圖像
    marker_img_3ch = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    return marker_img_3ch


def create_a4_with_marks():
    """
    在A4紙上分四份，每份中間產生一個標記
    A4尺寸: 210mm x 297mm (約 595 x 842 像素，以72 DPI計算)
    """
    # A4 紙張尺寸 (像素)
    a4_width = 595
    a4_height = 842

    # 創建白色背景
    img = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255

    # 計算分割線位置
    mid_x = a4_width // 2
    mid_y = a4_height // 2

    # 畫分割線 (黑色)
    line_color = (0, 0, 0)  # 黑色
    line_thickness = 4

    # 垂直分割線
    cv2.line(img, (mid_x, 0), (mid_x, a4_height), line_color, line_thickness)

    # 水平分割線
    # cv2.line(img, (0, mid_y), (a4_width, mid_y), line_color, line_thickness)

    # ArUco標記設定
    marker_size = 80  # ArUco標記大小

    # 四個區域的中心點
    centers = [
        (mid_x // 2, mid_y // 2),  # 左上
        (mid_x + mid_x // 2, mid_y // 2),  # 右上
        (mid_x // 2, mid_y + mid_y // 2),  # 左下
        (mid_x + mid_x // 2, mid_y + mid_y // 2),  # 右下
    ]

    # 在每個中心點放置ArUco標記
    for i, (center_x, center_y) in enumerate(centers):
        # 創建ArUco標記 (使用不同的ID: 0, 1, 2, 3)
        aruco_marker = create_aruco_marker(i, marker_size)

        # 計算標記的放置位置 (以中心點為準)
        start_x = center_x - marker_size // 2
        start_y = center_y - marker_size // 2
        end_x = start_x + marker_size
        end_y = start_y + marker_size

        # 確保標記不超出圖片邊界
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(a4_width, end_x)
        end_y = min(a4_height, end_y)

        # 將ArUco標記貼到圖片上
        marker_height = end_y - start_y
        marker_width = end_x - start_x

        if marker_height > 0 and marker_width > 0:
            # 調整標記大小以符合可用空間
            resized_marker = cv2.resize(aruco_marker, (marker_width, marker_height))
            img[start_y:end_y, start_x:end_x] = resized_marker

        # 在標記下方添加區域編號和ArUco ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 0)  # 黑色
        font_thickness = 2

        # 區域編號
        region_text = f"區域 {i+1}"
        aruco_text = f"ArUco ID: {i}"

        # 計算文字位置 (在標記下方)
        text_x = center_x - 40
        text_y1 = center_y + marker_size // 2 + 25
        text_y2 = text_y1 + 25

        # cv2.putText(
        #     img,
        #     region_text,
        #     (text_x, text_y1),
        #     font,
        #     font_scale,
        #     font_color,
        #     font_thickness,
        # )
        # cv2.putText(
        #     img,
        #     aruco_text,
        #     (text_x, text_y2),
        #     font,
        #     font_scale,
        #     font_color,
        #     font_thickness,
        # )

    return img


def save_and_display_image(img, filename="a4_with_aruco_marks.jpg"):
    """
    保存並顯示圖片
    """
    # 保存圖片
    cv2.imwrite(filename, img)
    print(f"圖片已保存為: {filename}")

    # 使用matplotlib顯示圖片
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 11))  # A4 比例
    plt.imshow(img_rgb)

    # 設定中文字型（如果有的話）
    try:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        plt.title("A4紙分四份並放置ArUco標記")
    except:
        # 如果中文字型不可用，使用英文標題
        plt.title("A4 Paper with ArUco Markers")

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return filename


def main(show_plot=True):
    """
    主函數

    Args:
        show_plot (bool): 是否顯示matplotlib視窗，默認True
    """
    print("正在創建A4紙分四份ArUco標記...")

    # 創建圖片
    img = create_a4_with_marks()

    if show_plot:
        # 保存並顯示
        filename = save_and_display_image(img)
    else:
        # 只保存，不顯示
        filename = "item60/a4_with_aruco_marks.jpg"
        cv2.imwrite(filename, img)
        print(f"圖片已保存為: {filename}")

    print("完成！")
    print(f"圖片尺寸: {img.shape[1]} x {img.shape[0]} 像素")
    print(f"ArUco標記ID: 0, 1, 2, 3 (左上、右上、左下、右下)")
    print(f"ArUco字典: DICT_4X4_50")

    return img, filename


if __name__ == "__main__":
    image, saved_file = main(show_plot=False)
