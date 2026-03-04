import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def detect_all_points_in_binary_image(image_path, output_path="point"):
    """
    在二值化圖片中偵測所有的點（像素點）

    Args:
        image_path (str): 二值化圖片的路徑
        output_path (str): 輸出結果圖片的路徑（可選）

    Returns:
        tuple: (點的座標列表, 點的數量)
    """
    # 讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"無法讀取圖像文件: {image_path}")
        return [], 0

    print(f"圖像尺寸: {image.shape[1]} x {image.shape[0]}")

    # 確保圖像是二值化的
    if len(np.unique(image)) > 2:
        print("警告: 圖像不是純二值化的，正在進行二值化處理...")
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 方法1: 找出所有白色像素點（假設白色為前景）
    white_points = np.where(image == 255)
    white_coordinates = list(zip(white_points[1], white_points[0]))  # (x, y) 格式

    print(f"白色像素點數量: {len(white_coordinates)}")

    # 創建可視化圖像
    if output_path or True:  # 總是顯示結果
        # 創建彩色圖像來標記點
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for x, y in white_coordinates:
            cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)  # 紅色點

        # 顯示結果
        # plt.figure(figsize=(12, 8))

        # plt.subplot(1, 2, 1)
        # plt.imshow(image, cmap="gray")
        # plt.title(f"原始二值化圖像\n總像素: {image.shape[0] * image.shape[1]}")
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"標記的白點: {len(white_coordinates)}")
        # plt.axis("off")

        # plt.tight_layout()

        if output_path:
            name = image_path.split("\\")[-1].split("_")[0]
            path = f"{output_path}\{name}_point.jpg"

            cv2.imwrite(path, vis_image)
            print(f"結果已儲存至: {path}")

        # plt.show()

    # 返回白色點作為主要結果（通常白色是前景）
    return white_coordinates, len(white_coordinates)


if __name__ == "__main__":
    image_path = r"adaptive\7_adaptive.jpg"  # 或者其他二值化圖像
    all_point, n_point = detect_all_points_in_binary_image(image_path)
    print(f"總白色點數量: {n_point}")
    print(f"所有點的座標: {all_point[:10]}... (總數: {n_point})")
