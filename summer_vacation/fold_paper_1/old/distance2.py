import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def detect_all_points_in_binary_image(image_path, output_path=None):
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

    # # 方法2: 找出所有黑色像素點（假設黑色為前景）
    # black_points = np.where(image == 0)
    # black_coordinates = list(zip(black_points[1], black_points[0]))  # (x, y) 格式

    print(f"白色像素點數量: {len(white_coordinates)}")
    # print(f"黑色像素點數量: {len(black_coordinates)}")

    # 創建可視化圖像
    if output_path or True:  # 總是顯示結果
        # 創建彩色圖像來標記點
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 在原圖上標記一些點（為了避免過於密集，只標記部分點）
        # step = max(1, len(white_coordinates) // 1000)  # 最多標記1000個點
        # marked_points = white_coordinates[::step]

        for x, y in white_coordinates:
            cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)  # 紅色點

        # 顯示結果
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"原始二值化圖像\n總像素: {image.shape[0] * image.shape[1]}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"標記的白點: {len(white_coordinates)}")
        plt.axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"結果已保存到: {output_path}")

        plt.show()

    # 返回白色點作為主要結果（通常白色是前景）
    return white_coordinates, len(white_coordinates)


def detect_contour_points(image_path, output_path=None):
    """
    使用輪廓檢測來找出二值化圖像中的關鍵點

    Args:
        image_path (str): 二值化圖片的路徑
        output_path (str): 輸出結果圖片的路徑（可選）

    Returns:
        tuple: (輪廓點列表, 輪廓數量)
    """
    # 讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"無法讀取圖像文件: {image_path}")
        return [], 0

    # 確保圖像是二值化的
    if len(np.unique(image)) > 2:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 找出輪廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print(f"找到 {len(contours)} 個輪廓")

    # 收集所有輪廓點
    all_contour_points = []
    for i, contour in enumerate(contours):
        # 將輪廓點轉換為 (x, y) 格式
        points = [(point[0][0], point[0][1]) for point in contour]
        all_contour_points.extend(points)
        print(f"輪廓 {i+1}: {len(points)} 個點")

    # 創建可視化
    if output_path or True:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 繪製輪廓
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)

        # 標記輪廓點
        for x, y in all_contour_points[:: max(1, len(all_contour_points) // 500)]:
            cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title("原始二值化圖像")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(
            f"輪廓檢測結果\n{len(contours)} 個輪廓, {len(all_contour_points)} 個點"
        )
        plt.axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.show()

    return all_contour_points, len(contours)


def detect_corner_points(image_path, output_path=None):
    """
    使用Harris角點檢測來找出圖像中的角點

    Args:
        image_path (str): 二值化圖片的路徑
        output_path (str): 輸出結果圖片的路徑（可選）

    Returns:
        tuple: (角點座標列表, 角點數量)
    """
    # 讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"無法讀取圖像文件: {image_path}")
        return [], 0

    # Harris角點檢測
    corners = cv2.cornerHarris(image, 2, 3, 0.04)

    # 標記角點
    corner_points = []
    corner_threshold = 0.01 * corners.max()
    corner_locations = np.where(corners > corner_threshold)

    for y, x in zip(corner_locations[0], corner_locations[1]):
        corner_points.append((x, y))

    print(f"檢測到 {len(corner_points)} 個角點")

    # 可視化
    if output_path or True:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 標記角點
        for x, y in corner_points:
            cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title("原始二值化圖像")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Harris角點檢測\n{len(corner_points)} 個角點")
        plt.axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.show()

    return corner_points, len(corner_points)


def main():
    """
    主函數：演示不同的點檢測方法
    """
    # 設定圖像路徑（請根據您的需要修改）
    image_path = r"adaptive\7_adaptive.jpg"  # 或者其他二值化圖像

    print("=== 方法1: 檢測所有像素點 ===")
    points1, count1 = detect_all_points_in_binary_image(image_path)

    print("\n=== 方法2: 輪廓點檢測 ===")
    points2, count2 = detect_contour_points(image_path)

    print("\n=== 方法3: Harris角點檢測 ===")
    points3, count3 = detect_corner_points(image_path)

    print(f"\n=== 總結 ===")
    print(f"像素點檢測: {count1} 個點")
    print(f"輪廓檢測: {count2} 個輪廓, {len(points2)} 個輪廓點")
    print(f"角點檢測: {count3} 個角點")


if __name__ == "__main__":
    main()
