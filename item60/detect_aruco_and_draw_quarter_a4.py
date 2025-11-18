import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class ArUcoQuarterA4Detector:
    """
    ArUco 標記偵測器，並畫出 1/4 個 A4 長方形
    """

    def __init__(self):
        """
        初始化偵測器
        """
        try:
            # 嘗試新版本的OpenCV API (4.x)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                # 嘗試舊版本的OpenCV API (3.x)
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                # 最新版本的API
                self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50)
                self.aruco_params = cv2.aruco.DetectorParameters()

        # ArUco 標記實際尺寸 (毫米)
        self.ARUCO_MARKER_SIZE_MM = 28  # 2.8cm = 28mm

        # A4 紙張實際尺寸 (毫米)
        self.A4_WIDTH_MM = 210
        self.A4_HEIGHT_MM = 297

        # 1/4 A4 尺寸 (毫米)
        self.QUARTER_A4_WIDTH_MM = self.A4_WIDTH_MM / 2  # 105 mm
        self.QUARTER_A4_HEIGHT_MM = self.A4_HEIGHT_MM / 2  # 148.5 mm

        # 像素到毫米的轉換比例 (將根據ArUco標記動態計算)
        self.px_to_mm_ratio = None

    def detect_aruco_markers(self, image):
        """
        偵測 ArUco 標記

        Args:
            image: 輸入圖像 (BGR 格式)

        Returns:
            corners: 標記角點
            ids: 標記ID
            rejected: 被拒絕的候選區域
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            # 嘗試最新版本的OpenCV API (4.7+)
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            try:
                # 嘗試較舊版本的OpenCV API (4.0-4.6)
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params
                )
            except:
                # 舊版本的OpenCV API
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params
                )

        return corners, ids, rejected

    def calculate_quarter_a4_rectangle(self, corner, marker_id=None):
        """
        根據 ArUco 標記計算 1/4 A4 長方形的位置

        Args:
            corner: ArUco 標記的四個角點
            marker_id: 標記ID (用於輸出詳細資訊)

        Returns:
            tuple: (rectangle_corners, scale_info) - 長方形角點和比例尺資訊
        """
        # 計算 ArUco 標記的中心點
        center_x = np.mean(corner[0][:, 0])
        center_y = np.mean(corner[0][:, 1])

        # 精確計算標記的四邊長度
        side_lengths = []
        for i in range(4):
            p1 = corner[0][i]
            p2 = corner[0][(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)

        # 使用平均邊長來計算比例尺
        avg_marker_size_px = np.mean(side_lengths)

        # 根據實際ArUco標記尺寸 (2.8cm = 28mm) 計算像素到毫米比例
        px_to_mm_ratio = avg_marker_size_px / self.ARUCO_MARKER_SIZE_MM

        # 儲存比例尺供其他方法使用
        self.px_to_mm_ratio = px_to_mm_ratio

        # 計算 1/4 A4 的像素尺寸 (毫米 × 像素/毫米比例 = 像素)
        quarter_width_px = self.QUARTER_A4_WIDTH_MM * px_to_mm_ratio
        quarter_height_px = self.QUARTER_A4_HEIGHT_MM * px_to_mm_ratio

        # 計算 ArUco 標記的旋轉角度
        # 使用標記的第一邊（從點0到點1）作為參考邊來計算角度
        edge_vector = corner[0][1] - corner[0][0]  # 從點0到點1的向量
        rotation_angle = np.arctan2(edge_vector[1], edge_vector[0])  # 計算角度（弧度）

        # 以 ArUco 標記中心為基準，先計算未旋轉的 1/4 A4 長方形角點
        half_width = quarter_width_px / 2
        half_height = quarter_height_px / 2

        # 未旋轉的四個角點 (相對於中心點的座標)
        unrotated_corners = np.array(
            [
                [-half_width, -half_height],  # 左上
                [half_width, -half_height],  # 右上
                [half_width, half_height],  # 右下
                [-half_width, half_height],  # 左下
            ]
        )

        # 建立旋轉矩陣
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # 應用旋轉並平移到正確位置
        rotated_corners = []
        for corner_point in unrotated_corners:
            # 旋轉角點
            rotated_point = rotation_matrix @ corner_point
            # 平移到 ArUco 中心位置
            final_point = [center_x + rotated_point[0], center_y + rotated_point[1]]
            rotated_corners.append(final_point)

        rectangle_corners = np.array(rotated_corners, dtype=np.int32)

        # 比例尺資訊
        scale_info = {
            "marker_id": marker_id,
            "marker_size_px": avg_marker_size_px,
            "marker_size_mm": self.ARUCO_MARKER_SIZE_MM,
            "px_to_mm_ratio": px_to_mm_ratio,
            "mm_to_px_ratio": 1 / px_to_mm_ratio,
            "side_lengths_px": side_lengths,
            "center": (center_x, center_y),
            "rotation_angle_rad": rotation_angle,
            "rotation_angle_deg": np.degrees(rotation_angle),
            "quarter_a4_size_px": (quarter_width_px, quarter_height_px),
            "quarter_a4_size_mm": (self.QUARTER_A4_WIDTH_MM, self.QUARTER_A4_HEIGHT_MM),
        }

        # 輸出詳細的比例尺資訊
        if marker_id is not None:
            print(f"\n--- 標記 ID {marker_id} 比例尺分析 ---")
            print(
                f"旋轉角度: {np.degrees(rotation_angle):.1f}° ({rotation_angle:.3f} 弧度)"
            )
            print(f"比例尺: 1 mm = {px_to_mm_ratio:.3f} px")
            print(
                f"1/4 A4 實際尺寸: {self.QUARTER_A4_WIDTH_MM} x {self.QUARTER_A4_HEIGHT_MM} mm"
            )

        return rectangle_corners, scale_info

    def draw_quarter_a4_rectangles(self, image, corners, ids):
        """
        在圖像上畫出 1/4 A4 長方形

        Args:
            image: 輸入圖像
            corners: ArUco 標記角點
            ids: ArUco 標記ID

        Returns:
            tuple: (output_image, detection_results) - 繪製後的圖像和偵測結果
        """
        output_image = image.copy()
        detection_results = []

        if ids is not None:
            # 繪製 ArUco 標記
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

            # 為每個偵測到的標記畫出 1/4 A4 長方形
            for i, corner in enumerate(corners):
                marker_id = ids[i][0] if ids[i] is not None else i

                # 計算 1/4 A4 長方形和比例尺資訊
                rectangle_corners, scale_info = self.calculate_quarter_a4_rectangle(
                    corner, marker_id
                )
                detection_results.append(scale_info)

                # 繪製長方形 (綠色邊框)
                cv2.polylines(output_image, [rectangle_corners], True, (0, 255, 0), 3)

                # 標記長方形的四個角點 (藍色圓點)
                for point in rectangle_corners:
                    cv2.circle(output_image, tuple(point), 6, (255, 0, 0), -1)

                # 添加標記ID和類型標籤
                text = f"ID:{marker_id} - 1/4 A4"
                text_position = (
                    int(rectangle_corners[0][0]),
                    int(rectangle_corners[0][1] - 15),
                )
                # cv2.putText(
                #     output_image,
                #     text,
                #     text_position,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 255),
                #     2,
                # )

                # 在中心點標記尺寸資訊
                center = np.mean(rectangle_corners, axis=0).astype(int)
                size_text = (
                    f"{self.QUARTER_A4_WIDTH_MM:.0f}x{self.QUARTER_A4_HEIGHT_MM:.0f}mm"
                )
                # cv2.putText(
                #     output_image,
                #     size_text,
                #     (center[0] - 40, center[1]),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (255, 255, 0),
                #     2,
                # )

                # 添加比例尺資訊
                scale_text = f"Scale: 1mm={scale_info['px_to_mm_ratio']:.2f}px"
                # cv2.putText(
                #     output_image,
                #     scale_text,
                #     (center[0] - 60, center[1] + 20),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (255, 255, 0),
                #     1,
                # )

                # 標記ArUco標記尺寸
                aruco_text = f"ArUco: 2.8cm"
                aruco_center = (
                    int(scale_info["center"][0]),
                    int(scale_info["center"][1] + 30),
                )
                # cv2.putText(
                #     output_image,
                #     aruco_text,
                #     aruco_center,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 255, 255),
                #     1,
                # )

                # 標記旋轉角度
                rotation_text = f"Angle: {scale_info['rotation_angle_deg']:.1f}°"
                rotation_center = (
                    int(scale_info["center"][0]),
                    int(scale_info["center"][1] + 45),
                )
                # cv2.putText(
                #     output_image,
                #     rotation_text,
                #     rotation_center,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 255, 255),
                #     1,
                # )

        return output_image, detection_results

    def process_image(self, image_path, save_result=True, show_result=True):
        """
        處理單張圖片

        Args:
            image_path: 圖片路徑
            save_result: 是否保存結果
            show_result: 是否顯示結果

        Returns:
            output_image: 處理後的圖像
        """
        # 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")

        print(f"處理圖片: {image_path}")
        print(f"圖片尺寸: {image.shape[1]} x {image.shape[0]} 像素")

        # 偵測 ArUco 標記
        corners, ids, rejected = self.detect_aruco_markers(image)

        if ids is not None:
            print(f"偵測到 {len(ids)} 個 ArUco 標記，ID: {ids.flatten()}")
        else:
            print("未偵測到任何 ArUco 標記")

        # 繪製結果
        output_image, detection_results = self.draw_quarter_a4_rectangles(
            image, corners, ids
        )

        # 輸出總結資訊
        if detection_results:
            print("\n=== 偵測結果總結 ===")
            for result in detection_results:
                print(
                    f"標記 ID {result['marker_id']}: 比例尺 1mm = {result['px_to_mm_ratio']:.3f}px"
                )

        # 保存結果
        if save_result:
            # 確保 result 目錄存在
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                print(f"已創建目錄: {result_dir}")

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_quarter_a4_detected.jpg"
            output_path = os.path.join(result_dir, output_filename)
            cv2.imwrite(output_path, output_image)
            print(f"結果已保存到: {output_path}")

        # 顯示結果
        if show_result:
            self.show_result(image, output_image)

        return output_image

    def show_result(self, original_image, result_image):
        """
        顯示原圖和結果的對比
        """
        plt.figure(figsize=(15, 8))

        # 原圖
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("原始圖像")
        plt.axis("off")

        # 結果圖
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("ArUco 偵測與 1/4 A4 長方形")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def process_directory(self, directory_path):
        """
        處理整個目錄中的圖片

        Args:
            directory_path: 目錄路徑
        """
        # 支援的圖片格式
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                try:
                    self.process_image(image_path, save_result=True, show_result=False)
                    print("-" * 50)
                except Exception as e:
                    print(f"處理 {filename} 時發生錯誤: {e}")


def main():
    """
    主函數 - 示範如何使用 ArUcoQuarterA4Detector
    """
    detector = ArUcoQuarterA4Detector()

    # 測試用的圖片路徑
    img_dir = "img"

    print("ArUco 偵測器與 1/4 A4 長方形繪製")
    print("長方形會與 ArUco 標記保持平行")
    print("=" * 50)

    # 處理 img 目錄中的所有圖片
    if os.path.exists(img_dir):
        print(f"處理 {img_dir} 目錄中的圖片...")
        detector.process_directory(img_dir)
    else:
        print(f"目錄 {img_dir} 不存在")

    # 如果有 a4_with_aruco_marks.jpg，也處理它
    # test_image = "a4_with_aruco_marks.jpg"
    # if os.path.exists(test_image):
    #     print(f"\n處理測試圖片: {test_image}")
    #     detector.process_image(test_image, save_result=True, show_result=True)

    print("\n處理完成！")


if __name__ == "__main__":
    main()
