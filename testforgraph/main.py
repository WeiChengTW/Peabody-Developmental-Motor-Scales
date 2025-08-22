import cv2
import os
from ultralytics import YOLO
import numpy as np


class ShapeRecognizer:
    def __init__(self, model_path=None):
        """初始化圖形辨識器"""
        if model_path is None:
            # 使用預設的模型路徑
            self.model_path = (
                r"best.pt"
            )
        else:
            self.model_path = model_path

        # 檢查模型檔案是否存在
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"成功載入模型: {self.model_path}")
        else:
            # 如果沒有訓練好的模型，使用預訓練模型
            self.model = YOLO("yolov8n-seg.pt")
            print("使用預訓練模型: yolov8n-seg.pt")

        # 根據data.yaml定義的類別
        self.class_names = [
            "circle",
            "cross",
            "diamond",
            "rectangle",
            "triangle",
            "other",
        ]
        self.colors = {
            0: (255, 0, 0),  # 藍色 - circle
            1: (0, 255, 0),  # 綠色 - cross
            2: (0, 0, 255),  # 紅色 - diamond
            3: (255, 255, 0),  # 青色 - rectangle
            4: (255, 0, 255),  # 洋紅色 - triangle
            5: (128, 128, 128),  # 灰色 - other
        }

    def calculate_iou(self, bbox1, bbox2):
        """計算兩個邊界框的IoU (Intersection over Union)"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 計算交集區域
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # 如果沒有交集
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        # 計算交集面積
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 計算各自面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # 計算並集面積
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def apply_nms(self, detections, iou_threshold=0.5):
        """應用非重複最大抑制"""
        if not detections:
            return []

        # 按信心度降序排序
        sorted_detections = sorted(
            detections, key=lambda x: x["confidence"], reverse=True
        )

        keep = []
        while sorted_detections:
            # 取出信心度最高的檢測結果
            best = sorted_detections.pop(0)
            keep.append(best)

            # 與剩餘的檢測結果比較IoU
            remaining = []
            for detection in sorted_detections:
                iou = self.calculate_iou(best["bbox"], detection["bbox"])
                # 如果IoU小於閾值，表示不重疊，保留
                if iou < iou_threshold:
                    remaining.append(detection)

            sorted_detections = remaining

        return keep

    def apply_nms_with_other(
        self, detections, iou_threshold=0.5, other_iou_threshold=0.3
    ):
        """應用非重複最大抑制，對other類別使用不同的閾值"""
        if not detections:
            return []

        # 按信心度降序排序
        sorted_detections = sorted(
            detections, key=lambda x: x["confidence"], reverse=True
        )

        keep = []
        while sorted_detections:
            # 取出信心度最高的檢測結果
            best = sorted_detections.pop(0)
            keep.append(best)

            # 與剩餘的檢測結果比較IoU
            remaining = []
            for detection in sorted_detections:
                iou = self.calculate_iou(best["bbox"], detection["bbox"])

                # 根據類別選擇不同的IoU閾值
                if best["class"] == "other" or detection["class"] == "other":
                    threshold = other_iou_threshold
                else:
                    threshold = iou_threshold

                # 如果IoU小於閾值，表示不重疊，保留
                if iou < threshold:
                    remaining.append(detection)

            sorted_detections = remaining

        return keep

    def save_cropped_shapes(self, image, detections, image_filename):
        """將檢測到的圖形切割並保留原始比例，然後等比例縮放成100x100像素"""
        result_dir = "result"
        os.makedirs(result_dir, exist_ok=True)

        # 為每個類別建立資料夾
        for class_name in self.class_names:
            class_dir = os.path.join(result_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        # 取得圖片檔名（不含副檔名）
        base_name = os.path.splitext(image_filename)[0]

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]

            # 計算邊界框的寬高
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # 增加邊距來保留完整圖形（邊距為邊界框尺寸的20%）
            margin_x = int(bbox_width * 0.2)
            margin_y = int(bbox_height * 0.2)

            # 計算擴展後的切割區域
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(image.shape[1], x2 + margin_x)
            crop_y2 = min(image.shape[0], y2 + margin_y)

            # 切割擴展後的區域（保留原始比例）
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # 如果切割區域為空，跳過
            if cropped.size == 0:
                print(f"警告: {class_name} 切割區域為空，跳過")
                continue

            # 計算原始寬高
            original_height, original_width = cropped.shape[:2]

            # 目標尺寸
            target_size = 100

            # 計算縮放比例（保持寬高比）
            scale = min(target_size / original_width, target_size / original_height)

            # 計算新的寬高
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # 等比例縮放
            resized = cv2.resize(
                cropped, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

            # 轉換為灰度圖
            if len(resized.shape) == 3:
                gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_resized = resized

            # 應用自適應閾值來處理不均勻照明
            binary = cv2.adaptiveThreshold(
                gray_resized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )

            # 創建較小的結構元素來保護細線
            kernel_small = np.ones((2, 2), np.uint8)
            kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

            # 輕微的"閉運算"來連接斷開的線條，但不會太過激進
            processed = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel_small, iterations=1
            )

            # 使用十字形結構元素進行輕微的"開運算"，減少對線條的破壞
            processed = cv2.morphologyEx(
                processed, cv2.MORPH_OPEN, kernel_cross, iterations=8
            )

            # 反轉顏色，使得結果為黑底白線
            processed = cv2.bitwise_not(processed)

            # 線條粗化 - 使用膨脹操作讓線條更粗
            thicken_kernel = np.ones((3, 3), np.uint8)
            processed = cv2.dilate(processed, thicken_kernel, iterations=1)

            # 建立100x100的黑色背景
            final_image = np.zeros((target_size, target_size), dtype=np.uint8)

            # 計算要放置的位置（居中）
            start_y = (target_size - new_height) // 2
            start_x = (target_size - new_width) // 2
            end_y = start_y + new_height
            end_x = start_x + new_width

            # 將處理後的圖形放到中心
            final_image[start_y:end_y, start_x:end_x] = processed

            # 建立檔案名稱
            crop_filename = f"{base_name}_{i:03d}_{confidence:.2f}.jpg"
            class_dir = os.path.join(result_dir, class_name)
            crop_path = os.path.join(class_dir, crop_filename)

            # 保存切割的圖形
            cv2.imwrite(crop_path, final_image)
            print(
                f"保存切割圖形: {crop_path} (原始: {original_width}x{original_height} -> 縮放: {new_width}x{new_height})"
            )

    def recognize_single_image(self, image_path, save_result=True):
        """辨識單張圖片中的圖形"""
        if not os.path.exists(image_path):
            print(f"圖片檔案不存在: {image_path}")
            return None

        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            return None

        print(f"正在辨識圖片: {image_path}")

        # 進行預測
        results = self.model(image)

        # 處理結果
        result_image = image.copy()
        detections = []

        # 先收集所有檢測結果
        all_detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # 獲取邊界框座標
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 處理所有檢測結果，信心度小於0.5的歸類為other
                    if confidence >= 0.5:
                        # 獲取類別名稱
                        if (
                            class_id < len(self.class_names) - 1
                        ):  # 減1是因為最後一個是other
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        actual_class_id = class_id
                    else:
                        # 信心度小於0.5的歸類為other
                        class_name = "other"
                        actual_class_id = 5  # other的class_id

                    all_detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "class_id": actual_class_id,
                        }
                    )

        # 應用非重複最大抑制，移除太相近的檢測框
        # 對於other類別使用較低的IoU閾值
        filtered_detections = self.apply_nms_with_other(
            all_detections, iou_threshold=0.5, other_iou_threshold=0.3
        )

        # 繪製結果
        for detection in filtered_detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            # 繪製邊界框
            color = self.colors.get(class_id, (0, 255, 255))
            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # 繪製標籤
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # 記錄檢測結果
            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )

            print(f"檢測到: {class_name} (信心度: {confidence:.2f})")

        # 顯示結果
        # cv2.imshow("原始圖片", image)
        # cv2.imshow("辨識結果", result_image)

        # 儲存結果
        if save_result and detections:
            result_dir = "result"
            os.makedirs(result_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            result_path = os.path.join(result_dir, f"{name}_detected{ext}")
            cv2.imwrite(result_path, result_image)
            print(f"結果已儲存至: {result_path}")

            # 切割並保存每個檢測到的圖形
            self.save_cropped_shapes(image, filtered_detections, filename)

        # print("按任意鍵繼續...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return detections

    def recognize_single_image_batch(self, image_path, save_result=True):
        """辨識單張圖片中的圖形（批量處理版本，不顯示窗口）"""
        if not os.path.exists(image_path):
            print(f"圖片檔案不存在: {image_path}")
            return None

        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            return None

        print(f"正在辨識圖片: {image_path}")

        # 進行預測
        results = self.model(image)

        # 處理結果
        result_image = image.copy()
        detections = []

        # 先收集所有檢測結果
        all_detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # 獲取邊界框座標
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 處理所有檢測結果，信心度小於0.5的歸類為other
                    if confidence >= 0.5:
                        # 獲取類別名稱
                        if (
                            class_id < len(self.class_names) - 1
                        ):  # 減1是因為最後一個是other
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        actual_class_id = class_id
                    else:
                        # 信心度小於0.5的歸類為other
                        class_name = "other"
                        actual_class_id = 5  # other的class_id

                    all_detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "class_id": actual_class_id,
                        }
                    )

        # 應用非重複最大抑制，移除太相近的檢測框
        # 對於other類別使用較低的IoU閾值
        filtered_detections = self.apply_nms_with_other(
            all_detections, iou_threshold=0.5, other_iou_threshold=0.3
        )

        # 繪製結果
        for detection in filtered_detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            # 繪製邊界框
            color = self.colors.get(class_id, (0, 255, 255))
            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # 繪製標籤
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # 記錄檢測結果
            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )

            print(f"檢測到: {class_name} (信心度: {confidence:.2f})")

        # 儲存結果
        if save_result and detections:
            result_dir = "result"
            os.makedirs(result_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            result_path = os.path.join(result_dir, f"{name}_detected{ext}")
            cv2.imwrite(result_path, result_image)
            print(f"結果已儲存至: {result_path}")

            # 切割並保存每個檢測到的圖形
            self.save_cropped_shapes(image, filtered_detections, filename)

        return detections

    def recognize_directory(self, directory_path, show_images=False):
        """辨識資料夾中的所有圖片"""
        if not os.path.exists(directory_path):
            print(f"資料夾不存在: {directory_path}")
            return

        # 支援的圖片格式
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

        # 獲取所有圖片檔案
        image_files = [
            f
            for f in os.listdir(directory_path)
            if f.lower().endswith(image_extensions)
        ]

        if not image_files:
            print(f"在 {directory_path} 中沒有找到圖片檔案")
            return

        print(f"找到 {len(image_files)} 張圖片，開始辨識...")

        all_detections = {}
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            detections = self.recognize_single_image_batch(image_path, save_result=True)
            if detections:
                all_detections[image_file] = detections

        # 統計結果
        self.print_statistics(all_detections)

    def print_statistics(self, all_detections):
        """列印統計結果"""
        print("\n=== 辨識統計結果 ===")
        total_shapes = 0
        shape_counts = {name: 0 for name in self.class_names}

        for filename, detections in all_detections.items():
            print(f"\n{filename}: 檢測到 {len(detections)} 個圖形")
            for detection in detections:
                shape_counts[detection["class"]] += 1
                total_shapes += 1

        print(f"\n總計檢測到 {total_shapes} 個圖形:")
        for shape, count in shape_counts.items():
            if count > 0:
                print(f"  {shape}: {count} 個")


def main():
    """主程式"""
    print("=== 圖形辨識系統 ===")

    # 建立辨識器
    recognizer = ShapeRecognizer()

    default_dir = "raw"
    if os.path.exists(default_dir):
        recognizer.recognize_directory(default_dir)
    else:
        print(f"預設資料夾 '{default_dir}' 不存在")
    # 測試單張圖片
    # url = r"raw\LINE_ALBUM_蠟筆圖_250728_584.jpg"
    # recognizer.recognize_single_image(url)
    # recognizer.recognize_single_image("raw/313_1.jpg")


if __name__ == "__main__":
    main()
