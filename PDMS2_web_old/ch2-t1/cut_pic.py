import cv2
import os
import shutil
from ultralytics import YOLO
import math
import glob


class Analyze_graphics:
    def __init__(
        self,
        model_path=r"ch2-t1\model\YOLO.pt",
        class_names=["Circle", "Cross", "Diamond", "Square", "Triangle"],
    ):
        self.model = YOLO(model_path)
        self.class_names = class_names

    def calculate_distance(self, center1, center2):
        return math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

    def non_max_suppression_custom(
        self, detections, distance_threshold=30, conf_threshold=0.1
    ):
        if not detections:
            return []
        detections = [d for d in detections if d["confidence"] > conf_threshold]
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                distance = self.calculate_distance(
                    detection["center"], existing["center"]
                )
                if (
                    distance < distance_threshold
                    and detection["cls_id"] == existing["cls_id"]
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(detection)
        return filtered

    def reset_dir(self, dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def get_image_files(self, folder_path):
        """獲取資料夾中所有圖片檔案"""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

        return image_files

    def infer_and_draw(self, image_path, output_dir="pic_result"):
        """處理單張圖片並將結果存到指定資料夾"""
        results = self.model(image_path, conf=0.3, iou=0.3, max_det=100, imgsz=640)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        ori_img = cv2.imread(image_path)

        # 確保輸出資料夾存在
        os.makedirs(output_dir, exist_ok=True)

        binary_paths = []  # 回傳二值化圖的完整路徑
        index = 0

        # 收集所有檢測結果
        all_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                md_px = 20
                all_detections.append(
                    {
                        "cls_id": cls_id,
                        "confidence": confidence,
                        "bbox": (x1 - md_px, y1 - md_px, x2 + md_px, y2 + md_px),
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    }
                )

        filtered_detections = self.non_max_suppression_custom(
            all_detections, distance_threshold=30
        )

        for detection in filtered_detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(ori_img.shape[1], int(x2)), min(ori_img.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped_img = ori_img[y1:y2, x1:x2]

            # 轉灰階 + Otsu 二值化 + 反轉黑底白字
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_inverted = cv2.bitwise_not(binary)

            # 儲存黑底白字二值化圖到 pic_result 資料夾
            binary_filename = rf"ch2-t1\{image_name}_{index}_binary.jpg"
            binary_path = os.path.join(output_dir, binary_filename)
            cv2.imwrite(binary_path, binary_inverted)
            binary_paths.append(binary_path)

            index += 1

        return binary_paths  # 回傳所有黑底白字圖路徑

    def process_folder(self, input_folder, output_folder="pic_result"):
        """批量處理資料夾中的所有圖片"""
        # 重置輸出資料夾
        self.reset_dir(output_folder)

        # 獲取所有圖片檔案
        image_files = self.get_image_files(input_folder)

        if not image_files:
            print(f"在資料夾 '{input_folder}' 中未找到任何圖片檔案")
            return []

        all_binary_paths = []
        total_images = len(image_files)

        print(f"找到 {total_images} 張圖片，開始處理...")

        for i, image_path in enumerate(image_files, 1):
            print(f"正在處理 ({i}/{total_images}): {os.path.basename(image_path)}")

            try:
                binary_paths = self.infer_and_draw(image_path, output_folder)
                all_binary_paths.extend(binary_paths)
                print(f"  -> 成功切割出 {len(binary_paths)} 個圖形")

            except Exception as e:
                print(f"  -> 處理失敗: {str(e)}")
                continue

        print(f"\n處理完成！")
        print(f"總共處理了 {total_images} 張圖片")
        print(f"切割出 {len(all_binary_paths)} 個圖形")
        print(f"結果已保存在 '{output_folder}' 資料夾")

        return all_binary_paths


if __name__ == "__main__":
    # 設定輸入資料夾路徑
    input_folder = "demo"  # 請修改為您的圖片資料夾路徑
    output_folder = "pic_result"  # 輸出資料夾

    # 建立分析器
    segmenter = Analyze_graphics()

    # 批量處理資料夾中的所有圖片
    binary_paths = segmenter.process_folder(input_folder, output_folder)

    print("\n所有二值化圖片路徑:")
    for p in binary_paths:
        print(p)
