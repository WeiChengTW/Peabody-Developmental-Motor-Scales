import cv2
import os
import shutil
from ultralytics import YOLO
import math

class Analyze_graphics:
    def __init__(
        self,
        model_path=r"C:\Users\hiimd\Desktop\vscode\Peabody-Developmental-Motor-Scales\draw_circle\circle2\runs\segment\train\weights\best.pt",
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
                if distance < distance_threshold and detection["cls_id"] == existing["cls_id"]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(detection)
        return filtered

    def reset_dir(self, dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def infer_and_draw(self, image_path, save_results=True):
        results = self.model(image_path, conf=0.3, iou=0.3, max_det=100, imgsz=640)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        ori_img = cv2.imread(image_path)

        # 建立 ready 資料夾
        ready_dir = "ready"
        if save_results:
            self.reset_dir(ready_dir)

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
                all_detections.append({
                    "cls_id": cls_id,
                    "confidence": confidence,
                    "bbox": (x1 - md_px, y1 - md_px, x2 + md_px, y2 + md_px),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                })

        filtered_detections = self.non_max_suppression_custom(all_detections, distance_threshold=30)

        for detection in filtered_detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(ori_img.shape[1], int(x2)), min(ori_img.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped_img = ori_img[y1:y2, x1:x2]

            # 儲存彩色裁切圖
            ready_path = os.path.join(ready_dir, f"{image_name}_{index}.jpg")
            cv2.imwrite(ready_path, cropped_img)

            # 轉灰階 + Otsu 二值化 + 反轉黑底白字
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_inverted = cv2.bitwise_not(binary)

            # 儲存二值化圖
            ready_binary_path = os.path.join(ready_dir, f"{image_name}_{index}_binary.jpg")
            cv2.imwrite(ready_binary_path, binary_inverted)
            binary_paths.append(ready_binary_path)

            index += 1

        return binary_paths  # 回傳所有黑底白字圖路徑


if __name__ == "__main__":
    image_path = r"demo\2.jpg"
    segmenter = Analyze_graphics()
    binary_paths = segmenter.infer_and_draw(image_path)

    print("Binary paths:")
    for p in binary_paths:
        print(p)
