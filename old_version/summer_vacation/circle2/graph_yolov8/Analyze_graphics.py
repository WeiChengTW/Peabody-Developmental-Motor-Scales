import torch
import cv2
import math
import os
import shutil
from ultralytics import YOLO


class Analyze_graphics:
    def __init__(
        self,
        model_path=r"runs\segment\train\weights\best.pt",
        class_names=["Circle", "Cross", "Diamond", "Square", "Triangle"],
    ):
        self.model = YOLO(model_path)
        self.class_names = class_names

    def calculate_distance(self, center1, center2):
        """計算兩個中心點之間的歐氏距離"""
        return math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

    def non_max_suppression_custom(
        self, detections, distance_threshold=30, conf_threshold=0.1
    ):
        """改進的非最大抑制，特別針對相連圖形"""
        if not detections:
            return []

        # 先過濾極低信心度的檢測
        detections = [d for d in detections if d["confidence"] > conf_threshold]

        # 按信心度排序
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                distance = self.calculate_distance(
                    detection["center"], existing["center"]
                )

                # 如果距離很近且類別相同，才視為重複
                if (
                    distance < distance_threshold
                    and detection["cls_id"] == existing["cls_id"]
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(detection)

        return filtered

    def reset_dir(self, dir_path="result"):
        """重置並清空指定資料夾"""
        if os.path.exists(dir_path):
            # 刪除整個資料夾及其內容
            shutil.rmtree(dir_path)
            # print(f"已刪除資料夾: {dir_path}")

        # 重新創建資料夾
        os.makedirs(dir_path)
        # print(f"已重新創建資料夾: {dir_path}")

        # 創建各類別子資料夾
        all_categories = self.class_names + ["Other"]
        for category in all_categories:
            category_dir = os.path.join(dir_path, category)
            os.makedirs(category_dir)
            # print(f"已創建子資料夾: {category_dir}")

    def infer_and_draw(self, image_path, save_results=True, thresh_path=None):
        results = self.model(
            image_path,
            conf=0.3,  # 降低信心度閾值
            iou=0.3,  # 降低IoU閾值，允許更多重疊檢測
            max_det=100,  # 增加最大檢測數量
            imgsz=640,  # 確保輸入圖像尺寸
        )

        # 獲取圖像檔名（不含路徑和副檔名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 只取檔名 . 以前的部分
        image_name = image_name.split("_")[0]
        # print(f"處理圖像: {image_name}")
        ori_path = rf"extracted/{image_name}_extracted_paper.jpg"
        print(f"處理圖像: {ori_path}")
        original_image = cv2.imread(ori_path)
        image = cv2.imread(ori_path)
        thresh_img = cv2.imread(thresh_path)

        # 創建result資料夾和各類別子資料夾
        if save_results:
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # 為每個類別創建子資料夾
            all_categories = self.class_names + ["Other"]
            for category in all_categories:
                category_dir = os.path.join(result_dir, category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)

        # 統計各類圖形數量
        shape_counts = {shape: 0 for shape in self.class_names}
        shape_counts["Other"] = 0  # 添加 Other 類別

        # 收集所有檢測結果
        all_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                all_detections.append(
                    {
                        "cls_id": cls_id,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    }
                )

        # 使用NMS過濾太相近的檢測結果
        filtered_detections = self.non_max_suppression_custom(
            all_detections, distance_threshold=30
        )

        # 保存檢測到的圖形到對應類別資料夾
        detection_index = 0
        for detection in filtered_detections:
            cls_id = detection["cls_id"]
            confidence = detection["confidence"]
            x1, y1, x2, y2 = detection["bbox"]

            # 信心度<0.5的歸類為Other
            if confidence < 0.5:
                shape_counts["Other"] += 1
                label = f"Other: {confidence:.2f}"
                category = "Other"
            else:
                # 增加對應類別的計數
                shape_counts[self.class_names[cls_id]] += 1
                label = f"{self.class_names[cls_id]}: {confidence:.2f}"
                category = self.class_names[cls_id]

            # 裁切檢測到的圖形區域並保存
            if save_results:
                # 確保座標在圖像範圍內
                x1, y1, x2, y2 = (
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(thresh_img.shape[1], int(x2)),
                    min(thresh_img.shape[0], int(y2)),
                )

                if x2 > x1 and y2 > y1:  # 確保有效的裁切區域
                    cropped_image = thresh_img[y1:y2, x1:x2]

                    # 將裁切的圖像調整為 100x100 像素
                    resized_image = cv2.resize(
                        cropped_image, (100, 100), interpolation=cv2.INTER_AREA
                    )

                    # 保存到對應類別的資料夾
                    category_dir = os.path.join(result_dir, category)
                    # 保留原始檔名（如 98_1.jpg）作為部分新檔名

                    filename = f"{image_name}_{category}_{detection_index:03d}_conf{confidence:.2f}.jpg"

                    save_path = os.path.join(category_dir, filename)
                    cv2.imwrite(save_path, resized_image)
                    detection_index += 1

            # 在原圖上繪製檢測框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                image,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                image,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        # 輸出統計結果
        # print("=== 模型圖形分類統計 ===")
        total_count = 0
        for shape in ["Circle", "Cross", "Diamond", "Square", "Triangle", "Other"]:
            count = shape_counts.get(shape, 0)
            if count > 0:
                # print(f"{shape}: {count} 個")
                total_count += count
        # print(f"總計: {total_count} 個圖形")
        image_name = image_name.split("_")[0]
        if save_results:
            # print(f"\n檢測結果已保存到 'result' 資料夾，按類別分類存放")
            # 保存標註後的完整圖像
            annotated_image_path = os.path.join(
                result_dir, f"{image_name}_annotated.jpg"
            )
            cv2.imwrite(annotated_image_path, image)
            # print(f"標註圖像已保存: {annotated_image_path}")

        # cv2.imshow("YOLOv8 Detection Results", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"extracted\img1_extracted_paper.jpg"

    segmenter = Analyze_graphics()

    # 重置result資料夾（可選）
    # segmenter.reset_dir()

    segmenter.infer_and_draw(image_path)
