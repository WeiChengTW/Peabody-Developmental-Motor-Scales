import cv2
import os
import shutil
from ultralytics import YOLO
import math

class Analyze_graphics:
    def __init__(
        self,
        model_path=r"model/YOLO.pt",
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
        """清空資料夾"""
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def ensure_dir(self, dir_path):
        """創建資料夾但不清空現有內容"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    def clear_multiple_dirs(self, dir_list):
        """清空多個資料夾"""
        for dir_path in dir_list:
            if os.path.exists(dir_path):
                print(f"清空資料夾: {dir_path}")
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            print(f"重新建立資料夾: {dir_path}")
    
    def initialize_workspace(self, clear_all=True):
        """初始化工作空間，清空所有相關資料夾"""
        workspace_dirs = ["quadrilateral", "Other", "cropped_a4", "ready"]
        
        if clear_all:
            print("=== 初始化工作空間，清空所有資料夾 ===")
            self.clear_multiple_dirs(workspace_dirs)
        else:
            print("=== 確保工作空間資料夾存在 ===")
            for dir_path in workspace_dirs:
                self.ensure_dir(dir_path)
                print(f"確保資料夾存在: {dir_path}")
        
        print("工作空間初始化完成！\n")

    def get_next_index(self, ready_dir, image_name):
        """取得下一個可用的索引號"""
        if not os.path.exists(ready_dir):
            return 0
        
        existing_files = [f for f in os.listdir(ready_dir) if f.startswith(image_name)]
        max_index = -1
        
        for filename in existing_files:
            try:
                # 從檔名中提取索引號 (例如: image_5_Circle.jpg -> 5)
                parts = filename.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    max_index = max(max_index, int(parts[1]))
            except:
                continue
        
        return max_index + 1

    def get_unique_filename(self, base_path):
        """產生不重複的檔名"""
        if not os.path.exists(base_path):
            return base_path
        
        # 分解檔名
        dir_path = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        
        counter = 1
        while True:
            new_filename = f"{name}_v{counter}{ext}"
            new_path = os.path.join(dir_path, new_filename)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def infer_and_draw(self, image_path, save_results=True, expand_ratio=0.15, clear_dir=False):
        """
        推論並切割圖形
        
        Args:
            image_path: 輸入圖片路徑
            save_results: 是否保存結果
            expand_ratio: 框框擴大比例 (0.15 = 擴大 15%)
            clear_dir: 是否清空目標資料夾 (預設False，保留現有檔案)
        
        Returns:
            binary_paths: 所有二值化圖片的路徑列表
        """

        
        results = self.model(image_path, conf=0.3, iou=0.3, max_det=100, imgsz=640)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        ori_img = cv2.imread(image_path)

        if ori_img is None:
            print(f"錯誤：無法讀取圖片 {image_path}")
            return []

        # 建立 ready 資料夾
        ready_dir = "ready"
        if save_results:
            if clear_dir:
                self.reset_dir(ready_dir)  # 清空資料夾
                index = 0
            else:
                self.ensure_dir(ready_dir)  # 只創建資料夾，不清空
                index = self.get_next_index(ready_dir, image_name)  # 取得下一個索引

        binary_paths = []  # 回傳二值化圖的完整路徑

        # 收集所有檢測結果
        all_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 計算原始框框的寬度和高度
                width = x2 - x1
                height = y2 - y1
                
                # 計算擴大後的座標
                expand_w = width * expand_ratio / 2
                expand_h = height * expand_ratio / 2
                
                # 獲取圖片尺寸
                img_height, img_width = ori_img.shape[:2]
                
                # 應用擴大並確保不超出圖片邊界
                x1_expanded = max(0, int(x1 - expand_w))
                y1_expanded = max(0, int(y1 - expand_h))
                x2_expanded = min(img_width, int(x2 + expand_w))
                y2_expanded = min(img_height, int(y2 + expand_h))
                
                all_detections.append({
                    "cls_id": cls_id,
                    "confidence": confidence,
                    "bbox": (x1_expanded, y1_expanded, x2_expanded, y2_expanded),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                })

        filtered_detections = self.non_max_suppression_custom(all_detections, distance_threshold=30)

        for detection in filtered_detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # 再次確認座標有效性
            if x2 <= x1 or y2 <= y1:
                print(f"跳過無效座標: ({x1}, {y1}) to ({x2}, {y2})")
                continue

            try:
                # 切割圖片
                cropped_img = ori_img[y1:y2, x1:x2]
                
                # 檢查切割結果
                if cropped_img.size == 0:
                    print(f"切割結果為空，跳過索引 {index}")
                    continue

                # 產生檔名
                ready_path = os.path.join(ready_dir, f"{image_name}_{index}_{class_name}.jpg")
                ready_binary_path = os.path.join(ready_dir, f"{image_name}_{index}_{class_name}_binary.jpg")
                
                # 確保檔名不重複
                ready_path = self.get_unique_filename(ready_path)
                ready_binary_path = self.get_unique_filename(ready_binary_path)

                # 儲存彩色裁切圖
                cv2.imwrite(ready_path, cropped_img)

                # 轉灰階 + Otsu 二值化 + 反轉黑底白字
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_inverted = cv2.bitwise_not(binary)

                # 儲存二值化圖
                cv2.imwrite(ready_binary_path, binary_inverted)
                binary_paths.append(ready_binary_path)

                print(f"成功處理 {class_name} (信心度: {confidence:.2f}), 索引: {index}")
                print(f"  - 彩色圖: {ready_path}")
                print(f"  - 二值圖: {ready_binary_path}")
                print(f"  - 切割尺寸: {cropped_img.shape}")

                index += 1
                
            except Exception as e:
                print(f"處理索引 {index} 時發生錯誤: {e}")
                print(f"  - 座標: ({x1}, {y1}) to ({x2}, {y2})")
                continue

        print(f"\n總共成功處理 {len(binary_paths)} 個圖形")
        return binary_paths  # 回傳所有黑底白字圖路徑

    def crop_with_padding(self, image_path, padding_pixels=20, clear_dir=False):
        """
        使用固定像素填充的方式切割（舊版本的 md_px 方法）
        
        Args:
            image_path: 輸入圖片路徑
            padding_pixels: 固定填充像素數
            clear_dir: 是否清空目標資料夾
        """
        results = self.model(image_path, conf=0.3, iou=0.3, max_det=100, imgsz=640)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        ori_img = cv2.imread(image_path)

        ready_dir = "ready"
        if clear_dir:
            self.reset_dir(ready_dir)
            index = 0
        else:
            self.ensure_dir(ready_dir)
            index = self.get_next_index(ready_dir, image_name)

        binary_paths = []

        # 收集所有檢測結果
        all_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                all_detections.append({
                    "cls_id": cls_id,
                    "confidence": confidence,
                    "bbox": (x1 - padding_pixels, y1 - padding_pixels, x2 + padding_pixels, y2 + padding_pixels),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                })

        filtered_detections = self.non_max_suppression_custom(all_detections, distance_threshold=30)

        for detection in filtered_detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(ori_img.shape[1], int(x2)), min(ori_img.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue

            cropped_img = ori_img[y1:y2, x1:x2]
            class_name = detection["class_name"]

            # 產生檔名並確保不重複
            ready_path = os.path.join(ready_dir, f"{image_name}_{index}_{class_name}.jpg")
            ready_binary_path = os.path.join(ready_dir, f"{image_name}_{index}_{class_name}_binary.jpg")
            
            ready_path = self.get_unique_filename(ready_path)
            ready_binary_path = self.get_unique_filename(ready_binary_path)

            # 儲存彩色裁切圖
            cv2.imwrite(ready_path, cropped_img)

            # 轉灰階 + Otsu 二值化 + 反轉黑底白字
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_inverted = cv2.bitwise_not(binary)

            # 儲存二值化圖
            cv2.imwrite(ready_binary_path, binary_inverted)
            binary_paths.append(ready_binary_path)

            index += 1

        return binary_paths


if __name__ == "__main__":
    image_path = r"input\S__75472901_0.jpg"
    segmenter = Analyze_graphics()
    
    # 方法1: 使用比例擴大，保留現有檔案（預設）
    print("=== 使用比例擴大方式（不清空資料夾）===")
    binary_paths = segmenter.infer_and_draw(image_path, expand_ratio=0.15, clear_dir=False)
    
    # 如果想要清空資料夾重新開始，設定 clear_dir=True
    # print("=== 使用比例擴大方式（清空資料夾）===")
    # binary_paths = segmenter.infer_and_draw(image_path, expand_ratio=0.15, clear_dir=True)
    
    # 方法2: 使用固定像素填充，保留現有檔案
    # print("=== 使用固定像素填充方式（不清空資料夾）===")
    # binary_paths = segmenter.crop_with_padding(image_path, padding_pixels=25, clear_dir=False)

    print("\nBinary paths:")
    for p in binary_paths:
        print(p)