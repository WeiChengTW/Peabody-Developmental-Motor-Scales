from ultralytics import YOLO
import cv2
import numpy as np

# 初始化模型
model = YOLO(r'bean_model.pt')

# 讀取圖片
img_path = r'1.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"無法讀取圖片: {img_path}")
    exit()

# 使用 YOLO 模型進行預測
results = model(img)

# 在原圖上標記豆子
for result in results:
    boxes = result.boxes  # 取得檢測框
    
    for box in boxes:
        # 取得邊界框座標
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # 計算中心點
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # 畫紅色圓點
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  # BGR格式，紅色是(0,0,255)

# 顯示結果
cv2.imshow('Bean Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 儲存結果
cv2.imwrite('1_detected.jpg', img)
print(f"已標記 {len(results[0].boxes)} 個豆子")
print("結果已儲存至: 1_detected.jpg")