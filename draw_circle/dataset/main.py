import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model



def pred(url):
    # 類別名稱
    class_names = ['Unlabeled', 'circle', 'oval']

    # 2. 讀取灰階圖片
    img_path = url
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 模糊 + Otsu 二值化 (灰階輸入)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 前處理 (假設模型輸入大小 100x100x3)
    img_resized = cv2.resize(binary, (100, 100))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 100, 100, 3)

    # 4. 預測
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions[0])
    pred_label = class_names[pred_class]
    pred_conf = predictions[0][pred_class]

    # 5. 標註結果 (用原始灰階圖來畫文字比較清楚)
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    text = f"{pred_label} ({pred_conf*100:.2f}%)"
    cv2.putText(output_img, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. 顯示
    cv2.imshow('Prediction', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 載入模型
    model = load_model(r'best_model.keras')
    for i in range(1, 9):
        try:
            pred(f'demo\\1_{i}.jpg')
        except:
            print('ERROR')