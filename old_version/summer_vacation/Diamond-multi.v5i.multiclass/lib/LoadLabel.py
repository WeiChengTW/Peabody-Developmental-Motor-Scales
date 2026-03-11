import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class LoadLabel:
    
    @staticmethod
    def load_images_and_labels(df, img_size, label_columns, img_dir):
        images = []
        labels = []
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])  # 根據你的圖片副檔名
            try:
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img) / 255.0
                label = row[label_columns].astype('float').values  # 多標籤
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"讀圖錯誤: {img_path}", e)
        return np.array(images), np.array(labels)