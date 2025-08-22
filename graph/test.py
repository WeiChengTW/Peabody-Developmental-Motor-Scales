import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 指定你要用的單一特徵（例如 'gap'）
target_label = "Sharp corner"

model = load_model(f"models/{target_label}_best_model.keras")

df = pd.read_csv(r"Data/test/_classes.csv")
df.columns = df.columns.str.strip()
label_columns = df.columns[1:].tolist()

# 拿到該特徵在 CSV 中的位置
label_index = label_columns.index(target_label)

test_folder = r"Data/test"
extensions = (".jpg", ".jpeg", ".png")

for idx, row in df.iterrows():
    img_path = os.path.join(test_folder, row[0])
    if not img_path.lower().endswith(extensions):
        continue

    img = image.load_img(img_path, target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred_prob = model.predict(x)[0][0]  # 取出 float 值
    pred_label = int(pred_prob >= 0.5)  # 0 或 1

    true_label = int(row[1:][label_index])

    print(f"Image: {row[0]}")
    print(f"Predicted ({target_label}): {pred_label} ({pred_prob:.3f})")
    print(f"Actual:    ({target_label}): {true_label}")
    print("-" * 30)
