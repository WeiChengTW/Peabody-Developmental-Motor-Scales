import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

def binary_focal_loss(gamma, alpha):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        modulating = tf.where(tf.equal(y_true, 1), (1 - y_pred) ** gamma, y_pred ** gamma)
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(alpha_weight * modulating * bce)
    return focal_loss

target_label = "Sharp corner"
label_loss_map = {
    "Sharp corner" : (0.8295, 0.1899)
}

gamma, alpha = label_loss_map[target_label]
model = load_model(
    f"pre_models/{target_label}_best_model.keras",
    custom_objects={'focal_loss': binary_focal_loss(gamma=gamma, alpha=alpha)}
)

df = pd.read_csv(r"test/_classes.csv")
df.columns = df.columns.str.strip()
label_columns = df.columns[1:].tolist()

# 拿到該特徵在 CSV 中的位置
label_index = label_columns.index(target_label)

test_folder = r"test"
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
