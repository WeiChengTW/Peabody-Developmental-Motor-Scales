import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("cross_multilabel_simple.h5")
labels = ["asymmetric", "nonstraight", "nonrightangle"]

test_folder = "test"
extensions = (".jpg", ".jpeg", ".png")

for fname in os.listdir(test_folder):
    if not fname.lower().endswith(extensions):
        continue
    img_path = os.path.join(test_folder, fname)
    img = image.load_img(img_path, target_size=(100, 100))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    result = []
    for i, p in enumerate(pred):
        flag = "有特徵" if p > 0.5 else "無"
        result.append(f"{labels[i]}: {flag} (信心度={p:.2f})")
    print(f"{fname} → {'，'.join(result)}")
