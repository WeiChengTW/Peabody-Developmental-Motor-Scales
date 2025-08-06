import os
import pandas as pd
from Cross.cross_asym_class import AsymmetryDetector
from Cross.cross_line_class import NonStraightDetector
from Cross.cross_angle_class import NonRightAngleDetector

src_folder = "practice"  # 圖片都在這裡
extensions = ('.jpg', '.jpeg', '.png')
img_paths = [f for f in os.listdir(src_folder) if f.lower().endswith(extensions)]

detector1 = AsymmetryDetector(threshold=0.27)
detector2 = NonStraightDetector(threshold=0.98)
detector3 = NonRightAngleDetector(threshold=13)

records = []
for fname in img_paths:
    img_path = os.path.join(src_folder, fname)
    label_asym = int(detector1.detect(img_path, debug=False))
    label_line = int(detector2.detect(img_path, debug=False))
    label_angle = int(detector3.detect(img_path, debug=False))
    records.append({
        "filename": fname,
        "asymmetric": label_asym,
        "nonstraight": label_line,
        "nonrightangle": label_angle
    })
df = pd.DataFrame(records)
df.to_csv("labels.csv", index=False)
print("已完成標註，labels.csv 產生！")
