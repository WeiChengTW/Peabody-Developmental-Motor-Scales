import os
from PIL import Image

raw_dir = "raw"
output_dir = "raw"

for idx, filename in enumerate(sorted(os.listdir(raw_dir)), 1):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        img_path = os.path.join(raw_dir, filename)
        img = Image.open(img_path)

        # 如果圖片是 RGBA 模式，轉換為 RGB 模式
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # 上下翻轉
        img_flip_ud = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_flip_ud.save(os.path.join(output_dir, f"{idx}_2.jpg"))

        # 左右翻轉
        img_flip_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip_lr.save(os.path.join(output_dir, f"{idx}_3.jpg"))
