import os

raw_dir = r"Dia/test"
for idx, filename in enumerate(os.listdir(raw_dir), 1):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        old_path = os.path.join(raw_dir, filename)
        new_filename = f"{idx}.jpg"
        new_path = os.path.join(raw_dir, new_filename)
        os.rename(old_path, new_path)
