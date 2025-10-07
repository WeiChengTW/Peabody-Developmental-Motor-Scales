import cv2
import numpy as np
import os

def is_asymmetric(img_path, threshold=0.27, debug=True, scale=3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"無法讀取: {img_path}")
        return False
    # 強制轉 float
    binary = (img > 127).astype(np.float32)
    flip_lr = np.fliplr(binary)
    diff_lr = np.abs(binary - flip_lr)
    asym_score_lr = np.sum(diff_lr) / binary.size
    flip_ud = np.flipud(binary)
    diff_ud = np.abs(binary - flip_ud)
    asym_score_ud = np.sum(diff_ud) / binary.size
    asym_score = max(asym_score_lr, asym_score_ud)
    # asym_score = (asym_score_lr + asym_score_ud) / 2


    if debug:
        print("binary min/max:", binary.min(), binary.max())
        print("diff_lr min/max:", diff_lr.min(), diff_lr.max())
        print(f"{os.path.basename(img_path)}: 左右分數={asym_score_lr:.3f}, 上下分數={asym_score_ud:.3f}, max={asym_score:.3f}")
        print("判斷：", "非對稱" if asym_score > threshold else "對稱")
        def resize(img):
            return cv2.resize((img*255).astype(np.uint8), (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Original", resize(binary))
        cv2.imshow("Flip_LR", resize(flip_lr))
        # cv2.imshow("Diff_LR", resize(diff_lr))
        cv2.imshow("Flip_UD", resize(flip_ud))
        # cv2.imshow("Diff_UD", resize(diff_ud))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return asym_score > threshold

src_folder = "test"
extensions = ('.jpg', '.jpeg', '.png')
for fname in os.listdir(src_folder):
    if not fname.lower().endswith(extensions):
        continue
    img_path = os.path.join(src_folder, fname)
    is_asymmetric(img_path, threshold=0.27, debug=True, scale=3)
