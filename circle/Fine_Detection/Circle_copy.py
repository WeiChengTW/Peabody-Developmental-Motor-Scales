import cv2
import os

input_dir = r"result\Circle"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, (500, 500))
            cv2.imshow(filename, resized_image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)
