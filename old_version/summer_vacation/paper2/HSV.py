import cv2
import numpy as np
import os


def nothing(x):
    pass


img_dir = r"paper2\test_img"
img_files = [
    os.path.join(img_dir, f)
    for f in os.listdir(img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]
img_files.sort()
if not img_files:
    print("No images found in the directory.")
    exit()

cv2.namedWindow("HSV")
mouse_x, mouse_y = 0, 0
clicked_hsv = None


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, clicked_hsv
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, _ = frame.shape
        x_, y_ = min(x, w - 1), min(y, h - 1)
        clicked_hsv = hsv[y_, x_]


idx = 0
while True:
    frame = cv2.imread(img_files[idx])
    if frame is None:
        print(f"Failed to load {img_files[idx]}")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w, _ = frame.shape
    x, y = min(mouse_x, w - 1), min(mouse_y, h - 1)
    hsv_value = hsv[y, x]
    bgr_value = frame[y, x]
    text = f"HSV: {hsv_value} BGR: {bgr_value}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if clicked_hsv is not None:
        cv2.putText(
            frame,
            f"Clicked HSV: {clicked_hsv}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    x1, y1 = max(x - 10, 0), max(y - 10, 0)
    x2, y2 = min(x + 10, w - 1), min(y + 10, h - 1)
    roi = hsv[y1:y2, x1:x2]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR)
    frame[0 : roi_bgr.shape[0], 0 : roi_bgr.shape[1]] = roi_bgr

    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.setMouseCallback("HSV", mouse_callback, {"frame": frame})

    cv2.imshow("HSV", frame)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == ord("d") or key == 83:  # 右鍵或d
        idx = (idx + 1) % len(img_files)
        clicked_hsv = None
    elif key == ord("a") or key == 81:  # 左鍵或a
        idx = (idx - 1) % len(img_files)
        clicked_hsv = None

cv2.destroyAllWindows()
