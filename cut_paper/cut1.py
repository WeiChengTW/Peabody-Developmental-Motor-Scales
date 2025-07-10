import cv2
import numpy as np


# 用來儲存所有點選的 HSV
clicked_hsv = []


def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        color = hsv[y, x]
        clicked_hsv.append(color)
        print(f"點選 HSV: {color}")


def count_pink_pixels_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return
    print("請用滑鼠左鍵點選你想要判斷為粉紅色的區域，按 q 結束並自動計算範圍...")
    global clicked_hsv
    clicked_hsv = []
    selected_frame = None

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color = hsv[y, x]
            clicked_hsv.append(color)
            print(f"點選 HSV: {color}")

    # 設定裁切區域（例如中央 1/2 區域）
    crop_ratio = 0.6  # 裁切比例（中央 50%）
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像")
            break
        h, w = frame.shape[:2]
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        frame_crop = frame[y1 : y1 + ch, x1 : x1 + cw]
        frame_zoom = cv2.resize(
            frame_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("原始鏡頭", frame_zoom)
        cv2.setMouseCallback("原始鏡頭", mouse_callback, frame_zoom)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            selected_frame = frame_zoom.copy()
            break
    cap.release()
    cv2.destroyWindow("原始鏡頭")
    if not clicked_hsv:
        print("未點選任何顏色，使用預設範圍")
        lower_pink = np.array([113, 56, 147])
        upper_pink = np.array([126, 84, 225])
    else:
        arr = np.array(clicked_hsv)
        lower_pink = arr.min(axis=0)
        upper_pink = arr.max(axis=0)
        print(
            f"自動計算粉紅色範圍: lower_pink = np.array({lower_pink.tolist()}), upper_pink = np.array({upper_pink.tolist()})"
        )
    if selected_frame is not None:
        hsv = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        result = selected_frame.copy()
        result[mask > 0] = [0, 0, 255]
        cv2.imshow("pink", result)
        pink_pixels = cv2.countNonZero(mask)
        print(f"粉紅色像素數量: {pink_pixels}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return pink_pixels
    return None


if __name__ == "__main__":
    count_pink_pixels_from_camera()
