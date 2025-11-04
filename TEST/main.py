import cv2
import numpy as np
from threading import Lock


class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = Lock()
        self.running = True

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        height, width = frame.shape[:2]

        # 設定文字參數
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        font_color = (0, 255, 0)  # 綠色

        # 計算文字大小和位置
        text = "HELLO"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # 在影像上繪製文字
        cv2.putText(
            frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness
        )
        return frame

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        height, width = frame.shape[:2]

        # 設定文字參數
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        font_color = (0, 255, 0)  # 綠色

        # 計算文字大小和位置
        text = "HELLO"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # 在影像上繪製文字
        cv2.putText(
            frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness
        )

        return frame

    def run(self):
        while self.running:
            self.get_frame()
            if not self.running:
                break

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


# 創建全域的攝影機物件
camera = CameraStream()

if __name__ == "__main__":
    camera.run()
