# 用 OpenCV 攝影機即時進行顏色區塊檢測
import cv2
import numpy as np


def camera_color_block_detection():
    # 開啟攝影機 (預設攝影機索引為 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    print("攝影機已開啟，按 'q' 或 'Q' 退出")

    while True:
        # 讀取攝影機畫面
        ret, frame = cap.read()

        if not ret:
            print("無法從攝影機讀取畫面")
            break

        # 轉換為 HSV 色彩空間，更適合顏色分割
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 增強對比度和飽和度
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)  # 增強飽和度
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.1)  # 略微增強亮度
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)

        # 高斯模糊去除雜訊
        blur = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 定義主要顏色範圍 (HSV) - 提高對比度和飽和度要求
        color_ranges = {
            "Orange": [(11, 120, 120), (22, 255, 255)],  # 縮小範圍，提高飽和度
            "Yellow": [(23, 150, 150), (35, 255, 255)],  # 提高飽和度和亮度要求
            "Green": [(36, 80, 80), (85, 255, 255)],
            "Cyan": [(86, 80, 80), (100, 255, 255)],
            "Blue": [(101, 80, 80), (130, 255, 255)],
            "Purple": [(131, 80, 80), (170, 255, 255)],
            "Pink": [(171, 80, 80), (179, 255, 255)],
        }

        total_blocks = 0
        result_frame = frame.copy()

        # 對每種顏色進行檢測
        for color_name, (lower, upper) in color_ranges.items():
            # 建立顏色遮罩
            mask = cv2.inRange(blur, np.array(lower), np.array(upper))

            # 形態學操作：先開運算去雜訊，再閉運算填補空洞
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 尋找輪廓
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 過濾掉太小的區域
            min_area = 700  # 最小面積閾值
            valid_contours = [
                cnt for cnt in contours if cv2.contourArea(cnt) > min_area
            ]

            if valid_contours:
                # 在影像上繪製輪廓和標籤
                cv2.drawContours(result_frame, valid_contours, -1, (0, 255, 0), 2)

                for i, cnt in enumerate(valid_contours):
                    # 獲取輪廓的邊界框
                    x, y, w, h = cv2.boundingRect(cnt)
                    # 在輪廓上標註顏色名稱和編號
                    cv2.putText(
                        result_frame,
                        f"{color_name}#{i+1}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                total_blocks += len(valid_contours)

        # 在畫面上顯示總區塊數
        cv2.putText(
            result_frame,
            f"Total Color Blocks: {total_blocks}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        # 邊緣檢測
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200)
        cv2.imshow("Edges", edges)
        cv2.imshow("Color Block Detection", result_frame)

        # 按 'q' 或 'Q' 退出
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q")]:
            break

    # 釋放攝影機資源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_color_block_detection()
