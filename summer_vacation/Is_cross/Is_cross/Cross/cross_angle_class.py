import cv2
import numpy as np
import os

class NonRightAngleDetector:
    def __init__(self, threshold=15):
        self.threshold = threshold

    def detect(self, img_path, debug=True, scale=3):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"無法讀取: {img_path}")
            return False
        binary = (img > 127).astype(np.uint8) * 255
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, minLineLength=0.3*min(binary.shape), maxLineGap=20)
        if lines is None or len(lines) < 2:
            print("主線不足兩條，無法計算交角")
            return True

        # 分成水平線（-20~+20度, 160~180度）和垂直線（70~110度）
        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            length = np.linalg.norm([x2-x1, y2-y1])
            if angle < 20 or angle > 160:
                h_lines.append((length, line[0]))
            elif 70 < angle < 110:
                v_lines.append((length, line[0]))

        # 各取最長一條
        h_line = max(h_lines, default=None, key=lambda x: x[0])
        v_line = max(v_lines, default=None, key=lambda x: x[0])

        if h_line is None or v_line is None:
            print("找不到一橫一直")
            return True

        # 計算交角
        lines_chosen = [h_line[1], v_line[1]]
        angles = []
        for x1, y1, x2, y2 in lines_chosen:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            angles.append(angle)
        cross_angle = abs((angles[0] - angles[1]) % 180)
        if cross_angle > 90:
            cross_angle = 180 - cross_angle
        angle_diff = abs(cross_angle - 90)
        is_not_rightangle = angle_diff > self.threshold

        if debug:
            print(f"{os.path.basename(img_path)}: 主橫線+主直線交角={cross_angle:.1f}° (偏差={angle_diff:.1f}°)")
            print("判斷：", "非直角" if is_not_rightangle else "直角")
            color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for idx, (x1, y1, x2, y2) in enumerate(lines_chosen):
                col = (0,0,255) if idx==0 else (0,255,0)
                cv2.line(color, (x1, y1), (x2, y2), col, 2)
            color = cv2.resize(color, (color.shape[1]*scale, color.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Main Horizontal & Vertical Cross Angle", color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return is_not_rightangle

