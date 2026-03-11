import cv2
import numpy as np
from Draw_square import Draw_square
import os


class BoxDistanceAnalyzer:
    def __init__(self, box1=None, image_path=None):
        self.box1 = box1
        self.image_path = image_path
        self.box2 = None

    def detect_main_contour_points(self, image_path, show_debug=False):
        """
        只取來自面積大於1000的輪廓的邊緣點
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖片: {image_path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 100, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                continue
            contour_points = contour.reshape(-1, 2)
            points.append(contour_points)
        if points:
            points = np.concatenate(points, axis=0)
        else:
            points = np.empty((0, 2), dtype=int)

        if show_debug:
            debug_image = img.copy()
            for pt in points:
                cv2.circle(debug_image, tuple(pt), 1, (0, 255, 255), 1)
            cv2.imshow("debug_edges_points", debug_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return points

    def detect_main_contours(self, image_path, min_area=3000):
        """
        取得主要(外部)輪廓清單，僅保留面積 >= min_area 的輪廓。
        回傳: list[np.ndarray]，每個為 (N,1,2) 的 contour。
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖片: {image_path}")
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 100, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        good = []
        for c in contours:
            if cv2.contourArea(c) >= float(min_area):
                good.append(c)
        return good

    @staticmethod
    def _closest_point_on_segment(p, a, b):
        """取得點 p 到線段 ab 的最近點（垂足，若超出端點則回端點）。
        p, a, b: np.array([x, y])
        回傳: (closest_point, distance)
        """
        ap = p - a
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq == 0.0:
            # 退化為單點
            return a, float(np.linalg.norm(p - a))
        t = float(np.dot(ap, ab)) / ab_len_sq
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return proj, float(np.linalg.norm(p - proj))

    def _min_distance_to_polygon_edges(self, p, poly):
        """計算點 p 到多邊形 poly 的各邊的最短距離與對應最近點。
        poly: (N,2) 順時針或逆時針
        回傳: (min_dist, closest_point)
        """
        min_dist = float("inf")
        closest = None
        n = len(poly)
        for i in range(n):
            a = poly[i].astype(float)
            b = poly[(i + 1) % n].astype(float)
            foot, d = self._closest_point_on_segment(p.astype(float), a, b)
            if d < min_dist:
                min_dist = d
                closest = foot
        return min_dist, closest

    def _min_distance_to_contours(self, p, contours):
        """
        計算點 p 到多條輪廓(折線)的最短距離與對應最近點。
        contours: list of contours (每個 contour 形如 (N,1,2))
        回傳: (min_dist, closest_point)
        """
        min_dist = float("inf")
        closest = None
        P = p.astype(float)
        for c in contours:
            if c is None or len(c) < 2:
                continue
            pts = c.reshape(-1, 2).astype(float)
            n = len(pts)
            for i in range(n - 1):  # 折線，不自動閉合
                a = pts[i]
                b = pts[i + 1]
                foot, d = self._closest_point_on_segment(P, a, b)
                if d < min_dist:
                    min_dist = d
                    closest = foot
        return min_dist, closest

    def draw_main_contour(self, img, color=(0, 255, 255), thickness=1):
        """
        在 img 上畫出最大輪廓的邊緣點（黃色）
        """
        points = self.detect_main_contour_points(self.image_path)
        if points is not None:
            for pt in points:
                cv2.circle(img, tuple(pt), 1, color, thickness)
        return img

    def max_distance_between_boxes(self, box1, box2):
        """
        box1, box2: shape (4,2) numpy array
        計算兩個四邊形所有點對的最大距離
        """
        max_dist = 0
        max_pair = (None, None)
        for pt1 in box1:
            for pt2 in box2:
                dist = np.linalg.norm(pt1 - pt2)
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (tuple(pt1), tuple(pt2))
        return max_dist, max_pair

    def analyze(self, pixel_per_cm=None, out_path="result"):
        """
        改為：紅點(預設紅框中心)到黃線(主要外部輪廓折線)的垂直最短距。
        - 偵測外部主要輪廓並以黃色線表示。
        - 取紅框中心作為紅點，量測其到黃線最近距離與垂足。
        - 標示距離(若提供 pixel_per_cm 則轉換為公分)。
        輸出結果圖於 out_path。
        """
        if pixel_per_cm is None:
            print("未提供每公分像素數")
            return
        if self.box1 is None or self.image_path is None:
            print("box1 或 image_path 尚未設定")
            return

        # 取得主要外部輪廓(黃線來源)
        contours = self.detect_main_contours(self.image_path, min_area=3000)
        if contours is None or len(contours) == 0:
            print("未偵測到任何輪廓，無法計算紅點到黃線距離")
            return

        # 紅點：取紅框中心
        red_center = np.mean(self.box1.astype(float), axis=0)

        # 紅點到黃線(輪廓折線)的最短距離與垂足
        d_unsigned, foot = self._min_distance_to_contours(red_center, contours)

        img = cv2.imread(self.image_path)
        if img is None:
            print(f"無法讀取圖片: {self.image_path}")
            return
        canvas = img.copy()

        # 畫紅框
        cv2.polylines(
            canvas,
            [self.box1.astype(np.int32)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=1,
        )
        # 畫黃線(主要輪廓)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)

        # 畫紅點與綠色垂線到垂足
        p = (int(red_center[0]), int(red_center[1]))
        foot_xy = (int(foot[0]), int(foot[1])) if foot is not None else None
        cv2.circle(canvas, p, 4, (0, 0, 255), -1)
        if foot_xy is not None:
            cv2.circle(canvas, foot_xy, 3, (255, 0, 255), -1)
            cv2.line(canvas, p, foot_xy, (0, 200, 0), 2)

        # 顯示距離（公分）
        dist_cm = d_unsigned / float(pixel_per_cm)
        text = f"{dist_cm:.2f} cm"
        cv2.putText(
            canvas,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        os.makedirs(out_path, exist_ok=True)
        name = self.image_path.split("\\")[-1].split("_")[0]
        path = f"{out_path}/{name}_min_dist_red_to_yellow.png"
        cv2.imwrite(path, canvas)
        print(f"紅點(紅框中心)到黃線的最短距離: {dist_cm:.2f} cm，結果已存 {path}")

    def interactive_click_distance(self, pixel_per_cm=None, snap_radius_px=10):
        """
        互動模式量測（紅點到黃線的垂直最短距）：
        - 顯示圖片並畫出黃色輪廓線(黃線)。
        - 滑鼠左鍵點選產生「紅點」，計算此紅點到黃線(主要輪廓折線)的最短距離，
          並畫出綠色垂線與數值標示。
        - 若 pixel_per_cm 提供，顯示公分；否則顯示像素。
        - 按 q 或 Esc 離開。
        """
        if self.image_path is None:
            print("image_path 尚未設定")
            return
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"無法讀取圖片: {self.image_path}")
            return

        # 取得黃色輪廓(黃線)
        contours = self.detect_main_contours(self.image_path, min_area=3000)
        if contours is None or len(contours) == 0:
            print("未偵測到任何輪廓線，無法互動量測")
            return

        # 預先畫底圖：黃線（與可選：黃點雜訊顯示）
        base = img.copy()
        cv2.drawContours(base, contours, -1, (0, 255, 255), 2)  # 畫黃線
        # 若仍希望顯示點雲可取消註解
        # pts_cloud = self.detect_main_contour_points(self.image_path)
        # if pts_cloud is not None:
        #     for pt in pts_cloud:
        #         cv2.circle(base, tuple(pt), 1, (0, 255, 255), 1)

        hint = "Click to measure (q/Esc to quit)"
        cv2.putText(
            base,
            hint,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            base,
            hint,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        win = "Measure Distance"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        state = {"last": None}

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 紅點為使用者點擊位置
                p = np.array([x, y], dtype=float)
                # 到黃線(輪廓折線)的最短距離
                d_unsigned, foot = self._min_distance_to_contours(p, contours)

                if pixel_per_cm and float(pixel_per_cm) > 0:
                    val = d_unsigned / float(pixel_per_cm)
                    label = f"{val:.2f} cm"
                else:
                    val = d_unsigned
                    label = f"{val:.1f} px"

                canvas = base.copy()
                # 畫紅點(點擊處)
                cv2.circle(canvas, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
                foot_xy = (int(foot[0]), int(foot[1]))
                cv2.circle(canvas, foot_xy, 3, (255, 0, 255), -1)
                cv2.line(canvas, (int(p[0]), int(p[1])), foot_xy, (0, 200, 0), 2)
                tx = int(min(max(10, p[0] + 8), canvas.shape[1] - 10))
                ty = int(min(max(30, p[1] - 10), canvas.shape[0] - 10))
                cv2.putText(
                    canvas,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                state["last"] = canvas
                cv2.imshow(win, canvas)

        cv2.setMouseCallback(win, on_mouse)
        cv2.imshow(win, base)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if state["last"] is None:
                cv2.imshow(win, base)
        cv2.destroyWindow(win)


if __name__ == "__main__":
    detector_path = r"extracted\img2_extracted_paper.jpg"
    D_sq_path, black_corners_int = Draw_square(detector_path)

    analyzer = BoxDistanceAnalyzer(box1=black_corners_int, image_path=detector_path)
    # 互動點擊量測：點黃點顯示到紅線距離
    analyzer.interactive_click_distance(pixel_per_cm=None)
