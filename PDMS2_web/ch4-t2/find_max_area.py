import cv2
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class MaxAreaQuadFinder:
    def __init__(self, image_path):
        try:
            json_path = BASE_DIR.parent / "px2cm.json"
            with open(json_path, "r") as f:
                data = json.load(f)
                self.px2cm = data["px2cm"]
        except FileNotFoundError:
            self.px2cm = 47.4416628993705  # 預設值
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"無法讀取影像: {image_path}")
        self.max_area = 0
        self.max_contour = None
        self.ordered_pts = None  # (tl, tr, br, bl)
        self.side_lengths = None  # dict

    def find_max_area_quad(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 120)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > self.max_area:
                    self.max_area = area
                    self.max_contour = approx
        if self.max_contour is not None:
            self.ordered_pts = self._order_points(self.max_contour.reshape(4, 2))
            self.side_lengths = self._compute_side_lengths(self.ordered_pts)

    def draw_and_show(self):
        if self.max_contour is not None:
            cv2.drawContours(self.img, [self.max_contour], -1, (0, 255, 0), 3)
            print("最大面積:", self.max_area)
            print("四個頂點座標:\n", self.max_contour.reshape(4, 2))
            # 顯示四邊長於左上角
            if self.side_lengths:
                print("邊長 :")
                lines = []
                for k in ["top", "right", "bottom", "left"]:
                    v = self.side_lengths[k]
                    lines.append(f"{k}: {(v / self.px2cm):.2f} cm")
                    print(f"  {k}: {v:.2f} 像素, {(v / self.px2cm):.2f} cm")
                # 標註邊長
                self._annotate_lengths()
                # 將四邊長資訊畫在左上角
                x0, y0 = 10, 30
                for i, text in enumerate(lines):
                    cv2.putText(
                        self.img,
                        text,
                        (x0 + 50, y0 + (i * 60) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            resized = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow("Max Area Quad", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            name = self.image_path.split("\\")[-1].split("_")[0]
            print(rf"儲存結果到 ch4-t2\result\{name}_max_area_quad.jpg")
            cv2.imwrite(rf"ch4-t2\result\{name}_max_area_quad.jpg", self.img)
            # 找出與7.5差距最大的邊
            if self.side_lengths:
                max_diff = 0
                max_diff_side = None
                for side_name in ["top", "right", "bottom", "left"]:
                    side_length_cm = self.side_lengths[side_name] / self.px2cm
                    diff = abs(side_length_cm - 7.5)
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_side = side_name
                print(f"與7.5cm差距最大的邊: {max_diff_side}, 差距: {max_diff:.2f} cm")
                return self.img, max_diff
        else:
            print("找不到四邊形")

    @staticmethod
    def _order_points(pts):
        # pts: (4,2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # tl
        rect[2] = pts[np.argmax(s)]  # br
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # tr
        rect[3] = pts[np.argmax(diff)]  # bl
        return rect

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(a - b))

    def _compute_side_lengths(self, rect):
        (tl, tr, br, bl) = rect
        return {
            "top": self._dist(tl, tr),
            "right": self._dist(tr, br),
            "bottom": self._dist(br, bl),
            "left": self._dist(bl, tl),
            "perimeter": self._dist(tl, tr)
            + self._dist(tr, br)
            + self._dist(br, bl)
            + self._dist(bl, tl),
        }

    def _annotate_lengths(self):
        (tl, tr, br, bl) = self.ordered_pts.astype(int)
        pairs = {
            "top": (tl, tr),
            "right": (tr, br),
            "bottom": (br, bl),
            "left": (bl, tl),
        }
        for name, (p1, p2) in pairs.items():
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            length = self.side_lengths[name]
            cv2.putText(
                self.img,
                f"{name}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def get_side_lengths(self):
        """取得邊長 (像素) 字典，如果尚未計算或找不到四邊形則回傳 None"""
        return self.side_lengths


# 使用範例
if __name__ == "__main__":
    finder = MaxAreaQuadFinder("edges/4_edges_paper.jpg")
    finder.find_max_area_quad()
    finder.draw_and_show()
