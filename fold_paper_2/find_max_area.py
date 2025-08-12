import cv2
import numpy as np


class MaxAreaQuadFinder:
    def __init__(self, image_path):
        self.px2cm = 47.4416628993705  # 像素轉公分的比例
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
        edges = cv2.Canny(gray, 20, 150)
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
            if self.side_lengths:
                print("邊長 :")
                for k, v in self.side_lengths.items():
                    print(f"  {k}: {v:.2f} 像素, {(v / self.px2cm):.2f} cm")
                # 標註邊長
                self._annotate_lengths()
            resized = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Max Area Quad", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # print(self.image_path)
            name = self.image_path.split("\\")[-1].split("_")[0]
            # print(name)
            print(rf"儲存結果到 result\{name}_max_area_quad.jpg")
            cv2.imwrite(rf"result\{name}_max_area_quad.jpg", self.img)
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
                f"{name}:{(length / self.px2cm):.1f}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
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
