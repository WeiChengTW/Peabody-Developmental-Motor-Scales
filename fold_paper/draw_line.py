import numpy as np
import cv2
from collections import defaultdict
import math
from distance3 import detect_all_points_in_binary_image


class LineDirectionDetector:
    def __init__(self):
        """線條方向檢測器 - 檢測4個主要方向並連接斷線"""
        # 定義四個主要方向：水平、垂直、對角線
        self.directions = {
            "horizontal": 0,  # 水平線
            "vertical": 1,  # 垂直線
            "diagonal_1": 2,  # 正對角線 /
            "diagonal_2": 3,  # 負對角線 \
        }

    def get_direction(self, x1, y1, x2, y2):
        """根據兩點計算方向"""
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-6:  # 垂直線
            return self.directions["vertical"]
        elif abs(dy) < 1e-6:  # 水平線
            return self.directions["horizontal"]
        else:
            slope = dy / dx
            if slope > 0:
                return self.directions["diagonal_2"]  # 正斜率 \
            else:
                return self.directions["diagonal_1"]  # 負斜率 /

    def analyze_local_direction(self, points, center_idx, window_size=5):
        """分析點周圍的方向"""
        if center_idx < window_size or center_idx >= len(points) - window_size:
            return None

        center_x, center_y = points[center_idx]
        directions = []

        # 檢查周圍5x5框內的點
        for i in range(
            max(0, center_idx - window_size),
            min(len(points), center_idx + window_size + 1),
        ):
            if i == center_idx:
                continue

            x, y = points[i]
            # 檢查是否在5x5框內
            if abs(x - center_x) <= 5 and abs(y - center_y) <= 5:
                direction = self.get_direction(center_x, center_y, x, y)
                directions.append(direction)

        if not directions:
            return None

        # 返回最常見的方向
        direction_counts = defaultdict(int)
        for d in directions:
            direction_counts[d] += 1

        return max(direction_counts.items(), key=lambda x: x[1])[0]

    def find_direction_from_neighbors(self, points, center_idx, neighbor_range=5):
        """從前後鄰居點找出方向"""
        directions = []
        center_x, center_y = points[center_idx]

        # 檢查前5個點
        for i in range(max(0, center_idx - neighbor_range), center_idx):
            x, y = points[i]
            direction = self.get_direction(center_x, center_y, x, y)
            directions.append(direction)

        # 檢查後5個點
        for i in range(
            center_idx + 1, min(len(points), center_idx + neighbor_range + 1)
        ):
            x, y = points[i]
            direction = self.get_direction(center_x, center_y, x, y)
            directions.append(direction)

        if not directions:
            return None

        # 返回最常見的方向
        direction_counts = defaultdict(int)
        for d in directions:
            direction_counts[d] += 1

        return max(direction_counts.items(), key=lambda x: x[1])[0]

    def connect_broken_lines(self, points, max_gap=10):
        """連接斷線"""
        if len(points) < 3:
            return points

        connected_points = []
        point_directions = {}

        # 為每個點計算方向
        for i in range(len(points)):
            # 先嘗試從周圍5x5框內的點分析方向
            direction = self.analyze_local_direction(points, i)

            # 如果無法從局部分析，則從前後鄰居找方向
            if direction is None:
                direction = self.find_direction_from_neighbors(points, i)

            point_directions[i] = direction

        # 按座標排序以便連接
        indexed_points = [(i, points[i]) for i in range(len(points))]
        indexed_points.sort(key=lambda x: (x[1][1], x[1][0]))  # 按y,x排序

        connected_points = [point[1] for point in indexed_points]

        # 補齊斷線
        final_points = []
        i = 0
        while i < len(connected_points):
            current_point = connected_points[i]
            final_points.append(current_point)

            # 查找下一個點
            if i < len(connected_points) - 1:
                next_point = connected_points[i + 1]
                current_idx = indexed_points[i][0]
                current_direction = point_directions.get(current_idx)

                # 計算距離
                distance = math.sqrt(
                    (next_point[0] - current_point[0]) ** 2
                    + (next_point[1] - current_point[1]) ** 2
                )

                # 如果距離太大且有方向信息，則插入中間點
                if distance > max_gap and current_direction is not None:
                    interpolated_points = self.interpolate_points(
                        current_point, next_point, current_direction
                    )
                    final_points.extend(interpolated_points)

            i += 1

        return final_points

    def interpolate_points(self, point1, point2, direction):
        """根據方向插入中間點"""
        x1, y1 = point1
        x2, y2 = point2

        interpolated = []

        if direction == self.directions["horizontal"]:
            # 水平方向插值
            if x1 != x2:
                steps = abs(x2 - x1)
                for step in range(1, steps):
                    x = x1 + (x2 - x1) * step / steps
                    interpolated.append((int(x), y1))

        elif direction == self.directions["vertical"]:
            # 垂直方向插值
            if y1 != y2:
                steps = abs(y2 - y1)
                for step in range(1, steps):
                    y = y1 + (y2 - y1) * step / steps
                    interpolated.append((x1, int(y)))

        else:
            # 對角線方向插值
            distance = max(abs(x2 - x1), abs(y2 - y1))
            if distance > 1:
                for step in range(1, distance):
                    x = x1 + (x2 - x1) * step / distance
                    y = y1 + (y2 - y1) * step / distance
                    interpolated.append((int(x), int(y)))

        return interpolated

    def draw_connected_lines(
        self,
        points,
        image_size=(50, 50),
        line_color=(255, 255, 255),
        background_color=(0, 0, 0),
        line_thickness=1,
    ):
        """繪製連接後的線條"""
        # 創建黑色背景圖像
        image = np.full(
            (image_size[1], image_size[0], 3), background_color, dtype=np.uint8
        )

        # 連接斷線
        connected_points = self.connect_broken_lines(points)

        # 繪製所有點
        for point in connected_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(image, (x, y), line_thickness, line_color, -1)

        # 繪製連接線
        if len(connected_points) > 1:
            for i in range(len(connected_points) - 1):
                pt1 = (int(connected_points[i][0]), int(connected_points[i][1]))
                pt2 = (int(connected_points[i + 1][0]), int(connected_points[i + 1][1]))

                # 確保點在圖像範圍內
                if (
                    0 <= pt1[0] < image_size[0]
                    and 0 <= pt1[1] < image_size[1]
                    and 0 <= pt2[0] < image_size[0]
                    and 0 <= pt2[1] < image_size[1]
                ):
                    cv2.line(image, pt1, pt2, line_color, line_thickness)

        return image, connected_points

    def analyze_and_visualize(self, points, save_path=None):
        """分析並視覺化線條"""
        print(f"總共有 {len(points)} 個點")

        # 計算圖像大小
        if points:
            max_x = max(point[0] for point in points)
            max_y = max(point[1] for point in points)
            image_size = (max_x + 50, max_y + 50)
        else:
            image_size = (800, 600)

        # 繪製連接後的線條
        result_image, connected_points = self.draw_connected_lines(points, image_size)

        print(f"連接後共有 {len(connected_points)} 個點")

        # 顯示結果
        # cv2.imshow("Connected Lines", result_image)

        # 保存結果
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"結果已保存到: {save_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return result_image, connected_points


def main():

    image_path = r"4.png"  # 或者其他二值化圖像
    all_point, n_point = detect_all_points_in_binary_image(image_path)

    print(f"總白色點數量: {n_point}")
    print(f"所有點的座標: {all_point}")
    # 創建檢測器
    detector = LineDirectionDetector()

    # 分析並視覺化
    result_image, connected_points = detector.analyze_and_visualize(
        all_point, save_path="connected_lines_result.jpg"
    )

    print("線條連接完成！")
    return result_image, connected_points


if __name__ == "__main__":
    main()
