import cv2
import numpy as np


class ImprovedPaperDetector:
    """
    改進的紙張輪廓偵測器，參考ch3-t2的方法
    解決中間分隔線誤判，實現基於ArUco連接區域的左右分區
    """

    def __init__(self):
        self.paper_contours = []
        self.rectangle_corners = []

    def detect_connected_paper_region(self, image, aruco_center):
        """
        偵測與ArUco標記連接的紙張區域
        使用flood fill來找到連接的區域，避免中間分隔線誤判

        Args:
            image: 輸入圖像
            aruco_center: ArUco標記中心點 (x, y)

        Returns:
            paper_contour: 連接的紙張輪廓
            paper_mask: 紙張區域遮罩
        """
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 邊緣偵測 - 使用更精細的參數
        edges = cv2.Canny(blurred, 20, 100, apertureSize=3)

        # 形態學操作來連接斷開的邊緣
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找到所有輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 找到包含ArUco中心點的輪廓
        target_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            # 過濾太小的輪廓
            if area < 3000:
                continue

            # 檢查ArUco中心是否在輪廓內或附近
            center_point = (float(aruco_center[0]), float(aruco_center[1]))
            if cv2.pointPolygonTest(contour, center_point, False) >= 0:
                if area > max_area:
                    max_area = area
                    target_contour = contour

        # 如果沒找到包含中心的輪廓，找最近的大輪廓
        if target_contour is None:
            min_distance = float("inf")
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 3000:
                    continue

                # 計算ArUco中心到輪廓的最短距離
                center_point = (float(aruco_center[0]), float(aruco_center[1]))
                distance = abs(cv2.pointPolygonTest(contour, center_point, True))
                if distance < min_distance:
                    min_distance = distance
                    target_contour = contour

        if target_contour is not None:
            # 創建遮罩
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [target_contour], -1, 255, -1)
            return target_contour, mask

        return None, None

    def get_contour_edge_points(self, contour, area_threshold=3000):
        """
        獲取輪廓的邊緣點，參考ch3-t2的方法

        Args:
            contour: 輪廓
            area_threshold: 面積閾值

        Returns:
            edge_points: 邊緣點集合 (N, 2)
        """
        if contour is None:
            return np.empty((0, 2), dtype=int)

        area = cv2.contourArea(contour)
        if area < area_threshold:
            return np.empty((0, 2), dtype=int)

        # 將輪廓點重塑為 (N, 2) 格式
        contour_points = contour.reshape(-1, 2)
        return contour_points

    @staticmethod
    def _closest_point_on_segment(p, a, b):
        """
        取得點 p 到線段 ab 的最近點（垂足，若超出端點則回端點）
        參考ch3-t2的實現

        Args:
            p, a, b: np.array([x, y])

        Returns:
            (closest_point, distance)
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
        """
        計算點 p 到多邊形 poly 的各邊的最短距離與對應最近點
        參考ch3-t2的實現

        Args:
            p: 點座標
            poly: 多邊形頂點 (N,2)

        Returns:
            (min_dist, closest_point)
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

    def calculate_distances_to_rectangle(self, paper_contour, rectangle_corners):
        """
        計算紙張輪廓到長方形的兩種距離：
        1. 紙張邊緣點到長方形邊框的距離
        2. 長方形角點到紙張邊緣的距離
        參考ch3-t2的方法

        Args:
            paper_contour: 紙張輪廓
            rectangle_corners: 長方形四個角點 (4, 2)

        Returns:
            edge_to_box_result: 紙張邊緣到長方形的距離結果
            corner_to_paper_result: 長方形角點到紙張的距離結果
        """
        # 獲取紙張邊緣點
        edge_points = self.get_contour_edge_points(paper_contour)

        if len(edge_points) == 0:
            return None, None

        red_poly = rectangle_corners.astype(np.int32)
        red_contour = red_poly.reshape(-1, 1, 2)

        # 1. 計算紙張邊緣點到長方形的距離（支援內凹外凸）
        best_abs_signed = 0.0
        best_sign = 0.0
        best_edge_point = None
        best_edge_foot = None

        for edge_point in edge_points:
            # 有號距離：>0 在多邊形內，<0 在外
            signed_d = cv2.pointPolygonTest(
                red_contour, (float(edge_point[0]), float(edge_point[1])), True
            )
            # 無號最近距離與垂足
            d_unsigned, foot = self._min_distance_to_polygon_edges(
                np.array(edge_point, dtype=float), red_poly
            )
            if abs(signed_d) > best_abs_signed:
                best_abs_signed = abs(signed_d)
                best_sign = 1.0 if signed_d >= 0 else -1.0
                best_edge_point = tuple(map(int, edge_point))
                best_edge_foot = tuple(map(int, foot))

        edge_to_box_result = {
            "distance": best_abs_signed,
            "direction": "內凹" if best_sign >= 0 else "外凸",
            "edge_point": best_edge_point,
            "foot_point": best_edge_foot,
            "type": "edge_to_box",
        }

        # 2. 計算長方形四角到紙張邊緣的距離
        best_corner_min_dist = -1.0
        best_corner_idx = -1
        best_corner_point = None
        best_corner_foot = None

        # 將紙張輪廓轉換為邊集合
        paper_points = edge_points
        n_points = len(paper_points)

        for idx, corner in enumerate(red_poly):
            p = corner.astype(float)
            # 角點到紙張邊緣線的最短距離
            min_d = float("inf")
            min_foot = None

            for i in range(n_points):
                a = paper_points[i].astype(float)
                b = paper_points[(i + 1) % n_points].astype(float)
                foot, d = self._closest_point_on_segment(p, a, b)
                if d < min_d:
                    min_d = d
                    min_foot = foot

            # 更新四角中的最大者（最短的最長距離）
            if min_d > best_corner_min_dist:
                best_corner_min_dist = min_d
                best_corner_point = tuple(map(int, corner))
                best_corner_foot = tuple(map(int, min_foot))
                best_corner_idx = idx

        corner_to_paper_result = {
            "distance": best_corner_min_dist,
            "corner_idx": best_corner_idx,
            "corner_point": best_corner_point,
            "foot_point": best_corner_foot,
            "type": "corner_to_paper",
        }

        return edge_to_box_result, corner_to_paper_result

    def process_image_with_rectangles(self, image, rectangles_info):
        """
        處理圖像，計算紙張輪廓與長方形的距離
        實現基於ArUco連接區域的分區計算

        Args:
            image: 輸入圖像
            rectangles_info: 長方形資訊列表

        Returns:
            result_image: 標註結果的圖像
            distance_results: 距離計算結果
        """
        result_image = image.copy()
        distance_results = []

        if not rectangles_info:
            print("沒有長方形資訊，無法進行分析")
            return result_image, distance_results

        print(f"\n=== 基於ArUco連接區域的紙張分析 ===")

        # 對每個ArUco標記及其連接的紙張區域進行分析
        for rect_info in rectangles_info:
            marker_id = rect_info["marker_id"]
            rectangle_corners = rect_info["corners"]

            # 計算ArUco標記中心點
            center_coords = np.mean(rectangle_corners, axis=0).astype(int)
            aruco_center = (int(center_coords[0]), int(center_coords[1]))

            print(f"\n--- 處理標記 ID {marker_id} ---")
            print(f"ArUco中心: {aruco_center}")

            # 偵測與此ArUco連接的紙張區域
            paper_contour, paper_mask = self.detect_connected_paper_region(
                image, aruco_center
            )

            if paper_contour is not None:
                paper_area = cv2.contourArea(paper_contour)
                print(f"偵測到連接的紙張區域，面積: {paper_area:.0f}")

                # 繪製紙張輪廓（根據標記ID用不同顏色）
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                color = colors[marker_id % len(colors)]
                cv2.drawContours(result_image, [paper_contour], -1, color, 2)

                # 計算兩種距離
                edge_result, corner_result = self.calculate_distances_to_rectangle(
                    paper_contour, rectangle_corners
                )

                if edge_result and corner_result:
                    # 繪製距離線和標記
                    self.draw_distance_annotations(
                        result_image, edge_result, corner_result, marker_id
                    )

                    # 記錄結果
                    result = {
                        "marker_id": marker_id,
                        "paper_area": paper_area,
                        "edge_to_box": edge_result,
                        "corner_to_paper": corner_result,
                        "rectangle_corners": rectangle_corners,
                        "aruco_center": aruco_center,
                    }
                    distance_results.append(result)

                    print(
                        f"邊緣到長方形距離: {edge_result['distance']:.2f} 像素 ({edge_result['direction']})"
                    )
                    print(f"角點到紙張距離: {corner_result['distance']:.2f} 像素")
            else:
                print(f"未偵測到與標記 ID {marker_id} 連接的紙張區域")

        return result_image, distance_results

    def draw_distance_annotations(
        self, result_image, edge_result, corner_result, marker_id
    ):
        """
        繪製兩種距離的標註

        Args:
            result_image: 結果圖像
            edge_result: 邊緣到長方形的距離結果
            corner_result: 角點到紙張的距離結果
            marker_id: 標記ID
        """
        # 1. 繪製邊緣到長方形的距離線（綠色）
        if edge_result["edge_point"] and edge_result["foot_point"]:
            cv2.line(
                result_image,
                edge_result["edge_point"],
                edge_result["foot_point"],
                (0, 255, 0),
                2,
            )
            cv2.circle(result_image, edge_result["edge_point"], 3, (255, 0, 0), -1)
            cv2.circle(result_image, edge_result["foot_point"], 3, (255, 0, 255), -1)

            # 標註文字
            text1 = f"ID{marker_id}-Edge: {edge_result['distance']:.1f}px"
            text_pos1 = (
                edge_result["edge_point"][0] + 10,
                edge_result["edge_point"][1] - 10,
            )
            cv2.putText(
                result_image,
                text1,
                text_pos1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # 2. 繪製角點到紙張的距離線（青色）
        if corner_result["corner_point"] and corner_result["foot_point"]:
            cv2.line(
                result_image,
                corner_result["corner_point"],
                corner_result["foot_point"],
                (255, 255, 0),
                2,
            )
            cv2.circle(
                result_image, corner_result["corner_point"], 4, (0, 165, 255), -1
            )
            cv2.circle(result_image, corner_result["foot_point"], 3, (0, 255, 255), -1)

            # 標註文字
            text2 = f"ID{marker_id}-Corner: {corner_result['distance']:.1f}px"
            text_pos2 = (
                corner_result["corner_point"][0] + 10,
                corner_result["corner_point"][1] + 15,
            )
            cv2.putText(
                result_image,
                text2,
                text_pos2,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

    def find_longest_distance(self, distance_results):
        """
        找出所有結果中的最長距離（考慮兩種距離類型）

        Args:
            distance_results: 距離計算結果列表

        Returns:
            longest_result: 最長距離的結果
        """
        if not distance_results:
            return None

        all_distances = []

        for result in distance_results:
            # 邊緣到長方形的距離
            if result["edge_to_box"]:
                all_distances.append(
                    {
                        "marker_id": result["marker_id"],
                        "distance": result["edge_to_box"]["distance"],
                        "type": "edge_to_box",
                        "details": result["edge_to_box"],
                    }
                )

            # 角點到紙張的距離
            if result["corner_to_paper"]:
                all_distances.append(
                    {
                        "marker_id": result["marker_id"],
                        "distance": result["corner_to_paper"]["distance"],
                        "type": "corner_to_paper",
                        "details": result["corner_to_paper"],
                    }
                )

        if not all_distances:
            return None

        # 找最大距離
        longest = max(all_distances, key=lambda x: x["distance"])

        print(f"\n=== 最長距離結果 ===")
        print(f"標記 ID: {longest['marker_id']}")
        print(f"距離類型: {longest['type']}")
        print(f"最長距離: {longest['distance']:.2f} 像素")

        return longest
