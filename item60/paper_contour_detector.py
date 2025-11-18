import cv2
import numpy as np


class PaperContourDetector:
    """
    紙張輪廓偵測器，計算與 1/4 A4 長方形的距離
    """

    def __init__(self):
        self.paper_contours = []
        self.rectangle_corners = []
        self.max_distances = []

    def detect_paper_contours(self, image, region_mask=None):
        """
        偵測圖片中的紙張輪廓

        Args:
            image: 輸入圖像 (BGR 格式)
            region_mask: 區域遮罩，只在指定區域偵測輪廓

        Returns:
            contours: 偵測到的輪廓
        """
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 如果有區域遮罩，只在指定區域處理
        if region_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=region_mask)

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 邊緣偵測
        edges = cv2.Canny(blurred, 50, 150)

        # 形態學操作，填補邊緣
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 尋找輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 過濾輪廓：只保留面積較大的
        min_area = 1000  # 最小面積閾值
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # 按面積排序，取最大的幾個
        filtered_contours.sort(key=cv2.contourArea, reverse=True)

        return filtered_contours[:5]  # 返回最大的5個輪廓

    def point_to_line_distance(self, point, line_start, line_end):
        """
        計算點到線段的距離

        Args:
            point: 點座標 (x, y)
            line_start: 線段起點 (x, y)
            line_end: 線段終點 (x, y)

        Returns:
            distance: 距離值
        """
        # 向量計算
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)

        # 線段長度
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return np.linalg.norm(point_vec)

        # 投影長度
        projection = np.dot(point_vec, line_vec) / line_length

        if projection < 0:
            # 點在線段起點之前
            return np.linalg.norm(point_vec)
        elif projection > line_length:
            # 點在線段終點之後
            return np.linalg.norm(np.array(point) - np.array(line_end))
        else:
            # 點在線段範圍內
            closest_point = np.array(line_start) + (projection / line_length) * line_vec
            return np.linalg.norm(np.array(point) - closest_point)

    def calculate_distances_to_rectangle(self, contour, rectangle_corners):
        """
        計算輪廓點到長方形邊和角點的距離，比較並保留較長的距離

        Args:
            contour: 輪廓點集
            rectangle_corners: 長方形的四個角點

        Returns:
            max_distance: 最大距離
            max_point: 最大距離對應的點
            distance_type: 距離類型 ('edge' 或 'corner')
            reference_info: 參考資訊 (邊索引或角點索引)
        """
        max_distance = 0
        max_point = None
        distance_type = None
        reference_info = None

        # 長方形的四條邊
        edges = [
            (rectangle_corners[0], rectangle_corners[1]),  # 上邊
            (rectangle_corners[1], rectangle_corners[2]),  # 右邊
            (rectangle_corners[2], rectangle_corners[3]),  # 下邊
            (rectangle_corners[3], rectangle_corners[0]),  # 左邊
        ]

        # 遍歷輪廓上的每個點
        for point in contour:
            point_coord = (point[0][0], point[0][1])

            # 1. 計算到四條邊的最短距離
            min_dist_to_edges = float("inf")
            nearest_edge = None

            for i, (edge_start, edge_end) in enumerate(edges):
                dist = self.point_to_line_distance(point_coord, edge_start, edge_end)
                if dist < min_dist_to_edges:
                    min_dist_to_edges = dist
                    nearest_edge = i

            # 2. 計算到四個角點的垂直距離 (歐幾里得距離)
            min_dist_to_corners = float("inf")
            nearest_corner = None

            for i, corner in enumerate(rectangle_corners):
                # 計算點到角點的直線距離
                dist = np.linalg.norm(np.array(point_coord) - np.array(corner))
                if dist < min_dist_to_corners:
                    min_dist_to_corners = dist
                    nearest_corner = i

            # 3. 比較邊距離和角點距離，選擇較長的
            current_max_dist = max(min_dist_to_edges, min_dist_to_corners)
            current_type = (
                "edge" if min_dist_to_edges >= min_dist_to_corners else "corner"
            )
            current_ref = nearest_edge if current_type == "edge" else nearest_corner

            # 4. 更新全局最大距離
            if current_max_dist > max_distance:
                max_distance = current_max_dist
                max_point = point_coord
                distance_type = current_type
                reference_info = current_ref

        return max_distance, max_point, distance_type, reference_info

    def create_region_masks(self, image_shape, rectangles_info):
        """
        根據ArUco標記位置創建左右分區遮罩

        Args:
            image_shape: 圖像形狀 (height, width)
            rectangles_info: 長方形資訊列表

        Returns:
            left_mask: 左側區域遮罩
            right_mask: 右側區域遮罩
            left_rectangles: 左側區域的長方形
            right_rectangles: 右側區域的長方形
        """
        height, width = image_shape[:2]
        image_center_x = width // 2

        # 創建左右遮罩
        left_mask = np.zeros((height, width), dtype=np.uint8)
        right_mask = np.zeros((height, width), dtype=np.uint8)

        # 左側區域：圖像左半部
        left_mask[:, :image_center_x] = 255
        # 右側區域：圖像右半部
        right_mask[:, image_center_x:] = 255

        # 根據長方形中心位置分組
        left_rectangles = []
        right_rectangles = []

        for rect_info in rectangles_info:
            corners = rect_info["corners"]
            # 計算長方形中心點
            center_x = np.mean([corner[0] for corner in corners])

            if center_x < image_center_x:
                left_rectangles.append(rect_info)
            else:
                right_rectangles.append(rect_info)

        return left_mask, right_mask, left_rectangles, right_rectangles

    def calculate_cutting_score(
        self, distance_pixels, px_to_mm_ratio=None, has_cutting_evidence=True
    ):
        """
        計算剪紙評分

        Args:
            distance_pixels: 距離（像素）
            px_to_mm_ratio: 像素到毫米的轉換比例（如果有的話）
            has_cutting_evidence: 是否有剪切證據（是否真的剪下去了）

        Returns:
            score: 評分 (0, 1, 2)
            score_description: 評分描述
        """
        if not has_cutting_evidence:
            return 0, "小朋友只是動動剪刀而沒剪下去"

        if px_to_mm_ratio:
            distance_cm = distance_pixels / px_to_mm_ratio / 10
            if distance_cm < 1.2:
                return 2, f"沿著線剪完，且間距{distance_cm:.2f}cm < 1.2cm"
            else:
                return 1, f"沿著線剪完，但間距{distance_cm:.2f}cm ≥ 1.2cm"
        else:
            # 沒有比例尺時，使用像素作為粗略估計
            return -1, f"距離{distance_pixels:.1f}px，無比例尺無法精確評分"

    def process_image_with_rectangles(self, image, rectangles_info):
        """
        處理圖像，計算紙張輪廓與長方形的距離
        實現左右分區計算：左框與左紙計算，右框與右紙計算

        Args:
            image: 輸入圖像
            rectangles_info: 長方形資訊列表 [{'corners': corners, 'marker_id': id}, ...]

        Returns:
            result_image: 標註結果的圖像
            distance_results: 距離計算結果
        """
        result_image = image.copy()
        distance_results = []

        if not rectangles_info:
            print("沒有長方形資訊，無法進行分析")
            return result_image, distance_results

        # 創建左右分區遮罩
        left_mask, right_mask, left_rectangles, right_rectangles = (
            self.create_region_masks(image.shape, rectangles_info)
        )

        print(f"\n=== 左右分區分析 ===")
        print(f"左側區域長方形數量: {len(left_rectangles)}")
        print(f"右側區域長方形數量: {len(right_rectangles)}")

        # 在結果圖像上繪製分區線
        center_x = image.shape[1] // 2
        cv2.line(
            result_image, (center_x, 0), (center_x, image.shape[0]), (128, 128, 128), 2
        )

        # 處理左側區域
        if left_rectangles:
            print(f"\n--- 處理左側區域 ---")
            left_contours = self.detect_paper_contours(image, left_mask)
            if left_contours:
                print(f"左側偵測到 {len(left_contours)} 個紙張輪廓")
                # 繪製左側紙張輪廓 (藍色)
                cv2.drawContours(result_image, left_contours, -1, (255, 0, 0), 2)

                # 處理左側長方形
                for rect_info in left_rectangles:
                    distance_result = self.calculate_rectangle_distance(
                        rect_info, left_contours, result_image, "左側"
                    )
                    if distance_result:
                        distance_results.append(distance_result)
            else:
                print("左側未偵測到紙張輪廓")

        # 處理右側區域
        if right_rectangles:
            print(f"\n--- 處理右側區域 ---")
            right_contours = self.detect_paper_contours(image, right_mask)
            if right_contours:
                print(f"右側偵測到 {len(right_contours)} 個紙張輪廓")
                # 繪製右側紙張輪廓 (綠色，與左側區別)
                cv2.drawContours(result_image, right_contours, -1, (0, 255, 0), 2)

                # 處理右側長方形
                for rect_info in right_rectangles:
                    distance_result = self.calculate_rectangle_distance(
                        rect_info, right_contours, result_image, "右側"
                    )
                    if distance_result:
                        distance_results.append(distance_result)
            else:
                print("右側未偵測到紙張輪廓")

        return result_image, distance_results

    def calculate_rectangle_distance(
        self, rect_info, contours, result_image, region_name, scale_info=None
    ):
        """
        計算單個長方形與輪廓的距離

        Args:
            rect_info: 長方形資訊
            contours: 輪廓列表
            result_image: 結果圖像
            region_name: 區域名稱

        Returns:
            distance_result: 距離計算結果
        """
        rectangle_corners = rect_info["corners"]
        marker_id = rect_info["marker_id"]

        print(f"\n計算{region_name}標記 ID {marker_id} 的距離:")

        max_overall_distance = 0
        best_contour_idx = -1
        best_point = None
        best_distance_type = None
        best_reference_info = None

        # 對每個紙張輪廓計算距離
        for contour_idx, contour in enumerate(contours):
            max_dist, max_pt, dist_type, ref_info = (
                self.calculate_distances_to_rectangle(contour, rectangle_corners)
            )

            type_text = "邊框" if dist_type == "edge" else "角點"
            print(f"  輪廓 {contour_idx}: 最大距離 {max_dist:.2f} 像素 (到{type_text})")

            if max_dist > max_overall_distance:
                max_overall_distance = max_dist
                best_contour_idx = contour_idx
                best_point = max_pt
                best_distance_type = dist_type
                best_reference_info = ref_info

        # 計算評分
        if scale_info and "px_to_mm_ratio" in scale_info:
            score, score_description = self.calculate_cutting_score(
                max_overall_distance, scale_info["px_to_mm_ratio"]
            )
        else:
            score, score_description = self.calculate_cutting_score(
                max_overall_distance
            )

        # 記錄結果
        result = {
            "marker_id": marker_id,
            "max_distance": max_overall_distance,
            "max_point": best_point,
            "contour_index": best_contour_idx,
            "distance_type": best_distance_type,
            "reference_info": best_reference_info,
            "rectangle_corners": rectangle_corners,
            "region": region_name,
            "score": score,
            "score_description": score_description,
        }

        # 在圖上標註最大距離點
        if best_point:
            self.draw_distance_annotation(result_image, result, rectangle_corners)

        type_text = "邊框" if best_distance_type == "edge" else "角點"
        print(
            f"  {region_name}標記 ID {marker_id} 最大距離: {max_overall_distance:.2f} 像素 (到{type_text})"
        )

        return result

    def draw_distance_annotation(self, result_image, result, rectangle_corners):
        """
        在圖像上繪製距離標註

        Args:
            result_image: 結果圖像
            result: 距離計算結果
            rectangle_corners: 長方形角點
        """
        best_point = result["max_point"]
        best_distance_type = result["distance_type"]
        best_reference_info = result["reference_info"]
        marker_id = result["marker_id"]
        max_overall_distance = result["max_distance"]

        # 標記最大距離點 (紅色圓圈)
        cv2.circle(result_image, best_point, 8, (0, 0, 255), -1)

        # 根據距離類型繪製連線
        if best_distance_type == "edge":
            # 繪製到邊的連線
            edges = [
                (rectangle_corners[0], rectangle_corners[1]),
                (rectangle_corners[1], rectangle_corners[2]),
                (rectangle_corners[2], rectangle_corners[3]),
                (rectangle_corners[3], rectangle_corners[0]),
            ]

            edge_start, edge_end = edges[best_reference_info]
            # 計算最近點
            line_vec = np.array(edge_end) - np.array(edge_start)
            point_vec = np.array(best_point) - np.array(edge_start)
            line_length = np.linalg.norm(line_vec)

            if line_length > 0:
                projection = np.dot(point_vec, line_vec) / line_length
                projection = max(0, min(line_length, projection))
                closest_point = (
                    np.array(edge_start) + (projection / line_length) * line_vec
                )
                closest_point = tuple(closest_point.astype(int))

                # 繪製到邊框的連線 (紅色虛線)
                cv2.line(result_image, best_point, closest_point, (0, 0, 255), 2)

        elif best_distance_type == "corner":
            # 繪製到角點的連線
            corner_point = tuple(rectangle_corners[best_reference_info].astype(int))

            # 繪製到角點的連線 (橙色線)
            cv2.line(result_image, best_point, corner_point, (0, 165, 255), 2)

            # 標記目標角點 (橙色圓圈)
            cv2.circle(result_image, corner_point, 10, (0, 165, 255), 2)

        # 標註距離文字和類型
        distance_type_text = "邊框" if best_distance_type == "edge" else "角點"
        region_text = result.get("region", "")
        text = f"{region_text}ID{marker_id}: {max_overall_distance:.1f}px({distance_type_text})"
        text_pos = (best_point[0] + 10, best_point[1] - 10)
        cv2.putText(
            result_image,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    def find_longest_distance(self, distance_results):
        """
        找出所有結果中的最長距離

        Args:
            distance_results: 距離計算結果列表

        Returns:
            longest_result: 最長距離的結果
        """
        if not distance_results:
            return None

        longest_result = max(distance_results, key=lambda x: x["max_distance"])

        print(f"\n=== 最長距離結果 ===")
        print(f"標記 ID: {longest_result['marker_id']}")
        print(f"最長距離: {longest_result['max_distance']:.2f} 像素")
        print(f"位置: {longest_result['max_point']}")

        return longest_result
