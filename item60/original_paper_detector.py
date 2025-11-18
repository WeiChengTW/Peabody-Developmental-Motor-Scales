import cv2
import numpy as np


class OriginalPaperDetector:
    """
    使用原本精確紙張輪廓偵測方法的改進版本
    保留左右分區邏輯和雙重距離計算，但使用原本的輪廓偵測方法
    """

    def __init__(self):
        self.paper_contours = []
        self.rectangle_corners = []

    def detect_paper_contours(self, image, region_mask=None, filter_center_line=True):
        """
        偵測圖片中的紙張輪廓 - 使用直接裁切區域的方法避免遮罩邊界問題

        Args:
            image: 輸入圖像 (BGR 格式)
            region_mask: 區域遮罩，用於確定裁切區域（不直接應用遮罩，而是裁切對應區域）
            filter_center_line: 是否過濾中間分隔線

        Returns:
            contours: 偵測到的輪廓（座標已轉換回原圖座標系）
        """
        # 如果有區域遮罩，直接裁切對應區域避免遮罩邊界問題
        if region_mask is not None:
            # 找到遮罩的邊界框
            coords = np.where(region_mask == 255)
            if len(coords[0]) == 0:
                return []  # 遮罩區域為空

            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # 直接裁切圖像區域，完全避免遮罩邊界問題
            cropped_image = image[y_min : y_max + 1, x_min : x_max + 1]
            offset_x, offset_y = x_min, y_min
        else:
            cropped_image = image
            offset_x, offset_y = 0, 0

        # 轉換為灰度圖
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 邊緣偵測
        edges = cv2.Canny(blurred, 50, 150)
        # cv2.imshow("Edges", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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

        # 如果使用了裁切，將輪廓座標轉換回原圖座標系
        if region_mask is not None:
            for i, contour in enumerate(filtered_contours):
                filtered_contours[i] = contour + np.array([offset_x, offset_y])

        # 過濾中間分隔線（使用原圖尺寸）
        if filter_center_line:
            filtered_contours = self.filter_center_dividing_line(
                filtered_contours, image.shape
            )

        # 按面積排序，取最大的幾個
        filtered_contours.sort(key=cv2.contourArea, reverse=True)

        return filtered_contours[:5]  # 返回最大的5個輪廓

    def filter_center_dividing_line(self, contours, image_shape):
        """
        過濾掉中間的分隔線輪廓

        Args:
            contours: 輪廓列表
            image_shape: 圖像形狀

        Returns:
            filtered_contours: 過濾後的輪廓
        """
        height, width = image_shape[:2]
        center_x = width // 2
        center_tolerance = width * 0.05  # 中心區域容忍度（圖像寬度的5%）

        filtered_contours = []

        for contour in contours:
            # 計算輪廓的邊界框
            x, y, w, h = cv2.boundingRect(contour)
            contour_center_x = x + w // 2

            # 檢查輪廓是否為垂直線（可能是分隔線）
            aspect_ratio = h / w if w > 0 else float("inf")

            # 過濾條件：
            # 1. 不在圖像中心附近
            # 2. 不是細長的垂直線（長寬比過大）
            # 3. 寬度不能太小（避免過濾掉真正的紙張邊緣）
            is_center_line = (
                abs(contour_center_x - center_x) < center_tolerance
                and aspect_ratio > 5.0  # 長寬比大於5:1的細長形狀
                and w < width * 0.02  # 寬度小於圖像寬度的2%
            )

            if not is_center_line:
                filtered_contours.append(contour)
            else:
                print(
                    f"過濾掉疑似分隔線的輪廓：中心x={contour_center_x}, 長寬比={aspect_ratio:.2f}, 寬度={w}"
                )

        return filtered_contours

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
        """
        ap = p - a
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq == 0.0:
            return a, float(np.linalg.norm(p - a))
        t = float(np.dot(ap, ab)) / ab_len_sq
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return proj, float(np.linalg.norm(p - proj))

    def _min_distance_to_polygon_edges(self, p, poly):
        """
        計算點 p 到多邊形 poly 的各邊的最短距離與對應最近點
        參考ch3-t2的實現
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

    def calculate_distances_to_rectangle(self, contour, rectangle_corners):
        """
        計算輪廓到長方形的兩種距離：邊緣到長方形 + 角點到輪廓
        使用原本的輪廓格式但採用ch3-t2的距離計算方法
        """
        # 獲取輪廓邊緣點
        edge_points = self.get_contour_edge_points(contour)

        if len(edge_points) == 0:
            return None, None

        red_poly = rectangle_corners.astype(np.int32)
        red_contour = red_poly.reshape(-1, 1, 2)

        # 1. 計算邊緣點到長方形的距離
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

        # 2. 計算角點到輪廓的距離
        best_corner_min_dist = -1.0
        best_corner_idx = -1
        best_corner_point = None
        best_corner_foot = None

        for idx, corner in enumerate(red_poly):
            p = corner.astype(float)
            # 角點到輪廓邊緣的最短距離
            min_d = float("inf")
            min_foot = None

            # 對輪廓的每條邊計算距離
            contour_points = edge_points
            n_points = len(contour_points)

            for i in range(n_points):
                a = contour_points[i].astype(float)
                b = contour_points[(i + 1) % n_points].astype(float)
                foot, d = self._closest_point_on_segment(p, a, b)
                if d < min_d:
                    min_d = d
                    min_foot = foot

            # 更新四角中的最大者
            if min_d > best_corner_min_dist:
                best_corner_min_dist = min_d
                best_corner_point = tuple(map(int, corner))
                best_corner_foot = (
                    tuple(map(int, min_foot)) if min_foot is not None else None
                )
                best_corner_idx = idx

        corner_to_paper_result = {
            "distance": best_corner_min_dist,
            "corner_idx": best_corner_idx,
            "corner_point": best_corner_point,
            "foot_point": best_corner_foot,
            "type": "corner_to_paper",
        }

        return edge_to_box_result, corner_to_paper_result

    def create_region_masks(self, image_shape, rectangles_info):
        """
        根據ArUco標記位置創建左右分區遮罩
        使用更智能的分區方法，基於ArUco標記連接的紙張區域
        """
        height, width = image_shape[:2]

        # 根據長方形中心位置分組
        left_rectangles = []
        right_rectangles = []

        # 先按ArUco標記的x座標分組
        rect_centers = []
        for rect_info in rectangles_info:
            corners = rect_info["corners"]
            center_x = np.mean([corner[0] for corner in corners])
            rect_centers.append((center_x, rect_info))

        # 按x座標排序
        rect_centers.sort(key=lambda x: x[0])

        # 如果只有一個或兩個標記，使用簡單分割
        if len(rect_centers) <= 2:
            image_center_x = width // 2
            for center_x, rect_info in rect_centers:
                if center_x < image_center_x:
                    left_rectangles.append(rect_info)
                else:
                    right_rectangles.append(rect_info)
        else:
            # 多個標記時，使用更智能的分組
            # 找到最大的x座標間隔作為分界點
            max_gap = 0
            split_index = len(rect_centers) // 2

            for i in range(len(rect_centers) - 1):
                gap = rect_centers[i + 1][0] - rect_centers[i][0]
                if gap > max_gap:
                    max_gap = gap
                    split_index = i + 1

            # 分割標記
            for i, (_, rect_info) in enumerate(rect_centers):
                if i < split_index:
                    left_rectangles.append(rect_info)
                else:
                    right_rectangles.append(rect_info)

        # 創建更精確的左右遮罩
        left_mask, right_mask = self.create_adaptive_masks(
            image_shape, left_rectangles, right_rectangles
        )

        return left_mask, right_mask, left_rectangles, right_rectangles

    def create_adaptive_masks(self, image_shape, left_rectangles, right_rectangles):
        """
        創建自適應的左右區域遮罩，基於ArUco標記的實際位置
        """
        height, width = image_shape[:2]

        # 創建遮罩
        left_mask = np.zeros((height, width), dtype=np.uint8)
        right_mask = np.zeros((height, width), dtype=np.uint8)

        # 如果有左側標記，擴展左側區域
        if left_rectangles:
            # 找到所有左側標記的最右邊界
            max_right_x = 0
            for rect_info in left_rectangles:
                corners = rect_info["corners"]
                right_x = np.max([corner[0] for corner in corners])
                max_right_x = max(max_right_x, right_x)

            # 左側遮罩延伸到最右邊界加一些緩衝
            buffer = width * 0.1  # 10%的緩衝區
            left_boundary = min(int(max_right_x + buffer), width)
            left_mask[:, :left_boundary] = 255

        # 如果有右側標記，擴展右側區域
        if right_rectangles:
            # 找到所有右側標記的最左邊界
            min_left_x = width
            for rect_info in right_rectangles:
                corners = rect_info["corners"]
                left_x = np.min([corner[0] for corner in corners])
                min_left_x = min(min_left_x, left_x)

            # 右側遮罩從最左邊界減一些緩衝開始
            buffer = width * 0.1  # 10%的緩衝區
            right_boundary = max(int(min_left_x - buffer), 0)
            right_mask[:, right_boundary:] = 255

        # 如果沒有標記，使用默認分割
        if not left_rectangles and not right_rectangles:
            center_x = width // 2
            left_mask[:, :center_x] = 255
            right_mask[:, center_x:] = 255
        elif not left_rectangles:
            # 只有右側標記，左側使用圖像左半部
            center_x = width // 2
            left_mask[:, :center_x] = 255
        elif not right_rectangles:
            # 只有左側標記，右側使用圖像右半部
            center_x = width // 2
            right_mask[:, center_x:] = 255

        return left_mask, right_mask

    def calculate_rectangle_distance(
        self, rect_info, contours, result_image, region_name
    ):
        """
        計算單個長方形與輪廓的距離 - 使用原本方法但增加雙重距離計算
        """
        rectangle_corners = rect_info["corners"]
        marker_id = rect_info["marker_id"]

        print(f"\n計算{region_name}標記 ID {marker_id} 的距離:")

        max_overall_distance = 0
        best_contour_idx = -1
        best_point = None
        best_distance_type = None
        best_reference_info = None

        # 用於儲存兩種距離類型的最佳結果
        best_edge_to_box = None
        best_corner_to_paper = None

        # 對每個紙張輪廓計算距離
        for contour_idx, contour in enumerate(contours):
            # 計算兩種距離
            edge_result, corner_result = self.calculate_distances_to_rectangle(
                contour, rectangle_corners
            )

            if edge_result and corner_result:
                print(
                    f"  輪廓 {contour_idx}: 邊緣距離 {edge_result['distance']:.2f}px, 角點距離 {corner_result['distance']:.2f}px"
                )

                # 比較並記錄最大距離
                if edge_result["distance"] > max_overall_distance:
                    max_overall_distance = edge_result["distance"]
                    best_contour_idx = contour_idx
                    best_point = edge_result["edge_point"]
                    best_distance_type = "edge_to_box"
                    best_reference_info = edge_result

                if corner_result["distance"] > max_overall_distance:
                    max_overall_distance = corner_result["distance"]
                    best_contour_idx = contour_idx
                    best_point = corner_result["corner_point"]
                    best_distance_type = "corner_to_paper"
                    best_reference_info = corner_result

                # 記錄最佳結果
                if (
                    not best_corner_to_paper
                    or corner_result["distance"] > best_corner_to_paper["distance"]
                ):
                    best_corner_to_paper = corner_result
                if (
                    not best_edge_to_box
                    or edge_result["distance"] > best_edge_to_box["distance"]
                ):
                    best_edge_to_box = edge_result

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
            "edge_to_box": best_edge_to_box,
            "corner_to_paper": best_corner_to_paper,
        }

        # 在圖上標註距離線和標記
        if best_edge_to_box and best_corner_to_paper:
            self.draw_distance_annotations(
                result_image,
                best_edge_to_box,
                best_corner_to_paper,
                marker_id,
                region_name,
            )

        type_text = "邊框" if best_distance_type == "edge_to_box" else "角點"
        print(
            f"  {region_name}標記 ID {marker_id} 最大距離: {max_overall_distance:.2f} 像素 (到{type_text})"
        )

        return result

    def calculate_rectangle_distance_no_draw(
        self, rect_info, contours, region_name, scale_info=None
    ):
        """
        計算單個長方形與輪廓的距離 - 不繪製距離線，只計算結果
        Args:
            scale_info: 比例尺資訊，包含 px_to_mm_ratio 用於單位轉換
        """
        rectangle_corners = rect_info["corners"]
        marker_id = rect_info["marker_id"]

        print(f"\n計算{region_name}標記 ID {marker_id} 的距離:")

        max_overall_distance = 0
        best_contour_idx = -1
        best_point = None
        best_distance_type = None
        best_reference_info = None

        # 用於儲存兩種距離類型的最佳結果
        best_edge_to_box = None
        best_corner_to_paper = None

        # 對每個紙張輪廓計算距離
        for contour_idx, contour in enumerate(contours):
            # 計算兩種距離
            edge_result, corner_result = self.calculate_distances_to_rectangle(
                contour, rectangle_corners
            )

            if edge_result and corner_result:
                # 轉換為公分單位顯示
                if scale_info:
                    edge_distance_cm = (
                        edge_result["distance"] / scale_info["px_to_mm_ratio"] / 10
                    )
                    corner_distance_cm = (
                        corner_result["distance"] / scale_info["px_to_mm_ratio"] / 10
                    )
                    print(
                        f"  輪廓 {contour_idx}: 邊緣距離 {edge_distance_cm:.2f}cm, 角點距離 {corner_distance_cm:.2f}cm"
                    )
                else:
                    print(
                        f"  輪廓 {contour_idx}: 邊緣距離 {edge_result['distance']:.2f}px, 角點距離 {corner_result['distance']:.2f}px"
                    )

                # 比較並記錄最大距離
                if edge_result["distance"] > max_overall_distance:
                    max_overall_distance = edge_result["distance"]
                    best_contour_idx = contour_idx
                    best_point = edge_result["edge_point"]
                    best_distance_type = "edge_to_box"
                    best_reference_info = edge_result

                if corner_result["distance"] > max_overall_distance:
                    max_overall_distance = corner_result["distance"]
                    best_contour_idx = contour_idx
                    best_point = corner_result["corner_point"]
                    best_distance_type = "corner_to_paper"
                    best_reference_info = corner_result

                # 記錄最佳結果
                if (
                    not best_corner_to_paper
                    or corner_result["distance"] > best_corner_to_paper["distance"]
                ):
                    best_corner_to_paper = corner_result
                if (
                    not best_edge_to_box
                    or edge_result["distance"] > best_edge_to_box["distance"]
                ):
                    best_edge_to_box = edge_result

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
            "edge_to_box": best_edge_to_box,
            "corner_to_paper": best_corner_to_paper,
        }

        type_text = "邊框" if best_distance_type == "edge_to_box" else "角點"
        print(
            f"  {region_name}標記 ID {marker_id} 最大距離: {max_overall_distance:.2f} 像素 (到{type_text})"
        )

        return result

    def draw_distance_annotations(
        self, result_image, edge_result, corner_result, marker_id, region_name=""
    ):
        """
        繪製兩種距離的標註 - 根據區域使用不同顏色
        左側：咖啡色系 (邊緣：咖啡色，角點：粉紅色)
        右側：橘紫系 (邊緣：橘色，角點：紫色)
        """
        # 根據區域選擇顏色
        if "左側" in region_name:
            edge_color = (42, 42, 165)  # 咖啡色 (BGR格式)
            corner_color = (147, 20, 255)  # 粉紅色 (BGR格式)
        else:  # 右側或其他
            edge_color = (0, 165, 255)  # 橘色 (BGR格式)
            corner_color = (255, 0, 255)  # 紫色 (BGR格式)

        # 1. 繪製邊緣到長方形的距離線
        if edge_result and edge_result["edge_point"] and edge_result["foot_point"]:
            cv2.line(
                result_image,
                edge_result["edge_point"],
                edge_result["foot_point"],
                edge_color,
                2,
            )
            cv2.circle(result_image, edge_result["edge_point"], 3, edge_color, -1)
            cv2.circle(result_image, edge_result["foot_point"], 3, edge_color, -1)

            # 標註文字
            region_code = "L" if "左側" in region_name else "R"
            # 計算公分距離用於顯示（保持原始像素值用於計算）
            distance_cm = (
                edge_result["distance"] / 49.92 / 10
                if "px_to_mm_ratio" in str(edge_result)
                else edge_result["distance"] / 50.0
            )
            text1 = f"{region_code}ID{marker_id}-Edge: {distance_cm:.1f}cm"
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
                edge_color,
                2,
            )

        # 2. 繪製角點到紙張的距離線
        if (
            corner_result
            and corner_result["corner_point"]
            and corner_result["foot_point"]
        ):
            cv2.line(
                result_image,
                corner_result["corner_point"],
                corner_result["foot_point"],
                corner_color,
                2,
            )
            cv2.circle(result_image, corner_result["corner_point"], 4, corner_color, -1)
            cv2.circle(result_image, corner_result["foot_point"], 3, corner_color, -1)

            # 標註文字
            region_code = "L" if "左側" in region_name else "R"
            # 計算公分距離用於顯示
            corner_distance_cm = (
                corner_result["distance"] / 49.92 / 10
                if "px_to_mm_ratio" in str(corner_result)
                else corner_result["distance"] / 50.0
            )
            text2 = f"{region_code}ID{marker_id}-Corner: {corner_distance_cm:.1f}cm"
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
                corner_color,
                2,
            )

    def process_image_with_rectangles(self, image, rectangles_info):
        """
        處理圖像，計算紙張輪廓與長方形的距離
        使用原本的紙張輪廓偵測方法，但採用左右分區邏輯
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

        print(f"\n=== 左右分區分析（使用原本輪廓偵測） ===")
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
                # 繪製右側紙張輪廓 (藍色)
                cv2.drawContours(result_image, right_contours, -1, (255, 0, 0), 2)

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

    def find_longest_distance(self, distance_results):
        """
        找出所有結果中的最長距離（考慮兩種距離類型）
        """
        if not distance_results:
            return None

        all_distances = []

        for result in distance_results:
            # 邊緣到長方形的距離
            if result.get("edge_to_box"):
                all_distances.append(
                    {
                        "marker_id": result["marker_id"],
                        "distance": result["edge_to_box"]["distance"],
                        "type": "edge_to_box",
                        "details": result["edge_to_box"],
                    }
                )

            # 角點到紙張的距離
            if result.get("corner_to_paper"):
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
