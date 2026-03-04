import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 設置中文字體
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False


def analyze_rectangle_edges(image_path, output_path=None):
    """
    分析 100x100 像素矩形圖片的邊緣，檢測連接處是否有突出

    Args:
        image_path: 輸入圖片路徑
        output_path: 輸出結果圖片路徑（可選）

    Returns:
        dict: 分析結果，包含突出點的資訊
    """
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return None

    # 轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 確保圖片是 100x100
    if gray.shape != (100, 100):
        print(f"圖片尺寸為 {gray.shape}，將調整為 100x100")
        gray = cv2.resize(gray, (100, 100))
        img = cv2.resize(img, (100, 100))

    # 適中的預處理 - 平衡降噪和保持細節
    # 輕微高斯模糊，避免過度平滑
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)  # 減少sigma值

    # 輕度形態學操作
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 調整邊緣檢測參數，更平衡
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)  # 調整到中等敏感度    # 查找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到輪廓")
        return None

    # 找到最大輪廓（假設為主要矩形）
    largest_contour = max(contours, key=cv2.contourArea)

    # 使用適中的道格拉斯-普克算法簡化輪廓
    # 調整epsilon值，既不會太粗糙也不會太敏感
    epsilon = 0.015 * cv2.arcLength(largest_contour, True)  # 調整到0.015，平衡精度
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 如果頂點仍然太多，適度嚴格化
    if len(approx) > 6:  # 調整閾值從8到6
        epsilon = 0.025 * cv2.arcLength(largest_contour, True)  # 適度增加而不是減少
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 合併距離太近的頂點 - 使用適應性距離
    # merge_distance = max(5, int(0.05 * cv2.arcLength(largest_contour, True)))
    merge_distance = 5
    # approx = merge_close_vertices(approx, min_distance=merge_distance)

    # 分析結果
    result = {
        "total_points": len(approx),
        "expected_points": 4,  # 矩形應該有4個頂點
        "is_rectangle": len(approx) == 4,
        "protrusions": [],
        "corner_angles": [],
    }

    # 如果點數超過4個，說明有突出
    if len(approx) > 4:
        result["has_protrusions"] = True
        result["protrusion_count"] = len(approx) - 4

        # 找到理想的矩形頂點（4個角點）- 平衡的參數
        rect_epsilon = 0.06 * cv2.arcLength(largest_contour, True)  # 調整到0.06
        rect_approx = cv2.approxPolyDP(largest_contour, rect_epsilon, True)

        # 如果還是無法得到4個點，適度調整
        if len(rect_approx) != 4:
            rect_epsilon = 0.09 * cv2.arcLength(largest_contour, True)  # 降低到0.09
            rect_approx = cv2.approxPolyDP(
                largest_contour, rect_epsilon, True
            )  # 分析每個突出點 - 更嚴格的判斷條件
        for i, point in enumerate(approx):
            x, y = point[0]
            point_tuple = tuple(point[0])

            # 檢查是否為突出點
            is_protrusion = False

            # 方法1：點集合比較（如果能找到4個角點）
            if len(rect_approx) == 4:
                rect_points = set(tuple(pt[0]) for pt in rect_approx)
                if point_tuple not in rect_points:
                    # 進一步檢查：是否真的是突出而不是角點的細微偏移
                    min_dist_to_corner = float("inf")
                    for rect_pt in rect_points:
                        dist = np.linalg.norm(np.array(point_tuple) - np.array(rect_pt))
                        min_dist_to_corner = min(min_dist_to_corner, dist)

                    # 如果距離最近角點超過閾值，才認為是突出
                    if min_dist_to_corner > 5:  # 降低距離閾值到5，更合理
                        is_protrusion = True

            # 方法2：平衡的邊緣位置判斷
            edge_threshold = 12  # 降低閾值到12
            corner_threshold = 18  # 降低角點閾值到18

            is_near_edge = (
                x < edge_threshold
                or x > (100 - edge_threshold)
                or y < edge_threshold
                or y > (100 - edge_threshold)
            )

            is_corner = (
                (x < corner_threshold and y < corner_threshold)
                or (x > (100 - corner_threshold) and y < corner_threshold)
                or (x < corner_threshold and y > (100 - corner_threshold))
                or (x > (100 - corner_threshold) and y > (100 - corner_threshold))
            )

            # 方法3：角度分析 - 檢查該點是否形成尖銳角度
            if len(approx) >= 3:
                prev_idx = (i - 1) % len(approx)
                next_idx = (i + 1) % len(approx)

                p1 = approx[prev_idx][0]
                p2 = point[0]
                p3 = approx[next_idx][0]

                angle = calculate_angle(p1, p2, p3)

                # 如果角度過於尖銳或過於平直，可能是突出
                is_sharp_angle = angle < 70 or angle > 110  # 放寬角度範圍，更合理

                if is_near_edge and not is_corner and is_sharp_angle:
                    is_protrusion = True

            if is_protrusion:
                result["protrusions"].append(
                    {
                        "point_index": i,
                        "coordinates": (x, y),
                        "edge_type": get_edge_type(x, y),
                        "angle": (
                            calculate_angle(
                                approx[(i - 1) % len(approx)][0],
                                point[0],
                                approx[(i + 1) % len(approx)][0],
                            )
                            if len(approx) >= 3
                            else None
                        ),
                    }
                )
    else:
        result["has_protrusions"] = False
        result["protrusion_count"] = 0

    # 計算角度（如果有足夠的點）
    if len(approx) >= 3:
        for i in range(len(approx)):
            p1 = approx[i - 1][0]
            p2 = approx[i][0]
            p3 = approx[(i + 1) % len(approx)][0]

            angle = calculate_angle(p1, p2, p3)
            result["corner_angles"].append(
                {"point_index": i, "angle": angle, "coordinates": tuple(p2)}
            )

    # 視覺化結果
    if output_path or True:  # 總是顯示結果
        visualize_analysis(img, approx, result, output_path)

    return result


def get_edge_type(x, y):
    """確定點在哪個邊緣"""
    edge_threshold = 12  # 調整回合理的閾值
    if x < edge_threshold:
        return "左邊緣"
    elif x > (100 - edge_threshold):
        return "右邊緣"
    elif y < edge_threshold:
        return "上邊緣"
    elif y > (100 - edge_threshold):
        return "下邊緣"
    else:
        return "內部"


def merge_close_vertices(approx, min_distance=8):
    """
    合併距離太近的頂點

    Args:
        approx: 檢測到的頂點數組
        min_distance: 最小距離閾值，小於此距離的頂點將被合併

    Returns:
        合併後的頂點數組
    """
    if len(approx) <= 4:
        return approx

    vertices = [point[0] for point in approx]
    merged_vertices = []
    used = [False] * len(vertices)

    for i in range(len(vertices)):
        if used[i]:
            continue

        current_group = [vertices[i]]
        used[i] = True

        # 找到所有與當前頂點距離過近的頂點
        for j in range(i + 1, len(vertices)):
            if used[j]:
                continue

            distance = np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j]))
            if distance < min_distance:
                current_group.append(vertices[j])
                used[j] = True

        # 如果有多個相近的頂點，選擇最佳代表點
        if len(current_group) > 1:
            # 策略1：選擇最靠近角落的點（優先保留角點）
            best_vertex = None
            min_corner_dist = float("inf")

            corners = [(0, 0), (100, 0), (0, 100), (100, 100)]

            for vertex in current_group:
                # 計算到最近角落的距離
                corner_distances = [
                    np.linalg.norm(np.array(vertex) - np.array(corner))
                    for corner in corners
                ]
                min_dist = min(corner_distances)

                if min_dist < min_corner_dist:
                    min_corner_dist = min_dist
                    best_vertex = vertex

            # 如果沒有明顯的角點，使用中心點
            if best_vertex is None:
                best_vertex = np.mean(current_group, axis=0).astype(int)

            merged_vertices.append(best_vertex)
        else:
            merged_vertices.append(vertices[i])

    # 轉換回原始格式
    merged_approx = np.array([[vertex] for vertex in merged_vertices])

    print(f"頂點合併：{len(approx)} -> {len(merged_approx)} (距離閾值: {min_distance})")

    return merged_approx


def calculate_angle(p1, p2, p3):
    """計算三點形成的角度"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def visualize_analysis(img, approx, result, output_path=None):
    """視覺化分析結果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始圖片
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始圖片 (100x100)")
    axes[0].axis("off")

    # 邊緣檢測結果
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("邊緣檢測")
    axes[1].axis("off")

    # 分析結果
    result_img = img.copy()

    # 繪製檢測到的多邊形
    cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)

    # 標記頂點
    for i, point in enumerate(approx):
        x, y = point[0]
        cv2.circle(result_img, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(
            result_img,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # 標記突出點（紅色圓圈和文字標記）
    if result["has_protrusions"]:
        for i, protrusion in enumerate(result["protrusions"]):
            x, y = protrusion["coordinates"]
            # 用紅色圓圈標記突出點
            cv2.circle(result_img, (x, y), 5, (0, 0, 255), 1)
            # 添加紅色文字標記
            cv2.putText(
                result_img,
                f"pro{i+1}",
                (x + 10, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
            # 在突出點周圍畫一個紅色方框
            cv2.rectangle(result_img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 1)

    axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f'分析結果 - {"有突出" if result["has_protrusions"] else "無突出"}'
    )
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"結果已保存到: {output_path}")

    plt.show()


def print_analysis_report(result):
    """打印詳細的分析報告"""
    print("=" * 50)
    print("矩形連接處突出分析報告")
    print("=" * 50)

    print(f"檢測到的頂點數量: {result['total_points']}")
    print(f"期望的頂點數量: {result['expected_points']}")
    print(f"是否為標準矩形: {'是' if result['is_rectangle'] else '否'}")

    if result["has_protrusions"]:
        print(f"\n⚠️  發現突出: {result['protrusion_count']} 個")
        print("突出點詳情:")
        for i, protrusion in enumerate(result["protrusions"], 1):
            coord = protrusion["coordinates"]
            edge = protrusion["edge_type"]
            angle = protrusion.get("angle")
            angle_str = f", 角度: {angle:.1f}°" if angle else ""
            print(f"  {i}. 位置: ({coord[0]}, {coord[1]}) - {edge}{angle_str}")
    else:
        print("\n✅ 未發現突出，矩形邊緣正常")

    if result["corner_angles"]:
        print(f"\n角度分析:")
        for angle_info in result["corner_angles"]:
            angle = angle_info["angle"]
            coord = angle_info["coordinates"]
            status = "正常" if 85 <= angle <= 95 else "異常"
            print(f"  頂點 ({coord[0]}, {coord[1]}): {angle:.1f}° - {status}")


def batch_analyze_rectangles(input_folder, output_folder=None):
    """批量分析矩形圖片"""
    import os

    if not os.path.exists(input_folder):
        print(f"輸入文件夾不存在: {input_folder}")
        return

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支援的圖片格式
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    results = {}

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(input_folder, filename)
            output_path = None

            if output_folder:
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}_analysis.png")

            print(f"\n正在分析: {filename}")
            result = analyze_rectangle_edges(image_path, output_path)

            if result:
                results[filename] = result
                print_analysis_report(result)

    return results


# 測試函數
def test_analysis():
    """測試分析功能"""
    # 創建一個測試圖片
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 繪製一個帶有小突出的矩形
    cv2.rectangle(test_img, (10, 10), (90, 90), (255, 255, 255), 2)

    # 添加一個小突出
    cv2.circle(test_img, (50, 10), 3, (255, 255, 255), -1)

    # 保存測試圖片
    test_path = "test_rectangle.png"
    cv2.imwrite(test_path, test_img)

    # 分析測試圖片
    result = analyze_rectangle_edges(test_path)
    if result:
        print_analysis_report(result)

    return result


if __name__ == "__main__":
    # 使用範例
    print("矩形連接處突出檢測工具 - 平衡模式 + 頂點合併")
    print("改進項目：")
    print("1. 平衡的輪廓簡化參數 (epsilon = 0.015)")
    print("2. 適中的預處理：輕微模糊 + 單次形態學操作")
    print("3. 合理的驗證閾值：距離=5, 角度=70-110°")
    print("4. 平衡的邊緣和角點閾值 (12, 18)")
    print("5. 智能頂點合併：優先保留角點，合併相近頂點")
    print("-" * 50)

    path = r"result/Square/96_1_Square_000_conf0.97.jpg"
    result = analyze_rectangle_edges(path)
    if result:
        print_analysis_report(result)

    # 可選：測試其他圖片
    # test_analysis()
