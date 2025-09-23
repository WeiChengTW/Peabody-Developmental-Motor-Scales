import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
from distance3 import detect_all_points_in_binary_image

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def get_neighbors_in_3x3(matrix, x, y):
    """取得點(x,y)周圍3x3範圍內的鄰居點"""
    neighbors = []
    rows, cols = len(matrix), len(matrix[0])

    for dy in range(-1, 2):  # 3x3範圍: -1到+1
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:  # 跳過中心點本身
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and matrix[ny][nx] == 1:
                neighbors.append((nx, ny))

    return neighbors


def get_direction(p1, p2):
    """計算兩點之間的方向"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 處理距離較遠的點，正規化方向
    if abs(dx) > abs(dy):
        # 水平方向為主
        if dx > 0:
            return "right"
        else:
            return "left"
    elif abs(dy) > abs(dx):
        # 垂直方向為主
        if dy > 0:
            return "down"
        else:
            return "up"
    else:
        # 對角線方向
        if dx > 0 and dy > 0:
            return "down-right"
        elif dx < 0 and dy < 0:
            return "up-left"
        elif dx > 0 and dy < 0:
            return "up-right"
        elif dx < 0 and dy > 0:
            return "down-left"

    return "unknown"


def get_line_direction_from_neighbors(neighbors, center_point):
    """從鄰居點分析主要線條方向"""
    direction_count = defaultdict(int)

    for neighbor in neighbors:
        direction = get_direction(center_point, neighbor)
        direction_count[direction] += 1

    # 找出最主要的方向
    if direction_count:
        main_direction = max(direction_count.items(), key=lambda x: x[1])
        return main_direction[0], direction_count

    return "unknown", {}


def check_consistent_direction(matrix, start_point, direction, min_points=5):
    """檢查從起始點開始，指定方向是否有連續的同向點"""
    direction_vectors = {
        "right": (1, 0),
        "left": (-1, 0),
        "down": (0, 1),
        "up": (0, -1),
        "down-right": (1, 1),
        "up-left": (-1, -1),
        "up-right": (1, -1),
        "down-left": (-1, 1),
    }

    if direction not in direction_vectors:
        return False, []

    dx, dy = direction_vectors[direction]
    x, y = start_point
    consecutive_points = []

    # 檢查正方向
    for i in range(1, min_points + 3):  # 多檢查幾個點
        next_x, next_y = x + dx * i, y + dy * i
        if 0 <= next_x < len(matrix[0]) and 0 <= next_y < len(matrix):
            if matrix[next_y][next_x] == 1:
                consecutive_points.append((next_x, next_y))
            else:
                break
        else:
            break

    return len(consecutive_points) >= min_points, consecutive_points


def check_extension_feasibility(matrix, center_point, direction, num_check=3):
    """檢查延伸方向是否能找到至少3個相連同向的點"""
    direction_vectors = {
        "right": (1, 0),
        "left": (-1, 0),
        "down": (0, 1),
        "up": (0, -1),
        "down-right": (1, 1),
        "up-left": (-1, -1),
        "up-right": (1, -1),
        "down-left": (-1, 1),
    }

    if direction not in direction_vectors:
        return False

    dx, dy = direction_vectors[direction]
    x, y = center_point
    found_points = 0

    # 檢查延伸方向的點
    for i in range(1, 10):  # 檢查前10個位置
        check_x, check_y = x + dx * i, y + dy * i
        if 0 <= check_x < len(matrix[0]) and 0 <= check_y < len(matrix):
            if matrix[check_y][check_x] == 1:
                found_points += 1
                if found_points >= num_check:
                    return True
            # 如果遇到空位，檢查是否在合理範圍內（可能需要補齊）
            elif i <= 3:  # 前3個位置的空位是可以接受的
                continue
            else:
                break
        else:
            break

    return found_points >= num_check


def predict_next_points(matrix, center_point, neighbors, num_predict=5):
    """根據鄰居點預測接下來可能的點位置"""
    predicted_points = []

    # 分析主要方向
    main_direction, direction_count = get_line_direction_from_neighbors(
        neighbors, center_point
    )

    if main_direction == "unknown":
        return predicted_points

    # 根據主要方向預測點
    direction_vectors = {
        "right": (1, 0),
        "left": (-1, 0),
        "down": (0, 1),
        "up": (0, -1),
        "down-right": (1, 1),
        "up-left": (-1, -1),
        "up-right": (1, -1),
        "down-left": (-1, 1),
    }

    if main_direction in direction_vectors:
        dx, dy = direction_vectors[main_direction]
        x, y = center_point

        # 預測前後各num_predict個點
        for i in range(1, num_predict + 1):
            # 正方向
            pred_x, pred_y = x + dx * i, y + dy * i
            if 0 <= pred_x < len(matrix[0]) and 0 <= pred_y < len(matrix):
                predicted_points.append((pred_x, pred_y))

            # 反方向
            pred_x, pred_y = x - dx * i, y - dy * i
            if 0 <= pred_x < len(matrix[0]) and 0 <= pred_y < len(matrix):
                predicted_points.append((pred_x, pred_y))

    return predicted_points


def analyze_point_direction(matrix, x, y):
    """分析點的走向"""
    neighbors = get_neighbors_in_3x3(matrix, x, y)
    directions = []

    for nx, ny in neighbors:
        direction = get_direction((x, y), (nx, ny))
        directions.append(direction)

    return directions, neighbors


def find_line_direction(matrix, points_list):
    """找出每個點的主要線條方向"""
    point_directions = {}

    for x, y in points_list:
        directions, neighbors = analyze_point_direction(matrix, x, y)
        point_directions[(x, y)] = {
            "directions": directions,
            "neighbors": neighbors,
            "neighbor_count": len(neighbors),
        }

    return point_directions


def predict_missing_points(matrix, point_directions):
    """預測並補齊缺失的點，使用5x5鄰域和前後5個點的走向分析"""
    new_matrix = [row[:] for row in matrix]  # 複製原矩陣
    added_points = []

    print(f"\n開始分析 {len(point_directions)} 個點...")

    for (x, y), info in point_directions.items():
        neighbors = info["neighbors"]

        # 分析主要線條方向
        main_direction, direction_stats = get_line_direction_from_neighbors(
            neighbors, (x, y)
        )

        print(f"點({x}, {y}): 鄰居={len(neighbors)}, 主要方向={main_direction}")

        # 調整補點條件，降低門檻
        if main_direction != "unknown" and len(neighbors) >= 1:
            # 檢查該方向是否有至少3個連續的同向點（降低要求）
            has_consistent_direction, consistent_points = check_consistent_direction(
                matrix, (x, y), main_direction, min_points=3
            )

            # 檢查反方向是否也有足夠的點
            opposite_direction = get_opposite_direction(main_direction)
            has_opposite_direction, opposite_points = check_consistent_direction(
                matrix, (x, y), opposite_direction, min_points=3
            )

            print(
                f"  - 正方向連續點: {len(consistent_points) if has_consistent_direction else 0}"
            )
            print(
                f"  - 反方向連續點: {len(opposite_points) if has_opposite_direction else 0}"
            )

            # 當某個方向有至少3個連續點時，考慮補齊
            if has_consistent_direction or has_opposite_direction:
                # 選擇更強的方向來補齊
                chosen_direction = (
                    main_direction if has_consistent_direction else opposite_direction
                )

                print(f"  - 選擇方向: {chosen_direction}")

                # 檢查延伸方向是否能找到至少2個相連同向的點（降低要求）
                if check_extension_feasibility(
                    matrix, (x, y), chosen_direction, num_check=2
                ):
                    print(f"  - 延伸方向可行，開始補點")

                    # 根據選定方向預測可能的點位置
                    predicted_points = predict_points_in_direction(
                        matrix, (x, y), chosen_direction, num_predict=2
                    )

                    # 添加符合條件的補齊點
                    points_added_for_this_center = 0
                    for px, py in predicted_points:
                        if 0 <= px < len(matrix[0]) and 0 <= py < len(matrix):
                            if new_matrix[py][px] == 0:  # 如果該位置為空
                                if should_add_point_relaxed(
                                    new_matrix, (px, py), (x, y), chosen_direction
                                ):
                                    new_matrix[py][px] = 1
                                    added_points.append((px, py))
                                    points_added_for_this_center += 1
                                    print(f"    -> 補點: ({px}, {py})")

                                    # 每個原點最多補5個點
                                    if points_added_for_this_center >= 5:
                                        break

    print(f"總共補了 {len(added_points)} 個點")
    return new_matrix, added_points


def get_opposite_direction(direction):
    """取得相反方向"""
    opposite_map = {
        "right": "left",
        "left": "right",
        "down": "up",
        "up": "down",
        "down-right": "up-left",
        "up-left": "down-right",
        "up-right": "down-left",
        "down-left": "up-right",
    }
    return opposite_map.get(direction, "unknown")


def predict_points_in_direction(matrix, center_point, direction, num_predict=3):
    """在指定方向預測點位置"""
    direction_vectors = {
        "right": (1, 0),
        "left": (-1, 0),
        "down": (0, 1),
        "up": (0, -1),
        "down-right": (1, 1),
        "up-left": (-1, -1),
        "up-right": (1, -1),
        "down-left": (-1, 1),
    }

    predicted_points = []
    if direction in direction_vectors:
        dx, dy = direction_vectors[direction]
        x, y = center_point

        # 在該方向預測前幾個點
        for i in range(1, num_predict + 1):
            pred_x, pred_y = x + dx * i, y + dy * i
            if 0 <= pred_x < len(matrix[0]) and 0 <= pred_y < len(matrix):
                predicted_points.append((pred_x, pred_y))

    return predicted_points


def should_add_point_relaxed(matrix, new_point, center_point, direction):
    """寬鬆判斷是否應該添加這個點"""
    px, py = new_point
    cx, cy = center_point

    # 計算與中心點的距離
    distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    # 距離太遠的點不添加（寬鬆一些）
    if distance > 3:
        return False

    # 檢查該點是否大致在指定方向上（允許一些偏差）
    point_direction = get_direction(center_point, new_point)
    if point_direction == "unknown":
        return False

    # 檢查附近是否已經有太多點（寬鬆條件）
    nearby_count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = px + dx, py + dy
            if 0 <= nx < len(matrix[0]) and 0 <= ny < len(matrix):
                if matrix[ny][nx] == 1:
                    nearby_count += 1

    # 如果周圍點太密集，不添加（比較寬鬆）
    if nearby_count > 4:
        return False

    return True


def should_add_point_strict(matrix, new_point, center_point, direction):
    """嚴格判斷是否應該添加這個點"""
    px, py = new_point
    cx, cy = center_point

    # 計算與中心點的距離
    distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    # 距離太遠的點不添加（更嚴格）
    if distance > 2:
        return False

    # 檢查該點是否在指定方向上
    point_direction = get_direction(center_point, new_point)
    if point_direction != direction:
        return False

    # 檢查附近是否已經有太多點
    nearby_count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = px + dx, py + dy
            if 0 <= nx < len(matrix[0]) and 0 <= ny < len(matrix):
                if matrix[ny][nx] == 1:
                    nearby_count += 1

    # 如果周圍點太密集，不添加（更嚴格）
    if nearby_count > 2:
        return False

    return True


def should_add_point(matrix, new_point, center_point, existing_neighbors):
    """判斷是否應該添加這個點"""
    px, py = new_point
    cx, cy = center_point

    # 計算與中心點的距離
    distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    # 距離太遠的點不添加
    if distance > 2:
        return False

    # 如果附近已經有很多點，不添加
    nearby_count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = px + dx, py + dy
            if 0 <= nx < len(matrix[0]) and 0 <= ny < len(matrix):
                if matrix[ny][nx] == 1:
                    nearby_count += 1

    # 如果周圍點太密集，不添加
    if nearby_count > 3:
        return False

    return True


def are_collinear(p1, p2, p3):
    """檢查三個點是否共線"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 使用向量叉積判斷共線
    cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return abs(cross_product) < 2  # 允許小的誤差


def get_extension_points(point, direction, reverse=False):
    """根據方向取得延伸點"""
    x, y = point
    extension_map = {
        "right": [(1, 0), (2, 0)],
        "left": [(-1, 0), (-2, 0)],
        "down": [(0, 1), (0, 2)],
        "up": [(0, -1), (0, -2)],
        "down-right": [(1, 1), (2, 2)],
        "up-left": [(-1, -1), (-2, -2)],
        "up-right": [(1, -1), (2, -2)],
        "down-left": [(-1, 1), (-2, 2)],
    }

    if direction in extension_map:
        extensions = extension_map[direction]
        if reverse:
            extensions = [(-dx, -dy) for dx, dy in extensions]
        return [(x + dx, y + dy) for dx, dy in extensions]

    return []


def are_opposite_directions(dir1, dir2):
    """檢查兩個方向是否相對"""
    opposite_pairs = [
        ("right", "left"),
        ("up", "down"),
        ("up-right", "down-left"),
        ("up-left", "down-right"),
    ]

    for d1, d2 in opposite_pairs:
        if (dir1 == d1 and dir2 == d2) or (dir1 == d2 and dir2 == d1):
            return True
    return False


def find_bridge_points(center, p1, p2):
    """在兩點之間找到橋接點"""
    # 簡單的線性插值
    bridge_points = []

    # 計算中點
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2

    if (mid_x, mid_y) != center:
        bridge_points.append((mid_x, mid_y))

    return bridge_points


def visualize_result(original_matrix, new_matrix, added_points):
    """視覺化結果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 原始矩陣
    ax1.imshow(original_matrix, cmap="binary", interpolation="nearest")
    ax1.set_title("原始點分布")
    ax1.grid(True, alpha=0.3)

    # 處理後的矩陣
    display_matrix = np.array(new_matrix, dtype=float)

    # 將新增的點用不同顏色標記
    for x, y in added_points:
        display_matrix[y][x] = 0.5  # 用灰色表示新增的點

    ax2.imshow(display_matrix, cmap="RdYlBu_r", interpolation="nearest")
    ax2.set_title("補齊線條後 (灰色為新增點)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("line_reconstruction_result.jpg", dpi=300, bbox_inches="tight")
    plt.show()


def print_matrix(matrix, title):
    """印出矩陣"""
    print(f"\n{title}:")
    for row in matrix:
        print(row)


# 主要執行流程
if __name__ == "__main__":
    image_path = r"1.png"  # 或者其他二值化圖像
    point, n_point = detect_all_points_in_binary_image(image_path)
    # 建立二維矩陣 - 自動計算所需的矩陣大小
    max_x = max(x for x, y in point) + 1
    max_y = max(y for x, y in point) + 1

    print(f"矩陣大小: {max_y} x {max_x}")

    matrix = [[0 for _ in range(max_x)] for _ in range(max_y)]
    for x, y in point:
        matrix[y][x] = 1

    # print("原始點座標:")
    # print(point)

    # print_matrix(matrix, "原始二維矩陣")

    # 分析每個點的方向
    point_directions = find_line_direction(matrix, point)

    print("\n每個點的鄰域分析:")
    for (x, y), info in point_directions.items():
        print(
            f"點({x}, {y}): 鄰居數量={info['neighbor_count']}, 方向={info['directions']}"
        )

    # 預測並補齊缺失的點
    new_matrix, added_points = predict_missing_points(matrix, point_directions)

    # print_matrix(new_matrix, "補齊線條後的矩陣")

    # if added_points:
    #     print(f"\n新增的點: {added_points}")
    # else:
    #     print("\n沒有新增任何點")

    # 視覺化結果
    visualize_result(matrix, new_matrix, added_points)
