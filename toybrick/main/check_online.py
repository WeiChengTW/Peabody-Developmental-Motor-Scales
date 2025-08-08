import numpy as np

class check_on_line:
    def __init__(self, gap_threshold=50, x_layer_threshold=30):
        self.gap_threshold = gap_threshold
        self.x_layer_threshold = x_layer_threshold

    @staticmethod
    def check_x(layers, offset):
        if len(layers) < 2:
            return 0

        xs = [np.mean([p[0] for p in layer]) for layer in layers]

        standard_x = xs[-1] #用最上面的積木當基準
        count = 1

        bottom_layer_valid = any(abs(p[0] - standard_x) <= offset for p in layers[0])
        if bottom_layer_valid:
            count += 1

        #處理倒數第2層 ~ 第2層 
        for x in xs[-2:0:-1]:
            if abs(standard_x - x) <= offset:
                count += 1
                        
        return count
    
    def check_y(self, centroids):
        if len(centroids) <= 2:
            return 0
        if len(centroids) == 3:
            return 1
        
        # 按 y 座標排序
        sorted_points = sorted(centroids, key=lambda point: point[1], reverse=True)
        
        max_count = 1  # 至少有一個點
        current_count = 1
        
        for i in range(3, len(sorted_points)):
            curr_y = sorted_points[i][1]
            prev_y = sorted_points[i-1][1]
            
            if abs(curr_y - prev_y) <= self.gap_threshold:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1  # 重新開始計算
        
        return max_count

    def check_gap(self, centroids):
        gap_pairs = []

        for i, (cx, cy) in enumerate(centroids):
            top_neighbor = None      # 上方鄰居
            bottom_neighbor = None   # 下方鄰居
            min_top_dy = float('inf')
            min_bottom_dy = float('inf')

            for j, (nx, ny) in enumerate(centroids):
                if i == j:
                    continue
                if abs(nx - cx) > self.x_layer_threshold:
                    continue  # 過濾非同列 (x軸範圍內)

                dy = ny - cy  # Y軸差距
                if dy < 0 and abs(dy) < min_top_dy:      # dy < 0 表示在上方
                    min_top_dy = abs(dy)
                    top_neighbor = (nx, ny)
                elif dy > 0 and abs(dy) < min_bottom_dy:  # dy > 0 表示在下方
                    min_bottom_dy = abs(dy)
                    bottom_neighbor = (nx, ny)

            # 檢查上方鄰居的間隙
            if top_neighbor:
                dist = abs(cy - top_neighbor[1])  # Y軸距離
                if dist > self.gap_threshold:
                    gap_pairs.append(((cx, cy), top_neighbor, dist))

            # 檢查下方鄰居的間隙
            if bottom_neighbor:
                dist = abs(cy - bottom_neighbor[1])  # Y軸距離
                if dist > self.gap_threshold:
                    gap_pairs.append(((cx, cy), bottom_neighbor, dist))

        return gap_pairs


    