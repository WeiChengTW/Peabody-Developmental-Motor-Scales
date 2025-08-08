class LayerGrouping:
    def __init__(self, layer_threshold=30):
        self.layer_threshold = layer_threshold

    def group_by_y(self, centroids):
        sorted_points = sorted(centroids, key=lambda p: p[1], reverse=True)
        layers = []
        for cx, cy in sorted_points:
            placed = False
            for layer in layers:
                if abs(layer[0][1] - cy) < self.layer_threshold:
                    layer.append((cx, cy))
                    placed = True
                    break
            if not placed:
                layers.append([(cx, cy)])
        return layers
