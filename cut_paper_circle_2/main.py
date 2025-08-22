from PaperDetector_edge import PaperDetector_edges
from BoxDistanceAnalyzer import BoxDistanceAnalyzer
from px2cm import get_pixel_per_cm_from_a4
import cv2

if __name__ == "__main__":

    # img = 1
    for img in range(1, 5):
        image_path = rf"raw\img{img}.jpg"
        _, json_path = get_pixel_per_cm_from_a4(rf"a4_2.jpg", show_debug=False)
        # 提取紙張區域
        print(f"\n正在處理圖片: {image_path}")
        print("====提取紙張區域====")
        detector = PaperDetector_edges(image_path)
        detector.detect_paper_by_color()
        if detector.original is not None:

            region = detector.extract_paper_region()
            if region is not None:
                height, width = region.shape[:2]
                if width > 600:
                    scale = 600 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    region = cv2.resize(region, (new_width, new_height))
                # cv2.imshow("提取的紙張區域", region)
                detector_path = detector.save_results()
                detector.show_results()

            if detector_path:
                min_dist_cm, max_dist_cm = BoxDistanceAnalyzer(detector_path)
            if min_dist_cm is not None and max_dist_cm is not None:
                correct = 4.0
                kid = max(abs(min_dist_cm - correct), abs(max_dist_cm - correct))
                if kid < 0.6:
                    print(f"kid = {kid:.2f}, score = 2")
                elif kid < 1.2:
                    print(f"kid = {kid:.2f}, score = 1")
                else:
                    print(f"kid = {kid:.2f}, score = 0")
