from PaperDetector_edge import PaperDetector_edges

from BoxDistanceAnalyzer import BoxDistanceAnalyzer

# from BoxDistanceAnalyzer_2 import BoxDistanceAnalyzer
from Draw_square import Draw_square
from px2cm import get_pixel_per_cm_from_a4

import cv2
import json

if __name__ == "__main__":

    # img = 1
    for img in range(1, 5):
        image_path = rf"raw\img{img}.jpg"
        _, json_path = get_pixel_per_cm_from_a4(rf"a4.jpg", show_debug=False)
        if json_path is not None:
            with open(json_path, "r") as f:
                data = json.load(f)
                pixel_per_cm = data.get("pixel_per_cm", 19.597376925845985)

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
                D_sq_path, black_corners_int = Draw_square(detector_path)
                if D_sq_path is not None:

                    analyzer = BoxDistanceAnalyzer(
                        box1=black_corners_int, image_path=detector_path
                    )
                    analyzer.analyze(pixel_per_cm=pixel_per_cm)
