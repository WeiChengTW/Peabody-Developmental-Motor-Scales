from PaperDetector_edge import PaperDetector_edges
from find_point import PointDetector
from find_d import ShapeEdgeDistanceCalculator
from px2cm import get_pixel_per_cm_from_a4
import cv2

if __name__ == "__main__":

    # img = 1
    for img in range(6, 10):
        image_path = rf"img\{img}.jpg"
        _, json_path = get_pixel_per_cm_from_a4(rf"a4/a4_2.jpg", show_debug=False)
        # 提取紙張區域
        print(f"正在處理圖片: {image_path}")
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
                # 接下來檢測點
                print("\n====檢測點====")
                point_detector = PointDetector(detector_path, image_path)
                point_detector.load_image(
                    json_path=json_path
                )  # 使用 px2cm.json 中的像素比例
                points = point_detector.detect_points()
                if len(points) == 1:
                    point_detector.draw_squares_at_points(
                        square_size_cm=8
                    )  # 畫出8cm的方塊
                    # point_detector.show_results()
                    point_detector.save_result()
                    given_points = point_detector.points
                    # print(points)
                else:
                    print(f"檢測到{len(points)}點")
                # 接下來計算距離
                print("\n====計算距離====")
                calculator = ShapeEdgeDistanceCalculator(detector_path, json_path)

                if calculator.load_image_and_config():

                    if calculator.detect_shape_edges(show_debug=True):

                        calculator.set_given_points(given_points)

                        if calculator.calculate_distances():

                            # 視覺化和保存結果

                            calculator.visualize_results()
