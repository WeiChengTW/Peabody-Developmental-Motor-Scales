from fold import AdvancedFoldDetector
from PaperDetector_edge import PaperDetector_edges
import cv2

if __name__ == "__main__":
    image_path = r"img\10.jpg"
    detector = PaperDetector_edges(image_path)
    detector.detect_paper_by_color()
    if detector.original is not None:
        detector.show_results()
        region = detector.extract_paper_region()
        if region is not None:
            height, width = region.shape[:2]
            if width > 600:
                scale = 600 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                region = cv2.resize(region, (new_width, new_height))
            # cv2.imshow("提取的紙張區域", region)
            # print("按任意鍵關閉視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            detector_path = detector.save_results()
    if detector_path:
        # Initialize the advanced fold detector
        fold_detector = AdvancedFoldDetector()
        result = fold_detector.adaptive_threshold(detector_path)
        cv2.imshow("Adaptive Threshold Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
