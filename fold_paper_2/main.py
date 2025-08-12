from PaperDetector_edge import PaperDetector_edges
from find_max_area import MaxAreaQuadFinder
import cv2

if __name__ == "__main__":
    for img in range(12, 13):
        image_path = rf"img\{img}.jpg"
        detector = PaperDetector_edges(image_path)
        edges_path = detector.detect_paper_by_color()
        if edges_path is not None:
            finder = MaxAreaQuadFinder(edges_path)
            finder.find_max_area_quad()
            finder.draw_and_show()
