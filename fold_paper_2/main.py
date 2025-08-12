from PaperDetector_edge import PaperDetector_edges
from find_max_area import MaxAreaQuadFinder
import cv2

if __name__ == "__main__":
    for img in range(1, 8):
        image_path = rf"img\{img}.jpg"
        img = cv2.imread(image_path)
        resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        detector = PaperDetector_edges(image_path)
        edges_path = detector.detect_paper_by_color()
        if edges_path is not None:
            finder = MaxAreaQuadFinder(edges_path)
            finder.find_max_area_quad()
            finder.draw_and_show()
