import cv2

from PrtSc import ScreenCapture
from PaperDetector_HSV import PaperDetector_HSV
from PaperDetector_edge import PaperDetector_edges
from ShapeAnalyzer import ShapeAnalyzer
from graph_yolov8.Analyze_graphics import Analyze_graphics
from DEL_IMG import reset_result_dir
import os


def PS():
    screen_capture = ScreenCapture()
    img_path = screen_capture.run()
    screen_capture.cleanup()
    return img_path


if __name__ == "__main__":
    # reset_result_dir()

    img_mode = "dir"  # 'PrtSc' or 'image' or 'dir'
    paper_detector_mode = "edge"  # HSV or edge
    if img_mode == "PrtSc":
        print("開始截圖...")
        image_path = PS()
    elif img_mode == "image":
        img = "img3.jpg"
        image_path = f"test_img\{img}"
    elif img_mode == "dir":
        img_dir = "raw"
        image_files = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not image_files:
            print("資料夾中沒有圖片。")
            exit()
        for img in image_files:
            image_path = os.path.join(img_dir, img)
            print(f"處理圖片: {image_path}")

            pattern_detector = ShapeAnalyzer(image_path=image_path)
            thresh_img_path, thresh, result_img, adaptive_thresh = (
                pattern_detector.process()
            )

            if thresh_img_path is not None:
                # print("\n開始圖形分析...")
                segmenter = Analyze_graphics()
                # segmenter.reset_dir()
                segmenter.infer_and_draw(thresh_img_path)
                continue
    if img_mode != "dir":
        if paper_detector_mode == "HSV":
            detector = PaperDetector_HSV(image_path)
            detector.detect_paper_by_color()
        elif paper_detector_mode == "edge":
            detector = PaperDetector_edges(image_path)
            detector.detect_paper_by_color()
        if detector.original is not None:

            region = detector.extract_paper_region()
            if region is not None:
                cv2.imshow("region", region)
                cv2.imshow("original", detector.original)
                cv2.imshow("result", detector.result)
                print("按任意鍵關閉視窗...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                detector_image_path = detector.save_results()

            pattern_detector = ShapeAnalyzer(image_path=detector_image_path)
            thresh_img_path, thresh, result_img, adaptive_thresh = (
                pattern_detector.process()
            )
            cv2.imshow("result_img", result_img)
            cv2.imshow("adaptive_thresh", adaptive_thresh)
            cv2.imshow("thresh", thresh)
            print("按任意鍵關閉視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if thresh_img_path is not None:
            print("\n開始圖形分析...")
            segmenter = Analyze_graphics()
            # 重置result資料夾（可選）
            # segmenter.reset_dir()
            segmenter.infer_and_draw(thresh_img_path)
