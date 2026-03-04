from PaperDetector_edge import PaperDetector_edges
from BoxDistanceAnalyzer import BoxDistanceAnalyzer
from px2cm import get_pixel_per_cm_from_a4
import cv2
import sys
import os
import json

if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        image_path = rf"kid\{uid}\{img_id}.jpg"
        # image_path = rf"C:\Users\chang\Downloads\web\kid\lull222\ch3-t1.jpg"
        _, json_path = get_pixel_per_cm_from_a4(rf"ch3-t1\a4_2.jpg", show_debug=False)
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
                    score = 2
                elif kid < 1.2:
                    print(f"kid = {kid:.2f}, score = 1")
                    score = 1
                else:
                    print(f"kid = {kid:.2f}, score = 0")
                    score = 0
                    # 讀取現有的 result.json 或建立新的
            result_file = "result.json"
            try:
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                else:
                    results = {}
            except (json.JSONDecodeError, FileNotFoundError):
                results = {}

            # 確保 uid 存在於結果中
            if uid not in results:
                results[uid] = {}

            # 更新對應 uid 的關卡分數
            results[uid][img_id] = score

            # 儲存到 result.json
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(
                f"結果已儲存到 {result_file} - 用戶 {uid} 的關卡 {img_id} 分數: {score}"
            )
