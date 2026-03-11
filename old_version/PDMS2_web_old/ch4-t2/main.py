from PaperDetector_edge import PaperDetector_edges
from find_max_area import MaxAreaQuadFinder
import cv2
import json
import sys
import os

if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        image_path = rf"kid\{uid}\{img_id}.jpg"
        # for img in range(1, 8):
        # image_path = rf"img\{img}.jpg"

        img = cv2.imread(image_path)
        detector = PaperDetector_edges(image_path)
        edges_path = detector.detect_paper_by_color()
        if edges_path is not None:
            finder = MaxAreaQuadFinder(edges_path)
            finder.find_max_area_quad()
            kid = finder.draw_and_show()
        if kid is not None:
            if kid < 0.3:
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

        print(f"結果已儲存到 {result_file} - 用戶 {uid} 的關卡 {img_id} 分數: {score}")
