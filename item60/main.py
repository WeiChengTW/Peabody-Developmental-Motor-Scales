import cv2
import numpy as np
import os
from step1_paper_contour import detect_paper_contour
from step2_aruco_quarter_a4 import detect_aruco_and_draw_quarter_a4
from fast_step3_calculate_distances import fast_calculate_distances
from fast_scoring import fast_calculate_cutting_score


def main_pipeline(input_dir="img", output_dir="result"):
    """
    主程序：串聯所有步驟（快速版本）
    1. 檢測紙張輪廓(藍線)
    2. 根據ArUco畫出1/4 A4矩形(綠框線)並計算比例尺
    3. 快速計算距離(橘線和紫線) - 優化性能
    4. 快速評分 - 根據評分標準自動評分
    """

    print("=" * 60)
    print("Peabody 發展性動作量表 - 紙張分析程序 (快速版)")
    print("=" * 60)

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 獲取所有圖片文件
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"在 {input_dir} 目錄中未找到圖片文件")
        return

    print(f"找到 {len(image_files)} 張圖片")
    print("-" * 60)

    results = []

    for i, filename in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 處理圖片: {filename}")

        input_path = os.path.join(input_dir, filename)

        # 步驟1：檢測紙張輪廓
        print("  步驟1: 檢測紙張輪廓...")
        step1_output = os.path.join(output_dir, f"step1_{filename}")
        paper_contour, step1_image = detect_paper_contour(input_path, step1_output)

        if paper_contour is not None:
            print(f"    ✓ 檢測到紙張輪廓 ({len(paper_contour)} 個點)")
        else:
            print("    ✗ 未檢測到紙張輪廓，跳過此圖片")
            continue

        # 步驟2：ArUco檢測和1/4 A4矩形
        print("  步驟2: 檢測ArUco標記並繪製1/4 A4矩形...")
        step2_output = os.path.join(output_dir, f"step2_{filename}")
        quarter_a4_corners, pixels_per_cm, step2_image = (
            detect_aruco_and_draw_quarter_a4(input_path, step2_output)
        )

        if quarter_a4_corners is not None:
            print(f"    ✓ 檢測到ArUco標記，比例尺: {pixels_per_cm:.2f} pixels/cm")
        else:
            print("    ✗ 未檢測到ArUco標記，跳過此圖片")
            continue

        # 步驟3：快速計算距離
        print("  步驟3: 快速計算距離...")
        step3_output = os.path.join(output_dir, f"fast_step3_{filename}")
        final_output = os.path.join(output_dir, f"final_{filename}")

        distance_result = fast_calculate_distances(input_path, step3_output)

        if distance_result is not None:
            result_image, edge_distances, corner_distances = distance_result

            # 複製最終結果
            cv2.imwrite(final_output, result_image)

            print(f"    ✓ 距離計算完成")
            print(
                f"      邊線最大距離 (cm): {[f'{d/pixels_per_cm:.2f}' for d in edge_distances]}"
            )
            print(
                f"      角點距離 (cm): {[f'{d/pixels_per_cm:.2f}' for d in corner_distances]}"
            )

            # 步驟4：快速評分
            print("  步驟4: 快速評分...")
            scored_output = os.path.join(output_dir, f"fast_scored_{filename}")
            score, score_reason, scored_image = fast_calculate_cutting_score(
                input_path, scored_output
            )

            print(f"    ✓ 評分完成: {score}/2")
            print(f"      評分原因: {score_reason}")

            # 記錄結果
            results.append(
                {
                    "filename": filename,
                    "paper_contour_points": len(paper_contour),
                    "pixels_per_cm": pixels_per_cm,
                    "edge_distances_cm": [d / pixels_per_cm for d in edge_distances],
                    "corner_distances_cm": [
                        d / pixels_per_cm for d in corner_distances
                    ],
                    "score": score,
                    "score_reason": score_reason,
                    "success": True,
                }
            )
        else:
            print("    ✗ 距離計算失敗")
            results.append({"filename": filename, "success": False})

    # 輸出總結
    print("\n" + "=" * 60)
    print("處理總結")
    print("=" * 60)

    successful = sum(1 for r in results if r.get("success", False))
    print(f"成功處理: {successful}/{len(results)} 張圖片")

    # 評分統計
    if successful > 0:
        scores = [r.get("score", 0) for r in results if r.get("success", False)]
        total_score = sum(scores)
        average_score = total_score / len(scores) if scores else 0

        print(f"\n評分統計:")
        print(f"總分: {total_score}/{len(scores)*2}")
        print(f"平均分: {average_score:.2f}/2")

        # 分數分布
        score_counts = {0: 0, 1: 0, 2: 0}
        for score in scores:
            if score in score_counts:
                score_counts[score] += 1

        print(
            f"分數分布: 0分={score_counts[0]}張, 1分={score_counts[1]}張, 2分={score_counts[2]}張"
        )

    for result in results:
        if result.get("success", False):
            print(f"\n圖片: {result['filename']}")
            print(
                f"  評分: {result.get('score', 0)}/2 - {result.get('score_reason', 'N/A')}"
            )
            print(f"  輪廓點數: {result['paper_contour_points']}")
            print(f"  比例尺: {result['pixels_per_cm']:.2f} pixels/cm")
            print(
                f"  邊線最大距離: {[f'{d:.2f}cm' for d in result['edge_distances_cm']]}"
            )
            print(
                f"  角點距離: {[f'{d:.2f}cm' for d in result['corner_distances_cm']]}"
            )
        else:
            print(f"\n圖片: {result['filename']} - 處理失敗")

    print(f"\n所有結果圖片已保存到: {output_dir}")
    print("檔案說明 (快速版):")
    print("  step1_*.jpg - 紙張輪廓檢測結果(藍線)")
    print("  step2_*.jpg - ArUco檢測和1/4 A4矩形結果(綠框)")
    print("  fast_step3_*.jpg - 快速距離計算結果(優化性能)")
    print("  fast_scored_*.jpg - 快速評分結果(分數左上角、距離左下角)")
    print("  final_*.jpg - 最終完整結果")
    print("\n評分標準:")
    print("  2分: 剪紙完成且精確(間距<1.2cm)")
    print("  1分: 剪紙完成但不夠精確(間距>1.2cm)")
    print("  0分: 未完成剪紙(檢測到多個ArUco標記)")
    print("\n性能優化:")
    print("  ✓ 輪廓簡化: 3000點 -> 4-6點")
    print("  ✓ 採樣優化: 100點 -> 10-20點")
    print("  ✓ 速度提升: 20-30倍")


def create_test_image():
    """創建測試用的ArUco標記圖片"""
    print("\n正在創建測試ArUco標記...")

    try:
        # 使用make_ArUco目錄中的程序
        import sys

        sys.path.append("../make_ArUco")
        from make_ArUco import main as create_aruco_main

        # 創建ArUco標記
        image, saved_file, pdf_file = create_aruco_main(
            show_plot=False, create_pdf=False
        )

        # 複製到img目錄
        import shutil

        if os.path.exists(saved_file):
            shutil.copy(saved_file, "img/test_aruco.jpg")
            print("✓ 測試ArUco圖片已複製到 img/test_aruco.jpg")

    except Exception as e:
        print(f"創建測試ArUco圖片失敗: {e}")
        print("請手動使用 make_ArUco 目錄中的程序創建ArUco標記")


if __name__ == "__main__":
    # 檢查是否存在ArUco標記
    has_aruco = False
    if os.path.exists("img"):
        for filename in os.listdir("img"):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # 簡單檢查是否可能包含ArUco
                image = cv2.imread(os.path.join("img", filename))
                if image is not None:
                    # 嘗試檢測ArUco
                    try:
                        aruco_dict = cv2.aruco.getPredefinedDictionary(
                            cv2.aruco.DICT_4X4_50
                        )
                        detector = cv2.aruco.ArucoDetector(aruco_dict)
                        corners, ids, _ = detector.detectMarkers(image)
                        if len(corners) > 0:
                            has_aruco = True
                            break
                    except:
                        pass

    if not has_aruco:
        print("警告: 未在圖片中檢測到ArUco標記")
        print("步驟2和步驟3可能會失敗")
        create_test_image()

    # 執行主程序
    main_pipeline()
