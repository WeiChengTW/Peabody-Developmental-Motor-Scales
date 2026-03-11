import cv2
import numpy as np
import os
from score_2_complete_cut import CompleteCutDetector
from score_1_partial_cut import PartialCutDetector
from score_0_no_cut import NoActualCutDetector


class PaperCuttingScorer:
    """
    整合所有三個評分檢測器的主要評分系統
    """

    def __init__(self):
        self.complete_cut_detector = CompleteCutDetector()
        self.partial_cut_detector = PartialCutDetector()
        self.no_cut_detector = NoActualCutDetector()

    def score_image(self, image_path: str) -> int:
        """
        對單張圖像進行評分
        返回值：
        2 - 把紙剪成均分的2等份
        1 - 只剪到紙的1/4或更少
        0 - 只動動剪刀未剪下去
        """
        if not os.path.exists(image_path):
            print(f"圖像文件不存在: {image_path}")
            return 0

        print(f"\n正在分析圖像: {image_path}")
        print("=" * 50)

        # 按優先級順序檢測
        # 首先檢測是否為完全剪切（評分2）
        if self.complete_cut_detector.detect_complete_cut(image_path):
            print("最終評分: 2 (完全剪切 - 均分成2等份)")
            return 2

        # 然後檢測是否為部分剪切（評分1）
        if self.partial_cut_detector.detect_partial_cut(image_path):
            print("最終評分: 1 (部分剪切 - 剪到1/4或更少)")
            return 1

        # 最後默認為無實際剪切（評分0）
        if self.no_cut_detector.detect_no_actual_cut(image_path):
            print("最終評分: 0 (無實際剪切 - 只動剪刀)")
            return 0

        # 如果都不符合，默認為0分
        print("最終評分: 0 (默認 - 無明確剪切證據)")
        return 0

    def batch_score(self, image_folder: str) -> dict:
        """批量評分資料夾中的所有圖像"""
        if not os.path.exists(image_folder):
            print(f"資料夾不存在: {image_folder}")
            return {}

        results = {}
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

        print(f"\n開始批量分析資料夾: {image_folder}")
        print("=" * 60)

        image_files = [
            f for f in os.listdir(image_folder) if f.lower().endswith(supported_formats)
        ]

        if not image_files:
            print("資料夾中沒有找到支援的圖像文件")
            return results

        for i, filename in enumerate(image_files, 1):
            print(f"\n進度: {i}/{len(image_files)}")
            image_path = os.path.join(image_folder, filename)
            score = self.score_image(image_path)
            results[filename] = score

        return results

    def generate_detailed_report(self, results: dict) -> str:
        """生成詳細的評分報告"""
        if not results:
            return "沒有可分析的結果"

        # 統計各評分的數量
        score_counts = {0: 0, 1: 0, 2: 0}
        for score in results.values():
            score_counts[score] += 1

        total_images = len(results)

        report = f"""
剪紙評分詳細報告
{'=' * 40}

總圖像數量: {total_images}

評分分布:
- 評分 2 (完全剪切): {score_counts[2]} 張 ({score_counts[2]/total_images*100:.1f}%)
- 評分 1 (部分剪切): {score_counts[1]} 張 ({score_counts[1]/total_images*100:.1f}%)  
- 評分 0 (無實際剪切): {score_counts[0]} 張 ({score_counts[0]/total_images*100:.1f}%)

詳細結果:
{'-' * 40}
"""

        # 按評分分組顯示
        for score in [2, 1, 0]:
            score_names = {2: "完全剪切", 1: "部分剪切", 0: "無實際剪切"}
            files_with_score = [
                filename for filename, s in results.items() if s == score
            ]

            if files_with_score:
                report += f"\n評分 {score} ({score_names[score]}):\n"
                for filename in sorted(files_with_score):
                    report += f"  - {filename}\n"

        return report

    def visualize_all_results(self, image_folder: str, output_folder: str = "results"):
        """為所有圖像生成視覺化結果"""
        if not os.path.exists(image_folder):
            print(f"資料夾不存在: {image_folder}")
            return

        # 創建輸出資料夾
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(image_folder, filename)
                score = self.score_image(image_path)

                # 根據評分選擇對應的視覺化方法
                name_without_ext = os.path.splitext(filename)[0]
                output_path = os.path.join(
                    output_folder, f"{name_without_ext}_score_{score}.jpg"
                )

                if score == 2:
                    self.complete_cut_detector.visualize_detection(
                        image_path, output_path
                    )
                elif score == 1:
                    self.partial_cut_detector.visualize_detection(
                        image_path, output_path
                    )
                else:
                    self.no_cut_detector.visualize_detection(image_path, output_path)

                print(f"已保存視覺化結果: {output_path}")


def main():
    """主函數 - 示範如何使用評分系統"""
    scorer = PaperCuttingScorer()

    print("剪紙評分系統")
    print("=" * 30)
    print("評分標準:")
    print("2 - 把紙剪成均分的2等份")
    print("1 - 只剪到紙的1/4或更少")
    print("0 - 只動動剪刀未剪下去")
    print()

    # 單張圖像測試
    # test_image = "test_image.jpg"
    # if os.path.exists(test_image):
    #     print("單張圖像測試:")
    #     score = scorer.score_image(test_image)
    #     print(f"圖像 {test_image} 的評分: {score}")

    # 批量測試
    image_folder = "images"
    if os.path.exists(image_folder):
        print("\n批量分析:")
        results = scorer.batch_score(image_folder)

        if results:
            # 生成報告
            report = scorer.generate_detailed_report(results)
            print(report)

            # 保存報告到文件
            with open("cutting_score_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            print("詳細報告已保存到: cutting_score_report.txt")

            # 生成視覺化結果
            scorer.visualize_all_results(image_folder, "cutting_analysis_results")
            print("所有視覺化結果已保存到: cutting_analysis_results/")

    # 互動模式
    while True:
        print("\n" + "=" * 50)
        print("互動模式選項:")
        print("1. 分析單張圖像")
        print("2. 批量分析資料夾")
        print("3. 退出")

        choice = input("請選擇操作 (1-3): ").strip()

        if choice == "1":
            image_path = input("請輸入圖像路徑: ").strip()
            if os.path.exists(image_path):
                score = scorer.score_image(image_path)
                print(f"\n評分結果: {score}")
            else:
                print("圖像文件不存在!")

        elif choice == "2":
            folder_path = input("請輸入資料夾路徑: ").strip()
            if os.path.exists(folder_path):
                results = scorer.batch_score(folder_path)
                if results:
                    report = scorer.generate_detailed_report(results)
                    print(report)
                else:
                    print("資料夾中沒有找到圖像文件!")
            else:
                print("資料夾不存在!")

        elif choice == "3":
            print("感謝使用剪紙評分系統!")
            break

        else:
            print("無效選擇，請重新輸入!")


if __name__ == "__main__":
    main()
