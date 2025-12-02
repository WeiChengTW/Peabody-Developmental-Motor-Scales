"""
測量標準紙張面積的工具程式
使用方式：
    1. 直接拍照模式：python measure_standard_area.py --camera
    2. 讀取圖片模式：python measure_standard_area.py <圖片路徑>
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime


def capture_from_camera(camera_index=0):
    """
    從攝影機拍攝標準紙張

    參數:
        camera_index: 攝影機編號 (預設 0)

    返回:
        img: 拍攝的圖片
    """
    print("\n=== 攝影機拍攝模式 ===")
    print("1. 請將完整的標準紙張放在攝影機前")
    print("2. 按下空白鍵 (Space) 拍照")
    print("3. 按下 ESC 取消退出")
    print("4. 拍照後可以按 'r' 重新拍攝，或按任意鍵繼續測量\n")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"錯誤: 無法開啟攝影機 {camera_index}")
        return None

    # 設定攝影機解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    captured_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤: 無法讀取攝影機畫面")
            break

        # 顯示即時畫面
        display_frame = frame.copy()

        # 添加提示文字
        if captured_img is None:
            text = "Press SPACE to capture | ESC to exit"
            color = (0, 255, 0)
        else:
            text = "Press R to retry | Any key to continue"
            color = (0, 255, 255)

        cv2.putText(
            display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        cv2.imshow("Camera - Capture Standard Paper", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC 鍵退出
        if key == 27:
            print("取消拍攝")
            cap.release()
            cv2.destroyAllWindows()
            return None

        # 空白鍵拍照
        elif key == 32:
            captured_img = frame.copy()
            print("✓ 拍照完成！")

            # 儲存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"standard_paper_{timestamp}.jpg"
            cv2.imwrite(save_path, captured_img)
            print(f"圖片已儲存至: {save_path}")

        # 'r' 鍵重新拍攝
        elif key == ord("r") or key == ord("R"):
            if captured_img is not None:
                print("重新拍攝...")
                captured_img = None

        # 其他按鍵確認並繼續
        elif key != 255 and captured_img is not None:
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_img


def measure_paper_area_from_image(img):
    """
    從圖片測量紙張面積的函式

    參數:
        img: 圖片 (numpy array)

    返回:
        area: 測量到的面積（像素平方）
    """
    if img is None:
        print("錯誤: 圖片為空")
        return None

    print(f"圖片尺寸: {img.shape[1]} x {img.shape[0]}")

    # 2. 影像預處理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化 (使用 OTSU 自動找閾值)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 顯示二值化結果 (方便調試)
    cv2.imshow("Threshold Image", thresh)

    # 3. 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("錯誤: 未偵測到任何輪廓")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    print(f"找到 {len(contours)} 個輪廓")

    # 4. 找出最符合紙張特徵的輪廓
    valid_contours = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # 濾除太小的雜訊 (面積小於 1000)
        if area < 1000:
            continue

        # 計算矩形充滿度 (Extent)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area

        # 計算縱橫比 (Aspect Ratio)
        aspect_ratio = float(w) / h if h > 0 else 0

        print(f"\n輪廓 {i}:")
        print(f"  面積: {area:.0f}")
        print(f"  矩形充滿度: {extent:.3f}")
        print(f"  縱橫比: {aspect_ratio:.3f}")
        print(f"  包圍框: {w} x {h}")

        # 紙張通常是矩形，Extent > 0.65
        if extent > 0.65:
            valid_contours.append(
                {
                    "contour": cnt,
                    "area": area,
                    "extent": extent,
                    "aspect_ratio": aspect_ratio,
                    "bbox": (x, y, w, h),
                }
            )

    if not valid_contours:
        print("\n警告: 找不到符合矩形特徵的輪廓，顯示所有大型輪廓供參考")
        # 顯示所有面積大於 1000 的輪廓
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 1000:
                valid_contours.append(
                    {
                        "contour": cnt,
                        "area": area,
                        "extent": 0,
                        "aspect_ratio": 0,
                        "bbox": cv2.boundingRect(cnt),
                    }
                )

    # 5. 選擇面積最大的輪廓作為標準紙張
    best_match = max(valid_contours, key=lambda x: x["area"])

    print(f"\n=== 選定的標準紙張 ===")
    print(f"面積: {best_match['area']:.0f} 像素²")
    print(f"矩形充滿度: {best_match['extent']:.3f}")
    print(f"縱橫比: {best_match['aspect_ratio']:.3f}")

    # 6. 視覺化結果
    display_img = img.copy()

    # 繪製所有候選輪廓 (淡藍色)
    for candidate in valid_contours:
        cv2.drawContours(display_img, [candidate["contour"]], -1, (255, 200, 100), 2)

    # 繪製最佳匹配 (綠色粗線)
    cv2.drawContours(display_img, [best_match["contour"]], -1, (0, 255, 0), 4)

    # 繪製包圍框 (紅色)
    x, y, w, h = best_match["bbox"]
    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 添加文字說明
    text = f"Area: {best_match['area']:.0f}"
    cv2.rectangle(display_img, (x, y - 40), (x + w, y), (0, 0, 0), -1)
    cv2.putText(
        display_img,
        text,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # 縮放顯示 (避免圖片太大)
    h_img, w_img = display_img.shape[:2]
    max_width = 1000
    if w_img > max_width:
        scale = max_width / w_img
        new_dim = (max_width, int(h_img * scale))
        display_img = cv2.resize(display_img, new_dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Detected Paper", display_img)

    print("\n按下任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best_match["area"]


def measure_paper_area(image_path):
    """
    從圖片檔案測量紙張面積

    參數:
        image_path: 圖片路徑

    返回:
        area: 測量到的面積（像素平方）
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤: 無法讀取圖片 {image_path}")
        return None

    return measure_paper_area_from_image(img)


def measure_multiple_captures(num_captures=3):
    """
    從攝影機拍攝多張圖片並計算平均值

    參數:
        num_captures: 要拍攝的次數

    返回:
        average_area: 平均面積
    """
    areas = []

    print(f"\n將進行 {num_captures} 次拍攝測量\n")

    for i in range(num_captures):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{num_captures} 次拍攝")
        print(f"{'='*60}")

        img = capture_from_camera()
        if img is None:
            print("取消測量")
            break

        area = measure_paper_area_from_image(img)
        if area is not None:
            areas.append(area)
            print(f"✓ 第 {i+1} 次測量完成，面積: {area:.0f} 像素²")

    if not areas:
        print("\n錯誤: 沒有成功測量任何圖片")
        return None

    # 計算統計資訊
    avg_area = np.mean(areas)
    std_area = np.std(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)

    print(f"\n{'='*60}")
    print(f"=== 測量結果統計 ===")
    print(f"{'='*60}")
    print(f"測量次數: {len(areas)}")
    print(f"平均面積: {avg_area:.0f} 像素²")
    print(f"標準差: {std_area:.0f}")
    print(f"最小值: {min_area:.0f}")
    print(f"最大值: {max_area:.0f}")
    print(f"變異係數: {(std_area/avg_area)*100:.2f}%")
    print(f"\n建議使用的 STANDARD_AREA = {int(avg_area)}")
    print(f"{'='*60}")

    return int(avg_area)
    """
    測量多張圖片並計算平均值

    參數:
        image_paths: 圖片路徑列表

    返回:
        average_area: 平均面積
    """


def measure_multiple_images(image_paths):
    """
    測量多張圖片檔案並計算平均值

    參數:
        image_paths: 圖片路徑列表

    返回:
        average_area: 平均面積
    """
    areas = []

    for i, path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"測量第 {i+1}/{len(image_paths)} 張圖片: {path}")
        print(f"{'='*60}")

        area = measure_paper_area(path)
        if area is not None:
            areas.append(area)

    if not areas:
        print("\n錯誤: 沒有成功測量任何圖片")
        return None

    # 計算統計資訊
    avg_area = np.mean(areas)
    std_area = np.std(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)

    print(f"\n{'='*60}")
    print(f"=== 測量結果統計 ===")
    print(f"{'='*60}")
    print(f"測量次數: {len(areas)}")
    print(f"平均面積: {avg_area:.0f} 像素²")
    print(f"標準差: {std_area:.0f}")
    print(f"最小值: {min_area:.0f}")
    print(f"最大值: {max_area:.0f}")
    print(f"變異係數: {(std_area/avg_area)*100:.2f}%")
    print(f"\n建議使用的 STANDARD_AREA = {int(avg_area)}")
    print(f"{'='*60}")

    return int(avg_area)


if __name__ == "__main__":
    print("=== 標準紙張面積測量工具 ===\n")

    # 攝影機模式
    if len(sys.argv) >= 2 and sys.argv[1] in ["--camera", "-c", "camera"]:
        # 檢查是否指定拍攝次數
        num_captures = 3  # 預設 3 次
        if len(sys.argv) >= 3:
            try:
                num_captures = int(sys.argv[2])
            except:
                print(f"警告: 無法解析拍攝次數 '{sys.argv[2]}'，使用預設值 3")

        # 多次拍攝測量
        measure_multiple_captures(num_captures)

    # 圖片檔案模式
    elif len(sys.argv) >= 2:
        image_paths = sys.argv[1:]

        # 檢查所有檔案是否存在
        for path in image_paths:
            if not os.path.exists(path):
                print(f"錯誤: 找不到檔案 {path}")
                sys.exit(1)

        if len(image_paths) == 1:
            # 測量單張圖片
            area = measure_paper_area(image_paths[0])
            if area is not None:
                print(f"\n建議使用的 STANDARD_AREA = {int(area)}")
        else:
            # 測量多張圖片並計算平均
            measure_multiple_images(image_paths)

    # 顯示使用說明
    else:
        print("使用方式:")
        print("  1. 攝影機拍攝模式:")
        print("     python measure_standard_area.py --camera [拍攝次數]")
        print("     範例: python measure_standard_area.py --camera 3")
        print()
        print("  2. 測量單張圖片:")
        print("     python measure_standard_area.py <圖片路徑>")
        print("     範例: python measure_standard_area.py standard_paper.jpg")
        print()
        print("  3. 測量多張圖片:")
        print("     python measure_standard_area.py <圖片1> <圖片2> <圖片3> ...")
        print(
            "     範例: python measure_standard_area.py paper1.jpg paper2.jpg paper3.jpg"
        )
        sys.exit(1)
