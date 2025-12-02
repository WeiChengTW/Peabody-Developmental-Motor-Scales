import cv2
import numpy as np
import os


def capture_photo():
    """
    使用攝影機拍攝照片
    """
    print("=== 標準面積測量工具 ===")
    print("請準備一張完整的 A4 紙放在拍攝區域")
    print()
    
    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機")
        return None
    
    print("攝影機已開啟，按下空白鍵拍照，按 ESC 退出")
    
    captured_img = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤: 無法讀取攝影機畫面")
            break
        
        # 在畫面上顯示提示文字
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            "Press SPACE to capture, ESC to exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        cv2.imshow("Capture Standard Paper", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 按空白鍵拍照
        if key == ord(' '):
            captured_img = frame.copy()
            print("照片已拍攝！")
            break
        
        # 按 ESC 退出
        elif key == 27:
            print("取消拍攝")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return captured_img


def calculate_paper_area(img):
    """
    計算圖片中紙張的面積
    """
    if img is None:
        print("錯誤: 沒有可用的圖片")
        return None
    
    # 影像預處理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 二值化 (使用 OTSU 自動找閾值)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("錯誤: 未偵測到任何輪廓")
        return None
    
    # 智慧過濾：找出「面積夠大」且「形狀像矩形」的輪廓
    best_cnt = None
    best_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 濾除太小的雜訊
        if area < 1000:
            continue
        
        # 計算矩形充滿度 (Extent)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area
        
        # 紙張通常是矩形，Extent 會比較高
        if extent > 0.65:
            if area > best_area:
                best_area = area
                best_cnt = cnt
    
    # 如果都沒找到像矩形的，使用最大輪廓
    if best_cnt is None:
        print("警告：找不到明顯的矩形物體，使用最大輪廓")
        best_cnt = max(contours, key=cv2.contourArea)
        best_area = cv2.contourArea(best_cnt)
    
    # 顯示結果
    display_img = img.copy()
    
    # 畫出輪廓 (綠色)
    cv2.drawContours(display_img, [best_cnt], -1, (0, 255, 0), 4)
    
    # 畫出包圍框 (紅色)
    x, y, w, h = cv2.boundingRect(best_cnt)
    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # 顯示面積資訊
    text = f"Area: {int(best_area)}"
    cv2.rectangle(display_img, (x, y - 40), (x + w, y), (0, 0, 0), -1)
    cv2.putText(
        display_img,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    
    # 縮放顯示
    h, w = display_img.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_dim = (max_width, int(h * scale))
        final_view = cv2.resize(display_img, new_dim, interpolation=cv2.INTER_AREA)
    else:
        final_view = display_img
    
    cv2.imshow("Measured Paper Area", final_view)
    print(f"\n偵測到的紙張面積: {int(best_area)}")
    print("按任意鍵確認...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return best_area


def save_result(area, img):
    """
    儲存結果到檔案
    """
    # 儲存照片
    save_dir = "iten57/img"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_path = os.path.join(save_dir, "standard_paper.jpg")
    cv2.imwrite(img_path, img)
    print(f"標準照片已儲存至: {img_path}")
    
    # 儲存面積數值到文字檔
    result_path = "iten57/standard_area.txt"
    with open(result_path, "w") as f:
        f.write(f"STANDARD_AREA = {int(area)}\n")
        f.write(f"\n# 測量日期: {os.popen('date').read()}")
        f.write(f"# 請將此數值複製到 cut_paper.py 中的 STANDARD_AREA 變數\n")
    
    print(f"標準面積數值已儲存至: {result_path}")
    print(f"\n=== 請複製以下數值到 cut_paper.py ===")
    print(f"STANDARD_AREA = {int(area)}")
    print(f"=====================================\n")


def main():
    """
    主程式
    """
    # 步驟 1: 拍攝照片
    img = capture_photo()
    
    if img is None:
        print("程式結束")
        return
    
    # 步驟 2: 計算面積
    area = calculate_paper_area(img)
    
    if area is None:
        print("無法計算面積，程式結束")
        return
    
    # 步驟 3: 確認是否儲存
    print("\n是否儲存此結果? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        save_result(area, img)
        print("完成！")
    else:
        print("結果未儲存")


if __name__ == "__main__":
    main()
