#裁切圖形 + 得出px->cm -> 分類圖形(圓 橢圓 其他) -> 標示端點&算距離

import cv2
import numpy as np
from skimage.morphology import skeletonize
import math
import json
from Analyze_graphics import Analyze_graphics
import glob
from PIL import Image
import os
from check_point import check_point
from circle_or_oval import ImageClassifier


def get_pixel_per_cm_from_a4(image_path, real_width_cm=29.7, show_debug=False, save_cropped=True, output_folder="cropped_a4"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("圖片讀取失敗，請確認路徑正確")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a4_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(a4_contour, True)
    approx = cv2.approxPolyDP(a4_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("無法偵測 A4 紙四邊形輪廓")

    if show_debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, [approx], -1, (0, 0, 255), 3)
        cv2.imshow("Detected A4 Contour", cv2.resize(debug_img, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 整理四個角點
    pts = approx.reshape(4, 2).astype(np.float32)
    pts = sorted(pts, key=lambda p: p[0])  # 先左右
    left = sorted(pts[0:2], key=lambda p: p[1])
    right = sorted(pts[2:4], key=lambda p: p[1])
    tl, bl = left
    tr, br = right

    # 計算像素/公分比例
    a4_pixel_width = np.linalg.norm(tr - tl)
    pixel_per_cm = float(a4_pixel_width / real_width_cm)  # 轉換為 Python 原生 float

    # 儲存裁切後的A4區域
    cropped_path = None
    if save_cropped:
        # 建立輸出資料夾
        os.makedirs(output_folder, exist_ok=True)
        
        # 計算原始A4區域的實際尺寸
        width1 = np.linalg.norm(tr - tl)  # 上邊長度
        width2 = np.linalg.norm(br - bl)  # 下邊長度
        height1 = np.linalg.norm(tl - bl) # 左邊長度
        height2 = np.linalg.norm(tr - br) # 右邊長度
        
        # 取平均值作為目標尺寸，保持原始比例
        target_width = int((width1 + width2) / 2)
        target_height = int((height1 + height2) / 2)
        
        # 原始四個角點（順序：左上、右上、右下、左下）
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        
        # 目標四個角點
        dst_pts = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ], dtype=np.float32)
        
        # 計算透視變換矩陣
        transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 進行透視變換
        warped = cv2.warpPerspective(img, transform_matrix, (target_width, target_height))
        
        # 儲存裁切後的圖片
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        cropped_filename = f"{image_name}_a4_cropped.jpg"
        cropped_path = os.path.join(output_folder, cropped_filename)
        cv2.imwrite(cropped_path, warped)
        
        print(f"A4區域已儲存至: {cropped_path}")

    # 儲存像素比例資料
    json_path = "px2cm.json"
    data = {
        "pixel_per_cm": pixel_per_cm, 
        "image_path": image_path,
        "cropped_path": cropped_path
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return pixel_per_cm, json_path, cropped_path

def read_all_images_from_folder(folder_path):
    """讀取資料夾中所有圖片（包含子資料夾）"""
    
    # 支援的圖片格式
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
    
    all_images = []
    
    # 使用 ** 進行遞迴搜尋
    for ext in image_extensions:
        # 搜尋當前資料夾
        pattern1 = os.path.join(folder_path, f"*.{ext}")
        pattern2 = os.path.join(folder_path, f"*.{ext.upper()}")
        
        # 搜尋所有子資料夾（遞迴）
        pattern3 = os.path.join(folder_path, "**", f"*.{ext}")
        pattern4 = os.path.join(folder_path, "**", f"*.{ext.upper()}")
        
        all_images.extend(glob.glob(pattern1))
        all_images.extend(glob.glob(pattern2))
        all_images.extend(glob.glob(pattern3, recursive=True))
        all_images.extend(glob.glob(pattern4, recursive=True))
    
    # 去除重複
    all_images = list(set(all_images))
    
    print(f"找到 {len(all_images)} 張圖片")
    
    # 處理每張圖片
    for image_path in all_images:
        try:
            image = Image.open(image_path)
            print(f"讀取: {os.path.basename(image_path)} - 尺寸: {image.size}")
            
            # 在這裡處理你的圖片
            # image.show()  # 顯示圖片
            
        except Exception as e:
            print(f"無法讀取 {image_path}: {e}")
    
    return all_images

if __name__ == "__main__":

    #==參數==#
    real_width_cm = 29.7
    SCALE = 2
    origin_img = r'demo\1.jpg'
    MODEL_PATH = r'circle_or_oval\Final_model.h5'
    CLASS_NAMES = ['Other', 'circle_or_oval']
    result = {}
    #==參數==#


    #得出px->cm
    print('\n==得出px -> cm==\n')
    pixel_per_cm, _, cropped_path = get_pixel_per_cm_from_a4(
        origin_img, 
        show_debug=True, 
        save_cropped=True,
        output_folder="cropped_a4"
    )
    print(pixel_per_cm)
    
    #裁切圖形
    print('\n==裁切圖形==')
    segmenter = Analyze_graphics()
    ready = segmenter.infer_and_draw(cropped_path)

    #分類圖形(圓 橢圓 其他)
    print('\n==分類圖形==\n')
    classifier = ImageClassifier(MODEL_PATH, CLASS_NAMES)

    for rb in ready:
        if "binary" in rb:
            predicted_class_name, conf = classifier.predict(rb)
            url = rb.replace('_binary', "")
            print(f"{url} → {predicted_class_name} ({conf*100:.2f}%)")
            result.update({url:predicted_class_name})
        else:
            continue

    #計算端點距離
    print('\n==計算端點距離==\n')
    cp = check_point(SCALE=SCALE)

    for url, u_type in result.items():
        if u_type == 'circle_or_oval':
            px = cp.check_point(url)
            if px == 0.0:
                print(f'{url} : Perfect!')
            else:    
                print(f'{url} : {px / pixel_per_cm}cm')
        else:
            img = cv2.imread(url)
            cv2.putText(img, 'Other !', 
                    (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('Other', img)
            print(f'{url} is {result[url]}!')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
       