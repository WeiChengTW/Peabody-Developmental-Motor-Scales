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
from cross_or_other import ImageClassifier
import shutil
from cross_detect import CrossScorer


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

def main(img_path):
    #==參數==#
    real_width_cm = 29.7
    SCALE = 2
    SCORE = -1

<<<<<<<< HEAD:draw_cross/main/main.py
    input_folder = "realtest"   # <-- 資料夾
    MODEL_PATH = r'model/cross_final.h5'
    CLASS_NAMES = ['cross', 'other']
========
    input_folder = "input"   # <-- 資料夾
    MODEL_PATH = r'model/circle_detect.h5'
    CLASS_NAMES = ['Other', 'circle_or_oval']
>>>>>>>> e176c5dcb783d5d7add1977640776f9d83438e4e:draw_circle/main/test.py
    #==參數==#

    # 讀取資料夾內所有圖片
    # all_images = read_all_images_from_folder(input_folder)

    ## 先建立分類資料夾
    os.makedirs("cross", exist_ok=True)
    os.makedirs("other", exist_ok=True)

    classifier = ImageClassifier(MODEL_PATH, CLASS_NAMES)
    cp = check_point(SCALE=SCALE)
<<<<<<<< HEAD:draw_cross/main/main.py

    #參數要改
    cs = CrossScorer(cm_per_pixel=0.02079, angle_min=70.0, angle_max=110.0, max_spread_cm=0.6)
========
>>>>>>>> e176c5dcb783d5d7add1977640776f9d83438e4e:draw_circle/main/test.py

    #初始化空間
    segmenter = Analyze_graphics()
    segmenter.initialize_workspace()


    # 逐張處理
    # for origin_img in all_images:
    #     print(f"\n=== 處理 {origin_img} ===\n")

    #     # 得出 px->cm
    #     try:
    #         pixel_per_cm, _, cropped_path = get_pixel_per_cm_from_a4(
    #             origin_img, 
    #             show_debug=False,  # 關掉視覺化避免卡住
    #             save_cropped=True,
    #             output_folder="cropped_a4"
    #         )
    #         print(f"{origin_img} pixel_per_cm = {pixel_per_cm}")
    #     except ValueError as e:
    #         print(f"⚠️ 跳過 {origin_img}：{e}")
    #         continue  # 直接跳過這張圖片
        
<<<<<<<< HEAD:draw_cross/main/main.py
    #     # 裁切圖形
    #     print('\n==裁切圖形==')
    #     segmenter = Analyze_graphics()
    #     # print(cropped_path)
    #     ready = segmenter.infer_and_draw(cropped_path, expand_ratio=0.15)
========
        # 裁切圖形
        print('\n==裁切圖形==')
        segmenter = Analyze_graphics()
        print(cropped_path)
        ready = segmenter.infer_and_draw(cropped_path, expand_ratio=0.15)
>>>>>>>> e176c5dcb783d5d7add1977640776f9d83438e4e:draw_circle/main/test.py

    #     # 分類圖形(圓 橢圓 其他)
    #     print('\n==分類圖形==\n')
    #     result = {}

    #     for rb in ready:
    #         if "binary" in rb:
    #             predicted_class_name, conf = classifier.predict(rb)
    #             url = rb.replace('_binary', "")
    #             print(f"{url} → {predicted_class_name} ({conf*100:.2f}%)")
    #             result[url] = predicted_class_name

<<<<<<<< HEAD:draw_cross/main/main.py
    #             # 直接分類存檔
    #             if predicted_class_name == "cross":
    #                 shutil.copy(url, os.path.join("cross", os.path.basename(url)))
    #             else:
    #                 # 讀取圖片並加上標記
    #                 img = cv2.imread(url)
    #                 cv2.putText(img, 'Other !', 
    #                             (30, 50), 
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #                 save_path = os.path.join("Other", os.path.basename(url))
    #                 cv2.imwrite(save_path, img)  # 直接存檔，不用手動關視窗
    #                 print(f"{url} 已存入 Other 資料夾並加上標記")
        
    # 單張處理
    print(f"\n=== 處理 {img_path} ===\n")

    # 得出 px->cm
    try:
        pixel_per_cm, _, cropped_path = get_pixel_per_cm_from_a4(
            img_path, 
            show_debug=False,  # 關掉視覺化避免卡住
            save_cropped=True,
            output_folder="cropped_a4"
        )
        print(f"{img_path} pixel_per_cm = {pixel_per_cm}")
    except ValueError as e:
        print(f"⚠️ 跳過 {img_path}：{e}")
        
    
    # 裁切圖形
    print('\n==裁切圖形==')
    # print(cropped_path)
    ready = segmenter.infer_and_draw(cropped_path, expand_ratio=0.15)

    # 分類圖形(圓 橢圓 其他)
    print('\n==分類圖形==\n')
    result = {}

    
    for rb in ready:
        if "binary" in rb:
            predicted_class_name, conf = classifier.predict(rb)
            url = rb.replace('_binary', "")
            print(f"{url} → {predicted_class_name} ({conf*100:.2f}%)")
            result[url] = predicted_class_name

            # 直接分類存檔
            if predicted_class_name == "cross":
                shutil.copy(url, os.path.join("cross", os.path.basename(url)))
                results, _, _, _ = cs.score_image(url)
                return results['score']
                
            else:
                # 讀取圖片並加上標記
========
        for rb in ready:
            if "binary" in rb:
                predicted_class_name, conf = classifier.predict(rb)
                url = rb.replace('_binary', "")
                print(f"{url} → {predicted_class_name} ({conf*100:.2f}%)")
                result[url] = predicted_class_name

                # 直接分類存檔
                if predicted_class_name == "circle_or_oval":
                    shutil.copy(url, os.path.join("circle_or_oval", os.path.basename(url)))
                else:
                    # 讀取圖片並加上標記
                    img = cv2.imread(url)
                    cv2.putText(img, 'Other !', 
                                (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    save_path = os.path.join("Other", os.path.basename(url))
                    cv2.imwrite(save_path, img)  # 直接存檔，不用手動關視窗
                    print(f"{url} 已存入 Other 資料夾並加上標記")


        # 計算端點距離 & 複製到對應資料夾
        print('\n==計算端點距離==\n')
        for url, u_type in result.items():
            if u_type == 'circle_or_oval':
                px = cp.check_point(url)
                if px == 0.0:
                    print(f'{url} : Perfect!')
                else:
                    offset = px / pixel_per_cm
                    print(f'{url} : {offset}cm')
                    if offset <= 1.2:
                        SCORE = 2
                    elif offset > 1.2 and offset <= 2.5:
                        SCORE = 1
                    else:
                        SCORE = 0

            else:
>>>>>>>> e176c5dcb783d5d7add1977640776f9d83438e4e:draw_circle/main/test.py
                img = cv2.imread(url)
                cv2.putText(img, 'Other !', 
                            (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
<<<<<<<< HEAD:draw_cross/main/main.py
                save_path = os.path.join("other", os.path.basename(url))
                cv2.imwrite(save_path, img)  # 直接存檔，不用手動關視窗
                print(f"{url} 已存入 Other 資料夾並加上標記")
                return 0
        
========
                # cv2.imshow('Other', img)
                print(f'{url} is {result[url]}!')
                SCORE = 0
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        


>>>>>>>> e176c5dcb783d5d7add1977640776f9d83438e4e:draw_circle/main/test.py
    return SCORE


if __name__ == "__main__":
<<<<<<<< HEAD:draw_cross/main/main.py
    img_path = r'realtest\S__75628563.jpg'
========
    img_path = 'test01.jpg'
>>>>>>>> e176c5dcb783d5d7add1977640776f9d83438e4e:draw_circle/main/test.py
    score = main(img_path)
    print(score)