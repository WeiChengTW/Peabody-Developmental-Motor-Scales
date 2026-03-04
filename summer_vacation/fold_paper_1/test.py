from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np


def enhance_fold_lines(image_path, output_path, contrast_factor=5.0):
    """
    增強影像中的摺線，使其更加明顯

    參數:
    - image_path: 輸入影像路徑
    - output_path: 輸出影像路徑
    - contrast_factor: 對比度增強倍數，預設為5.0
    """

    # 步驟1：載入影像
    original_image = Image.open(image_path)
    print(f"原始影像大小: {original_image.size}")

    # 步驟2：轉換為灰階
    grayscale_image = original_image.convert("L")
    print("已轉換為灰階影像")

    # 步驟3：應用邊緣檢測濾鏡
    edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
    print("已應用邊緣檢測濾鏡")

    # 步驟4：增強對比度
    enhancer = ImageEnhance.Contrast(edge_image)
    enhanced_image = enhancer.enhance(contrast_factor)
    print(f"已增強對比度 {contrast_factor} 倍")

    # 步驟5：顏色反轉（讓摺線更明顯）
    inverted_image = ImageOps.invert(enhanced_image)
    print("已反轉顏色")

    # 步驟6：儲存結果
    inverted_image.save(output_path)
    print(f"處理完成，已儲存至: {output_path}")

    return inverted_image


def enhance_fold_lines_advanced(
    image_path, output_path, blur_radius=1, contrast_factor=5.0
):
    """
    進階版本：加入高斯模糊來減少雜訊

    參數:
    - image_path: 輸入影像路徑
    - output_path: 輸出影像路徑
    - blur_radius: 高斯模糊半徑，預設為1
    - contrast_factor: 對比度增強倍數，預設為5.0
    """

    # 載入影像
    original_image = Image.open(image_path)

    # 轉換為灰階
    grayscale_image = original_image.convert("L")

    # 可選：先進行輕微的高斯模糊來減少雜訊
    if blur_radius > 0:
        blurred_image = grayscale_image.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
        print(f"已應用高斯模糊，半徑: {blur_radius}")
    else:
        blurred_image = grayscale_image

    # 應用邊緣檢測
    edge_image = blurred_image.filter(ImageFilter.FIND_EDGES)

    # 增強對比度
    enhancer = ImageEnhance.Contrast(edge_image)
    enhanced_image = enhancer.enhance(contrast_factor)

    # 顏色反轉
    final_image = ImageOps.invert(enhanced_image)

    # 儲存結果
    final_image.save(output_path)
    print(f"進階處理完成，已儲存至: {output_path}")

    return final_image


def batch_process_images(input_folder, output_folder, contrast_factor=5.0):
    """
    批次處理多個影像檔案

    參數:
    - input_folder: 輸入資料夾路徑
    - output_folder: 輸出資料夾路徑
    - contrast_factor: 對比度增強倍數
    """
    import os

    # 支援的影像格式
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 處理資料夾中的所有影像
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"enhanced_{filename}"
            output_path = os.path.join(output_folder, output_filename)

            print(f"正在處理: {filename}")
            enhance_fold_lines(input_path, output_path, contrast_factor)


# 使用範例
if __name__ == "__main__":
    # 基本使用方法
    input_image = r"img\7.jpg"  # 輸入影像路徑
    output_image = "7_enhanced.jpg"  # 輸出影像路徑

    # 方法1：基本處理
    # enhanced_result = enhance_fold_lines(input_image, output_image, contrast_factor=5.0)

    # 方法2：進階處理（包含雜訊減少）
    # enhanced_result = enhance_fold_lines_advanced(
    #     input_image, "7_advanced.jpg", blur_radius=1, contrast_factor=5.0
    # )

    # 方法3：批次處理多個檔案
    # batch_process_images("input_folder", "output_folder", contrast_factor=5.0)

    print("所有處理完成！")
