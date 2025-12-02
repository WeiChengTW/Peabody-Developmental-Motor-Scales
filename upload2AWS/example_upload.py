"""
示範如何使用 S3Uploader class 上傳資料夾
"""

import sys
import os

# 添加 upload2AWS 路徑以便 import
sys.path.append(os.path.join(os.path.dirname(__file__), "upload2AWS"))

from upload2AWS.s3_upload_main import S3Uploader

# ========== 設定 ==========
BUCKET_NAME = "cgu-pdms2"
REGION = "ap-southeast-2"
# ==========================

# 固定的 S3 網址前綴
S3_URL_PREFIX = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/"


def upload_folder_example(folder_path, s3_prefix=""):
    """
    上傳資料夾到 S3 並回傳檔案 URL 列表

    Args:
        folder_path: 本地資料夾路徑 (例如: "PDMS2_web/kid/cc22")
        s3_prefix: S3 中的前綴路徑 (可選)

    Returns:
        list: 包含所有上傳檔案的公開 URL
    """
    # 建立上傳器
    uploader = S3Uploader(BUCKET_NAME, REGION)

    # 檢查 bucket 是否存在
    if not uploader.bucket_exists():
        print(f"❌ Bucket '{BUCKET_NAME}' 不存在")
        return []

    # 檢查資料夾是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 資料夾 '{folder_path}' 不存在")
        return []

    print(f"📤 開始上傳: {folder_path}")
    print(f"   目標 Bucket: {BUCKET_NAME}")
    print(f"   S3 前綴: {s3_prefix if s3_prefix else '(根目錄)'}\n")

    # 收集上傳的檔案 URL
    uploaded_urls = []

    # 遍歷資料夾
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 本地檔案路徑
            local_path = os.path.join(root, file)

            # 計算相對路徑
            relative_path = os.path.relpath(local_path, folder_path)

            # 組合 S3 Key
            if s3_prefix:
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            else:
                s3_key = relative_path.replace("\\", "/")

            # 上傳檔案
            if uploader.upload_file(local_path, s3_key):
                # 產生公開 URL
                url = uploader.get_public_url(s3_key)
                uploaded_urls.append(url)
                print(f"   🔗 {url}\n")

    print(f"✅ 上傳完成! 共 {len(uploaded_urls)} 個檔案\n")

    return uploaded_urls


def get_s3_url(s3_key):
    """
    直接產生 S3 檔案的公開 URL (不需要上傳)

    Args:
        s3_key: S3 中的檔案路徑

    Returns:
        str: 公開 URL
    """
    return f"{S3_URL_PREFIX}{s3_key}"


# ========== 使用範例 ==========

if __name__ == "__main__":
    # 範例 1: 上傳 cc22 資料夾,保持資料夾結構
    print("=" * 70)
    print("範例 1: 上傳 PDMS2_web/kid/cc22 資料夾")
    print("=" * 70)

    folder_path = r"PDMS2_web\kid\cc22"
    s3_prefix = "kid/cc22"  # 在 S3 中的路徑

    urls = upload_folder_example(folder_path, s3_prefix)

    print("\n上傳的檔案 URLs:")
    for url in urls:
        print(f"  • {url}")

    print("\n" + "=" * 70)
    print("範例 2: 直接產生 S3 URL (已知檔案在 S3 中)")
    print("=" * 70)

    # 假設檔案已經在 S3 中,直接產生 URL
    example_files = [
        "kid/cc22/image1.jpg",
        "kid/cc22/image2.jpg",
        "kid/cc22/result.json",
    ]

    print("\n固定的 S3 URL 前綴:")
    print(f"  {S3_URL_PREFIX}")

    print("\n檔案 URLs:")
    for file_key in example_files:
        url = get_s3_url(file_key)
        print(f"  • {url}")

    print("\n" + "=" * 70)
    print("✅ 是的!URL 前綴是固定的:")
    print(f"   https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/")
    print("\n只要知道檔案在 S3 中的路徑,就可以直接組合出 URL")
    print("=" * 70)
