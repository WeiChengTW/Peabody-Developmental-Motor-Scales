"""
簡單的 S3 上傳和 URL 產生工具
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "upload2AWS"))
from upload2AWS.s3_upload_main import S3Uploader

# 固定設定
BUCKET_NAME = "cgu-pdms2"
REGION = "ap-southeast-2"

# ✅ 是的!這個 URL 前綴是固定的
S3_BASE_URL = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/"


def upload_and_get_urls(folder_path, s3_prefix=""):
    """
    上傳資料夾並返回所有檔案的 URL

    參數:
        folder_path: 例如 "PDMS2_web/kid/cc22"
        s3_prefix: 例如 "kid/cc22"

    返回:
        dict: {"檔案名稱": "完整URL", ...}
    """
    uploader = S3Uploader(BUCKET_NAME, REGION)

    if not uploader.bucket_exists():
        print(f"❌ Bucket '{BUCKET_NAME}' 不存在")
        return {}

    if not os.path.exists(folder_path):
        print(f"❌ 資料夾 '{folder_path}' 不存在")
        return {}

    result = {}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, folder_path)

            # S3 路徑
            if s3_prefix:
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            else:
                s3_key = relative_path.replace("\\", "/")

            # 上傳
            if uploader.upload_file(local_path, s3_key):
                # 產生 URL
                url = f"{S3_BASE_URL}{s3_key}"
                result[file] = url

    return result


def get_s3_url(file_path):
    """
    直接產生 S3 URL (不上傳,假設檔案已在 S3 中)

    參數:
        file_path: S3 中的檔案路徑,例如 "kid/cc22/image.jpg"

    返回:
        str: 完整的公開 URL
    """
    return f"{S3_BASE_URL}{file_path}"


# ========== 使用範例 ==========

if __name__ == "__main__":
    # 範例 1: 上傳 cc22 資料夾
    print("🔹 上傳 cc22 資料夾")
    urls = upload_and_get_urls(r"PDMS2_web\kid\cc22", "kid/cc22")

    for filename, url in urls.items():
        print(f"  {filename}: {url}")

    # 範例 2: 直接產生 URL (檔案已在 S3)
    print("\n🔹 直接產生 URL (固定前綴)")
    print(f"  基礎 URL: {S3_BASE_URL}")
    print(f"  檔案 URL: {get_s3_url('kid/cc22/test.jpg')}")
