import boto3
import os

# 設定 AWS S3 客戶端
s3 = boto3.client("s3", region_name="ap-southeast-2")

# 設定您的 S3 bucket 名稱
BUCKET_NAME = "chang-pdms-testdata-2024"

# 要上傳的資料夾路徑
LOCAL_FOLDER = "testforupload"


def upload_folder_to_s3(local_folder, bucket_name, s3_prefix="testforupload"):
    """
    上傳整個資料夾到 S3

    Args:
        local_folder: 本地資料夾路徑
        bucket_name: S3 bucket 名稱
        s3_prefix: S3 中的前綴路徑(資料夾名稱)
    """
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            # 本地檔案完整路徑
            local_path = os.path.join(root, file)

            # 計算相對路徑,用於 S3 的 Key
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                print(f"正在上傳: {local_path} -> s3://{bucket_name}/{s3_key}")
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"✓ 成功上傳: {s3_key}")
            except Exception as e:
                print(f"✗ 上傳失敗 {local_path}: {str(e)}")


if __name__ == "__main__":
    print(f"開始上傳資料夾 '{LOCAL_FOLDER}' 到 S3 bucket '{BUCKET_NAME}'...")
    upload_folder_to_s3(LOCAL_FOLDER, BUCKET_NAME)
    print("上傳完成!")
