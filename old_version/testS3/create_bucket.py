import boto3
from botocore.exceptions import ClientError

# 設定 bucket 名稱 (必須全球唯一)
# 建議格式: your-name-pdms-data-2024
BUCKET_NAME = "chang-pdms-testdata-2024"  # 請修改這裡
REGION = "ap-southeast-2"  # 雪梨區域

try:
    s3 = boto3.client("s3", region_name=REGION)

    print(f"正在建立 S3 bucket: {BUCKET_NAME}")
    print(f"區域: {REGION}")

    # 如果是 us-east-1 以外的區域,需要指定 LocationConstraint
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=BUCKET_NAME)
    else:
        s3.create_bucket(
            Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": REGION}
        )

    print(f"✓ 成功建立 bucket: {BUCKET_NAME}")
    print(f"\n現在可以將這個名稱填入 upload2AWS.py 的 BUCKET_NAME")

except ClientError as e:
    error_code = e.response["Error"]["Code"]
    if error_code == "BucketAlreadyExists":
        print(f"❌ 錯誤: Bucket 名稱 '{BUCKET_NAME}' 已被其他人使用")
        print("請換一個唯一的名稱")
    elif error_code == "BucketAlreadyOwnedByYou":
        print(f"ℹ Bucket '{BUCKET_NAME}' 已存在於您的帳戶中")
    else:
        print(f"❌ 建立失敗: {e}")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
