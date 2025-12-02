import boto3
import json

s3 = boto3.client("s3", region_name="ap-southeast-2")
BUCKET_NAME = "chang-pdms-testdata-2024"

try:
    # 1. 移除公開存取封鎖
    print("正在設定 Bucket 公開存取...")
    s3.put_public_access_block(
        Bucket=BUCKET_NAME,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        },
    )

    # 2. 設定 Bucket Policy 讓所有檔案公開可讀
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{BUCKET_NAME}/*",
            }
        ],
    }

    s3.put_bucket_policy(Bucket=BUCKET_NAME, Policy=json.dumps(bucket_policy))

    print(f"✓ Bucket '{BUCKET_NAME}' 已設定為公開讀取")
    print(f"\n圖片 URL 格式:")
    print(
        f"https://{BUCKET_NAME}.s3.ap-southeast-2.amazonaws.com/testforupload/檔案名稱"
    )

except Exception as e:
    print(f"❌ 設定失敗: {e}")
