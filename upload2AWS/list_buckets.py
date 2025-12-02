import boto3
from botocore.exceptions import NoCredentialsError, ClientError

try:
    # 建立 S3 客戶端
    s3 = boto3.client("s3")

    # 列出所有 buckets
    response = s3.list_buckets()

    print("您的 S3 Buckets:")
    print("=" * 60)
    for bucket in response["Buckets"]:
        print(f"名稱: {bucket['Name']}")
        print(f"建立時間: {bucket['CreationDate']}")
        print("-" * 60)

except NoCredentialsError:
    print("❌ 錯誤: 找不到 AWS 憑證")
    print("請先設定 AWS 憑證:")
    print("1. 在 AWS Console 中取得 Access Key")
    print("2. 設定環境變數或建立憑證檔案")

except ClientError as e:
    print(f"❌ AWS 錯誤: {e}")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
