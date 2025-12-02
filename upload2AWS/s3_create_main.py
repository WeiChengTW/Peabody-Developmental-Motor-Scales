import boto3
import json
import os
from botocore.exceptions import ClientError

# ========== 全域設定 (只需要改這裡!) ==========
BUCKET_NAME = "cgu-pdms2"  # 修改您的 bucket 名稱 (必須小寫)
REGION = "ap-southeast-2"  # 修改您的區域
# =============================================


class S3Manager:
    """AWS S3 管理工具"""

    def __init__(self, bucket_name, region):
        self.bucket_name = bucket_name
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)

    def list_buckets(self):
        """列出所有 buckets"""
        try:
            response = self.s3.list_buckets()
            print("\n📦 您目前的 S3 Buckets:")
            print("=" * 70)
            if response["Buckets"]:
                for i, bucket in enumerate(response["Buckets"], 1):
                    print(f"{i}. {bucket['Name']}")
                    print(f"   建立時間: {bucket['CreationDate']}")
            else:
                print("(沒有 bucket)")
            print("=" * 70)
            return response["Buckets"]
        except Exception as e:
            print(f"❌ 列出 buckets 失敗: {e}")
            return []

    def bucket_exists(self):
        """檢查 bucket 是否存在"""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            return True
        except:
            return False

    def create_bucket(self):
        """建立 bucket"""
        try:
            if self.bucket_exists():
                print(f"ℹ️  Bucket '{self.bucket_name}' 已經存在")
                return True

            print(f"\n🔨 正在建立 Bucket: {self.bucket_name}")
            print(f"   區域: {self.region}")

            if self.region == "us-east-1":
                self.s3.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )

            print(f"✓ 成功建立 Bucket: {self.bucket_name}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "BucketAlreadyExists":
                print(f"❌ 錯誤: Bucket 名稱 '{self.bucket_name}' 已被其他人使用")
                print("   請修改 BUCKET_NAME 為其他唯一名稱")
            elif error_code == "BucketAlreadyOwnedByYou":
                print(f"ℹ️  Bucket '{self.bucket_name}' 已存在於您的帳戶中")
                return True
            else:
                print(f"❌ 建立失敗: {e}")
            return False

        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
            return False

    def make_bucket_public(self):
        """設定 bucket 為公開讀取"""
        try:
            print(f"\n🌐 正在設定 Bucket 公開存取...")

            # 1. 移除公開存取封鎖
            self.s3.put_public_access_block(
                Bucket=self.bucket_name,
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
                        "Resource": f"arn:aws:s3:::{self.bucket_name}/*",
                    }
                ],
            }

            self.s3.put_bucket_policy(
                Bucket=self.bucket_name, Policy=json.dumps(bucket_policy)
            )

            print(f"✓ Bucket '{self.bucket_name}' 已設定為公開讀取")
            return True

        except Exception as e:
            print(f"❌ 設定公開存取失敗: {e}")
            return False

    def upload_folder(self, local_folder, s3_prefix=None):
        """上傳整個資料夾到 S3"""
        if s3_prefix is None:
            s3_prefix = os.path.basename(local_folder)

        if not os.path.exists(local_folder):
            print(f"❌ 錯誤: 本地資料夾 '{local_folder}' 不存在")
            return False

        try:
            print(f"\n📤 正在上傳資料夾: {local_folder}")
            print(f"   目標: s3://{self.bucket_name}/{s3_prefix}/")

            upload_count = 0
            for root, dirs, files in os.walk(local_folder):
                for file in files:
                    # 本地檔案完整路徑
                    local_path = os.path.join(root, file)

                    # 計算相對路徑,用於 S3 的 Key
                    relative_path = os.path.relpath(local_path, local_folder)
                    s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                    print(f"   上傳: {file} -> {s3_key}")
                    self.s3.upload_file(local_path, self.bucket_name, s3_key)
                    upload_count += 1

            print(f"✓ 成功上傳 {upload_count} 個檔案")
            return True

        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
            return False

    def get_public_url(self, s3_key):
        """取得檔案的公開 URL"""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

    def list_files(self, prefix=""):
        """列出 bucket 中的檔案"""
        try:
            print(f"\n📁 Bucket '{self.bucket_name}' 中的檔案:")
            print("=" * 70)

            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    size = obj["Size"]
                    print(f"   {key} ({size} bytes)")
                    print(f"   🔗 {self.get_public_url(key)}")
                    print()
            else:
                print("   (沒有檔案)")

            print("=" * 70)
            return True

        except Exception as e:
            print(f"❌ 列出檔案失敗: {e}")
            return False


def main():
    """主程式 - 只負責建立和設定 Bucket"""
    print("=" * 70)
    print("🔨 AWS S3 Bucket 建立工具")
    print("=" * 70)
    print(f"\n⚙️  設定資訊:")
    print(f"   Bucket 名稱: {BUCKET_NAME}")
    print(f"   區域: {REGION}")
    print()

    # 建立 S3 管理器
    s3_manager = S3Manager(BUCKET_NAME, REGION)

    # 步驟 1: 列出現有的 buckets
    print("\n" + "=" * 70)
    print("步驟 1: 檢查現有 Buckets")
    print("=" * 70)
    s3_manager.list_buckets()

    # 步驟 2: 建立 bucket
    print("\n" + "=" * 70)
    print("步驟 2: 建立 Bucket")
    print("=" * 70)
    if not s3_manager.create_bucket():
        print("\n❌ 建立 Bucket 失敗,程式終止")
        return

    # 步驟 3: 設定公開存取
    print("\n" + "=" * 70)
    print("步驟 3: 設定公開存取")
    print("=" * 70)
    if not s3_manager.make_bucket_public():
        print("\n⚠️  警告: 設定公開存取失敗")
        return

    # 完成
    print("\n" + "=" * 70)
    print("✅ Bucket 建立完成!")
    print("=" * 70)
    print(f"\n📝 下一步:")
    print(f"   使用 upload_main.py 上傳檔案到這個 Bucket")
    print(f"\n🔗 公開 URL 格式:")
    print(f"   https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/檔案路徑")
    print()


if __name__ == "__main__":
    main()
