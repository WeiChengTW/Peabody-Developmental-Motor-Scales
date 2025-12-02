import boto3
import os
from botocore.exceptions import ClientError

# ========== 全域設定 (只需要改這裡!) ==========
BUCKET_NAME = "cgu-pdms2"  # 修改您的 bucket 名稱 (必須小寫)
REGION = "ap-southeast-2"  # 修改您的區域
LOCAL_PATH = "testforupload"  # 要上傳的檔案或資料夾路徑
S3_PREFIX = ""  # S3 中的前綴路徑 (留空表示根目錄)
# =============================================


class S3Uploader:
    """AWS S3 上傳工具"""

    def __init__(self, bucket_name, region):
        self.bucket_name = bucket_name
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)

    def bucket_exists(self):
        """檢查 bucket 是否存在"""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            return True
        except:
            return False

    def upload_file(self, local_file, s3_key):
        """上傳單一檔案"""
        try:
            file_size = os.path.getsize(local_file)
            print(f"   📄 {os.path.basename(local_file)} ({file_size} bytes)")
            print(f"      → s3://{self.bucket_name}/{s3_key}")

            self.s3.upload_file(local_file, self.bucket_name, s3_key)
            print(f"      ✓ 上傳成功")
            return True

        except Exception as e:
            print(f"      ❌ 上傳失敗: {e}")
            return False

    def upload_folder(self, local_folder, s3_prefix=""):
        """上傳整個資料夾"""
        if not os.path.exists(local_folder):
            print(f"❌ 錯誤: 資料夾 '{local_folder}' 不存在")
            return False

        try:
            print(f"📂 正在掃描資料夾: {local_folder}")

            # 計算總檔案數
            total_files = sum(len(files) for _, _, files in os.walk(local_folder))
            print(f"   找到 {total_files} 個檔案\n")

            upload_count = 0
            fail_count = 0

            for root, dirs, files in os.walk(local_folder):
                for file in files:
                    # 本地檔案完整路徑
                    local_path = os.path.join(root, file)

                    # 計算相對路徑,用於 S3 的 Key
                    relative_path = os.path.relpath(local_path, local_folder)

                    # 組合 S3 Key
                    if s3_prefix:
                        s3_key = os.path.join(s3_prefix, relative_path).replace(
                            "\\", "/"
                        )
                    else:
                        s3_key = relative_path.replace("\\", "/")

                    if self.upload_file(local_path, s3_key):
                        upload_count += 1
                    else:
                        fail_count += 1

                    print()  # 空行分隔

            print("=" * 70)
            print(f"✅ 上傳完成: {upload_count} 成功, {fail_count} 失敗")
            return True

        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
            return False

    def upload_single_file(self, local_file, s3_key=None):
        """上傳單一檔案 (可指定 S3 路徑)"""
        if not os.path.exists(local_file):
            print(f"❌ 錯誤: 檔案 '{local_file}' 不存在")
            return False

        if not os.path.isfile(local_file):
            print(f"❌ 錯誤: '{local_file}' 不是檔案")
            return False

        # 如果沒有指定 S3 Key,使用檔案名稱
        if s3_key is None:
            s3_key = os.path.basename(local_file)

        return self.upload_file(local_file, s3_key)

    def get_public_url(self, s3_key):
        """取得檔案的公開 URL"""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

    def list_uploaded_files(self, prefix=""):
        """列出已上傳的檔案"""
        try:
            print("\n" + "=" * 70)
            print(f"📁 Bucket '{self.bucket_name}' 中的檔案:")
            print("=" * 70)

            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    size = obj["Size"]
                    modified = obj["LastModified"]
                    print(f"\n📄 {key}")
                    print(f"   大小: {size} bytes")
                    print(f"   更新: {modified}")
                    print(f"   🔗 {self.get_public_url(key)}")
            else:
                print("\n(沒有檔案)")

            print("\n" + "=" * 70)
            return True

        except Exception as e:
            print(f"❌ 列出檔案失敗: {e}")
            return False


def main():
    """主程式 - 上傳檔案到 S3"""
    print("=" * 70)
    print("📤 AWS S3 上傳工具")
    print("=" * 70)
    print(f"\n⚙️  設定資訊:")
    print(f"   Bucket 名稱: {BUCKET_NAME}")
    print(f"   區域: {REGION}")
    print(f"   本地路徑: {LOCAL_PATH}")
    print(f"   S3 前綴: {S3_PREFIX if S3_PREFIX else '(根目錄)'}")
    print()

    # 建立上傳器
    uploader = S3Uploader(BUCKET_NAME, REGION)

    # 檢查 Bucket 是否存在
    print("🔍 檢查 Bucket 是否存在...")
    if not uploader.bucket_exists():
        print(f"❌ 錯誤: Bucket '{BUCKET_NAME}' 不存在")
        print(f"   請先執行 s3_main.py 建立 Bucket")
        return

    print(f"✓ Bucket '{BUCKET_NAME}' 存在\n")

    # 檢查本地路徑是否存在
    if not os.path.exists(LOCAL_PATH):
        print(f"❌ 錯誤: 路徑 '{LOCAL_PATH}' 不存在")
        return

    # 開始上傳
    print("=" * 70)
    print("開始上傳...")
    print("=" * 70)
    print()

    if os.path.isfile(LOCAL_PATH):
        # 上傳單一檔案
        print("📄 上傳單一檔案\n")
        s3_key = (
            os.path.join(S3_PREFIX, os.path.basename(LOCAL_PATH)).replace("\\", "/")
            if S3_PREFIX
            else os.path.basename(LOCAL_PATH)
        )
        uploader.upload_single_file(LOCAL_PATH, s3_key)

    elif os.path.isdir(LOCAL_PATH):
        # 上傳整個資料夾
        print("📂 上傳整個資料夾\n")
        uploader.upload_folder(LOCAL_PATH, S3_PREFIX)

    # 列出已上傳的檔案
    uploader.list_uploaded_files(S3_PREFIX)

    # 完成
    print("\n✅ 上傳作業完成!")
    print("\n📝 在 HTML 中使用以下格式顯示:")
    print(f'   <img src="https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/檔案路徑">')
    print()


if __name__ == "__main__":
    main()
