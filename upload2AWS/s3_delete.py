import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3", region_name="ap-southeast-2")

# 要刪除的 Bucket 名稱
BUCKET_NAME = "chang-pdms-testdata-2024"  # 請修改為要刪除的 bucket 名稱


def delete_all_objects(bucket_name):
    """刪除 bucket 中的所有物件(包含版本)"""
    try:
        print(f"正在列出 {bucket_name} 中的所有物件...")

        # 刪除所有物件
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name)

        delete_count = 0
        for page in pages:
            if "Contents" in page:
                objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                if objects:
                    s3.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
                    delete_count += len(objects)
                    print(f"  已刪除 {len(objects)} 個物件...")

        # 刪除所有版本(如果啟用了版本控制)
        try:
            paginator = s3.get_paginator("list_object_versions")
            pages = paginator.paginate(Bucket=bucket_name)

            for page in pages:
                versions = []
                if "Versions" in page:
                    versions.extend(
                        [
                            {"Key": v["Key"], "VersionId": v["VersionId"]}
                            for v in page["Versions"]
                        ]
                    )
                if "DeleteMarkers" in page:
                    versions.extend(
                        [
                            {"Key": d["Key"], "VersionId": d["VersionId"]}
                            for d in page["DeleteMarkers"]
                        ]
                    )

                if versions:
                    s3.delete_objects(Bucket=bucket_name, Delete={"Objects": versions})
                    delete_count += len(versions)
                    print(f"  已刪除 {len(versions)} 個版本...")
        except:
            pass  # 如果沒有版本控制就跳過

        print(f"✓ 總共刪除了 {delete_count} 個物件")
        return True

    except ClientError as e:
        print(f"❌ 刪除物件失敗: {e}")
        return False


def delete_bucket(bucket_name):
    """刪除指定的 S3 bucket"""
    try:
        # 確認是否要刪除
        print(f"\n⚠️  警告: 即將刪除 Bucket '{bucket_name}'")
        print("這個操作無法復原!")

        confirm = input("確定要刪除嗎? 請輸入 'YES' 確認: ")

        if confirm != "YES":
            print("❌ 取消刪除操作")
            return

        # 先刪除所有物件
        print(f"\n步驟 1: 刪除 Bucket 中的所有物件...")
        if not delete_all_objects(bucket_name):
            return

        # 刪除 bucket
        print(f"\n步驟 2: 刪除 Bucket '{bucket_name}'...")
        s3.delete_bucket(Bucket=bucket_name)

        print(f"✓ 成功刪除 Bucket: {bucket_name}")

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            print(f"❌ 錯誤: Bucket '{bucket_name}' 不存在")
        elif error_code == "BucketNotEmpty":
            print(f"❌ 錯誤: Bucket 不是空的,請先刪除所有物件")
        else:
            print(f"❌ 刪除失敗: {e}")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")


def list_buckets():
    """列出所有 buckets"""
    try:
        response = s3.list_buckets()
        print("\n您目前的 S3 Buckets:")
        print("=" * 60)
        if response["Buckets"]:
            for i, bucket in enumerate(response["Buckets"], 1):
                print(f"{i}. {bucket['Name']} (建立於 {bucket['CreationDate']})")
        else:
            print("(沒有 bucket)")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 列出 buckets 失敗: {e}")


if __name__ == "__main__":
    print("=== AWS S3 Bucket 刪除工具 ===\n")

    # 先列出所有 buckets
    list_buckets()

    # 刪除指定的 bucket
    delete_bucket(BUCKET_NAME)

    # 再次列出確認
    print("\n刪除後的 Bucket 列表:")
    list_buckets()
