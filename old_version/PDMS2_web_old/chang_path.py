import os
import shutil
import sys
import json
from pathlib import Path


def move_photos_to_uid_folder(uid):
    """
    將 kid/ 目錄下的照片移動到 kid/{uid}/ 資料夾內
    """
    try:
        # 設定路徑
        root_dir = Path(__file__).parent.resolve()
        kid_dir = root_dir / "kid"
        uid_dir = kid_dir / uid

        # 確保 uid 資料夾存在
        uid_dir.mkdir(parents=True, exist_ok=True)

        # 支援的圖片格式
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        moved_files = []

        # 搜尋 kid/ 目錄下的所有圖片檔案（不包含子資料夾）
        for file_path in kid_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # 目標路徑
                target_path = uid_dir / file_path.name

                # 如果目標檔案已存在，可以選擇覆蓋或跳過
                if target_path.exists():
                    print(
                        f"警告：檔案 {file_path.name} 已存在於 {uid} 資料夾中，跳過移動"
                    )
                    continue

                # 移動檔案
                shutil.move(str(file_path), str(target_path))
                moved_files.append(file_path.name)
                print(f"已移動：{file_path.name} -> kid/{uid}/{file_path.name}")

        if moved_files:
            print(f"成功移動 {len(moved_files)} 個檔案到 kid/{uid}/ 資料夾")
            return {
                "success": True,
                "moved_files": moved_files,
                "count": len(moved_files),
            }
        else:
            print("沒有找到需要移動的圖片檔案")
            return {
                "success": True,
                "moved_files": [],
                "count": 0,
                "message": "沒有找到需要移動的圖片檔案",
            }

    except Exception as e:
        error_msg = f"移動檔案時發生錯誤：{str(e)}"
        print(error_msg)
        return {"success": False, "error": error_msg}


def move_specific_photos(uid, photo_names):
    """
    移動指定的照片到 uid 資料夾
    """
    try:
        root_dir = Path(__file__).parent.resolve()
        kid_dir = root_dir / "kid"
        uid_dir = kid_dir / uid

        # 確保 uid 資料夾存在
        uid_dir.mkdir(parents=True, exist_ok=True)

        moved_files = []
        not_found_files = []

        for photo_name in photo_names:
            source_path = kid_dir / photo_name
            target_path = uid_dir / photo_name

            if source_path.exists() and source_path.is_file():
                if target_path.exists():
                    print(f"警告：檔案 {photo_name} 已存在於 {uid} 資料夾中，跳過移動")
                    continue

                shutil.move(str(source_path), str(target_path))
                moved_files.append(photo_name)
                print(f"已移動：{photo_name} -> kid/{uid}/{photo_name}")
            else:
                not_found_files.append(photo_name)
                print(f"找不到檔案：{photo_name}")

        return {
            "success": True,
            "moved_files": moved_files,
            "not_found_files": not_found_files,
            "moved_count": len(moved_files),
        }

    except Exception as e:
        error_msg = f"移動指定檔案時發生錯誤：{str(e)}"
        print(error_msg)
        return {"success": False, "error": error_msg}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  python chang_path.py <uid>                    # 移動所有圖片")
        print("  python chang_path.py <uid> <photo1> <photo2>  # 移動指定圖片")
        sys.exit(1)

    uid = sys.argv[1].strip()

    if not uid:
        print("錯誤：UID 不能為空")
        sys.exit(1)

    # 檢查UID是否包含無效字符
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(char in uid for char in invalid_chars):
        print("錯誤：UID 包含無效字符")
        sys.exit(1)

    if len(sys.argv) > 2:
        # 移動指定的照片
        photo_names = sys.argv[2:]
        result = move_specific_photos(uid, photo_names)
    else:
        # 移動所有照片
        result = move_photos_to_uid_folder(uid)

    # 輸出 JSON 結果（供前端解析）
    print("=" * 50)
    print("RESULT_JSON:", json.dumps(result, ensure_ascii=False))
