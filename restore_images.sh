#!/bin/bash
# 還原所有 LFS 圖片（執行於專案根目錄）

DIRS=(
  "PDMS2_web/images"
)

for dir in "${DIRS[@]}"; do
  echo "=== 處理 $dir ==="
  for f in "$dir"/*.jpg "$dir"/*.png 2>/dev/null; do
    [ -f "$f" ] || continue
    # 判斷是否為 LFS 指標（檔案小於 200 bytes）
    size=$(wc -c < "$f")
    if [ "$size" -lt 200 ]; then
      echo "下載: $f"
      tmp=$(mktemp)
      git lfs smudge < "$f" > "$tmp" 2>/dev/null
      if [ $? -eq 0 ] && [ $(wc -c < "$tmp") -gt 200 ]; then
        mv "$tmp" "$f"
        echo "  OK: $(wc -c < "$f") bytes"
      else
        rm "$tmp"
        echo "  FAILED: $f"
      fi
    else
      echo "已是真實圖片: $f ($size bytes)"
    fi
  done
done

echo "完成！"
