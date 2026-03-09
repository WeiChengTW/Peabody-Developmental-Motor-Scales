import cv2
import numpy as np
from PIL import Image

# A4 size in mm: 210 x 297
# A5 size in mm: 148 x 210
# Convert mm to pixels (assuming 300 DPI)
DPI = 300
MM_PER_INCH = 25.4


def mm2px(mm):
    return int((mm / MM_PER_INCH) * DPI)


# A4 size in pixels
a4_width_px = mm2px(210)
a4_height_px = mm2px(297)

# A5 size in pixels
a5_width_px = mm2px(148)
a5_height_px = mm2px(210)

# A5 區域在 A4 上的偏移（置中）
a5_offset_x = (a4_width_px - a5_width_px) // 2
a5_offset_y = (a4_height_px - a5_height_px) // 2

# 在 A5 區域內放置 2 個圓環/ArUco（上下分布）
margin_from_edge_mm = 30  # 距離 A5 邊緣 3 cm
margin_px = mm2px(margin_from_edge_mm)

# 計算 A5 區域內的圓心位置（相對於 A4）
centers = [
    (a5_offset_x + a5_width_px // 2, a5_offset_y + margin_px),  # A5 上方中央
    (
        a5_offset_x + a5_width_px // 2,
        a5_offset_y + a5_height_px - margin_px,
    ),  # A5 下方中央
]

# 圓環參數
outer_radius_mm = 40  # 4 cm
border_width_mm = 6  # 0.6 cm
inner_radius_mm = outer_radius_mm - border_width_mm  # 3.4 cm
outer_radius_px = mm2px(outer_radius_mm)
inner_radius_px = mm2px(inner_radius_mm)

# ArUco marker parameters
aruco_size_mm = 35  # 3.5 cm（較大，因為背面沒有圓環）
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

print("=== 生成 A4 雙面頁面（A5 區域置中）===")
print(f"A4 尺寸: {a4_width_px} x {a4_height_px} px ({DPI} DPI)")
print(f"A5 尺寸: {a5_width_px} x {a5_height_px} px")
print(f"A5 偏移: ({a5_offset_x}, {a5_offset_y})")
print(f"外圓半徑: {outer_radius_px} px ({outer_radius_mm} mm)")
print(f"內圓半徑: {inner_radius_px} px ({inner_radius_mm} mm)")
print(f"ArUco 尺寸: {aruco_size_px} px ({aruco_size_mm} mm)")
print()

# ==================== 第一份 ====================
print("--- 生成第一份 ---")

# 正面：只有圓環（無 ArUco），A5 區域置中
front_page_1 = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# 繪製 A5 區域邊框（虛線，用於參考，可選）
# cv2.rectangle(front_page_1,
#               (a5_offset_x, a5_offset_y),
#               (a5_offset_x + a5_width_px, a5_offset_y + a5_height_px),
#               (200, 200, 200), 2)

for idx, (center_x, center_y) in enumerate(centers):
    # 畫黑色外圓
    cv2.circle(
        front_page_1, (center_x, center_y), outer_radius_px, (0, 0, 0), thickness=-1
    )
    # 畫白色內圓
    cv2.circle(
        front_page_1,
        (center_x, center_y),
        inner_radius_px,
        (255, 255, 255),
        thickness=-1,
    )
    print(
        f"圓環 {idx+1}: 圓心 ({center_x}, {center_y}), 外圓半徑 {outer_radius_px} px, 內圓半徑 {inner_radius_px} px"
    )

# 背面：只有 ArUco（無圓環），A5 區域置中
back_page_1 = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

for idx, (center_x, center_y) in enumerate(centers):
    # 產生 ArUco marker（ID 都是 0）
    aruco_marker = cv2.aruco.generateImageMarker(aruco_dict, 0, aruco_size_px)
    # 貼上 ArUco marker
    start_x = center_x - aruco_size_px // 2
    start_y = center_y - aruco_size_px // 2
    marker_bgr = cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2BGR)
    # 防止 marker 超出邊界
    end_x = min(start_x + aruco_size_px, back_page_1.shape[1])
    end_y = min(start_y + aruco_size_px, back_page_1.shape[0])
    marker_w = end_x - start_x
    marker_h = end_y - start_y
    back_page_1[start_y:end_y, start_x:end_x] = marker_bgr[:marker_h, :marker_w]

    print(f"ArUco ID 0: 位置 {idx+1} ({center_x}, {center_y})")

print()

# 儲存圖片
cv2.imwrite("a4_front_page1.png", front_page_1)
cv2.imwrite("a4_back_page1.png", back_page_1)

print("=== 圖片已儲存 ===")
print("✓ a4_front_page1.png (正面 - A5區域置中，2個黑色圓環)")
print("✓ a4_back_page1.png (背面 - A5區域置中，2個ArUco ID 0)")
print()

# 建立 PDF（需要 PIL/Pillow）
try:
    from PIL import Image

    # 將 OpenCV BGR 轉換為 PIL RGB
    front1_pil = Image.fromarray(cv2.cvtColor(front_page_1, cv2.COLOR_BGR2RGB))
    back1_pil = Image.fromarray(cv2.cvtColor(back_page_1, cv2.COLOR_BGR2RGB))

    # 建立 PDF（頁面順序：正面, 背面）
    front1_pil.save(
        "a4_double_sided.pdf",
        save_all=True,
        append_images=[back1_pil],
        resolution=DPI,
    )

    print("✓ a4_double_sided.pdf (2頁PDF: 正面, 背面)")
    print()
    print("💡 列印提示：")
    print("   使用雙面列印功能，或：")
    print("   1. 先列印第 1 頁（正面 - 圓環）")
    print("   2. 將紙張翻面，再列印第 2 頁（背面 - ArUco）")
    print()
    print("📐 設計說明：")
    print(f"   - A4 紙張尺寸：210 x 297 mm")
    print(f"   - A5 有效區域：148 x 210 mm (置中於 A4)")
    print(f"   - 2 個圓環/ArUco 位於 A5 區域內上下分布")
    print(f"   - 正面：黑色圓環（外徑 {outer_radius_mm}mm，內徑 {inner_radius_mm}mm）")
    print(f"   - 背面：ArUco 標記 ID 0（尺寸 {aruco_size_mm}mm）")

except ImportError:
    print("⚠️ 未安裝 Pillow，無法生成 PDF")
    print("   請執行: pip install Pillow")

print()
print("✅ 完成！")
