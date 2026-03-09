import cv2
import numpy as np
from PIL import Image

# A4 size in mm: 210 x 297
# Convert mm to pixels (assuming 300 DPI)
DPI = 300
MM_PER_INCH = 25.4


def mm2px(mm):
    return int((mm / MM_PER_INCH) * DPI)


# A4 size in pixels
a4_width_px = mm2px(210)
a4_height_px = mm2px(297)

# 線的粗細
line_thickness_mm = 3  # 3 mm
line_thickness_px = mm2px(line_thickness_mm)

# ArUco marker parameters
aruco_size_mm = 30  # 3 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 1/4 A4 的中心點
quarter_a4_x_left = a4_width_px // 4
quarter_a4_x_right = 3 * a4_width_px // 4
quarter_a4_y_upper = a4_height_px // 4
quarter_a4_y_lower = 3 * a4_height_px // 4

print("=== 生成 A4 雙面頁面 ===")
print(f"A4 尺寸: {a4_width_px} x {a4_height_px} px ({DPI} DPI)")
print(f"正面：中間垂直線，粗度 {line_thickness_mm} mm")
print(f"背面：4 個 ArUco 在各 1/4 A4 中心點")
print()

# ==================== 正面：中間垂直線 ====================
print("--- 生成正面（中間垂直線）---")
front_page = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# 計算中線位置
center_x = a4_width_px // 2

# 畫垂直中線（黑色）
line_x1 = center_x - line_thickness_px // 2
line_x2 = center_x + line_thickness_px // 2
cv2.rectangle(front_page, (line_x1, 0), (line_x2, a4_height_px), (0, 0, 0), -1)

print(f"正面：中線位置 x = {center_x} px，範圍 {line_x1} ~ {line_x2} px")
print()

# ==================== 背面：4 個 ArUco ====================
print("--- 生成背面（4 個 ArUco 在各 1/4 中心點）---")

# 建立白色 A4 畫布（背面）
back_page = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# ArUco marker parameters
aruco_size_mm = 30  # 3 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 1/4 A4 的中心點
quarter_a4_x_left = a4_width_px // 4
quarter_a4_x_right = 3 * a4_width_px // 4
quarter_a4_y_upper = a4_height_px // 4
quarter_a4_y_lower = 3 * a4_height_px // 4

# 4 個 ArUco 的位置
aruco_positions = [
    (quarter_a4_x_left, quarter_a4_y_upper, 0, "上左"),  # 上左 1/4 中心, ID 0
    (3 * a4_width_px // 4, a4_height_px // 4, 1),  # 上右 1/4 中心, ID 1
    (a4_width_px // 4, 3 * a4_height_px // 4, 0),  # 下左 1/4 中心, ID 0
    (3 * a4_width_px // 4, 3 * a4_height_px // 4, 1),  # 下右 1/4 中心, ID 1
]

# ArUco marker parameters
aruco_size_mm = 30  # 3 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

print("=== 生成 A4 雙面頁面 ===")
print(f"A4 尺寸: {a4_width_px} x {a4_height_px} px ({DPI} DPI)")
print(f"正面：中間垂直線（粗度 {line_thickness_mm} mm）")
print(f"背面：4 個 ArUco（各 1/4 A4 中心點）")
print()

# ==================== 正面：中間垂直線 ====================
print("--- 正面：中間垂直線 ---")
front_page = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# 計算中線位置
center_x = a4_width_px // 2

# 畫垂直中線（黑色）
line_x1 = center_x - line_thickness_px // 2
line_x2 = center_x + line_thickness_px // 2
cv2.rectangle(front_page, (line_x1, 0), (line_x2, a4_height_px), (0, 0, 0), -1)

print(f"正面 - 中線位置: x = {center_x} px")
print(f"中線範圍: x = {line_x1} ~ {line_x2} px")
print()

# ==================== 背面：4 個 ArUco ====================
print("--- 背面：4 個 ArUco 在各 1/4 A4 中心點 ---")

# 建立白色 A4 畫布（背面）
back_page = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# ArUco marker parameters
aruco_size_mm = 30  # 3 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 1/4 A4 的中心點
quarter_a4_x_left = a4_width_px // 4
quarter_a4_x_right = 3 * a4_width_px // 4
quarter_a4_y_upper = a4_height_px // 4
quarter_a4_y_lower = 3 * a4_height_px // 4

# 上半部分 ArUco 位置
upper_centers = [
    (quarter_a4_x_left, quarter_a4_y_upper),  # 左側 1/4 中心
    (quarter_a4_x_right, quarter_a4_y_upper),  # 右側 1/4 中心
]

# 下半部分 ArUco 位置
lower_centers = [
    (quarter_a4_x_left, quarter_a4_y_lower),  # 左側 1/4 中心
    (quarter_a4_x_right, quarter_a4_y_lower),  # 右側 1/4 中心
]

aruco_ids = [0, 1]  # 左邊 ID 0，右邊 ID 1
aruco_size_mm = 30  # 3 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

print("=== 生成 A4 雙面頁面 ===")
print(f"A4 尺寸: {a4_width_px} x {a4_height_px} px ({DPI} DPI)")
print(f"正面：中間垂直線（{line_thickness_mm} mm）")
print(f"背面：4個 ArUco 在各 1/4 中心點")
print()

# ==================== 正面：中間線 ====================
print("--- 正面 ---")
front_page = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# 計算中線位置
center_x = a4_width_px // 2
line_x1 = center_x - line_thickness_px // 2
line_x2 = center_x + line_thickness_px // 2
cv2.rectangle(front_page, (line_x1, 0), (line_x2, a4_height_px), (0, 0, 0), -1)

print(f"中線位置: x = {center_x} px")
print(f"中線範圍: x = {line_x1} ~ {line_x2} px")
print()

# ==================== 背面：4 個 ArUco ====================
print("--- 背面 ---")
back_page = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# 計算 1/4 位置
quarter_a4_x_left = a4_width_px // 4
quarter_a4_x_right = 3 * a4_width_px // 4
quarter_a4_y_upper = a4_height_px // 4
quarter_a4_y_lower = 3 * a4_height_px // 4

# 上半部分
for idx, (center_x, center_y) in enumerate(upper_centers):
    aruco_marker = cv2.aruco.generateImageMarker(
        aruco_dict, aruco_ids[idx], aruco_size_px
    )
    start_x = center_x - aruco_size_px // 2
    start_y = center_y - aruco_size_px // 2
    marker_bgr = cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2BGR)
    end_x = min(start_x + aruco_size_px, back_page.shape[1])
    end_y = min(start_y + aruco_size_px, back_page.shape[0])
    marker_w = end_x - start_x
    marker_h = end_y - start_y
    back_page[start_y:end_y, start_x:end_x] = marker_bgr[:marker_h, :marker_w]
    print(
        f"ArUco ID {aruco_ids[idx]} (上-{'左' if idx == 0 else '右'}): ({center_x}, {center_y})"
    )

# 下半部分
for idx, (center_x, center_y) in enumerate(lower_centers):
    aruco_marker = cv2.aruco.generateImageMarker(
        aruco_dict, aruco_ids[idx], aruco_size_px
    )
    start_x = center_x - aruco_size_px // 2
    start_y = center_y - aruco_size_px // 2
    marker_bgr = cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2BGR)
    end_x = min(start_x + aruco_size_px, back_page.shape[1])
    end_y = min(start_y + aruco_size_px, back_page.shape[0])
    marker_w = end_x - start_x
    marker_h = end_y - start_y
    back_page[start_y:end_y, start_x:end_x] = marker_bgr[:marker_h, :marker_w]
    print(
        f"ArUco ID {aruco_ids[idx]} (下-{'左' if idx == 0 else '右'}): ({center_x}, {center_y})"
    )

print()

# 儲存圖片
cv2.imwrite("a4_front_centerline.png", front_page)
cv2.imwrite("a4_back_aruco.png", back_page)

print("=== 圖片已儲存 ===")
print("✓ a4_front_centerline.png (正面 - 中間垂直線)")
print("✓ a4_back_aruco.png (背面 - 4個ArUco)")
print()

# 建立 PDF（需要 PIL/Pillow）
try:
    from PIL import Image

    # 將 OpenCV BGR 轉換為 PIL RGB
    front_pil = Image.fromarray(cv2.cvtColor(front_page, cv2.COLOR_BGR2RGB))
    back_pil = Image.fromarray(cv2.cvtColor(back_page, cv2.COLOR_BGR2RGB))

    # 建立 PDF（頁面順序：正面, 背面）
    front_pil.save(
        "a4_double_sided_line_aruco.pdf",
        save_all=True,
        append_images=[back_pil],
        resolution=DPI,
    )

    print("✓ a4_double_sided_line_aruco.pdf (2頁PDF: 正面線條, 背面ArUco)")
    print()
    print("💡 列印提示：")
    print("   使用雙面列印功能，或：")
    print("   1. 先列印第 1 頁（正面 - 中間線）")
    print("   2. 將紙張翻面，再列印第 2 頁（背面 - ArUco）")
    print()
    print("📐 設計說明：")
    print(f"   - A4 紙張尺寸：210 x 297 mm")
    print(f"   - 正面：垂直中線（寬度 {line_thickness_mm} mm）")
    print(f"   - 背面：4 個 ArUco 標記")
    print(f"     * 上左 1/4 中心: ID 0")
    print(f"     * 上右 1/4 中心: ID 1")
    print(f"     * 下左 1/4 中心: ID 0")
    print(f"     * 下右 1/4 中心: ID 1")

except ImportError:
    print("⚠️ 未安裝 Pillow，無法生成 PDF")
    print("   請執行: pip install Pillow")

print()
print("✅ 完成！")
