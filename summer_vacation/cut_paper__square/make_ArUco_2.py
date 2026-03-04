import cv2
import numpy as np

# A4 size in mm: 210 x 297
# Convert mm to pixels (assuming 300 DPI)
DPI = 300
MM_PER_INCH = 25.4


def mm2px(mm):
    return int((mm / MM_PER_INCH) * DPI)


# A4 size in pixels
a4_width_px = mm2px(210)
a4_height_px = mm2px(297)

# Create white A4 canvas
canvas = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# Square parameters
square_size_mm = 80  # 8 cm
border_width_mm = 6  # 0.6 cm
square_size_px = mm2px(square_size_mm)
border_width_px = mm2px(border_width_mm)


# 六個點的中心座標（2x3 排列）
offset_x = mm2px(30)  # 水平距離邊緣 3cm
offset_y = mm2px(30)  # 垂直距離邊緣 3cm

# 計算垂直間距
vertical_spacing = (a4_height_px - 2 * offset_y - square_size_px) // 2

# 計算水平位置（增加左右間距，更靠近邊緣）
margin = mm2px(15)  # 只留 1.5cm 的邊距
left_x = margin + square_size_px // 2
right_x = a4_width_px - margin - square_size_px // 2

# 計算三個垂直位置（也稍微調整垂直間距）
top_margin = mm2px(20)  # 上下邊距 2cm
available_height = a4_height_px - 2 * top_margin - square_size_px
vertical_spacing = available_height // 2  # 平均分配剩餘空間

top_y = top_margin + square_size_px // 2
middle_y = top_y + vertical_spacing
bottom_y = middle_y + vertical_spacing

centers = [
    (left_x, top_y),  # 左上
    (right_x, top_y),  # 右上
    (left_x, middle_y),  # 左中
    (right_x, middle_y),  # 右中
    (left_x, bottom_y),  # 左下
    (right_x, bottom_y),  # 右下
]

# ArUco marker parameters
aruco_size_mm = 20  # 1 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

for i, (center_x, center_y) in enumerate(centers):
    # 外框座標
    top_left = (center_x - square_size_px // 2, center_y - square_size_px // 2)
    bottom_right = (center_x + square_size_px // 2, center_y + square_size_px // 2)
    # 畫黑色方形
    cv2.rectangle(canvas, top_left, bottom_right, (0, 0, 0), thickness=-1)
    # 畫內白色方形
    inner_top_left = (top_left[0] + border_width_px, top_left[1] + border_width_px)
    inner_bottom_right = (
        bottom_right[0] - border_width_px,
        bottom_right[1] - border_width_px,
    )
    cv2.rectangle(
        canvas, inner_top_left, inner_bottom_right, (255, 255, 255), thickness=-1
    )
    # 產生 ArUco marker
    aruco_marker = cv2.aruco.generateImageMarker(aruco_dict, 0, aruco_size_px)
    # 放置 ArUco marker 於方形中心
    start_x = center_x - aruco_size_px // 2
    start_y = center_y - aruco_size_px // 2
    canvas[start_y : start_y + aruco_size_px, start_x : start_x + aruco_size_px] = (
        cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2BGR)
    )


# Save result
cv2.imwrite("a4_aruco.png", canvas)
