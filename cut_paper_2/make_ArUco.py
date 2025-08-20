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

# Center of A4
center_x = a4_width_px // 2
center_y = a4_height_px // 2

# Outer square coordinates
top_left = (center_x - square_size_px // 2, center_y - square_size_px // 2)
bottom_right = (center_x + square_size_px // 2, center_y + square_size_px // 2)

# Draw outer square (black border, sharp corners, no missing corners)
# 先畫大黑色方形，再畫小白色方形於內部，產生實心外框
cv2.rectangle(canvas, top_left, bottom_right, (0, 0, 0), thickness=-1)
# 計算內方形座標
inner_top_left = (top_left[0] + border_width_px, top_left[1] + border_width_px)
inner_bottom_right = (
    bottom_right[0] - border_width_px,
    bottom_right[1] - border_width_px,
)
cv2.rectangle(canvas, inner_top_left, inner_bottom_right, (255, 255, 255), thickness=-1)

# ArUco marker parameters
aruco_size_mm = 30  # 1 cm
aruco_size_px = mm2px(aruco_size_mm)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_id = 0  # You can change the ID if needed

# Generate ArUco marker
aruco_marker = cv2.aruco.generateImageMarker(aruco_dict, aruco_id, aruco_size_px)

# Place ArUco marker at center
start_x = center_x - aruco_size_px // 2
start_y = center_y - aruco_size_px // 2
canvas[start_y : start_y + aruco_size_px, start_x : start_x + aruco_size_px] = (
    cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2BGR)
)

# 紀錄外框四邊座標
print("Outer black border coordinates:")
print(f"Top:    y = {top_left[1]}, from x = {top_left[0]} to x = {bottom_right[0]}")
print(f"Bottom: y = {bottom_right[1]}, from x = {top_left[0]} to x = {bottom_right[0]}")
print(f"Left:   x = {top_left[0]}, from y = {top_left[1]} to y = {bottom_right[1]}")
print(f"Right:  x = {bottom_right[0]}, from y = {top_left[1]} to y = {bottom_right[1]}")

# Save result
cv2.imwrite("a4_aruco.png", canvas)
