import cv2
import numpy as np


def get_cm_per_pixel(image_path: str, cm_length: float = 16, show: bool = False):
    """
    計算圖片中比例尺的像素與公分比例。
    :param image_path: 圖片路徑
    :param cm_length: 比例尺實際長度（公分）
    :param show: 是否顯示中間處理結果視窗
    :return: (cm_per_pixel, long_side, objects) 或 None
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        objects.append(
            {"rect": (x, y, w, h), "area": area, "aspect": aspect_ratio, "cnt": cnt}
        )

    objects = sorted(objects, key=lambda o: o["rect"][0])

    edges = cv2.Canny(blur, 5, 150)
    if show:
        cv2.imshow("edges", edges)

    contours_edges, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_edges = sorted(contours_edges, key=cv2.contourArea, reverse=True)[:2]
    colors = [(0, 0, 255), (0, 255, 0)]
    for i, cnt in enumerate(contours_edges):
        cv2.drawContours(img, [cnt], -1, colors[i], 3)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(
            img,
            f"EdgeObj{i+1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[i],
            2,
        )

    if len(objects) >= 2:
        x2, y2, w2, h2 = objects[1]["rect"]
        long_side = max(w2, h2)
        # pixel_per_cm = long_side / cm_length
        cm_per_pixel = cm_length / long_side  # 新增：一像素是幾公分
        if show:
            cv2.imshow("result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cm_per_pixel, long_side, objects
    else:
        if show:
            cv2.imshow("result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None


if __name__ == "__main__":
    # 範例用法
    result = get_cm_per_pixel("img15.png", cm_length=16, show=True)
    if result:
        cm_per_pixel, long_side, objects = result
        print(f"Obj2 長邊像素: {long_side}")
        # print(f"像素與公分比例: 1 cm = {1/cm_per_pixel:.2f} px")
        print(f"公分與像素比例: 1 px = {cm_per_pixel:.4f} cm")
    else:
        print("找不到足夠的物件來計算比例")
