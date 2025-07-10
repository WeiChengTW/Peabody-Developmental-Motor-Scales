import cv2


def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print("HSV:", hsv[y, x])


img = cv2.imread("1.jpg")
cv2.imshow("image", img)
cv2.setMouseCallback("image", show_hsv, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
