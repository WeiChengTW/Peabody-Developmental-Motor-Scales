import cv2


for index in range(10):
    # 嘗試開啟相機
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        # 嘗試讀取一個畫面來確認真的可以用
        ret, frame = cap.read()
        
        if ret:
            height, width, _ = frame.shape
            print(f"[V] 相機 {index}: 偵測成功！ (解析度: {width}x{height})")
        else:
            print(f"[!] 相機 {index}: 能夠開啟連接，但讀不到畫面 (可能被其他程式占用)")
        
        # 記得釋放資源
        cap.release()
    else:
        print(f"[X] 相機 {index}: 未偵測到")

print("\n檢測結束。")