# ch5-t1/main.py (修改為讀取相機串流、實時分析與內建錄影)
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

# 初始化模型
model_path = Path(__file__).parent / 'bean_model.pt'
try:
    model = YOLO(str(model_path))
except Exception as e:
    print(f"錯誤：無法載入模型 {model_path}。請檢查路徑是否正確。錯誤訊息：{e}")
    sys.exit(-1)

CONF = 0.45
DIST_THRESHOLD = 10
CHECK_INTERVAL = 0.5
GAME_DURATION = 60 # 總遊戲時間 60 秒
SIDE = 1
# 影片設定
OUTPUT_FPS = 30
OUTPUT_WIDTH = 1280 # 【修正】從 1920 降到 1280
OUTPUT_HEIGHT = 720 # 【修正】從 1080 降到 720
# 使用 H.264 編碼器（Windows 上常見的 FOURCC）
FOURCC = cv2.VideoWriter_fourcc(*'mp4v') # mp4v 是一個常見的 MP4 編碼器，或嘗試 'XVID', 'MJPG'

class RaisinsGameEngine:
    def __init__(self):
        self.previous_count = 0
        self.warning_flag = False
        self.total_count = 0
        self.current_score = 0
        self.game_over = False
        # 【新增】追蹤連續快速新增的幀數 (3 幀寬鬆度)
        self.multi_add_frames = 0
        self.MULTI_ADD_THRESHOLD = 3 

    def calculate_score(self, total_count, warning_flag, elapsed):
        # 這裡的 elapsed 是遊戲的實際流逝時間
        print(f"Final elapsed time: {elapsed:.2f}s") 
        
        # 邏輯不變
        if elapsed <= 30:
            if total_count >= 10:
                return 1 if warning_flag else 2
            elif total_count >= 5:
                return 1
        else:
            if total_count >= 5:
                return 1
            else:
                return 0

    # 接收當前遊戲流逝時間作為參數
    def process_frame(self, frame, current_game_elapsed):
        # 1. 遊戲結束檢查
        if self.game_over:
            annotated = frame.copy()
            cv2.putText(annotated, "Game Over!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(annotated, f"Final Score: {self.current_score}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated, f'Time Elapsed: {int(current_game_elapsed)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            return annotated, {"status": "FINISHED", "score": self.current_score}

        results = model.predict(source=frame, conf=CONF, verbose=False)
        annotated = frame.copy()
        centers = []

        # ... (省略 masks, merged, circles 繪製等計算邏輯) ...
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                ys, xs = np.where(mask > CONF)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    centers.append((cx, cy))

        merged = []
        used = set()
        for i, (x1, y1) in enumerate(centers):
            if i in used:
                continue
            group = [(x1, y1)]
            for j, (x2, y2) in enumerate(centers):
                if i != j and j not in used:
                    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if dist < DIST_THRESHOLD:
                        group.append((x2, y2))
                        used.add(j)
            merged.append(np.mean(group, axis=0))

        for (cx, cy) in merged:
            cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        # ... (計算邏輯結束) ...

        count = len(merged)
        
        # 【修正】直接基於幀間差異更新計數和警告 (改為 3 幀寬鬆度)
        if count > self.previous_count:
            added = count - self.previous_count
            if added > 1:
                self.multi_add_frames += 1
                if self.multi_add_frames >= self.MULTI_ADD_THRESHOLD:
                    self.warning_flag = True
            else:
                self.multi_add_frames = 0 # 小於等於 1 個新增，重設計數
        else:
            self.multi_add_frames = 0 # 數量減少或不變，重設計數
        
        # 更新最大豆子數
        if count > self.total_count:
            self.total_count = count
            
        self.previous_count = count
        
        elapsed = current_game_elapsed 
        
        # 2. 遊戲結束判斷
        if self.total_count >= 10 or elapsed >= GAME_DURATION:
            self.game_over = True
            self.current_score = self.calculate_score(self.total_count, self.warning_flag, elapsed)
            status = {"status": "FINISHED", "score": self.current_score, "total_placed": self.total_count, "time": int(elapsed)}
            cv2.putText(annotated, "Game Over!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            return annotated, status

        # 3. 顯示資訊
        cv2.putText(annotated, f'Time Elapsed: {int(elapsed)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(annotated, f'SoyBean count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f'Total placed: {self.total_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if self.warning_flag:
            cv2.putText(annotated, "Warning !", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        status = {"status": "RUNNING", "score": None, "total_placed": self.total_count, "time": int(elapsed)}
        return annotated, status


if __name__ == "__main__":
    
    # === [修正點] 參數讀取：接收 UID 和 相機索引 (SIDE=2) ===
    UID = None
    CAMERA_INDEX = SIDE
    
    if len(sys.argv) >= 3:
        try:
            UID = sys.argv[1]
            CAMERA_INDEX = int(sys.argv[2])
            print(f"從 app.py 接收到 UID: {UID}, 相機索引: {CAMERA_INDEX}")
        except Exception as e:
            print(f"錯誤：無法解析參數: {e}")
            sys.exit(-1)
    else:
        print(f"錯誤：缺少 UID 和相機索引參數")
        sys.exit(-1)
        
    WINDOW_NAME = "Ch5-t1 Raisins Game"
    DISPLAY_WIDTH = 1280

    # 建立輸出路徑
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "kid" / UID
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_PATH = OUTPUT_DIR / "Ch5-t1_result.mp4"
    
    # === [修正點] 開啟相機 (使用 MSMF 後端並設置合理解析度/FPS) ===
    cap = cv2.VideoCapture(CAMERA_INDEX + cv2.CAP_MSMF) 
    
    if not cap.isOpened():
        print(f"警告：MSMF 無法開啟相機 {CAMERA_INDEX}，嘗試預設後端。")
        cap = cv2.VideoCapture(CAMERA_INDEX) # 嘗試預設後端
        if not cap.isOpened():
            print(f"錯誤：無法開啟指定的相機索引 {CAMERA_INDEX}")
            sys.exit(-1)
    
    # 設置相機參數
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, OUTPUT_FPS)
    
    # 實際寬高
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if actual_width != OUTPUT_WIDTH or actual_height != OUTPUT_HEIGHT:
        print(f"警告：實際相機解析度 {actual_width}x{actual_height} 與預期 {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} 不符。")

    # === [修正點] 建立 VideoWriter (內建錄影) ===
    # 輸出路徑、FOURCC、FPS、尺寸
    out = cv2.VideoWriter(
        str(VIDEO_PATH), 
        FOURCC, 
        OUTPUT_FPS, 
        (actual_width, actual_height) # 使用實際的相機解析度
    )
    
    if not out.isOpened():
        print(f"錯誤：無法建立影片寫入器 ({VIDEO_PATH})。請檢查 FFmpeg/編碼器支援 (FourCC: {FOURCC})。")
        cap.release()
        sys.exit(-1)
        
    print(f"錄影已開始: {VIDEO_PATH}")

    aspect_ratio = actual_height / actual_width
    DISPLAY_HEIGHT = int(DISPLAY_WIDTH * aspect_ratio)
    
    game_engine = RaisinsGameEngine()
    
    print("遊戲開始... 按 'q' 鍵結束。")
    
    final_score = 0 
    
    # === [修正點] 實時計時器初始化 ===
    start_time = time.time()
    final_elapsed_time = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    try:
        # 遊戲迴圈
        while True:
            ret, frame = cap.read()
            if not ret:
                print("相機畫面讀取失敗，結束中...")
                break
                
            # === [修正點] 計算當前遊戲流逝時間 (實時時間) ===
            current_game_elapsed = time.time() - start_time
            
            # 將流逝時間傳入 process_frame
            annotated_frame, status = game_engine.process_frame(frame, current_game_elapsed)
            
            # 寫入影片 (寫入的是未縮放的原始幀)
            out.write(annotated_frame)
            
            # 顯示到視窗 (顯示的是縮放後的幀)
            annotated_frame_resized = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow(WINDOW_NAME, annotated_frame_resized)
            
            # 【修正】將 waitKey 降到 1ms 以保持流暢
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1) 
            key = cv2.waitKey(1) & 0xFF 
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0) 

            # 遊戲提前結束 (總豆數 >= 10 或時間到 60 秒)
            if status.get("status") == "FINISHED":
                final_score = status.get("score")
                final_elapsed_time = current_game_elapsed
                print(f"遊戲結束！ 總豆數: {status.get('total_placed')}, 最終分數: {final_score}, 最終時間: {final_elapsed_time:.2f}s")
                cv2.waitKey(3000) # 顯示最終畫面 3 秒
                break

            # 允許手動退出
            if key == ord('q'):
                # 手動退出時也計算分數
                if not game_engine.game_over:
                    final_score = game_engine.calculate_score(
                        game_engine.total_count, 
                        game_engine.warning_flag, 
                        current_game_elapsed
                    )
                    final_elapsed_time = current_game_elapsed
                break
                
    except Exception as e:
        print(f"遊戲執行時發生錯誤: {e}")
        final_score = -1 
        final_elapsed_time = -1 
    finally:
        # 釋放資源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # 輸出分數
    if final_score > 2:
        final_score = 0  # 確保分數不超過 2 分
        
    print(f"程式結束。最終分數: {final_score}, 最終時間: {final_elapsed_time:.2f}s")
    sys.exit(final_score)