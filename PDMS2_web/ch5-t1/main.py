# ch5-t1/main.py (修改為讀取影片檔案)

from ultralytics import YOLO
import cv2
import numpy as np
import time
import sys

# 初始化模型
model_path = 'ch5-t1/bean_model.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"錯誤：無法載入模型 {model_path}。請檢查路徑是否正確。錯誤訊息：{e}")
    sys.exit(-1)

CONF = 0.5
DIST_THRESHOLD = 3
CHECK_INTERVAL = 0.5
GAME_DURATION = 63

class RaisinsGameEngine:
    def __init__(self):
        self.previous_count = 0
        self.last_check_time = time.time()
        self.start_time = time.time()
        self.warning_flag = False
        self.total_count = 0
        self.current_score = 0
        self.game_over = False

    def calculate_score(self, total_count, warning_flag, elapsed):
        if elapsed <= 30:
            if total_count >= 10:
                return 1 if warning_flag else 2
            else:
                return 0
        else:
            if total_count >= 5:
                return 1
            else:
                return 0

    def process_frame(self, frame):
        if self.game_over:
            annotated = frame.copy()
            cv2.putText(annotated, "Game Over!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(annotated, f"Final Score: {self.current_score}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return annotated, {"status": "FINISHED", "score": self.current_score}

        results = model.predict(source=frame, conf=CONF, verbose=False)
        annotated = frame.copy()
        centers = []

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

        count = len(merged)
        current_time = time.time()
        
        if current_time - self.last_check_time >= CHECK_INTERVAL:
            if count > self.previous_count:
                added = count - self.previous_count
                if added > 1:
                    self.warning_flag = True
            if count > self.total_count:
                self.total_count = count
            self.previous_count = count
            self.last_check_time = current_time

        elapsed = current_time - self.start_time
        remaining = max(0, GAME_DURATION - int(elapsed))

        if self.total_count >= 10 or elapsed >= GAME_DURATION:
            self.game_over = True
            self.current_score = self.calculate_score(self.total_count, self.warning_flag, elapsed)
            status = {"status": "FINISHED", "score": self.current_score, "total_placed": self.total_count, "time": int(elapsed)}
            cv2.putText(annotated, "Game Over!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            return annotated, status

        cv2.putText(annotated, f'SoyBean count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f'Total placed: {self.total_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated, f'Time Left: {remaining}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        if self.warning_flag:
            cv2.putText(annotated, "Warning !", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        status = {"status": "RUNNING", "score": None, "total_placed": self.total_count, "time": int(elapsed)}
        return annotated, status


if __name__ == "__main__":
    
    # ==========================================================
    # === 修改點：從命令列參數讀取影片路徑 ===
    # ==========================================================
    VIDEO_PATH = None
    
    # app.py 會傳入 3 個參數: script_name, uid, video_path
    if len(sys.argv) >= 3:
        try:
            # sys.argv[0] 是腳本名稱 ('ch5-t1/main.py')
            # sys.argv[1] 是 uid 
            # sys.argv[2] 是影片路徑
            VIDEO_PATH = sys.argv[2]
            print(f"從 app.py 接收到影片路徑: {VIDEO_PATH}")
        except Exception as e:
            print(f"錯誤：無法解析參數: {e}")
            sys.exit(-1)
    else:
        print(f"錯誤：缺少影片路徑參數")
        sys.exit(-1)
    
    WINDOW_NAME = "Ch5-t1 Raisins Game"
    DISPLAY_WIDTH = 1280

    # 開啟影片檔案（而不是相機）
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {VIDEO_PATH}")
        sys.exit(-1)

    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # 檢查是否成功讀取到寬高
    if original_width == 0 or original_height == 0:
        print(f"警告：無法讀取影片的原始解析度。")
        aspect_ratio = 9.0 / 16.0 
    else:
        aspect_ratio = original_height / original_width
        
    DISPLAY_HEIGHT = int(DISPLAY_WIDTH * aspect_ratio)
    
    game_engine = RaisinsGameEngine()
    
    print("遊戲開始... 按 'q' 鍵結束。")
    
    final_score = 0 

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    try:
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow(WINDOW_NAME, frame_resized)
            
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(100)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("影片已播放完畢或讀取失敗，結束中...")
                # 影片結束也當作遊戲結束
                if not game_engine.game_over:
                    game_engine.game_over = True
                    game_engine.current_score = game_engine.calculate_score(
                        game_engine.total_count, 
                        game_engine.warning_flag, 
                        time.time() - game_engine.start_time
                    )
                    final_score = game_engine.current_score
                break
                
            annotated_frame, status = game_engine.process_frame(frame)
            
            annotated_frame_resized = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow(WINDOW_NAME, annotated_frame_resized)
            
            if status.get("status") == "FINISHED":
                final_score = status.get("score")
                print(f"遊戲結束！ 總豆數: {status.get('total_placed')}, 最終分數: {final_score}")
                cv2.waitKey(3000)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"遊戲執行時發生錯誤: {e}")
        final_score = -1 
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"程式結束。最終分數: {final_score}")
    sys.exit(final_score)