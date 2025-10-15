# Ch5-t1/main.py
# -*- coding: utf-8 -*-
from pathlib import Path
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"[RaisinScorer] 無法匯入 ultralytics：{e}")

# ========= 參數 =========
CONF            = 0.45          # 偵測信心門檻
DIST_THRESHOLD  = 2             # 中心點合併距離閾值（像素）
CHECK_INTERVAL  = 0.5           # 新增數量檢查的間隔秒數
GAME_DURATION   = 60            # 倒數秒數
TARGET_COUNT    = 10            # 達標數量

# ========= 規則 =========
def _calc_score(total_count: int, warning_flag: bool, elapsed_sec: float) -> int:
    """
    依原本規則計分；elapsed_sec 已被 clamp 在 [0, GAME_DURATION]。
    0~30 秒：>=10 顆 -> 2 分（若 warning_flag=true -> 1 分）；否則 0
    30+ 秒 ：>=5  顆 -> 1 分；否則 0
    """
    if elapsed_sec <= 30:
        if total_count >= 10:
            return 1 if warning_flag else 2
        return 0
    else:
        return 1 if total_count >= 5 else 0


class RaisinScorer:
    """
    process(frame) -> (score:int, done:bool, overlay:ndarray)
    finalize() -> int

    公開屬性： total_count, warning_flag
    """
    def __init__(self, uid=None):
        self.uid = uid

        # 計時 / 狀態
        self.start_time: float | None = None  # 第一幀到時才起錶
        self.end_time: float | None = None    # 第一次達標/時間到時記錄
        self.duration = GAME_DURATION
        self.done = False

        # 統計
        self.last_score = 0
        self.total_count = 0
        self.warning_flag = False
        self.previous_count = 0
        self.last_check_time: float | None = None

        # 模型
        self.model = None
        self.model_loaded = False
        self._load_model()

        print(f"[RaisinScorer] 任務就緒，等待第一幀開始計時，UID={uid}")

    # ---------- 模型載入 ----------
    def _load_model(self):
        try:
            if YOLO is None:
                print("[RaisinScorer] YOLO 套件不可用，將只顯示倒數與 HUD。")
                return
            model_path = Path(__file__).parent / "bean_model.pt"
            if not model_path.exists():
                print(f"[RaisinScorer] 找不到模型：{model_path}，將只顯示倒數與 HUD。")
                return
            self.model = YOLO(str(model_path))
            self.model_loaded = True
            print("[RaisinScorer] YOLO 模型載入完成")
        except Exception as e:
            self.model = None
            self.model_loaded = False
            print(f"[RaisinScorer] 模型載入失敗：{e}，將只顯示倒數與 HUD。")

    # ---------- HUD/字幕 ----------
    def _draw_hud(self, frame, count, total, remaining, warn=False):
        """
        在 frame 上疊半透明 HUD 與文字，回傳 overlay。
        確保任何情況都能回傳可見字幕。
        """
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # 半透明黑底條（上方）
        hud = overlay.copy()
        top_h = 140
        cv2.rectangle(hud, (0, 0), (w, top_h), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.35, overlay, 0.65, 0, overlay)

        # 左上資訊
        try:
            cv2.putText(overlay, f"UID: {self.uid or '-'}", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"SoyBean count: {int(count)}", (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"Total placed: {int(total)}", (20, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3, cv2.LINE_AA)
        except Exception:
            pass

        # 右上倒數
        try:
            text = f"Time Left: {int(max(0, remaining))}s"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(overlay, text, (w - tw - 20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3, cv2.LINE_AA)
        except Exception:
            pass

        # 警示
        if warn:
            try:
                cv2.putText(overlay, "HURRY UP!",
                            (w - 260, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3, cv2.LINE_AA)
            except Exception:
                pass

        return overlay

    # ---------- 偵測 ----------
    def _detect_centers(self, frame):
        """
        回傳 (centers, annotated)：
          centers: [(cx, cy), ...]
          annotated: 在原圖上畫出偵測到的標記（紅點），僅作輔助；真正的 HUD 另畫
        """
        annotated = frame.copy()
        if not self.model_loaded:
            return [], annotated

        try:
            # Ultralytics YOLO: 直接丟 ndarray，關閉冗長輸出
            res = self.model.predict(source=frame, conf=CONF, verbose=False)[0]
            centers = []

            # masks（語意/實例分割）
            if getattr(res, "masks", None) is not None and res.masks is not None:
                masks = res.masks.data
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()
                else:
                    masks = np.asarray(masks)
                for mk in masks:
                    ys, xs = np.where(mk > CONF)
                    if xs.size and ys.size:
                        centers.append((int(xs.mean()), int(ys.mean())))

            # boxes（偵測框）
            if getattr(res, "boxes", None) is not None and res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

            # 畫原始偵測中心（紅點）
            for (cx, cy) in centers:
                cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            return centers, annotated
        except Exception as e:
            print(f"[RaisinScorer] 偵測錯誤：{e}，本幀退回 HUD")
            return [], annotated

    def _merge_centers(self, centers):
        """
        把相近中心點合併，降低重複計數。
        """
        if not centers:
            return []
        merged = []
        used = set()
        for i, (x1, y1) in enumerate(centers):
            if i in used:
                continue
            group = [(x1, y1)]
            for j, (x2, y2) in enumerate(centers):
                if i != j and j not in used:
                    if np.hypot(x1 - x2, y1 - y2) < DIST_THRESHOLD:
                        group.append((x2, y2))
                        used.add(j)
            merged.append(np.mean(group, axis=0))
        return merged

    # ---------- 每幀處理 ----------
    def process(self, frame):
        # 起錶
        if self.start_time is None:
            self.start_time = time.time()
            self.last_check_time = self.start_time
            print("[RaisinScorer] 收到第一幀，開始計時")

        now = time.time()
        elapsed = now - self.start_time

        # 如果已經記錄過 end_time，就用 end_time 計算 remaining，確保倒數不跳動
        logical_elapsed = (self.end_time - self.start_time) if self.end_time else elapsed
        remaining = max(0, int(self.duration - logical_elapsed))

        # 預設 overlay = 原圖（任何情況都會再畫 HUD）
        overlay = frame.copy()

        # 偵測
        count = self.previous_count  # 預設沿用上一幀（若沒模型或偵測壞）
        try:
            centers, _ann = self._detect_centers(frame)
            if centers:
                merged = self._merge_centers(centers)
                count = len(merged)
                # 在 overlay 上把合併後中心畫成綠點（可視化）
                for (cx, cy) in merged:
                    cv2.circle(overlay, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        except Exception as e:
            # 偵測出錯時就用上一幀的 count，HUD 仍會顯示
            print(f"[RaisinScorer] process detect exception: {e}")

        # 每 CHECK_INTERVAL 秒更新統計（避免小抖動）
        if (self.last_check_time is None) or (now - self.last_check_time >= CHECK_INTERVAL):
            if count > self.previous_count:
                added = count - self.previous_count
                if added > 1:
                    self.warning_flag = True
            if count > self.total_count:
                self.total_count = count
            self.previous_count = count
            self.last_check_time = now

        # 這一幀的即時分數（僅做即時顯示；最終用 finalize 的規則）
        self.last_score = int(count)

        # 疊 HUD/字幕（**關鍵**：確保每幀都有 overlay）
        overlay = self._draw_hud(
            overlay,
            count=self.last_score,
            total=self.total_count,
            remaining=remaining,
            warn=self.warning_flag
        )

        # 結束條件（只記第一次）
        if not self.done:
            if self.total_count >= TARGET_COUNT or elapsed >= self.duration:
                self.done = True
                self.end_time = self.start_time + min(elapsed, self.duration)

        return int(self.last_score), bool(self.done), overlay

    # ---------- 最終分數 ----------
    def finalize(self):
        if self.start_time is None:
            print("[RaisinScorer] finalize() 在沒有幀時被呼叫 -> 0 分")
            return 0

        # 使用 end_time（若尚未記錄則用現在時間），並 clamp 在 [0, duration]
        effective_end = self.end_time or time.time()
        elapsed = max(0.0, min(self.duration, effective_end - self.start_time))

        score = _calc_score(self.total_count, self.warning_flag, elapsed)
        print(f"[RaisinScorer] finalize() -> total={self.total_count}, "
              f"warning={self.warning_flag}, elapsed={int(elapsed)}s, score={score}")
        print("[DEBUG] thread 結束, camera released")
        return int(score)


if __name__ == "__main__":
    print("RaisinScorer class ready (Ultralytics YOLO).")
