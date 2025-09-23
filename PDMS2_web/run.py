# app.py
# -*- coding: utf-8 -*-
from pathlib import Path
from flask import Flask, send_from_directory, redirect, request, jsonify, session
import webbrowser
import threading  # 修正：使用正確的 import
from threading import Thread  # 新增：用於背景執行
import subprocess
import sys
import logging
import json
import secrets
from datetime import datetime
import uuid
import os
import cv2
import numpy as np
import io
from PIL import Image
import base64
from flask_cors import CORS

# 設定環境變數強制使用 UTF-8（可選）
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"


# 設定日誌輸出到 console.txt
def setup_console_logging():
    console_path = Path(__file__).parent / "console.txt"

    # 創建一個格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 創建文件處理器，只記錄我們的應用程式日誌
    file_handler = logging.FileHandler(console_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 禁用 Flask 和 Werkzeug 的 HTTP 請求日誌
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)  # 只記錄錯誤，不記錄 INFO 級別的請求日誌

    # 設定應用程式專用的日誌記錄器
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(file_handler)

    # 防止日誌向上傳播到根記錄器
    app_logger.propagate = False

    return app_logger


# 寫入結果到 console.txt 的函數
def write_to_console(message, level="INFO"):
    """將訊息寫入 console.txt，不包含 HTTP 請求日誌"""
    console_path = Path(__file__).parent / "console.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"{timestamp} - {level} - {message}\n"

    with open(console_path, "a", encoding="utf-8") as f:
        f.write(formatted_message)


PORT = 8000
HOST = "127.0.0.1"  # 只允許本機存取；要給同網段看可改 "0.0.0.0"
ROOT = Path(__file__).parent.resolve()  # 專案根：有 css/, js/, html/

app = Flask(__name__, static_folder=None)  # 不用預設 /static，改用我們自己的資料夾
app.secret_key = secrets.token_hex(16)  # 為 session 設置安全密鑰
CORS(app)

# 清除上次的 console.txt 內容
def clear_console_log():
    """在應用程式啟動時清除上次的日誌內容"""
    console_path = Path(__file__).parent / "console.txt"
    try:
        with open(console_path, "w", encoding="utf-8") as f:
            f.write("")  # 清空文件內容
    except Exception:
        pass  # 如果文件不存在或無法寫入，忽略錯誤


# 啟動時清除上次的日誌
clear_console_log()

# 初始化日誌系統
logger = setup_console_logging()

# 禁用 Flask 應用程式的日誌輸出
app.logger.disabled = True
logging.getLogger("flask.app").disabled = True

write_to_console("Flask 應用程式啟動", "INFO")

# 用於儲存任務狀態的字典
processing_tasks = {}


# === 改動 1：首頁直送入口頁 start.html（原本是導到 /html/index.html） ===
@app.route("/")
def home():
    return send_from_directory(ROOT / "html", "start.html")


# （可選）若有人直接打 /index 或 /index.html，都能進到故事首頁
@app.route("/index")
@app.route("/index.html")
def index_shortcut():
    return send_from_directory(ROOT / "html", "index.html")


# ---- 靜態檔路由（沿用你的資料夾結構）----
@app.route("/html/<path:filename>")
def html_files(filename):
    return send_from_directory(ROOT / "html", filename)


@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(ROOT / "css", filename)


@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(ROOT / "js", filename)


@app.route("/images/<path:filename>")
def images_files(filename):
    return send_from_directory(ROOT / "images", filename)


@app.route("/video/<path:filename>")
def video_files(filename):
    return send_from_directory(ROOT / "video", filename)


# （可選）favicon，如果你把檔案放在 html/ 或專案根
@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# 處理 Chrome 開發者工具的請求
@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    return ("", 204)


# === 改動 2：啟動時開首頁 "/"（原本開 /html/index.html） ===
def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/")


# Session 管理 API
@app.route("/session/set-uid", methods=["POST"])
def set_session_uid():
    """設置當前 session 的 UID"""
    try:
        data = request.get_json()
        uid = data.get("uid", "").strip()

        if not uid:
            write_to_console("設置 UID 失敗：UID 不能為空", "ERROR")
            return jsonify({"success": False, "error": "UID 不能為空"}), 400

        # 檢查UID是否包含無效字符
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in uid for char in invalid_chars):
            write_to_console(f"設置 UID 失敗：UID '{uid}' 包含無效字符", "ERROR")
            return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

        # 將 UID 存儲在 session 中
        session["uid"] = uid
        write_to_console(f"成功設置 UID：{uid}", "INFO")
        return jsonify({"success": True, "message": f"UID {uid} 已設置到 session"})

    except Exception as e:
        write_to_console(f"設置 UID 時發生錯誤：{str(e)}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/session/get-uid", methods=["GET"])
def get_session_uid():
    """獲取當前 session 的 UID"""
    uid = session.get("uid")
    if uid:
        return jsonify({"success": True, "uid": uid})
    else:
        return jsonify({"success": False, "message": "未找到 UID"}), 404


@app.route("/session/clear-uid", methods=["POST"])
def clear_session_uid():
    """清除當前 session 的 UID"""
    if "uid" in session:
        del session["uid"]
    return jsonify({"success": True, "message": "UID 已從 session 中清除"})


@app.route("/create-uid-folder", methods=["POST"])
def create_uid_folder():
    try:
        data = request.get_json()
        uid = data.get("uid", "").strip()

        if not uid:
            write_to_console("創建 UID 資料夾失敗：UID 不能為空", "ERROR")
            return jsonify({"success": False, "error": "UID 不能為空"}), 400

        # 檢查UID是否包含無效字符
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in uid for char in invalid_chars):
            write_to_console(f"創建 UID 資料夾失敗：UID '{uid}' 包含無效字符", "ERROR")
            return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

        # 將 UID 存儲在 session 中
        session["uid"] = uid

        # 創建 kid/{uid} 資料夾
        kid_dir = ROOT / "kid" / uid
        folder_already_exists = kid_dir.exists()

        if not folder_already_exists:
            kid_dir.mkdir(parents=True, exist_ok=True)
            write_to_console(f"成功創建使用者資料夾：{kid_dir}", "INFO")

        # 創建或更新結果JSON文件
        result_file = ROOT / f"result.json"

        # 讀取現有的JSON文件或創建新的
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                try:
                    result_data = json.load(f)
                except json.JSONDecodeError:
                    result_data = {}
        else:
            result_data = {}

        # 如果該UID還沒有記錄，添加初始化進度
        if uid not in result_data:
            result_data[uid] = {
                "ch1-t1": -1,
                "ch1-t2": -1,
                "ch1-t3": -1,
                "ch2-t1": -1,
                "ch2-t2": -1,
                "ch2-t3": -1,
                "ch2-t4": -1,
                "ch2-t5": -1,
                "ch2-t6": -1,
                "ch3-t1": -1,
                "ch3-t2": -1,
                "ch4-t1": -1,
                "ch4-t2": -1,
                "ch5-t1": -1,
            }

            # 寫回JSON文件
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            write_to_console(f"為使用者 {uid} 初始化進度記錄", "INFO")

        if folder_already_exists:
            write_to_console(f"資料夾 {uid} 已存在，確認結果文件", "INFO")
            return jsonify(
                {"success": True, "message": f"資料夾 {uid} 已存在，結果文件已確認"}
            )
        else:
            write_to_console(f"成功創建資料夾 {uid} 和結果文件", "INFO")
            return jsonify(
                {"success": True, "message": f"成功創建資料夾 {uid} 和結果文件"}
            )

    except Exception as e:
        write_to_console(f"創建 UID 資料夾時發生錯誤：{str(e)}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


# 安全的 subprocess 呼叫函數（解決編碼問題）
def safe_subprocess_run(cmd, **kwargs):
    """安全的 subprocess 呼叫，處理編碼問題"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    default_kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    default_kwargs.update(kwargs)

    return subprocess.run(cmd, **default_kwargs)

def run_analysis_in_background(task_id, uid, img_id, script_path):
    """在背景執行分析的函數"""
    try:
        # 更新任務狀態為進行中
        processing_tasks[task_id] = {
            "status": "running",
            "uid": uid,
            "img_id": img_id,
            "start_time": datetime.now().isoformat(),
            "progress": 0,
        }

        write_to_console(
            f"開始背景分析任務 {task_id}：使用者={uid}, 圖片ID={img_id}", "INFO"
        )

        # 執行 Python 腳本
        if uid:
            cmd = [sys.executable, str(script_path), uid, img_id]
        else:
            cmd = [sys.executable, str(script_path), img_id]

        write_to_console(f"執行命令：{' '.join(cmd)}", "INFO")
        result = safe_subprocess_run(cmd, cwd=ROOT)

        # 將腳本輸出寫入 console.txt
        if result.stdout:
            write_to_console(f"腳本輸出 (任務 {task_id})：\n{result.stdout}", "INFO")
        if result.stderr:
            write_to_console(
                f"腳本錯誤輸出 (任務 {task_id})：\n{result.stderr}", "ERROR"
            )

        # 更新任務狀態為完成
        processing_tasks[task_id] = {
            "status": "completed",
            "uid": uid,
            "img_id": img_id,
            "start_time": processing_tasks[task_id]["start_time"],
            "end_time": datetime.now().isoformat(),
            "progress": 100,
            "result": {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
        }

        status = "成功" if result.returncode == 0 else "失敗"
        write_to_console(
            f"背景分析任務 {task_id} 完成，狀態：{status} (返回碼: {result.returncode})",
            "INFO",
        )

    except Exception as e:
        # 更新任務狀態為錯誤
        processing_tasks[task_id] = {
            "status": "error",
            "uid": uid,
            "img_id": img_id,
            "start_time": processing_tasks[task_id].get("start_time"),
            "end_time": datetime.now().isoformat(),
            "progress": 0,
            "error": str(e),
        }

        write_to_console(f"背景分析任務 {task_id} 發生錯誤：{str(e)}", "ERROR")


# 修正：只保留一個 /run-python 路由（非同步版本）
@app.route("/run-python", methods=["POST"])
def run_python_script():
    try:
        data = request.get_json()
        img_id = data.get("id", "")

        # 優先從 session 獲取 UID
        uid = session.get("uid")
        if not uid:
            uid = data.get("uid", "") if data else ""

        write_to_console(
            f"收到 Python 腳本執行請求：使用者={uid}, 圖片ID={img_id}", "INFO"
        )

        # 檢查腳本是否存在
        script_path = ROOT / f"{img_id}" / "main.py"
        if not script_path.exists():
            write_to_console(f"腳本檔案不存在：{script_path}", "ERROR")
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        # 生成唯一的任務 ID
        task_id = str(uuid.uuid4())

        # 初始化任務狀態
        processing_tasks[task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": img_id,
            "progress": 0,
        }

        write_to_console(f"創建新任務：{task_id}，腳本路徑：{script_path}", "INFO")

        # 在背景執行任務
        thread = Thread(
            target=run_analysis_in_background, args=(task_id, uid, img_id, script_path)
        )
        thread.daemon = True
        thread.start()

        # 立即返回任務 ID
        return jsonify(
            {
                "success": True,
                "task_id": task_id,
                "message": "分析已開始，正在背景處理中...",
            }
        )

    except Exception as e:
        write_to_console(f"執行 Python 腳本時發生錯誤：{str(e)}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


# 新增：查詢任務狀態的 API
@app.route("/check-task/<task_id>", methods=["GET"])
def check_task_status(task_id):
    """查詢任務執行狀態"""
    if task_id not in processing_tasks:
        return jsonify({"success": False, "error": "任務不存在"}), 404

    task = processing_tasks[task_id]
    return jsonify(
        {
            "success": True,
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", 0),
            "img_id": task.get("img_id"),
            "uid": task.get("uid"),
            "start_time": task.get("start_time"),
            "end_time": task.get("end_time"),
            "result": task.get("result"),
            "error": task.get("error"),
        }
    )


# 新增：獲取用戶所有任務狀態
@app.route("/user-tasks", methods=["GET"])
def get_user_tasks():
    """獲取當前用戶的所有任務"""
    uid = session.get("uid")
    if not uid:
        return jsonify({"success": False, "error": "未找到用戶 UID"}), 400

    user_tasks = {}
    for task_id, task in processing_tasks.items():
        if task.get("uid") == uid:
            user_tasks[task_id] = task

    return jsonify(
        {
            "success": True,
            "uid": uid,
            "tasks": user_tasks,
            "task_count": len(user_tasks),
        }
    )


# 新增：清理已完成的任務（可選）
@app.route("/cleanup-tasks", methods=["POST"])
def cleanup_completed_tasks():
    """清理已完成超過一小時的任務"""
    current_time = datetime.now()
    tasks_to_remove = []

    for task_id, task in processing_tasks.items():
        if task["status"] in ["completed", "error"] and "end_time" in task:
            end_time = datetime.fromisoformat(task["end_time"])
            if (current_time - end_time).total_seconds() > 3600:  # 1小時
                tasks_to_remove.append(task_id)

    for task_id in tasks_to_remove:
        del processing_tasks[task_id]

    return jsonify(
        {
            "success": True,
            "cleaned_tasks": len(tasks_to_remove),
            "remaining_tasks": len(processing_tasks),
        }
    )


# 全局相機對象
camera = None
camera_active = False

def release_camera():
    """釋放相機資源"""
    global camera, camera_active
    if camera is not None:
        camera.release()
        camera = None
    camera_active = False

def init_camera(camera_index=1):
    """初始化相機"""
    global camera, camera_active
    try:
        release_camera()
        camera = cv2.VideoCapture(camera_index)  # 改用參數

        if not camera.isOpened():
            # 如果指定的失敗，嘗試其他索引
            for i in range(0, 4):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    break
            else:
                raise Exception("無法找到可用的相機")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 120)

        camera_active = True
        return True

    except Exception as e:
        print(f"相機初始化失敗: {e}")
        release_camera()
        return False



camera_lock = threading.Lock()

def get_frame():
    """獲取一幀圖像"""
    global camera, camera_active
    
    if not camera_active or camera is None:
        return None
    
    try:
        with camera_lock:
            ret, frame = camera.read()
            if not ret:
                return None

            # 轉換為 JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
        
    except Exception as e:
        print(f"獲取幀錯誤: {e}")
        return None

# === OpenCV 相機路由 ===

@app.route("/opencv-camera/start", methods=["POST"])
def start_opencv_camera():
    try:
        data = request.get_json()
        task_id = data.get("task_id", "")
        cam_index = data.get("camera_index", 1)  # 新增

        if init_camera(cam_index):
            global camera_active
            camera_active = True
            return jsonify({"success": True, "message": "相機已成功開啟", "task_id": task_id})
        else:
            return jsonify({"success": False, "error": "無法開啟相機"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/opencv-camera/frame", methods=["GET"])
def get_opencv_frame():
    """獲取相機幀，回傳 base64 編碼的 JPEG 圖像"""
    try:
        if not camera_active:
            return jsonify({"success": False, "error": "相機尚未啟動"}), 400

        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500

        # 將 bytes 轉成 base64 字串
        img_base64 = base64.b64encode(frame_data).decode('utf-8')

        return jsonify({
            "success": True,
            "image": img_base64
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/opencv-camera/capture", methods=["POST"])
def capture_opencv_photo():
    """拍照並儲存"""
    try:
        data = request.get_json()
        task_id = data.get("task_id", "")
        uid = data.get("uid", "") or session.get("uid", "default")
        
        if not task_id:
            return jsonify({"success": False, "error": "缺少任務 ID"}), 400
        
        # 獲取當前畫面
        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500
        
        target_dir = ROOT / "kid" / uid
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 加上 timestamp 避免覆蓋
        from datetime import datetime
        filename = f"{task_id}.jpg"
        file_path = target_dir / filename
        
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if not cv2.imwrite(str(file_path), img):
            return jsonify({"success": False, "error": "圖像儲存失敗"}), 500
        
        return jsonify({
            "success": True,
            "filename": filename,
            "path": str(file_path),
            "uid": uid,
            "task_id": task_id,
            "message": f"照片已成功儲存到 {file_path}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/opencv-camera/stop", methods=["POST"])
def stop_opencv_camera():
    """關閉 OpenCV 相機"""
    try:
        release_camera()
        global camera_active
        camera_active = False
        return jsonify({"success": True, "message": "相機已關閉"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 在應用程式關閉時清理資源
import atexit
atexit.register(release_camera)

if __name__ == "__main__":
    try:
        write_to_console(f"準備啟動 Flask 應用程式，HOST={HOST}, PORT={PORT}", "INFO")
        threading.Timer(0.5, _open_browser).start()
        write_to_console("已設定瀏覽器自動開啟", "INFO")
    except Exception as e:
        write_to_console(f"設定瀏覽器自動開啟時發生錯誤：{str(e)}", "ERROR")

    try:
        write_to_console("Flask 應用程式開始運行", "INFO")
        # 禁用 Werkzeug 開發伺服器的啟動訊息和請求日誌
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        write_to_console("接收到中斷信號，正在關閉應用程式", "INFO")
    except Exception as e:
        write_to_console(f"應用程式運行時發生錯誤：{str(e)}", "ERROR")
    finally:
        write_to_console("Flask 應用程式已關閉", "INFO")
