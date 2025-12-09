# run.py (已修正：相機開啟問題 + 資料表名稱錯誤)
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import uuid
import base64
import secrets
import logging
import traceback
import subprocess
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, date
from typing import Optional
from threading import Thread

import cv2
import numpy as np
import pymysql
from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS

# ====== 相機參數 =====
TOP = 0
SIDE = 6 # Ch5-t1 使用
CROP_RATE = 0.8  # 預設裁切比例
# ====================

# =========================
# 1) 資料庫設定（PyMySQL 模式）
# =========================
DB = dict(
    host="13.238.239.23",
    port=3306,
    user="project",
    password="project",
    database="pdms2",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

def db_exec(sql, params=None, fetch="none"):
    """簡易 DB 執行器 (PyMySQL)"""
    conn = None
    try:
        conn = pymysql.connect(**DB)
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None
    except Exception as e:
        write_to_console(f"[DB] PyMySQL 執行失敗: {sql}\nParams: {params}\nError: {e}", "ERROR")
        raise
    finally:
        if conn:
            conn.close()

# ★★★★★ 修正 1：Ch1-t4 對應 build_wall ★★★★★
TASK_MAP = {
    "Ch1-t1": "string_blocks",
    "Ch1-t2": "pyramid",
    "Ch1-t3": "stair",
    "Ch1-t4": "build_wall",  # 已修正為正確表名
    "Ch2-t1": "draw_circle",
    "Ch2-t2": "draw_square",
    "Ch2-t3": "draw_cross",
    "Ch2-t4": "draw_line",
    "Ch2-t5": "color",
    "Ch2-t6": "connect_dots",
    "Ch3-t1": "cut_circle",
    "Ch3-t2": "cut_square",
    "Ch3-t3": "cut_paper",
    "Ch3-t4": "cut_line",
    "Ch4-t1": "one_fold",
    "Ch4-t2": "two_fold",
    "Ch5-t1": "collect_raisins",
}

# def ensure_user(uid: str, name: Optional[str] = None, birthday: Optional[str] = None):
#     try:
#         db_exec(
#             "INSERT INTO user_list(uid, name, birthday) VALUES (%s,%s,%s) "
#             "ON DUPLICATE KEY UPDATE name=COALESCE(VALUES(name),name), birthday=COALESCE(VALUES(birthday),birthday)",
#             (uid, name, birthday),
#         )
#         write_to_console(f"[DB] ensure_user ok: uid={uid}", "INFO")
#     except Exception as e:
#         raise
def user_exists(uid: str) -> bool:
    """回傳這個 uid 是否存在於 user_list"""
    row = db_exec(
        "SELECT 1 FROM user_list WHERE uid=%s",
        (uid,),
        fetch="one",   # 如果你的 db_exec 寫法不一樣，這裡用你原本查一筆資料的方式
    )
    return row is not None

def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP:
        return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")

def insert_task_payload(
    task_id: str,
    uid: str,
    test_date: date,
    score: int,
    result_img_path: str,
    data1: Optional[str] = None,
) -> None:
    table = task_id_to_table(task_id)
    # 獲取當前時間
    current_time = datetime.now().strftime("%H:%M:%S")

    sql = f"""
        INSERT INTO `{table}` (uid, test_date, time, score, result_img_path, data1)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            score           = VALUES(score),
            result_img_path = VALUES(result_img_path),
            data1           = VALUES(data1),
            time            = VALUES(time)
    """
    try:
        db_exec(sql, (uid, test_date, current_time, score, result_img_path, data1))
    except Exception:
        raise

def ensure_task(task_id: str):
    if task_id not in TASK_MAP:
        raise ValueError(f"未知的 task_id：{task_id}")
    task_name = TASK_MAP[task_id]
    try:
        db_exec(
            "INSERT INTO task_list(task_id, task_name) VALUES (%s,%s) "
            "ON DUPLICATE KEY UPDATE task_name=VALUES(task_name)",
            (task_id, task_name),
        )
        write_to_console(f"[DB] ensure_task ok: {task_id} -> {task_name}", "INFO")
    except Exception as e:
        raise



def insert_score(
    uid: str,
    task_id: str,
    test_date: Optional[date] = None,
) -> date:

    if not user_exists(uid):
        write_to_console(f"[DB] insert_score: UID 不存在 -> {uid}", "WARN")
        # 丟一個明確的錯誤，讓呼叫的人去決定要怎麼回應前端
        raise ValueError("USER_NOT_FOUND")

    # task 邏輯照舊（如果你希望只有管理者能新增 task，也可以之後再改 ensure_task）
    ensure_task(task_id)

    if test_date is None:
        test_date = date.today()

    current_time = datetime.now().strftime("%H:%M:%S")

    db_exec(
        """
        INSERT INTO score_list (uid, task_id, test_date, time)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            test_date = VALUES(test_date),
            time = VALUES(time)
        """,
        (uid, task_id, test_date, current_time),
    )
    write_to_console(
        f"[DB] insert_score ok: uid={uid}, task_id={task_id}, date={test_date}, time={current_time}",
        "INFO",
    )
    return test_date

# =========================
# 2) 基礎環境/日誌/靜態路由
# =========================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

PORT = 8000
HOST = "127.0.0.1"
ROOT = Path(__file__).parent.resolve()

app = Flask(
    __name__,
    static_folder=str(ROOT / "static"),
    static_url_path="/static",
)
app.secret_key = secrets.token_hex(16)
CORS(app)

def setup_console_logging():
    console_path = Path(__file__).parent / "console.txt"
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(console_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    lg = logging.getLogger("app")
    lg.setLevel(logging.INFO)
    lg.addHandler(fh)
    lg.propagate = False
    return lg

def write_to_console(message, level="INFO"):
    console_path = ROOT / "console.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(console_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {level} - {message}\n")
    except Exception as e:
        print(f"寫入 console.txt 失敗: {e}")

def clear_console_log():
    console_path = ROOT / "console.txt"
    try:
        with open(console_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

clear_console_log()
logger = setup_console_logging()
app.logger.disabled = True
logging.getLogger("flask.app").disabled = True
write_to_console("=== 遠端 PyMySQL 模式 ===", "INFO")
write_to_console("Flask 應用程式啟動", "INFO")
processing_tasks = {}

@app.route("/")
def home():
    return send_from_directory(ROOT / "html", "start.html")

@app.route("/index")
@app.route("/index.html")
def index_shortcut():
    return send_from_directory(ROOT / "html", "index.html")

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

@app.route("/kid/<path:filename>")
def kid_files(filename):
    # 讓網頁可以讀取 kid 資料夾內的照片
    return send_from_directory(ROOT / "kid", filename)

@app.route("/video/<path:filename>")
def video_files(filename):
    return send_from_directory(ROOT / "video", filename)

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    return ("", 204)

@app.get("/logs/tail")
def logs_tail():
    try:
        n = int(request.args.get("n", 200))
        p = ROOT / "console.txt"
        if not p.exists():
            return jsonify({"ok": False, "msg": "console.txt not found"}), 404
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return jsonify({"ok": True, "lines": lines})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500

@app.before_request
def _log_request():
    if request.path.startswith(("/css/", "/js/", "/images/", "/video/", "/favicon.ico", "/opencv-camera/")):
        return
    try:
        if request.path.startswith(("/run-python", "/create-uid-folder", "/test-score")):
            write_to_console(f"[REQ] {request.method} {request.path}")
    except Exception as e:
        write_to_console(f"[REQ] log failed: {e}", "ERROR")

@app.after_request
def _log_response(resp):
    if request.path.startswith(("/css/", "/js/", "/images/", "/video/", "/favicon.ico", "/opencv-camera/")):
        return resp
    try:
        if resp.status_code >= 400 or request.path.startswith(("/run-python", "/create-uid-folder", "/test-score")):
            write_to_console(f"[RESP] {request.method} {request.path} -> {resp.status_code}")
    except Exception as e:
        write_to_console(f"[RESP] log failed: {e}", "ERROR")
    return resp

@app.errorhandler(Exception)
def _handle_err(e):
    tb = traceback.format_exc()
    write_to_console(f"[ERR] {request.method} {request.path}\n{tb}", "ERROR")
    return jsonify({"success": False, "error": str(e)}), 500

def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/")

# =========================
# 3) Session：UID
# =========================
@app.post("/session/set-uid")
def set_session_uid():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        if not uid:
            return jsonify({"success": False, "error": "UID 不能為空"}), 400
        if any(c in uid for c in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
            return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

        # ⭐ 新增：只能用資料庫裡已存在的 UID
        if not user_exists(uid):
            write_to_console(f"set_session_uid: UID 不存在 -> {uid}", "WARN")
            return jsonify({
                "success": False,
                "error": "此使用者不存在，請請管理者建立帳號",
                "code": "USER_NOT_FOUND",
            }), 404

        session["uid"] = uid
        write_to_console(f"成功設置 UID：{uid}", "INFO")
        return jsonify({"success": True, "uid": uid})
    except Exception as e:
        write_to_console(f"設置 UID 時發生錯誤：{e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/session/get-uid", methods=['GET'])
def get_session_uid():
    uid = session.get("uid")
    return jsonify({"success": True, "uid": uid})

@app.post("/create-uid-folder")
def create_uid_folder():
    write_to_console("[REQ] 進入 create_uid_folder", "INFO")
    data = request.get_json(silent=True) or {}
    uid = (data.get("uid") or "").strip()
    if not uid:
        write_to_console("create_uid_folder: UID 不能為空", "ERROR")
        return jsonify({"success": False, "error": "UID 不能為空"}), 400

    bad = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(c in uid for c in bad):
        write_to_console(f"create_uid_folder: UID 非法 -> {uid}", "ERROR")
        return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

    # ⭐ 不再自動新增，只允許已存在的 UID
    if not user_exists(uid):
        write_to_console(f"create_uid_folder: UID 不存在 -> {uid}", "WARN")
        return jsonify({
            "success": False,
            "error": "此使用者不存在，請請管理者建立帳號",
            "code": "USER_NOT_FOUND",
        }), 404

    kid_dir = ROOT / "kid" / uid
    if not kid_dir.exists():
        kid_dir.mkdir(parents=True, exist_ok=True)
        write_to_console(f"[FS] 建立資料夾：{kid_dir}", "INFO")

    session["uid"] = uid
    return jsonify({"success": True, "uid": uid, "message": "UID 已載入"})


@app.post("/session/clear-uid")
def clear_session_uid():
    if "uid" in session:
        del session["uid"]
    return jsonify({"success": True, "message": "UID 已清除"})

@app.route("/test-score", methods=["POST"])
def test_score():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        task_id = (data.get("task_id") or "").strip()

        if not uid or not task_id:
            return jsonify({"success": False, "error": "uid 與 task_id 不可為空"}), 400

        score = 3  # 你目前先寫死 3 分

        try:
            # 這裡可能會因為 UID 不存在而丟 ValueError("USER_NOT_FOUND")
            test_date = insert_score(uid=uid, task_id=task_id)
        except ValueError as e:
            if str(e) == "USER_NOT_FOUND":
                return jsonify({
                    "success": False,
                    "error": "此使用者不存在，請管理者建立帳號",
                    "code": "USER_NOT_FOUND",
                }), 404
            # 其他 ValueError 再往上丟，交給外層 except
            raise

        insert_task_payload(
            task_id=task_id,
            uid=uid,
            test_date=test_date,
            score=score,
            result_img_path="",
            data1=None,
        )
        return jsonify({
            "success": True,
            "uid": uid,
            "task_id": task_id,
            "test_date": test_date.isoformat(),
            "score": score,
        })
    except Exception as e:
        write_to_console(f"/test-score 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# 4) 背景執行 main.py
# =========================
def safe_subprocess_run(cmd, **kwargs):
    """
    用於靜態分析：擷取 stdout/stderr，隱藏視窗
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # 為靜態分析加入 CREATE_NO_WINDOW
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW  # 0x08000000

    default_kwargs = dict(
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=creation_flags,
    )
    default_kwargs.update(kwargs)
    return subprocess.run(cmd, **default_kwargs)


def normalize_task_id(task_code_raw: str) -> str:
    if task_code_raw in TASK_MAP:
        return task_code_raw
    parts = task_code_raw.split("-")
    if len(parts) == 2:
        guess = parts[0][:1].upper() + parts[0][1:] + "-" + parts[1]
        if guess in TASK_MAP:
            return guess
    return task_code_raw


def resolve_script_path(task_code: str) -> Optional[Path]:
    guesses = [
        ROOT / task_code / "main.py",
        ROOT / task_code.lower() / "main.py",
        ROOT / normalize_task_id(task_code) / "main.py",
        ROOT / normalize_task_id(task_code).lower() / "main.py",
    ]
    for p in guesses:
        if p.exists():
            return p
    return None


def run_analysis_in_background(
    task_id, uid, img_id, script_path, stair_type=None, cam_index_input=None
):
    try:
        processing_tasks[task_id] = {
            "status": "running",
            "uid": uid,
            "img_id": img_id,
            "start_time": datetime.now().isoformat(),
            "progress": 0,
        }
        write_to_console(f"開始背景任務 {task_id}: uid={uid}, task={img_id}", "INFO")

        # 基礎命令
        base_cmd = [sys.executable, str(script_path)]

        # 判斷是否為遊戲，並決定參數
        is_game = normalize_task_id(img_id) == "Ch5-t1"

        camera_to_use = SIDE  # Ch5-t1 的預設值
        if is_game and cam_index_input is not None:
            try:
                camera_to_use = int(cam_index_input)
            except ValueError:
                write_to_console(
                    f"無效的 cam_index: {cam_index_input}，使用預設 SIDE={SIDE}", "WARN"
                )
                pass

        if is_game:
            # ===== 重要：Ch5-t1 遊戲模式，先確保前端相機已釋放 =====
            write_to_console(
                f"Ch5-t1 遊戲模式：準備使用相機索引 {camera_to_use}", "INFO"
            )

            # 強制釋放前端相機
            release_camera()
            write_to_console("[Ch5-t1] 前端相機已釋放", "INFO")

            # 等待相機資源完全釋放
            write_to_console("[Ch5-t1] 等待相機資源釋放...", "INFO")
            time.sleep(1.5)

            # 遊戲模式：傳遞 uid 和相機索引
            cmd = base_cmd + [uid, str(camera_to_use)]
            write_to_console(f"[Ch5-t1] 啟動遊戲命令: {' '.join(cmd)}", "INFO")
        else:
            # 靜態分析模式：傳遞 uid 和 img_id (檔名)
            cmd = base_cmd + [uid, img_id]
            if stair_type:
                cmd.append(stair_type)

        write_to_console(f"執行命令：{' '.join(cmd)}", "INFO")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        # 決定 creationflags
        creation_flags = 0
        if sys.platform == "win32":
            # 靜態任務使用 CREATE_NO_WINDOW 隱藏主控台
            if not is_game:
                creation_flags = subprocess.CREATE_NO_WINDOW
            # 遊戲任務不隱藏，讓 OpenCV 視窗可以正常顯示
            else:
                creation_flags = 0

        # 遊戲任務也擷取輸出，以便看到錯誤訊息
        capture_output_flag = True

        # 執行子程序
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=capture_output_flag,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )

        # 從執行結果中取得分數（exit code）
        score = int(result.returncode)

        stdout_str = result.stdout if result.stdout else ""
        stderr_str = result.stderr if result.stderr else ""

        if stdout_str:
            write_to_console(f"腳本輸出 (任務 {task_id})：\n{stdout_str}", "INFO")
        if stderr_str:
            write_to_console(f"腳本錯誤輸出 (任務 {task_id})：\n{stderr_str}", "ERROR")

        # 正規化成 Ch1-t1 / Ch2-t1 這種
        task_id_std = normalize_task_id(img_id)
        uid_eff = uid or "unknown"

        # 🔸 這裡開始：同時拿 test_date + test_time
        test_date = None
        test_time = None
        try:
            test_date, test_time = insert_score(uid_eff, task_id_std)
        except Exception as e:
            write_to_console(f"寫入 score_list 失敗：{e}", "ERROR")

        # 靜態任務才寫任務子表（遊戲任務目前不寫）
        if (test_date is not None) and (test_time is not None) and (not is_game):
            try:
                insert_task_payload(
                    task_id=task_id_std,
                    uid=uid_eff,
                    test_date=test_date,
                    test_time=test_time,   # 🔸 新增：時間一起寫入
                    score=score,
                    result_img_path="",
                    data1=None,
                )
            except Exception as e:
                write_to_console(
                    f"寫入任務子表失敗 (uid={uid_eff}, task={task_id_std}): {e}",
                    "ERROR",
                )

        processing_tasks[task_id] = {
            "status": "completed",
            "uid": uid_eff,
            "img_id": img_id,
            "start_time": processing_tasks[task_id]["start_time"],
            "end_time": datetime.now().isoformat(),
            "progress": 100,
            "result": {
                "success": True,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "returncode": score,
                "task_id": task_id_std,
                "test_date": test_date.isoformat() if test_date else None,
                "test_time": test_time.strftime("%H:%M:%S") if test_time else None,
            },
        }
        write_to_console(
            f"任務 {task_id} 完成：uid={uid_eff}, task={task_id_std}, "
            f"score={score}, test_date={test_date}, test_time={test_time}",
            "INFO",
        )

    except Exception as e:
        tb = traceback.format_exc()
        processing_tasks[task_id] = {
            "status": "error",
            "uid": uid,
            "img_id": img_id,
            "start_time": processing_tasks[task_id].get("start_time"),
            "end_time": datetime.now().isoformat(),
            "progress": 0,
            "error": str(e),
        }
        write_to_console(f"背景任務 {task_id} 發生嚴重錯誤：{e}\n{tb}", "ERROR")


@app.post("/run-python")
def run_python_script():
    try:
        data = request.get_json() or {}
        img_id = (data.get("id") or "").strip()
        uid = (data.get("uid") or "").strip() or session.get("uid")

        cam_index_input = data.get("cam_index")

        if not img_id:
            return jsonify({"success": False, "error": "缺少 id(task_id)"}), 400
        if not uid:
            return jsonify({"success": False, "error": "缺少 uid"}), 400

        script_path = resolve_script_path(img_id)
        if not script_path or not script_path.exists():
            write_to_console(f"腳本不存在: {script_path}", "ERROR")
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        task_id = str(uuid.uuid4())
        stair_type = session.get("stair_type")

        processing_tasks[task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": img_id,
            "progress": 0,
        }

        # 所有任務，包括 Ch5-t1，都導向 run_analysis_in_background
        t = Thread(
            target=run_analysis_in_background,
            args=(task_id, uid, img_id, script_path, stair_type, cam_index_input),
        )

        t.daemon = True
        t.start()

        return jsonify(
            {
                "success": True,
                "task_id": task_id,
                "message": "分析已開始，背景處理中...",
            }
        )
    except Exception as e:
        write_to_console(f"/run-python 發生錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/check-task/<task_id>")
def check_task_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({"success": False, "error": "任務不存在"}), 404
    return jsonify({"success": True, **processing_tasks[task_id], "task_id": task_id})


@app.post("/save-stair-type")
def save_stair_type():
    try:
        data = request.get_json() or {}
        stair_type = data.get("stair_type", "").strip()

        if stair_type not in ["L", "R"]:
            return jsonify({"success": False, "error": "stair_type 只能是 L 或 R"}), 400

        session["stair_type"] = stair_type
        write_to_console(f"stair_type: {stair_type}", "INFO")

        return jsonify({"success": True, "stair_type": stair_type})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# 5) 便利檢查 API (PyMySQL 版本)
# =========================
@app.get("/db/ping")
def db_ping():
    try:
        v = db_exec("SELECT VERSION() AS v", fetch="one")
        version_str = v["v"] if v else "無法取得版本"
        write_to_console(f"[DB] PyMySQL ping ok: {version_str}", "INFO")
        return jsonify({"ok": True, "version": version_str})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500


# @app.get("/scores")
# def list_scores():
#     try:
#         rows = db_exec(
#             "SELECT score_id, uid, task_id, score, no, test_date "
#             "FROM score_list ORDER BY test_date DESC, score_id DESC LIMIT 50",
#             fetch="all",
#         )
#         return jsonify(rows or [])
#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500


# =========================
# 6) OpenCV 相機
# =========================
camera = None
camera_active = False


def release_camera():
    global camera, camera_active
    if camera is not None:
        try:
            camera.release()
            write_to_console("[相機] 相機已釋放", "INFO")
        except Exception as e:
            write_to_console(f"[相機] 釋放相機時發生錯誤: {e}", "WARN")
        camera = None
    camera_active = False
    # 給系統一點時間完全釋放資源
    time.sleep(0.3)


CROP_RATE = 0.7


def init_camera(camera_index=TOP):
    global camera, camera_active
    try:
        release_camera()

        target_w, target_h = 1280, 720
        target_fps = 30

        # 1) 嘗試先用 v4l2-ctl 設定硬體格式（如果系統有 v4l2-ctl）
        try:
            cmd = [
                "v4l2-ctl",
                "--device=/dev/video{}".format(camera_index),
                "--set-fmt-video=width={},height={},pixelformat=MJPG".format(target_w, target_h),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            subprocess.run(["v4l2-ctl", "--device=/dev/video{}".format(camera_index), "--set-parm={}".format(target_fps)], check=False)
            write_to_console(f"使用 v4l2-ctl 設定格式為 {target_w}x{target_h} MJPG", "INFO")
        except Exception as e:
            # 不能依賴 v4l2-ctl 就忽略，但記錄
            write_to_console(f"v4l2-ctl 設定失敗或不存在: {e}", "WARN")

        # 2) 優先嘗試直接用 V4L2 後端開相機
        try:
            camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        except Exception:
            camera = None

        # 3) 如果沒開成功，嘗試使用 GStreamer pipeline (libcamera / v4l2 都能派上用場，視系統而定)
        if camera is None or not camera.isOpened():
            write_to_console("V4L2 開啟失敗，嘗試使用 GStreamer pipeline 或預設後端...", "WARN")
            # GStreamer pipeline 範例（如果 OpenCV 支援 CAP_GSTREAMER）
            gst = (
                f"v4l2src device=/dev/video{camera_index} ! "
                f"image/jpeg, width={target_w}, height={target_h}, framerate={target_fps}/1 ! "
                "jpegdec ! videoconvert ! appsink"
            )
            try:
                camera = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            except Exception as e:
                write_to_console(f"GStreamer 開啟嘗試失敗: {e}", "WARN")
                camera = None

        # 4) 如果還是沒開，回退到任何後端（CAP_ANY）
        if camera is None or not camera.isOpened():
            write_to_console("嘗試預設後端 (CAP_ANY)...", "WARN")
            camera = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
            if not camera.isOpened():
                raise Exception(f"無法開啟指定的相機索引: {camera_index}")

        # 5) 嘗試設定 FOURCC 與解析度（FOURCC 在設定解析度之前設定）
        try:
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        except Exception:
            pass

        # 實際設定解析度與 FPS
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        camera.set(cv2.CAP_PROP_FPS, target_fps)

        # 給 driver 一點時間應用設定
        time.sleep(0.3)

        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = camera.get(cv2.CAP_PROP_FPS)

        write_to_console(
            f"相機實際設定：{actual_width}x{actual_height} @ {actual_fps:.1f} FPS (目標: {target_w}x{target_h})",
            "INFO",
        )

        # 若仍然 640x480，可再嘗試一次用 v4l2-ctl 強制設為 YUYV (退路)
        if actual_width == 640 and actual_height == 480:
            write_to_console("注意：相機仍回傳 640x480，嘗試改用 YUYV 或檢查相機是否只支援此解析度", "WARN")
            # optional: 嘗試改 pixel format 為 YUYV
            try:
                subprocess.run(
                    [
                        "v4l2-ctl",
                        "--device=/dev/video{}".format(camera_index),
                        "--set-fmt-video=width={},height={},pixelformat=YUYV".format(target_w, target_h),
                    ],
                    check=False,
                )
            except Exception:
                pass

        ret, frame = camera.read()
        if not ret:
            raise Exception(f"成功開啟相機 {camera_index} 但無法讀取畫面")

        # 計算裁切區域
        h, w = frame.shape[:2]
        crop_w = int(w * CROP_RATE)
        crop_h = int(h * CROP_RATE)
        write_to_console(f"裁切後尺寸：{crop_w}x{crop_h} (保留中間{int(CROP_RATE*100)}%)", "INFO")

        camera_active = True
        return True
    except Exception as e:
        print(f"相機初始化失敗: {e}")
        write_to_console(f"相機初始化失敗: {e}", "ERROR")
        release_camera()
        return False


def crop_center(frame, rate):
    """裁切畫面中間區域"""

    h, w = frame.shape[:2]
    crop_w = int(w * rate)
    crop_h = int(h * rate)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    return frame[start_y : start_y + crop_h, start_x : start_x + crop_w]


def get_frame():
    global camera, camera_active
    if not camera_active or camera is None:
        return None
    try:
        ret, frame = camera.read()
        if not ret:
            return None

        # 切割畫面 CROP_RATE
        frame = crop_center(frame, CROP_RATE)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    except Exception as e:
        write_to_console(f"get_frame 錯誤: {e}", "ERROR")
        return None


# 相機路由
@app.post("/opencv-camera/stop")
def stop_opencv_camera():
    try:
        release_camera()
        return jsonify({"success": True})
    except Exception:
        return jsonify({"success": False}), 500


@app.post("/opencv-camera/start")
def start_opencv_camera():
    try:
        data = request.get_json() or {}
        cam_index = data.get("camera_index", TOP)
        if init_camera(cam_index):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False}), 500
    except Exception:
        return jsonify({"success": False}), 500


@app.get("/opencv-camera/frame")
def get_opencv_frame():
    try:
        if not camera_active:
            return jsonify({"success": False}), 400
        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False}), 500
        img_b64 = base64.b64encode(frame_data).decode("utf-8")
        return jsonify({"success": True, "image": img_b64})
    except Exception:
        return jsonify({"success": False}), 500


@app.post("/opencv-camera/capture")
def capture_opencv_photo():
    """拍照並存儲,存儲成功後立即啟動小應任務的 main.py 做評分"""
    try:
        data = request.get_json() or {}
        task_id_input = (data.get("task_id") or "").strip()
        uid = (data.get("uid") or "").strip() or session.get("uid", "default")

        if not task_id_input:
            return jsonify({"success": False}), 400

        frame_data = get_frame()
        if frame_data is None:
            write_to_console("capture: 無法取得畫面", "ERROR")
            return jsonify({"success": False}), 500

        # ===== 重點：拍照後立即釋放相機 =====
        release_camera()
        write_to_console("[相機] 拍照後已釋放相機資源", "INFO")
        # =====================================

        target_dir = ROOT / "kid" / uid
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task_id_input}.jpg"
        file_path = target_dir / filename

        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if not cv2.imwrite(str(file_path), img):
            write_to_console(f"圖像存儲失敗: {file_path}", "ERROR")
            return jsonify({"success": False}), 500

        write_to_console(f"圖像存儲成功: {file_path}", "INFO")

        script_task_id = task_id_input.replace("-side", "").replace("-top", "")
        script_path = resolve_script_path(script_task_id)

        if not script_path:
            return jsonify(
                {
                    "success": True,
                    "uid": uid,
                    "task_id": task_id_input,
                    "filename": filename,
                    "analysis_started": False,
                }
            )

        bg_task_id = str(uuid.uuid4())
        processing_tasks[bg_task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": script_task_id,
            "progress": 0,
        }

        stair_type = session.get("stair_type")

        # 靜態拍照任務,執行背景分析
        t = Thread(
            target=run_analysis_in_background,
            args=(bg_task_id, uid, script_task_id, script_path, stair_type),
        )
        t.daemon = True
        t.start()

        return jsonify(
            {
                "success": True,
                "uid": uid,
                "task_id": task_id_input,
                "filename": filename,
                "analysis_started": True,
                "analysis_task_id": bg_task_id,
            }
        )

    except Exception as e:
        write_to_console(f"/opencv-camera/capture 錯誤: {e}", "ERROR")
        # 發生錯誤時也要確保釋放相機
        release_camera()
        return jsonify({"success": False}), 500

@app.get("/game-state/<uid>")
def get_game_state(uid):
    """取得 Ch5-t1 遊戲狀態"""
    try:
        state_file = ROOT / "kid" / uid / "Ch5-t1_state.json"
        if not state_file.exists():
            return jsonify({"success": False, "error": "狀態檔案不存在"}), 404

        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        return jsonify({"success": True, "state": state})
    except Exception as e:
        write_to_console(f"讀取遊戲狀態失敗: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/clear-game-state")
def clear_game_state():
    """清空 Ch5-t1 遊戲狀態 JSON"""
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()

        if not uid:
            return jsonify({"success": False, "error": "缺少 UID"}), 400

        state_file = ROOT / "kid" / uid / "Ch5-t1_state.json"

        # 寫入初始狀態
        initial_state = {
            "running": False,
            "bean_count": 0,
            "remaining_time": 60,
            "warning": False,
            "game_over": False,
            "score": -1,
        }

        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, ensure_ascii=False, indent=2)

        write_to_console(f"[Ch5-t1] 遊戲狀態已清空: {uid}", "INFO")
        return jsonify({"success": True, "message": "遊戲狀態已重置"})

    except Exception as e:
        write_to_console(f"[Ch5-t1] 清空遊戲狀態失敗: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    try:
        write_to_console(f"準備啟動 Flask 應用程式，HOST={HOST}, PORT={PORT}", "INFO")
        threading.Timer(0.5, _open_browser).start()
        write_to_console("已設定瀏覽器自動開啟", "INFO")
    except Exception as e:
        write_to_console(f"設定瀏覽器自動開啟時發生錯誤：{str(e)}", "ERROR")

    try:
        write_to_console("Flask 應用程式開始運行", "INFO")
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        write_to_console("接收到中斷信號，正在關閉應用程式", "INFO")
    except Exception as e:
        write_to_console(f"應用程式運行時發生錯誤：{str(e)}", "ERROR")
    finally:
        write_to_console("Flask 應用程式已關閉", "INFO")