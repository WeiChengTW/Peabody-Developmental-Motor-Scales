# runFortest.py (已修正：傳遞 SIDE 參數給 Ch5-t1)
# -*- coding: utf-8 -*-
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify, session
import webbrowser, threading
from threading import Thread
import subprocess, sys, logging, json, secrets, uuid, os, base64, re, sqlite3
from datetime import datetime, date
import cv2
import numpy as np
from PIL import Image  # 保留相機相關依賴
from flask_cors import CORS
import traceback
from typing import Optional

# ======相機參數=====
TOP = 2
SIDE = 1 # <-- Ch5-t1 會使用這個索引
# =================


# =========================
# 1) 資料庫設定（本地 SQLite 模式）
# =========================
DB_PATH = Path(__file__).parent / "pdms2.db"


def get_db_conn():
    """取得 SQLite 連線"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化資料庫結構"""
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        
        # user_list
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_list (
                uid TEXT PRIMARY KEY,
                name TEXT,
                birthday TEXT
            )
        """)
        
        # task_list
        cur.execute("""
            CREATE TABLE IF NOT EXISTS task_list (
                task_id TEXT PRIMARY KEY,
                task_name TEXT
            )
        """)
        
        # score_list
        cur.execute("""
            CREATE TABLE IF NOT EXISTS score_list (
                score_id TEXT PRIMARY KEY,
                task_id TEXT,
                uid TEXT,
                score INTEGER,
                no INTEGER,
                test_date TEXT,
                FOREIGN KEY(uid) REFERENCES user_list(uid),
                FOREIGN KEY(task_id) REFERENCES task_list(task_id)
            )
        """)
        
        # 為每個任務建子表（以 task_name 為表名）
        tables_to_create = [
            "string_blocks", "pyramid", "stair",
            "draw_circle", "draw_square", "draw_cross", "draw_line", "color", "connect_dots",
            "cut_circle", "cut_square",
            "one_fold", "two_fold",
            "collect_raisins"
        ]
        
        for table in tables_to_create:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS `{table}` (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    score_id TEXT,
                    data1 TEXT,
                    data2 TEXT,
                    FOREIGN KEY(score_id) REFERENCES score_list(score_id)
                )
            """)
        
        conn.commit()
        write_to_console("[DB] 本地資料庫初始化完成", "INFO")
    except Exception as e:
        write_to_console(f"[DB] 初始化失敗：{e}", "ERROR")
        raise
    finally:
        conn.close()


# runFortest.py (Modified db_exec function)
def db_exec(sql, params=None, fetch="none"):
    """簡易 DB 執行器"""
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        if fetch == "one":
            row = cur.fetchone() 
            return dict(row) if row else None
        if fetch == "all":
            return [dict(row) for row in cur.fetchall()]
        conn.commit()
        return None
    except Exception as e:
        write_to_console(f"[DB] 執行失敗：{sql}\n錯誤：{e}", "ERROR")
        raise
    finally:
        conn.close()


# 任務對照
TASK_MAP = {
    "Ch1-t1": "string_blocks",
    "Ch1-t2": "pyramid",
    "Ch1-t3": "stair",
    "Ch2-t1": "draw_circle",
    "Ch2-t2": "draw_square",
    "Ch2-t3": "draw_cross",
    "Ch2-t4": "draw_line",
    "Ch2-t5": "color",
    "Ch2-t6": "connect_dots",
    "Ch3-t1": "cut_circle",
    "Ch3-t2": "cut_square",
    "Ch4-t1": "one_fold",
    "Ch4-t2": "two_fold",
    "Ch5-t1": "collect_raisins",
}


def ensure_user(uid: str, name: Optional[str] = None, birthday: Optional[str] = None):
    """如果 user_list 沒有該 uid，就建立；有則略過/可補 name/birthday"""
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO user_list(uid, name, birthday) VALUES (?, ?, ?)",
            (uid, name, birthday)
        )
        if name or birthday:
            cur.execute(
                "UPDATE user_list SET name = COALESCE(?, name), birthday = COALESCE(?, birthday) WHERE uid = ?",
                (name, birthday, uid)
            )
        conn.commit()
        conn.close()
        write_to_console(f"[DB] ensure_user ok: uid={uid}", "INFO")
    except Exception as e:
        write_to_console(f"[DB] ensure_user failed: uid={uid}, err={e}", "ERROR")
        raise


def get_conn():
    """相容舊的 get_conn() 呼叫"""
    return get_db_conn()


def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP:
        return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")


def insert_task_payload(task_id: str, score_id: str, data1=None, data2=None):
    table = task_id_to_table(task_id)
    db_exec(
        f"INSERT INTO `{table}` (score_id, data1, data2) VALUES (?, ?, ?)",
        (score_id, data1, data2),
    )


def ensure_task(task_id: str):
    """如果 task_list 沒有該 task_id，就依 TASK_MAP 補上"""
    if task_id not in TASK_MAP:
        raise ValueError(f"未知的 task_id：{task_id}")
    task_name = TASK_MAP[task_id]
    try:
        existing = db_exec("SELECT task_id FROM task_list WHERE task_id = ?", (task_id,), fetch="one")
        if not existing:
            db_exec(
                "INSERT INTO task_list(task_id, task_name) VALUES (?, ?)",
                (task_id, task_name),
            )
        write_to_console(f"[DB] ensure_task ok: {task_id} -> {task_name}", "INFO")
    except Exception as e:
        write_to_console(
            f"[DB] ensure_task failed: task_id={task_id}, err={e}", "ERROR"
        )
        raise

# ... (中略，_read_score_from_result_json 和 _parse_score_from_stdout 不變) ...

def _read_score_from_result_json(root: Path, uid: str, img_id: str):
    p = root / "result.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if uid in data and img_id in data[uid]:
            return int(data[uid][img_id])
        return None
    except Exception:
        return None

def _parse_score_from_stdout(stdout: str):
    if not stdout:
        return None
    m = re.search(r"score\s*[:=]\s*(\d+)", stdout, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def insert_score(
    uid: str,
    task_id: str,
    score: int,
    no: Optional[int] = None,
    test_date: Optional[date] = None,
) -> str:
    ensure_user(uid)
    ensure_task(task_id)
    if no is None:
        row = db_exec(
            "SELECT COUNT(*) AS cnt FROM score_list WHERE uid=? AND task_id=?",
            (uid, task_id),
            fetch="one",
        )
        no = int(row["cnt"]) + 1 if row else 1
    score_id = uuid.uuid4().hex
    if test_date is None:
        test_date = date.today()
    try:
        db_exec(
            "INSERT INTO score_list(score_id, task_id, uid, score, no, test_date) VALUES (?,?,?,?,?,?)",
            (score_id, task_id, uid, int(score), no, str(test_date)),
        )
        write_to_console(
            f"[DB] insert_score ok: uid={uid}, task_id={task_id}, score={score}, score_id={score_id}",
            "INFO",
        )
        return score_id
    except Exception as e:
        write_to_console(
            f"[DB] insert_score failed: uid={uid}, task_id={task_id}, err={e}", "ERROR"
        )
        raise

# ... (中略，Flask 基礎路由不變) ...

# =========================
# 2) 基礎環境/日誌/靜態路由
# =========================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

PORT = 8000
HOST = "127.0.0.1"
ROOT = Path(__file__).parent.resolve()

app = Flask(__name__, static_folder=None)
app.secret_key = secrets.token_hex(16)
CORS(app)

def setup_console_logging():
    console_path = Path(__file__).parent / "console.txt"
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(console_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    lg = logging.getLogger("app")
    lg.setLevel(logging.INFO)
    lg.addHandler(fh)
    lg.propagate = False
    return lg

def write_to_console(message, level="INFO"):
    console_path = Path(__file__).parent / "console.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(console_path, "a", encoding="utf-8") as f:
        f.write(f"{ts} - {level} - {message}\n")

def clear_console_log():
    console_path = Path(__file__).parent / "console.txt"
    try:
        with open(console_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

clear_console_log()
logger = setup_console_logging()
app.logger.disabled = True
logging.getLogger("flask.app").disabled = True
init_db()
write_to_console("=== 本地 SQLite 測試模式 ===", "INFO")
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
        p = Path(__file__).parent / "console.txt"
        if not p.exists():
            return jsonify({"ok": False, "msg": "console.txt not found"}), 404
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return jsonify({"ok": True, "lines": lines})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500

@app.before_request
def _log_request():
    try:
        write_to_console(
            f"[REQ] {request.method} {request.path} "
            f"CT={request.headers.get('Content-Type')} "
            f"Body={request.get_data(as_text=True)[:300]}"
        )
    except Exception as e:
        write_to_console(f"[REQ] log failed: {e}", "ERROR")

@app.after_request
def _log_response(resp):
    try:
        write_to_console(
            f"[RESP] {request.method} {request.path} -> {resp.status_code}"
        )
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

# ... (中略，Session 路由不變) ...

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
        session["uid"] = uid
        write_to_console(f"成功設置 UID：{uid}", "INFO")
        return jsonify({"success": True, "uid": uid})
    except Exception as e:
        write_to_console(f"設置 UID 時發生錯誤：{e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/session/get-uid")
def get_session_uid():
    uid = session.get("uid")
    return (
        jsonify({"success": True, "uid": uid})
        if uid
        else (jsonify({"success": False, "message": "未找到 UID"}), 404)
    )

@app.post("/create-uid-folder")
def create_uid_folder():
    data = request.get_json(silent=True) or {}
    uid = (data.get("uid") or "").strip()
    if not uid:
        write_to_console("create_uid_folder: UID 不能為空", "ERROR")
        return jsonify({"success": False, "error": "UID 不能為空"}), 400
    bad = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(c in uid for c in bad):
        write_to_console(f"create_uid_folder: UID 非法 -> {uid}", "ERROR")
        return jsonify({"success": False, "error": "UID 包含無效字符"}), 400
    ensure_user(uid)
    kid_dir = ROOT / "kid" / uid
    if not kid_dir.exists():
        kid_dir.mkdir(parents=True, exist_ok=True)
        write_to_console(f"[FS] 建立資料夾：{kid_dir}", "INFO")
    session["uid"] = uid
    return jsonify({"success": True, "uid": uid, "message": "使用者建立完成"})

@app.post("/session/clear-uid")
def clear_session_uid():
    if "uid" in session:
        del session["uid"]
    return jsonify({"success": True, "message": "UID 已清除"})

@app.route("/test-score", methods=["POST"])
def test_score():
    try:
        data = request.get_json()
        uid = data["uid"]
        task_id = data["task_id"]
        score = 3
        score_id = insert_score(uid, task_id, score)
        insert_task_payload(task_id, score_id, None, None)
        return jsonify({"success": True, "score_id": score_id, "score": score})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


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
    
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW # 0x08000000

    default_kwargs = dict(
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        creationflags=creation_flags
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


def run_analysis_in_background(task_id, uid, img_id, script_path, stair_type=None):
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
        is_game = (normalize_task_id(img_id) == "Ch5-t1")
        
        if is_game:
            # 遊戲模式：傳遞 uid 和 SIDE 相機索引
            cmd = base_cmd + [uid, str(SIDE)] # <--- 修正點 1：加入 str(SIDE)
        else:
            # 靜態分析模式：傳遞 uid 和 img_id (檔名)
            cmd = base_cmd + [uid, img_id]
            # 如果是階梯任務，再加入 stair_type
            if stair_type:
                cmd.append(stair_type)

        write_to_console(f"執行命令：{' '.join(cmd)}", "INFO")
        
        if is_game:
            # Ch5-t1 遊戲模式：不擷取輸出，隱藏終端機
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NO_WINDOW
            
            result = subprocess.run(
                cmd, 
                cwd=ROOT, 
                env=env, 
                capture_output=False, 
                text=True, 
                encoding="utf-8", 
                errors="replace",
                creationflags=creation_flags
            )
            
            score = int(result.returncode)
            stdout_str = "Game executed in foreground."
            stderr_str = ""
        else:
            # 原始模式：背景擷取，用於靜態分析
            result = safe_subprocess_run(cmd, cwd=ROOT)
            stdout_str = result.stdout
            stderr_str = result.stderr
            score = int(result.returncode) 

        if stdout_str:
            write_to_console(f"腳本輸出 (任務 {task_id})：\n{stdout_str}", "INFO")
        if stderr_str:
            write_to_console(
                f"腳本錯誤輸出 (任務 {task_id})：\n{stderr_str}", "ERROR"
            )

        task_id_std = normalize_task_id(img_id)
        uid_eff = uid or "unknown"

        score_id = None
        try:
            score_id = insert_score(uid_eff, task_id_std, score)
        except Exception as e:
            write_to_console(f"寫分數失敗：{e}", "ERROR")

        if score_id:
            try:
                if not is_game:
                     insert_task_payload(task_id_std, score_id, None, None)
            except Exception as e:
                write_to_console(f"寫入子表失敗：{e}", "ERROR")

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
                "score_id": score_id,
                "task_id": task_id_std,
            },
        }
        write_to_console(
            f"任務 {task_id} 完成：uid={uid_eff}, task={task_id_std}, score={score}, score_id={score_id}",
            "INFO",
        )

    except Exception as e:
        processing_tasks[task_id] = {
            "status": "error",
            "uid": uid,
            "img_id": img_id,
            "start_time": processing_tasks[task_id].get("start_time"),
            "end_time": datetime.now().isoformat(),
            "progress": 0,
            "error": str(e),
        }
        write_to_console(f"背景任務 {task_id} 發生錯誤：{e}", "ERROR")


@app.post("/run-python")
def run_python_script():
    try:
        data = request.get_json() or {}
        img_id = (data.get("id") or "").strip()
        uid = (data.get("uid") or "").strip() or session.get("uid")

        if not img_id:
            return jsonify({"success": False, "error": "缺少 id(task_id)"}), 400
        if not uid:
            return jsonify({"success": False, "error": "缺少 uid"}), 400
        
        script_path = resolve_script_path(img_id)
        if not script_path or not script_path.exists():
            write_to_console(f"腳本不存在：{script_path}", "ERROR")
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        task_id = str(uuid.uuid4())
        stair_type = session.get("stair_type")

        processing_tasks[task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": img_id,
            "progress": 0,
        }

        t = Thread(
            target=run_analysis_in_background, args=(task_id, uid, img_id, script_path, stair_type)
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
        write_to_console(f"/run-python 發生錯誤：{e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/check-task/<task_id>")
def check_task_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({"success": False, "error": "任務不存在"}), 404
    return jsonify({"success": True, **processing_tasks[task_id], "task_id": task_id})

# 接收階梯狀態 左或右階梯
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
# 5) 便利檢查 API
# =========================
@app.get("/db/ping")
def db_ping():
    try:
        db_exec("SELECT 1 AS ok")
        write_to_console(f"[DB] 本地 SQLite 連線正常", "INFO")
        return jsonify({"ok": True, "version": "SQLite (local)", "mode": "本地測試模式"})
    except Exception as e:
        write_to_console(f"/db/ping 失敗：{e}", "ERROR")
        return jsonify({"ok": False, "err": str(e)}), 500


@app.get("/scores")
def list_scores():
    rows = db_exec(
        "SELECT score_id, uid, task_id, score, no, test_date "
        "FROM score_list ORDER BY test_date DESC, score_id DESC LIMIT 50",
        fetch="all",
    )
    return jsonify(rows or [])


# =========================
# 6) OpenCV 相機
# =========================
camera = None
camera_active = False
camera_lock = threading.Lock()

def release_camera():
    global camera, camera_active
    if camera is not None:
        camera.release()
        camera = None
    camera_active = False


def init_camera(camera_index=TOP):
    global camera, camera_active
    try:
        release_camera()
        camera = cv2.VideoCapture(camera_index)
        
        if not camera.isOpened():
            raise Exception(f"無法開啟指定的相機索引: {camera_index}")
            
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 120)
        camera_active = True
        return True
    except Exception as e:
        print(f"相機初始化失敗: {e}")
        write_to_console(f"相機初始化失敗 (Index={camera_index}): {e}", "ERROR") 
        release_camera()
        return False


def get_frame():
    global camera, camera_active
    if not camera_active or camera is None:
        return None
    try:
        with camera_lock:
            ret, frame = camera.read()
            if not ret:
                write_to_console("camera.read() 失敗, ret=False", "ERROR")
                return None
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
    except Exception as e:
        print(f"獲取幀錯誤: {e}")
        write_to_console(f"獲取幀錯誤: {e}", "ERROR") 
        return None


@app.post("/opencv-camera/start")
def start_opencv_camera():
    try:
        data = request.get_json() or {}
        cam_index = data.get("camera_index", TOP)
        if init_camera(cam_index):
            return jsonify({"success": True, "message": "相機已成功開啟"})
        else:
            return jsonify({"success": False, "error": "無法開啟相機"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/opencv-camera/frame")
def get_opencv_frame():
    try:
        if not camera_active:
            return jsonify({"success": False, "error": "相機尚未啟動"}), 400
        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500
        img_b64 = base64.b64encode(frame_data).decode("utf-8")
        return jsonify({"success": True, "image": img_b64})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/opencv-camera/capture")
def capture_opencv_photo():
    """拍照並儲存，儲存成功後立即啟動對應任務的 main.py 做評分"""
    try:
        data = request.get_json() or {}
        task_id = (data.get("task_id") or "").strip()
        uid = (data.get("uid") or "").strip() or session.get("uid", "default")

        if not task_id:
            return jsonify({"success": False, "error": "缺少任務 ID"}), 400

        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500

        target_dir = ROOT / "kid" / uid
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task_id}.jpg" 
        file_path = target_dir / filename

        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if not cv2.imwrite(str(file_path), img):
            return jsonify({"success": False, "error": "圖像儲存失敗"}), 500

        script_path = resolve_script_path(task_id)
        if not script_path:
            return jsonify(
                {
                    "success": True,
                    "message": "照片已儲存，但找不到對應的 main.py（未啟動評分）",
                    "uid": uid,
                    "task_id": task_id,
                    "filename": filename,
                    "analysis_started": False,
                }
            )

        bg_task_id = str(uuid.uuid4())
        processing_tasks[bg_task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": task_id,
            "progress": 0,
        }

        t = Thread(
            target=run_analysis_in_background,
            args=(bg_task_id, uid, task_id, script_path),
        )
        t.daemon = True
        t.start()

        return jsonify(
            {
                "success": True,
                "message": "照片已儲存並開始評分",
                "uid": uid,
                "task_id": task_id,
                "filename": filename,
                "analysis_started": True,
                "analysis_task_id": bg_task_id,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/opencv-camera/stop")
def stop_opencv_camera():
    try:
        release_camera()
        return jsonify({"success": True, "message": "相機已關閉"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# 7) 啟動
# =========================
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
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        write_to_console("接收到中斷信號，正在關閉應用程式", "INFO")
    except Exception as e:
        write_to_console(f"應用程式運行時發生錯誤：{str(e)}", "ERROR")
    finally:
        write_to_console("Flask 應用程式已關閉", "INFO")
