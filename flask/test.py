# app.py
# -*- coding: utf-8 -*-
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify, session, Response, stream_with_context
import webbrowser, threading
from threading import Thread
import subprocess, sys, logging, json, secrets, uuid, os, base64
from datetime import datetime, date
import cv2, numpy as np
from PIL import Image  # 保留相機相關依賴
from flask_cors import CORS
import traceback
import time
import re
import importlib.util
import uuid
import pymysql
from werkzeug.utils import secure_filename
# =========================
# 1) 資料庫設定（請改成你的值）
# =========================

DB = dict(
    host="16.176.187.101",
    port=3306,
    user="project",
    password="project",
    database="pdms2",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

def db_exec(sql, params=None, fetch="none"):
    """簡易 DB 執行器：fetch = none / one / all；會在 console.txt 記錄錯誤"""
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None
    finally:
        conn.close()
CAM_LOCK = threading.Lock()        
CAM = {
    "running": False,
    "cap": None,
    "index": 0,
    "last_err": None,
    "last_frame_ts": 0,
}

def _cam_release():
    cap = CAM.get("cap")
    if cap is not None:
        try: cap.release()
        except: pass
    CAM.update({"cap": None, "running": False})
TEXT_SHIFT_UP = 100   # 整塊字幕往上移 100px；想更上就增加
LINE_H = 28           # 行高（原本就是 28）

# ---- 放在檔案最上面 imports 附近 ----
def _make_loading_frame(text="Loading..."):
    import numpy as np, cv2, time
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None

# 任務對照（照你的 task_list）
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
# 統一 task_id 大小寫／格式
TASK_CANON = {
    "ch2-t1": "Ch2-t1",
    "Ch2-t1": "Ch2-t1",
    "ch5-t1": "Ch5-t1",
    "Ch5-t1": "Ch5-t1",
}
def canon_task_id(t: str) -> str:
    return TASK_CANON.get((t or "").strip(), (t or "").strip())


# ===== Collect Raisins（Ch5-t1）專用狀態 =====
beans_state = {
    "running": False,
    "start_ts": None,
    "duration": 60,
    "ui_anchor_ts": None,
    "last_frame": None,     
    "last_raw": None,        
    "last_overlay": None,    
    "score": None,
    "score_id": None,
    "uid": None,
    "task_id": "Ch5-t1",
    "warning": False,
    "total_count": 0,
    "abort": False,
}

beans_lock = threading.Lock()
stop_event = threading.Event()
def upsert_score_and_payload(uid: str, task_id: str, score: int, data1=None, data2=None):
    task_id = normalize_task_id(task_id)
    table = task_id_to_table(task_id)

    MAX_RETRY = 3
    for attempt in range(1, MAX_RETRY + 1):
        conn = pymysql.connect(**{**DB, "autocommit": False})
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(no), 0) AS max_no FROM score_list WHERE uid=%s AND task_id=%s FOR UPDATE",
                    (uid, task_id)
                )
                row = cur.fetchone() or {"max_no": 0}
                next_no = int(row.get("max_no", 0)) + 1

                cur.execute("SELECT UUID() AS uuid")
                row = cur.fetchone() or {}
                score_id = row.get("uuid")
                if not score_id:
                    raise RuntimeError("取得 UUID 失敗")

                cur.execute(
                    """INSERT INTO score_list (score_id, task_id, uid, score, no, test_date)
                       VALUES (%s, %s, %s, %s, %s, CURDATE())""",
                    (score_id, task_id, uid.strip(), int(score), next_no)
                )

                cur.execute(
                    f"INSERT INTO `{table}` (score_id, data1, data2) VALUES (%s, %s, %s)",
                    (score_id, data1, data2)
                )

            conn.commit()
            write_to_console(f"[DB] upsert_score_and_payload ok: uid={uid}, task={task_id}, score={score}, no={next_no}, score_id={score_id}", "INFO")
            return score_id, next_no

        except pymysql.err.OperationalError as e:
            conn.rollback()
            if getattr(e, "args", [None])[0] == 1213 and attempt < MAX_RETRY:
                backoff = 0.15 * attempt
                write_to_console(f"[DB] deadlock, retry {attempt}/{MAX_RETRY} after {backoff:.2f}s", "ERROR")
                time.sleep(backoff)
                continue
            write_to_console(f"[DB] upsert_score_and_payload error: {e}", "ERROR")
            raise
        except Exception as e:
            conn.rollback()
            write_to_console(f"[DB] upsert_score_and_payload error: {e}", "ERROR")
            raise
        finally:
            conn.close()


   
def _load_raisin_scorer_class():
    """
    從 Ch5-t1/main.py 動態載入 RaisinScorer 類別
        class RaisinScorer:
            def __init__(self, uid=None): ...
            def process(self, frame) -> tuple[int, bool, np.ndarray|None]
            def finalize(self) -> int
    """
    script_path = ROOT / "Ch5-t1" / "main.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到評分腳本: {script_path}")

    spec = importlib.util.spec_from_file_location("ch5_t1_main", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "RaisinScorer"):
        raise RuntimeError("Ch5-t1/main.py 內找不到 RaisinScorer 類別")
    return mod.RaisinScorer

def ensure_user(uid: str, name: str | None = None, birthday: str | None = None):
    """如果 user_list 沒有該 uid，就建立；有則略過/可補 name/birthday"""
    try:
        db_exec(
            "INSERT INTO user_list(uid, name, birthday) VALUES (%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE name=COALESCE(VALUES(name),name), birthday=COALESCE(VALUES(birthday),birthday)",
            (uid, name, birthday),
        )
        write_to_console(f"[DB] ensure_user ok: uid={uid}", "INFO")
    except Exception as e:
        write_to_console(f"[DB] ensure_user failed: uid={uid}, err={e}", "ERROR")
        raise

def get_conn():
    return pymysql.connect(**DB)

def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP:
        return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")


def ensure_task(task_id: str):
    """如果 task_list 沒有該 task_id，就依 TASK_MAP 補上"""
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
        write_to_console(f"[DB] ensure_task failed: task_id={task_id}, err={e}", "ERROR")
        raise

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
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
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
        write_to_console(f"[REQ] {request.method} {request.path} "
                         f"CT={request.headers.get('Content-Type')} "
                         f"Origin={request.headers.get('Origin')} "
                         f"Body={request.get_data(as_text=True)[:300]}")
    except Exception as e:
        write_to_console(f"[REQ] log failed: {e}", "ERROR")

@app.after_request
def _log_response(resp):
    try:
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
        if any(c in uid for c in ["/","\\",":","*","?","\"","<",">","|"]):
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
    return (jsonify({"success": True, "uid": uid}) if uid
            else (jsonify({"success": False, "message": "未找到 UID"}), 404))

@app.post("/create-uid-folder")
def create_uid_folder():
    data = request.get_json(silent=True) or {}
    uid = (data.get("uid") or "").strip()
    if not uid:
        write_to_console("create_uid_folder: UID 不能為空", "ERROR")
        return jsonify({"success": False, "error": "UID 不能為空"}), 400
    bad = ["/","\\",":","*","?","\"","<",">","|"]
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

@app.route("/test-score", methods=["POST"])  # 測試用的路由
def test_score():
    try:
        data = request.get_json()
        uid = data["uid"]
        task_id = data["task_id"]

        score = 3

        score_id = str(uuid.uuid4())
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO score_list (score_id, task_id, uid, score, no, test_date)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (score_id, task_id, uid, score, 1, date.today()))

            cursor.execute(f"""
                INSERT INTO `{task_id_to_table(task_id)}` (score_id, data1, data2)
                VALUES (%s, %s, %s)
            """, (score_id, None, None))

        conn.commit()
        conn.close()

        return jsonify({"success": True, "score_id": score_id, "score": score})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =========================
# 4) 背景執行 main.py + 寫入你的 schema
# =========================
def safe_subprocess_run(cmd, **kwargs):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    default_kwargs = dict(capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    default_kwargs.update(kwargs)
    return subprocess.run(cmd, **default_kwargs)

def normalize_task_id(task_code_raw: str) -> str:
    """把 ch2-t6 → Ch2-t6；若本來就正確大小寫則原樣返回"""
    if task_code_raw in TASK_MAP:
        return task_code_raw
    parts = task_code_raw.split("-")
    if len(parts) == 2:
        guess = parts[0][:1].upper() + parts[0][1:] + "-" + parts[1]
        if guess in TASK_MAP:
            return guess
    return task_code_raw

def resolve_script_path(task_code: str) -> Path | None:
    """根據傳入的 task_code 嘗試找到 {task}/main.py（兼容大小寫）"""
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

def run_analysis_in_background(task_id, uid, img_id, script_path):
    try:
        processing_tasks[task_id] = {
            "status": "running", "uid": uid, "img_id": img_id,
            "start_time": datetime.now().isoformat(), "progress": 0
        }
        write_to_console(f"開始背景任務 {task_id}: uid={uid}, task={img_id}", "INFO")

        if uid:
            cmd = [sys.executable, str(script_path), uid, img_id]
        else:
            cmd = [sys.executable, str(script_path), img_id]
    except Exception as e:
        # 至少加個 except，把錯誤記錄或 pass
        print(f"Error: {e}")
    try:
        result = safe_subprocess_run(cmd, cwd=ROOT)

        if result.stdout:
            write_to_console(f"腳本輸出 (任務 {task_id})：\n{result.stdout}", "INFO")
        if result.stderr:
            write_to_console(f"腳本錯誤輸出 (任務 {task_id})：\n{result.stderr}", "ERROR")

        task_id_std = normalize_task_id(img_id)
        uid_eff = uid or "unknown"
        score = int(result.returncode)

        try:
            score_id, _no = upsert_score_and_payload(uid_eff, task_id_std, score, None, None)
        except Exception as e:
            score_id = None
            write_to_console(f"寫分數/子表失敗：{e}", "ERROR")


        processing_tasks[task_id] = {
            "status": "completed",
            "uid": uid_eff,
            "img_id": img_id,
            "start_time": processing_tasks[task_id]["start_time"],
            "end_time": datetime.now().isoformat(),
            "progress": 100,
            "result": {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": score,
                "score_id": score_id,
                "task_id": task_id_std,
                "next_url": None,
            },
        }
        write_to_console(f"任務 {task_id} 完成：uid={uid_eff}, task={task_id_std}, score={score}, score_id={score_id}", "INFO")

    except Exception as e:
        processing_tasks[task_id] = {
            "status": "error", "uid": uid, "img_id": img_id,
            "start_time": processing_tasks[task_id].get("start_time"),
            "end_time": datetime.now().isoformat(), "progress": 0,
            "error": str(e),
        }
        write_to_console(f"背景任務 {task_id} 發生錯誤：{e}", "ERROR")

def run_collect_raisins_background(task_id, uid):
    """撿葡萄乾專用：同步跑 main.py，結束後入庫。"""
    try:
        img_id = "Ch5-t1"
        task_id_std = normalize_task_id(img_id)
        uid_eff = uid or "unknown"

        processing_tasks[task_id] = {
            "status": "running",
            "uid": uid_eff,
            "img_id": img_id,
            "start_time": datetime.now().isoformat(),
            "progress": 0
        }

        script_path = ROOT / "Ch5-t1" / "main.py"
        if not script_path.exists():
            raise FileNotFoundError(f"撿葡萄乾 main.py 不存在：{script_path}")

        cmd = [sys.executable, str(script_path), uid_eff]
        write_to_console(f"【撿葡萄乾】執行命令：{' '.join(cmd)}", "INFO")
        result = safe_subprocess_run(cmd, cwd=ROOT)

        if result.stdout:
            write_to_console(f"【撿葡萄乾】stdout：\n{result.stdout}", "INFO")
        if result.stderr:
            write_to_console(f"【撿葡萄乾】stderr：\n{result.stderr}", "ERROR")

        score = int(result.returncode)

        score_id = None
        try:
            score_id, _no = upsert_score_and_payload(uid_eff, task_id_std, score, None, None)
        except Exception as e:
            write_to_console(f"【撿葡萄乾】寫分數失敗：{e}", "ERROR")


        processing_tasks[task_id] = {
            "status": "completed",
            "uid": uid_eff,
            "img_id": img_id,
            "start_time": processing_tasks[task_id]["start_time"],
            "end_time": datetime.now().isoformat(),
            "progress": 100,
            "result": {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": score,
                "score_id": score_id,
                "task_id": task_id_std,
                "next_url": None,
            },
        }
        write_to_console(f"【撿葡萄乾】完成：uid={uid_eff}, score={score}, score_id={score_id}", "INFO")

    except Exception as e:
        processing_tasks[task_id] = {
            "status": "error",
            "uid": uid,
            "img_id": "Ch5-t1",
            "start_time": processing_tasks.get(task_id, {}).get("start_time"),
            "end_time": datetime.now().isoformat(),
            "progress": 0,
            "error": str(e),
        }
        write_to_console(f"【撿葡萄乾】背景任務發生錯誤：{e}", "ERROR")

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

        # === 撿葡萄乾：走 beans 流程（瀏覽器即時顯示） ===
        if img_id.lower() in ["ch5-t1", "collect_raisins"]:
            with beans_lock:
                already = beans_state["running"]
            if not already:
                t = Thread(target=_beans_thread_entry, args=(uid,), daemon=True)
                t.start()
                with beans_lock:
                    beans_state["running"]      = True
                    beans_state["uid"]          = uid
                    beans_state["task_id"]      = "Ch5-t1"
                    beans_state["start_ts"]     = time.time()   # ← 用 float
                    beans_state["duration"]     = 60
                    beans_state["ui_anchor_ts"] = None          # 讓 /beans/stream 連上時重設
                    # 清空緩衝
                    beans_state["last_frame"]   = None
                    beans_state["last_overlay"] = None
                    beans_state["last_overlay_ts"] = 0.0
                    beans_state["last_raw"]     = None
                    beans_state["last_raw_ts"]  = 0.0
                    beans_state["score"]        = None
                    beans_state["score_id"]     = None
            return jsonify({
                "success": True,
                "message": "撿葡萄乾任務已啟動（即時推流中）",
                "stream": "/beans/stream",
                "status": "/beans/status"
            })


        # === 其他任務：維持你的原本流程 ===
        script_path = resolve_script_path(img_id)
        if not script_path or not script_path.exists():
            write_to_console(f"腳本不存在：{script_path}", "ERROR")
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        task_id = str(uuid.uuid4())
        processing_tasks[task_id] = {"status": "pending", "uid": uid, "img_id": img_id, "progress": 0}
        t = Thread(target=run_analysis_in_background, args=(task_id, uid, img_id, script_path), daemon=True)
        t.start()

        return jsonify({"success": True, "task_id": task_id, "message": "分析已開始，背景處理中..."})
    except Exception as e:
        write_to_console(f"/run-python 發生錯誤：{e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/check-task/<task_id>")
def check_task_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({"success": False, "error": "任務不存在"}), 404
    return jsonify({"success": True, **processing_tasks[task_id], "task_id": task_id})

# =========================
# 5) 便利檢查 API
# =========================
@app.get("/db/ping")
def db_ping():
    try:
        v = db_exec("SELECT VERSION() AS v", fetch="one")
        write_to_console(f"[DB] ping ok: {v['v']}", "INFO")
        return jsonify({"ok": True, "version": v["v"]})
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
    return jsonify(rows)

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

def init_camera(camera_index=0):
    global camera, camera_active
    try:
        release_camera()
        camera = cv2.VideoCapture(int(camera_index))
        if not camera.isOpened():
            for i in range(0, 4):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    break
            else:
                raise Exception("無法找到可用的相機")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera_active = True
        return True
    except Exception as e:
        write_to_console(f"相機初始化失敗: {e}", "ERROR")
        release_camera()
        return False

def get_frame():
    """回傳一幀的 JPEG bytes；失敗回 None"""
    global camera, camera_active
    if not camera_active or camera is None:
        return None
    try:
        with camera_lock:
            ret, frame = camera.read()
            if not ret:
                return None
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            return buffer.tobytes()
    except Exception as e:
        write_to_console(f"獲取幀錯誤: {e}", "ERROR")
        return None

def _beans_thread_entry(uid: str):
    cap = None
    aborted = False
    try:
        # 1) 建 scorer（scorer 仍然由 main.py 控制「第一幀才開錶」）
        scorer = None
        try:
            Scorer = _load_raisin_scorer_class()
            scorer = Scorer(uid=uid) if "uid" in Scorer.__init__.__code__.co_varnames else Scorer()
            write_to_console("【撿葡萄乾】RaisinScorer 建立成功", "INFO")
        except Exception as e:
            write_to_console(f"【撿葡萄乾】RaisinScorer 建立/模型載入失敗：{e}", "ERROR")
            scorer = None  # 退回純相機直推

        # 2) 開相機（失敗就結束，不會開始倒數）
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            write_to_console("【撿葡萄乾】無法開啟相機", "ERROR")
            with beans_lock:
                beans_state["running"] = False
                beans_state["error"] = "camera_open_failed"
            return

        # 低延遲設定（不見得每個驅動都支援，但試試看）
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

        write_to_console(f"【撿葡萄乾】開始鏡頭任務 (UID={uid})", "INFO")

        # 3) 共享變數：讀取線放最新的 np.ndarray，推論線拿最新的用
        latest_np = {"img": None, "ts": 0.0}
        JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 75]  # 稍降畫質換順暢
        INFER_EVERY = 0.25  # 每 0.25s 做一次推論（丟掉中間多餘幀）
        last_infer_ts = 0.0
        last_score = 0

        # 4) 讀取線：只負責把 raw 幀推到前端，保持畫面流暢
        def reader_loop():
            first_raw_sent = False
            while True:
                if stop_event.is_set():
                    break

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # 覆蓋最新幀
                latest_np["img"] = frame
                latest_np["ts"] = time.time()

                # --- 送 raw 幀 ---
                try:
                    ok_raw, buf_raw = cv2.imencode(".jpg", frame, JPEG_PARAMS)
                    if ok_raw:
                        now_ts = time.time()  # ★ 新增
                        with beans_lock:
                            beans_state["last_raw"] = buf_raw.tobytes()
                            beans_state["last_raw_ts"] = now_ts
                            # ★ 第一次真正把畫面送出去時，記錄成「瀏覽器看見的 T0」
                            if not beans_state.get("start_ts"):
                                beans_state["start_ts"] = time.time()
                                # ★ 確保有 duration（跟 main.py 的 GAME_DURATION 對齊；沒有就用 60）
                                beans_state.setdefault("duration", 60)

                except Exception:
                    pass

                # --- Fallback 字幕（overlay 還沒來或過期，就先自己疊字） ---
                # --- Fallback 字幕（overlay 還沒來或過期，就先自己疊字） ---
                try:
                    with beans_lock:
                        over        = beans_state.get("last_overlay")
                        over_ts     = beans_state.get("last_overlay_ts", 0.0)
                        # 用 UI 的起點（沒有就退回 start_ts）
                        anchor_ts   = beans_state.get("ui_anchor_ts") or beans_state.get("start_ts")
                        total_count = int(beans_state.get("total_count", 0))
                        warning_flag= bool(beans_state.get("warning", False))
                        curr_count  = int(beans_state.get("current_count", 0))
                        duration    = int(beans_state.get("duration", 60))

                    need_fallback = (over is None) or (time.time() - over_ts > 0.5)
                    if need_fallback:
                        # 用 anchor_ts 讓畫面從 60 開始
                        if isinstance(anchor_ts, (int, float)) and anchor_ts > 0:
                            elapsed = max(0.0, time.time() - float(anchor_ts))
                        else:
                            elapsed = 0.0
                        remaining = max(0, int(duration - elapsed))

                        fallback = frame.copy()
                        h, w = fallback.shape[:2]
                        base_y = max(40, h - 20 - TEXT_SHIFT_UP)
                        line_h = LINE_H


                        # # 半透明底條
                        # try:
                        #     _bar = fallback.copy()
                        #     cv2.rectangle(_bar, (0, base_y - line_h*3 - 12), (w, h), (0, 0, 0), -1)
                        #     fallback = cv2.addWeighted(_bar, 0.35, fallback, 0.65, 0)
                        # except Exception:
                        #     pass

                        cv2.putText(fallback, f'SoyBean count: {curr_count}', (10, base_y - line_h*2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(fallback, f'Total placed: {total_count}', (10, base_y - line_h),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(fallback, f'Time Left: {remaining}s', (10, base_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        if warning_flag:
                            cv2.putText(fallback, 'Warning !', (10, base_y - line_h*3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        ok_fb, buf_fb = cv2.imencode(".jpg", fallback, JPEG_PARAMS)
                        if ok_fb:
                            with beans_lock:
                                beans_state["last_overlay"] = buf_fb.tobytes()
                                beans_state["last_overlay_ts"] = time.time()
                except Exception:
                    pass

                time.sleep(0.01)

        reader_thr = Thread(target=reader_loop, daemon=True)
        reader_thr.start()

        # 5) 推論線（在當前函式中做）：固定節奏抓「最新幀」做 process()
        while True:
            # 手動中止
            if stop_event.is_set():
                aborted = True
                write_to_console("【撿葡萄乾】偵測到手動中止，準備退出", "INFO")
                break

            frame = latest_np["img"]
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            # 沒到推論時間 -> 稍等（畫面仍由 reader 線在更新，不會卡）
            if now - last_infer_ts < INFER_EVERY:
                time.sleep(0.01)
                continue
            last_infer_ts = now

            show_frame = frame  # 預設原圖
            if scorer is not None:
                try:
                    out = scorer.process(frame)
                    if not isinstance(out, tuple):
                        raise RuntimeError("RaisinScorer.process 必須回傳 tuple")
                    if len(out) == 3:
                        score, done, overlay = out
                    elif len(out) == 2:
                        score, done = out
                        overlay = None
                    else:
                        raise RuntimeError("RaisinScorer.process 回傳格式需為 (score, done) 或 (score, done, overlay)")

                    last_score = int(score)
                    if overlay is not None:
                        show_frame = overlay

                    # 更新狀態（讓 /beans/status 即時反映）
                    with beans_lock:
                        beans_state["current_count"] = int(last_score)                  # ✅ 目前幀偵測數
                        beans_state["total_count"]   = int(getattr(scorer, "total_count", 0))
                        beans_state["warning"]       = bool(getattr(scorer, "warning_flag", False))


                    # 把 overlay 推一次，覆蓋 reader 剛剛送出去的 raw（使用者看到即時疊字）
                    # === 強制重畫字幕（以 start_ts 為準） ===
                    # === 強制重畫字幕（用 UI anchor 作為顯示起點） ===
                    try:
                        with beans_lock:
                            anchor_ts   = beans_state.get("ui_anchor_ts") or beans_state.get("start_ts")
                            total_count = int(beans_state.get("total_count", 0))
                            curr_count  = int(beans_state.get("current_count", 0))
                            warning     = bool(beans_state.get("warning", False))
                            duration    = int(beans_state.get("duration", 60))

                        if isinstance(anchor_ts, (int, float)) and anchor_ts > 0:
                            elapsed = max(0.0, time.time() - float(anchor_ts))
                        else:
                            elapsed = 0.0
                        remaining = max(0, int(duration - elapsed))

                        h, w = show_frame.shape[:2]
                        base_y = max(40, h - 20 - TEXT_SHIFT_UP)
                        line_h = LINE_H


                        try:
                            _bar = show_frame.copy()
                            cv2.rectangle(_bar, (0, base_y - line_h*3 - 12), (w, h), (0, 0, 0), -1)
                            show_frame = cv2.addWeighted(_bar, 0.35, show_frame, 0.65, 0)
                        except Exception:
                            pass

                        cv2.putText(show_frame, f'SoyBean count: {curr_count}', (10, base_y - line_h*2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(show_frame, f'Total placed: {total_count}', (10, base_y - line_h),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(show_frame, f'Time Left: {remaining}s', (10, base_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        if warning:
                            cv2.putText(show_frame, 'Warning !', (10, base_y - line_h*3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    except Exception:
                        pass



                    if bool(done):
                        write_to_console("【撿葡萄乾】scorer 回報完成", "INFO")
                        # 最後再推一次 overlay，確保終端畫面
                        try:
                            ok2, buf2 = cv2.imencode(".jpg", show_frame, JPEG_PARAMS)
                            if ok2:
                                with beans_lock:
                                    beans_state["last_frame"] = buf2.tobytes()
                        except Exception:
                            pass
                        break

                except Exception as e:
                    write_to_console(f"【撿葡萄乾】process 發生錯誤（保留 scorer 繼續重試）：{e}", "ERROR")
                    time.sleep(0.05)
                    continue  # 下個迴圈再試，不要把 scorer 取消

            # 控制推論線迴圈頻率（reader 線仍在快速送 raw）
            time.sleep(0.005)

        # 6) 收尾：若是手動中止或沒有 scorer，就不要寫 DB
        if aborted or scorer is None:
            with beans_lock:
                beans_state["running"] = False
                beans_state["last_frame"] = None
            return

        # 正常結束 → finalize + 寫 DB
        try:
            final_score = int(scorer.finalize())
        except Exception as e:
            write_to_console(f"【撿葡萄乾】finalize() 發生錯誤：{e}，改用最後分數", "ERROR")
            final_score = int(last_score)

        score_id, _no = upsert_score_and_payload(uid, "Ch5-t1", final_score, None, None)

        with beans_lock:
            beans_state["score"] = final_score
            beans_state["score_id"] = score_id
            beans_state["running"] = False
            beans_state["last_frame"] = None
            beans_state.pop("error", None)

        write_to_console(f"【撿葡萄乾】結束: score={final_score}, score_id={score_id}", "INFO")

    except Exception as e:
        write_to_console(f"【撿葡萄乾】執行錯誤: {e}", "ERROR")
        with beans_lock:
            beans_state["running"] = False
            beans_state["last_frame"] = None
            beans_state["error"] = "runtime_error"
    finally:
        try:
            if cap is not None:
                cap.release()
        except:
            pass
        with beans_lock:
            if not beans_state.get("running"):
                beans_state["last_frame"] = None


@app.get("/camera/stream")
def camera_stream():
    """前端 <img src="/camera/stream"> 就能即時預覽"""
    # 確保已開相機
    if not camera_active:
        init_camera(0)
    return Response(stream_with_context(_mjpeg_generator()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")



@app.post("/beans/start")
def beans_start():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip() or session.get("uid")
        if not uid:
            return jsonify({"success": False, "error": "缺少 uid"}), 400

        # 允許強制重啟：/beans/start?force=true 或 body: {"force": true}
        force = str(request.args.get("force", data.get("force", "false"))).lower() in ("1", "true", "yes")

        with beans_lock:
            if beans_state.get("running"):
                if not force:
                    # 已在跑且不強制 -> 直接回覆
                    return jsonify({"success": True, "message": "已在執行中"}), 200
                else:
                    # 強制重啟：先要求舊執行緒停
                    beans_state["abort"] = True
                    stop_event.set()
                    write_to_console("【撿葡萄乾】收到 force，舊任務將中止並重啟", "INFO")

            # 重置狀態，準備開新任務
            beans_state["running"]   = True
            beans_state["uid"]       = uid
            beans_state["task_id"]   = "Ch5-t1"
            beans_state["start_ts"]  = time.time()            # 用 float 當起點（你已改）
            beans_state["duration"]  = 60

            # ✨ 清空影像緩衝（最重要）
            beans_state["last_frame"]    = None
            beans_state["last_overlay"]  = None
            beans_state["last_overlay_ts"]= 0.0
            beans_state["last_raw"]      = None
            beans_state["last_raw_ts"]   = 0.0

            beans_state["score"]     = None
            beans_state["score_id"]  = None
            beans_state["warning"]   = False
            beans_state["total_count"]= 0
            beans_state["abort"]     = False

        # 清除停止事件（一定要在開新 thread 前清掉）
        stop_event.clear()

        t = Thread(target=_beans_thread_entry, args=(uid,), daemon=True)
        t.start()
        return jsonify({"success": True, "running": True, "stream": "/beans/stream"})
    except Exception as e:
        with beans_lock:
            beans_state["running"] = False
        return jsonify({"success": False, "error": str(e)}), 500


def _beans_mjpeg_generator():
    boundary = b"--frame"
    loading  = _make_loading_frame("Loading...") or b""

    while True:
        with beans_lock:
            running = beans_state["running"]
            over    = beans_state.get("last_overlay")
            raw     = beans_state.get("last_raw")

        # 只要有 overlay 就用 overlay；沒有才用 raw；最後才 loading
        frame = over if over is not None else (raw if raw is not None else loading)

        if frame:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            time.sleep(0.03)

        if not running and (over is None and raw is None):
            break


@app.get("/beans/stream")
def beans_stream():
    with beans_lock:
        # 接上串流時，才設定 UI 起點
        beans_state["ui_anchor_ts"] = time.time()

    resp = Response(stream_with_context(_beans_mjpeg_generator()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp




@app.get("/beans/status")
def beans_status():
    with beans_lock:
        running = beans_state["running"]
        score = beans_state["score"]
        done = (not running) and (score is not None)
        err  = beans_state.get("error")

        return jsonify({
            "running": running,
            "uid": beans_state["uid"],
            "task_id": beans_state["task_id"],
            "score": score,
            "score_id": beans_state["score_id"],
            "warning": beans_state["warning"],
            "total_count": beans_state["total_count"],
            "done": done,
            "error": err,
            "nextUrl": "/html/index.html"
        })

@app.post("/beans/stop")
def beans_stop():
    with beans_lock:
        stop_event.set()
        beans_state["running"]      = False
        beans_state["uid"]          = None
        beans_state["task_id"]      = None
        beans_state["error"]        = None
        beans_state["last_frame"]   = None
        # ✨ 一併清空緩衝
        beans_state["last_overlay"] = None
        beans_state["last_overlay_ts"] = 0.0
        beans_state["last_raw"]     = None
        beans_state["last_raw_ts"]  = 0.0
    return jsonify({"success": True})


# ===== 舊版相機相容層：/opencv-camera/* =====
_legacy_cam = {"stop": False, "index": 0}

def _opencv_stream_gen(cam_index: int):
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        import numpy as np, cv2
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "Camera open failed", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        ok, buf = cv2.imencode(".jpg", img)
        frame = buf.tobytes() if ok else b""
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _legacy_cam["stop"] = False
    while not _legacy_cam["stop"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03); continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)
    cap.release()

@app.route("/opencv-camera/start", methods=["POST"])
def start_opencv_camera():
    data = request.get_json()
    cam_index = data.get("camera_index", 0)
    if init_camera(cam_index):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "無法開啟相機"}), 500



@app.post("/opencv-camera/stop")
def opencv_stop():
    # 這支是給 legacy camera 用的，必須呼叫 release_camera()
    try:
        with camera_lock:
            release_camera()  # 正確關閉並把 camera=None, camera_active=False
        return jsonify({"success": True}), 200
    except Exception as e:
        write_to_console(f"/opencv-camera/stop error: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = get_frame()
        if frame is None:
            # 給前端機會重試，不要 500
            time.sleep(0.05)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)
@app.get("/opencv-camera/stream")
def legacy_opencv_stream():
    resp = Response(stream_with_context(_opencv_stream_gen(_legacy_cam["index"])),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/opencv-camera/frame")
def opencv_frame():
    global camera, camera_active

    # 第一次快速檢查（避免不必要加鎖）
    if not camera_active or camera is None:
        return ("", 404)

    with camera_lock:
        # 進入臨界區後再檢查一次（解決競態）
        if not camera_active or camera is None:
            return ("", 404)

        # 丟掉第一幀，避免黑畫面（要先確認 camera 仍在）
        try:
            _ = camera.read()
        except Exception:
            return ("", 404)

        ok, jpg = False, None
        for _ in range(3):  # 最多試 3 次
            time.sleep(0.05)
            # 讀取前再次確認
            if not camera_active or camera is None:
                return ("", 404)
            ret, frame = camera.read() if camera is not None else (False, None)
            if not ret or frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpg = buf.tobytes()
                break

    if not ok or jpg is None:
        # 失敗不要 500，回 404 讓前端重試或提示重新開啟相機
        return ("", 404)

    return Response(jpg, mimetype="image/jpeg")

def _run_task_main_bg(uid: str, file_suffix: str):
    """背景：解析任務、執行 main.py，然後寫入資料庫"""
    try:
        parts = file_suffix.split("-")
        task_id_for_main = "-".join(parts[:2])           # ch1-t2-side -> ch1-t2
        script_path = resolve_script_path(task_id_for_main)
        if not script_path or not script_path.exists():
            write_to_console(f"[BG] main.py 不存在: {task_id_for_main}", "ERROR")
            return

        # 跑 main.py（讀 kid/<uid>/<file_suffix>.jpg）
        cmd = [sys.executable, str(script_path), uid, file_suffix]
        result = safe_subprocess_run(cmd, cwd=ROOT)

        # 取分數（優先 returncode；也可改成從 stdout 解析）
        score = int(result.returncode)

        # 正規化 task_id（寫入 score_list 用）
        task_id_std = normalize_task_id(task_id_for_main)

        # 寫 DB：score_list + 對應子表
        try:
            upsert_score_and_payload(uid, task_id_std, score, None, None)
            write_to_console(f"[BG] 寫入分數完成 uid={uid}, task={task_id_std}, score={score}", "INFO")
        except Exception as e:
            write_to_console(f"[BG] 寫入 DB 失敗: {e}", "ERROR")

    except Exception as e:
        write_to_console(f"[BG] _run_task_main_bg 發生錯誤: {e}", "ERROR")

@app.route("/opencv-camera/capture", methods=["POST"])
def capture_opencv_photo():
    try:
        data = request.get_json(silent=True) or {}
        uid = (data.get("uid") or "").strip() or session.get("uid", "").strip()
        task_id = (data.get("task_id") or "").strip()

        if not uid:
            write_to_console("[CAPTURE] 缺少 uid", "ERROR")
            return jsonify({"success": False, "error": "缺少 uid"}), 400
        if not task_id:
            write_to_console("[CAPTURE] 缺少任務 ID", "ERROR")
            return jsonify({"success": False, "error": "缺少任務 ID"}), 400

        # 相機確保已啟動（避免 camera=None）
        if not camera_active or camera is None:
            write_to_console("[CAPTURE] 相機未啟動，嘗試 init_camera(0)", "INFO")
            if not init_camera(0):
                write_to_console("[CAPTURE] 相機啟動失敗", "ERROR")
                return jsonify({"success": False, "error": "相機未開啟"}), 500

        # 直接拿一幀的 JPEG bytes
        frame_data = get_frame()
        if not frame_data:
            write_to_console("[CAPTURE] 無法取得相機影像 (get_frame=None)", "ERROR")
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500

        # 確保路徑存在
        target_dir = ROOT / "kid" / uid
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{task_id}.jpg"
        file_path = target_dir / filename

        # 直接把 JPEG bytes 寫到檔案
        with open(file_path, "wb") as f:
            f.write(frame_data)

        # 再保險驗證檔案大小
        if (not file_path.exists()) or file_path.stat().st_size == 0:
            write_to_console(f"[CAPTURE] 寫檔後檢查失敗: {file_path}", "ERROR")
            return jsonify({"success": False, "error": "寫入檔案失敗"}), 500

        write_to_console(f"[CAPTURE] 存檔成功: {file_path}", "INFO")

        # ✅ 不再自動啟動背景任務；由前端另行呼叫 /run-python 啟動
        return jsonify({
            "success": True,
            "filename": filename,
            "path": str(file_path),
            "message": "影像已存。請呼叫 /run-python 啟動分析。",
            "run_endpoint": "/run-python",
            "run_payload": {"id": task_id, "uid": uid}
        }), 200

    except Exception as e:
        write_to_console(f"[CAPTURE] 發生未預期錯誤: {e}", "ERROR")
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


