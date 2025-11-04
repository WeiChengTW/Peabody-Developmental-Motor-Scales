# run.py (已合併 Ch5-t1 彈出視窗 + PyMySQL)
# -*- coding: utf-8 -*-
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify, session
import webbrowser, threading
from threading import Thread
import subprocess, sys, logging, json, secrets, uuid, os, base64, re
from datetime import datetime, date
import cv2, numpy as np
from PIL import Image  # 保留相機相關依賴
from flask_cors import CORS
import traceback
from typing import Optional
from flask import Flask, send_from_directory, request, jsonify, session, send_file


# ======相機參數 (使用 runFortest.py 的值) =====
TOP = 1
SIDE = 2 # <-- Ch5-t1 會使用這個索引
# ============================================

# =========================
# 1) 資料庫設定（PyMySQL 模式）
# =========================
import pymysql

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

def query_all(sql, params=()):
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()

def execute(sql, params=()):
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount
    finally:
        conn.close()


def db_exec(sql, params=None, fetch="none"):
    """簡易 DB 執行器 (PyMySQL)"""
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None
    except Exception as e:
        # 增加錯誤日誌
        write_to_console(f"[DB] PyMySQL 執行失敗: {sql}\nParams: {params}\nError: {e}", "ERROR")
        raise # 重新拋出錯誤，讓 Flask 的 error handler 處理
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
    """如果 user_list 沒有該 uid，就建立；有則略過/可補 name/birthday (PyMySQL)"""
    try:
        db_exec(
            "INSERT INTO user_list(uid, name, birthday) VALUES (%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE name=COALESCE(VALUES(name),name), birthday=COALESCE(VALUES(birthday),birthday)",
            (uid, name, birthday),
        )
        write_to_console(f"[DB] ensure_user ok: uid={uid}", "INFO")
    except Exception as e:
        # 錯誤已在 db_exec 中記錄，此處只需 raise
        raise

def get_conn():
    """相容舊的 get_conn() 呼叫 (PyMySQL)"""
    return pymysql.connect(**DB)


def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP:
        return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")


def insert_task_payload(task_id: str, score_id: str, data1=None, data2=None):
    """寫入任務子表 (PyMySQL)"""
    table = task_id_to_table(task_id)
    try:
        db_exec(
            f"INSERT INTO `{table}` (score_id, data1, data2) VALUES (%s, %s, %s)",
            (score_id, data1, data2),
        )
    except Exception as e:
        # 錯誤已在 db_exec 中記錄，此處只需 raise
        raise


def ensure_task(task_id: str):
    """如果 task_list 沒有該 task_id，就依 TASK_MAP 補上 (PyMySQL)"""
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
        # 錯誤已在 db_exec 中記錄，此處只需 raise
        raise

# ... (_read_score_from_result_json 和 _parse_score_from_stdout 不變) ...
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
    """在 score_list 新增一筆分數 (PyMySQL)"""
    ensure_user(uid)
    ensure_task(task_id)

    if no is None:
        row = db_exec(
            "SELECT COUNT(*) AS cnt FROM score_list WHERE uid=%s AND task_id=%s",
            (uid, task_id),
            fetch="one",
        )
        # PyMySQL 的 COUNT(*) 可能回傳 None 或 {'cnt': 0}
        no = (int(row["cnt"]) + 1) if row and row.get("cnt") is not None else 1

    score_id = uuid.uuid4().hex
    if test_date is None:
        test_date = date.today()

    try:
        db_exec(
            "INSERT INTO score_list(score_id, task_id, uid, score, no, test_date) VALUES (%s,%s,%s,%s,%s,%s)",
            (score_id, task_id, uid, int(score), no, test_date),
        )
        write_to_console(
            f"[DB] insert_score ok: uid={uid}, task_id={task_id}, score={score}, score_id={score_id}",
            "INFO",
        )
        return score_id
    except Exception as e:
        # 錯誤已在 db_exec 中記錄，此處只需 raise
        raise


# =========================
# 2) 基礎環境/日誌/靜態路由 (不變)
# =========================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

PORT = 8000
HOST = "127.0.0.1"
ROOT = Path(__file__).parent.resolve()

app = Flask(__name__, static_folder=None)
app.secret_key = secrets.token_hex(16)
CORS(app)

# ===== 角色表名（沿用你舊版）=====
TABLE_USERS = "admin_users"          # 欄位至少要有 account,password,email,level
TABLE_RECORDS = "records"            # 成績/紀錄表
TABLE_PARENT_CHILD = "parent_child"  # 家長綁小孩

# === 權限裝飾器（舊版原樣）===
def login_required(f):
    from functools import wraps
    @wraps(f)
    def _wrap(*args, **kwargs):
        if not session.get("uid"):
            return jsonify({"ok": False, "msg": "未登入"}), 401
        return f(*args, **kwargs)
    return _wrap

def require_level(min_level: int):
    from functools import wraps
    def deco(f):
        @wraps(f)
        def _wrap(*args, **kwargs):
            if session.get("level", 0) < min_level:
                return jsonify({"ok": False, "msg": "權限不足"}), 403
            return f(*args, **kwargs)
        return _wrap
    return deco

# ===== 診斷：列出所有請求的路徑（方便看到 /admin 有沒有被打到）=====
@app.before_request
def _log_path():
    try:
        print(">>> got request:", request.method, request.path, flush=True)
    except Exception:
        pass

# ===== 首頁（你新版已經有 / 指到 start.html，就略過這個；保留 admin 與 index 快捷）=====
# ===== 管理者介面（支援 /admin 與 /admin/）=====
@app.route("/admin")
@app.route("/admin/")
def admin_page():
    p = ROOT / "html" / "admin.html"
    if not p.exists():
        return f"找不到檔案：{p}", 404
    # 直接送檔，避免相對路徑問題
    return send_file(p)

# ===== /index 快捷（你新版已有 /index 與 /index.html，重名會由先宣告者生效；此處保留一份相容）=====
@app.route("/index/")
def index_shortcut_slash():
    p = ROOT / "html" / "index.html"
    if not p.exists():
        return f"找不到檔案：{p}", 404
    return send_from_directory(p.parent, p.name)

# ===== 簡單健康檢查路由（診斷用）=====
@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/admin-test")
def admin_test():
    return "admin-test-ok", 200

@app.route("/test-db")
def test_db():
    try:
        rows = query_all("SHOW TABLES;")
        return jsonify({"ok": True, "tables": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ===== 管理者登入 / 身分查詢 / 登出 =====
@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json() or {}
    account = str(data.get("account", "")).strip()
    password = str(data.get("password", "")).strip()
    if not account or not password:
        return jsonify({"ok": False, "msg": "缺少帳號或密碼"}), 400

    rows = query_all(
        f"SELECT account, email, level, account AS uid FROM {TABLE_USERS} WHERE account=%s AND password=%s LIMIT 1",
        (account, password),
    )
    if not rows:
        return jsonify({"ok": False, "msg": "帳號或密碼錯誤"}), 401

    user = rows[0]
    session["uid"] = user["uid"]          # 先用 account 當 uid
    session["account"] = user["account"]
    session["email"] = user.get("email")
    session["level"] = int(user["level"])
    return jsonify({"ok": True, "user": {"account": user["account"], "email": user["email"], "level": user["level"]}})

@app.route("/api/auth/whoami")
def whoami():
    if "uid" not in session:
        return jsonify({"ok": False, "logged_in": False})
    return jsonify({"ok": True, "logged_in": True, "user": {
        "uid": session.get("uid"),
        "account": session.get("account"),
        "email": session.get("email"),
        "level": session.get("level"),
    }})

@app.route("/api/auth/logout", methods=["POST"])
def logout_api():
    session.clear()
    return jsonify({"ok": True})

# === 主管介面（Level 3 專用） ===
@app.route("/api/admin/list")
@require_level(3)
def list_admins():
    """列出所有醫療人員帳號（無 name 欄位，就用 account 當作顯示名稱）"""
    try:
        rows = query_all("SELECT account, email, level FROM admin_users WHERE level=2")
        for r in rows:
            r["name"] = r["account"]  # 前端需要 name，就用 account 代替
        return jsonify(ok=True, admins=rows)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

@app.route("/api/admin/add", methods=["POST"])
@require_level(3)
def add_admin():
    """新增醫療人員帳號"""
    data = request.get_json() or {}
    account = str(data.get("account", "")).strip()
    password = str(data.get("password", "123456")).strip()
    email = str(data.get("email", "")).strip() or None
    if not account:
        return jsonify(ok=False, msg="缺少 account"), 400

    try:
        execute(
            "INSERT INTO admin_users (account, password, email, level) VALUES (%s, %s, %s, %s)",
            (account, password, email, 2),
        )
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

@app.route("/api/admin/update/<account>", methods=["PUT"])
@require_level(3)
def update_admin(account):
    """修改醫療人員資料（account/email/password）"""
    data = request.get_json() or {}
    new_account = data.get("account")
    email = data.get("email")
    password = data.get("password")

    fields, params = [], []
    if new_account:
        fields.append("account=%s")
        params.append(new_account)
    if email is not None:
        fields.append("email=%s")
        params.append(email)
    if password:
        fields.append("password=%s")
        params.append(password)

    if not fields:
        return jsonify(ok=False, msg="沒有要更新的欄位"), 400

    sql = f"UPDATE admin_users SET {', '.join(fields)} WHERE account=%s AND level=2"
    params.append(account)

    try:
        affected = execute(sql, tuple(params))
        return jsonify(ok=(affected == 1))
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

@app.route("/api/admin/delete/<account>", methods=["DELETE"])
@require_level(3)
def delete_admin(account):
    """刪除醫療人員帳號（用 account 當 key）"""
    try:
        affected = execute("DELETE FROM admin_users WHERE account=%s AND level=2", (account,))
        return jsonify(ok=(affected == 1))
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

# ========== Records：查詢 ==========
@app.route("/api/records", methods=["GET"])
@login_required
def list_records():
    level = session.get("level", 0)
    uid = session.get("uid")

    # 篩選條件（可選）
    child_uid = request.args.get("child_uid", "").strip()
    task_id = request.args.get("task_id", "").strip()
    limit = int(request.args.get("limit", 200))

    if level == 1:
        # 家長：只能看自己綁定的小孩
        binds = query_all(
            f"SELECT child_uid FROM {TABLE_PARENT_CHILD} WHERE parent_uid=%s",
            (uid,)
        )
        allow = [b["child_uid"] for b in binds]
        if not allow:
            return jsonify({"ok": True, "records": []})

        # 用安全方式組成 IN 子句
        in_clause = ",".join(["%s"] * len(allow))
        sql = f"SELECT * FROM {TABLE_RECORDS} WHERE child_uid IN ({in_clause})"
        params = tuple(allow)
        if child_uid:
            sql += " AND child_uid=%s"
            params += (child_uid,)
        if task_id:
            sql += " AND task_id=%s"
            params += (task_id,)
        sql += " ORDER BY id DESC LIMIT %s"
        params += (limit,)

        rows = query_all(sql, params)
        return jsonify({"ok": True, "records": rows})

    else:
        # 醫療人員 / 主管：看全部（可篩選）
        sql = f"SELECT * FROM {TABLE_RECORDS} WHERE 1=1"
        params = []
        if child_uid:
            sql += " AND child_uid=%s"
            params.append(child_uid)
        if task_id:
            sql += " AND task_id=%s"
            params.append(task_id)
        sql += " ORDER BY id DESC LIMIT %s"
        params.append(limit)
        rows = query_all(sql, tuple(params))
        return jsonify({"ok": True, "records": rows})

# ========== Records：新增（LV >= 2）==========
@app.route("/api/records", methods=["POST"])
@require_level(2)
def create_record():
    js = request.get_json() or {}
    child_uid = js.get("child_uid")
    task_id = js.get("task_id")
    score = js.get("score")
    data = js.get("data")

    if not child_uid or not task_id:
        return jsonify({"ok": False, "msg": "缺少 child_uid 或 task_id"}), 400

    sql = f"INSERT INTO {TABLE_RECORDS} (child_uid, task_id, score, data) VALUES (%s, %s, %s, %s)"
    affected = execute(sql, (child_uid, task_id, score, str(data) if data is not None else None))
    return jsonify({"ok": affected == 1})

# ========== Records：修改（LV >= 2）==========
@app.route("/api/records/<int:rid>", methods=["PUT"])
@require_level(2)
def update_record(rid):
    js = request.get_json() or {}
    fields = []
    params = []

    if "score" in js:
        fields.append("score=%s")
        params.append(js["score"])
    if "data" in js:
        fields.append("data=%s")
        params.append(str(js["data"]))

    if not fields:
        return jsonify({"ok": False, "msg": "沒有可更新欄位"}), 400

    sql = f"UPDATE {TABLE_RECORDS} SET {', '.join(fields)} WHERE id=%s"
    params.append(rid)
    affected = execute(sql, tuple(params))
    return jsonify({"ok": affected == 1})

# ========== Records：刪除（LV >= 2）==========
@app.route("/api/records/<int:rid>", methods=["DELETE"])
@require_level(2)
def delete_record(rid):
    sql = f"DELETE FROM {TABLE_RECORDS} WHERE id=%s"
    affected = execute(sql, (rid,))
    return jsonify({"ok": affected == 1})


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
    # 確保 console.txt 路徑正確
    console_path = ROOT / "console.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(console_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {level} - {message}\n")
    except Exception as e:
        print(f"寫入 console.txt 失敗: {e}") # 如果連 log 都寫不了，印在主控台

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
# init_db() # PyMySQL 不需要本地初始化
write_to_console("=== 遠端 PyMySQL 模式 ===", "INFO")
write_to_console("Flask 應用程式啟動", "INFO")
processing_tasks = {}

@app.route("/")
def home():
    return send_from_directory(ROOT / "html", "start.html")

# ... (其他靜態路由不變) ...
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
    try:
        write_to_console(
            f"[REQ] {request.method} {request.path} "
            f"CT={request.headers.get('Content-Type')} "
            f"Origin={request.headers.get('Origin')} " # 保留 Origin
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

# ... (Session 路由不變) ...
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

# /test-score 路由使用抽象函數 insert_score，保持不變
@app.route("/test-score", methods=["POST"])
def test_score():
    try:
        data = request.get_json()
        uid = data["uid"]
        task_id = data["task_id"]
        score = 3
        score_id = insert_score(uid, task_id, score)
        insert_task_payload(task_id, score_id, None, None) # 寫入子表
        return jsonify({"success": True, "score_id": score_id, "score": score})
    except Exception as e:
        # insert_score/insert_task_payload 失敗時會 raise，讓 @errorhandler 處理
        # 但如果 request.get_json() 或其他地方出錯，這裡會捕捉
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
    
    # [修正] 為靜態分析也加入 CREATE_NO_WINDOW
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW # 0x08000000

    default_kwargs = dict(
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        creationflags=creation_flags # <-- 增加此行
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


# === [修正] run_analysis_in_background (套用 Ch5-t1 彈出視窗邏輯) ===
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
            cmd = base_cmd + [uid, str(SIDE)] # <-- 使用全域 SIDE
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
            stderr_str = "" # 錯誤會直接印在執行 app.py 的主控台
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
            write_to_console(f"寫分數失敗：{e}", "ERROR") # 這裡只記錄錯誤，不中斷

        if score_id:
            try:
                # 遊戲(Ch5-t1)不需要寫入子表
                if not is_game:
                     insert_task_payload(task_id_std, score_id, None, None)
            except Exception as e:
                 # 子表寫入失敗也只記錄錯誤
                write_to_console(f"寫入子表失敗 (score_id={score_id}, task={task_id_std}): {e}", "ERROR")

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
                # "next_url": None, # 從舊版移除
            },
        }
        write_to_console(
            f"任務 {task_id} 完成：uid={uid_eff}, task={task_id_std}, score={score}, score_id={score_id}",
            "INFO",
        )

    except Exception as e:
        # 這個 except 捕捉 subprocess 執行前的錯誤，或 insert_score 失敗時的 raise
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


# === [修正] run_python_script (移除 Ch5-t1 特殊處理) ===
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
        
        # Ch5-t1 的特殊 if 區塊已被移除
        
        # 所有任務，包括 Ch5-t1，都走這個標準流程
        script_path = resolve_script_path(img_id)
        if not script_path or not script_path.exists():
            write_to_console(f"腳本不存在：{script_path}", "ERROR")
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        task_id = str(uuid.uuid4())
        stair_type = session.get("stair_type") # 取階梯參數

        processing_tasks[task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": img_id,
            "progress": 0,
        }

        t = Thread(
            target=run_analysis_in_background, args=(task_id, uid, img_id, script_path, stair_type) # 傳入 stair_type
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


# 接收階梯狀態 (不變)
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
        version_str = v['v'] if v else "無法取得版本"
        write_to_console(f"[DB] PyMySQL ping ok: {version_str}", "INFO")
        return jsonify({"ok": True, "version": version_str})
    except Exception as e:
        # db_exec 內部已記錄詳細錯誤
        return jsonify({"ok": False, "err": str(e)}), 500


@app.get("/scores")
def list_scores():
    try:
        rows = db_exec(
            "SELECT score_id, uid, task_id, score, no, test_date "
            "FROM score_list ORDER BY test_date DESC, score_id DESC LIMIT 50",
            fetch="all",
        )
        return jsonify(rows or []) # 回傳空列表而非 None
    except Exception as e:
        # db_exec 內部已記錄詳細錯誤
        return jsonify({"success": False, "error": str(e)}), 500

# === [移除] Ch5-t1 網頁串流相關程式碼 ===
# (import RaisinsGameEngine, global_game_engine, generate_collect_raisins_feed, /video_feed/Ch5-t1 路由)


# =========================
# 6) OpenCV 相機 (套用 runFortest.py 的修正)
# =========================
camera = None
camera_active = False
camera_lock = threading.Lock()

def release_camera():
    global camera, camera_active
    if camera is not None:
        try: # 增加 try-except
            camera.release()
        except Exception as e:
            write_to_console(f"釋放相機時發生錯誤: {e}", "WARN")
        camera = None
    camera_active = False

# === [修正] init_camera (使用 runFortest.py 的版本) ===
def init_camera(camera_index=TOP):
    global camera, camera_active
    try:
        release_camera()
        camera = cv2.VideoCapture(camera_index)
        
        if not camera.isOpened():
            raise Exception(f"無法開啟指定的相機索引: {camera_index}")
            
        # 設定解析度和 FPS (保留)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 120) 
        
        # 短暫讀取一幀以確認相機正常工作 (可選但建議)
        ret, _ = camera.read()
        if not ret:
             raise Exception(f"成功開啟相機 {camera_index} 但無法讀取畫面")

        camera_active = True
        write_to_console(f"相機 {camera_index} 初始化成功", "INFO")
        return True
    except Exception as e:
        print(f"相機初始化失敗: {e}") # 保留 print 給開發者看
        write_to_console(f"相機初始化失敗 (Index={camera_index}): {e}", "ERROR") 
        release_camera() # 確保失敗時釋放資源
        return False

# === [修正] get_frame (使用 runFortest.py 的版本) ===
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
        # print(f"獲取幀錯誤: {e}") # 避免洗版主控台
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
            # init_camera 內部已記錄詳細錯誤
            return jsonify({"success": False, "error": "無法開啟相機"}), 500
    except Exception as e:
        write_to_console(f"/opencv-camera/start 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/opencv-camera/frame")
def get_opencv_frame():
    try:
        if not camera_active:
            return jsonify({"success": False, "error": "相機尚未啟動"}), 400
        frame_data = get_frame()
        if frame_data is None:
            # get_frame 內部已記錄詳細錯誤
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500
        img_b64 = base64.b64encode(frame_data).decode("utf-8")
        return jsonify({"success": True, "image": img_b64})
    except Exception as e:
        write_to_console(f"/opencv-camera/frame 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/opencv-camera/capture")
def capture_opencv_photo():
    """拍照並儲存，儲存成功後立即啟動對應任務的 main.py 做評分"""
    try:
        data = request.get_json() or {}
        task_id_input = (data.get("task_id") or "").strip() # 使用 task_id_input 避免覆蓋
        uid = (data.get("uid") or "").strip() or session.get("uid", "default")

        if not task_id_input:
            return jsonify({"success": False, "error": "缺少任務 ID"}), 400

        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False, "error": "無法獲取相機畫面"}), 500

        target_dir = ROOT / "kid" / uid
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task_id_input}.jpg" # 使用輸入的 task_id (可能包含 -side/-top)
        file_path = target_dir / filename

        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if not cv2.imwrite(str(file_path), img):
             write_to_console(f"圖像儲存失敗: {file_path}", "ERROR")
             return jsonify({"success": False, "error": "圖像儲存失敗"}), 500
        else:
             write_to_console(f"圖像儲存成功: {file_path}", "INFO")

        # 使用原始的任務 ID (不含 -side/-top) 去找腳本
        # 例如，即使 task_id_input 是 "ch1-t2-side"，script_task_id 仍是 "ch1-t2"
        script_task_id = task_id_input.replace("-side", "").replace("-top", "")
        script_path = resolve_script_path(script_task_id)

        if not script_path:
            return jsonify(
                {
                    "success": True,
                    "message": f"照片已儲存({filename})，但找不到對應的 main.py ({script_task_id})（未啟動評分）",
                    "uid": uid,
                    "task_id": task_id_input, # 回傳原始輸入 ID
                    "filename": filename,
                    "analysis_started": False,
                }
            )

        bg_task_id = str(uuid.uuid4())
        processing_tasks[bg_task_id] = {
            "status": "pending",
            "uid": uid,
            "img_id": script_task_id, # 分析任務使用原始 ID
            "progress": 0,
        }
        
        # 取得階梯參數 (如果有的話)
        stair_type = session.get("stair_type")

        t = Thread(
            target=run_analysis_in_background,
            args=(bg_task_id, uid, script_task_id, script_path, stair_type), # 傳入 stair_type
        )
        t.daemon = True
        t.start()

        return jsonify(
            {
                "success": True,
                "message": f"照片已儲存({filename})並開始評分({script_task_id})",
                "uid": uid,
                "task_id": task_id_input, # 回傳原始輸入 ID
                "filename": filename,
                "analysis_started": True,
                "analysis_task_id": bg_task_id,
            }
        )

    except Exception as e:
        write_to_console(f"/opencv-camera/capture 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/opencv-camera/stop")
def stop_opencv_camera():
    try:
        release_camera()
        return jsonify({"success": True, "message": "相機已關閉"})
    except Exception as e:
        write_to_console(f"/opencv-camera/stop 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# 7) 啟動 (不變)
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
        # 隱藏 Flask 啟動訊息
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None
        # 啟動伺服器
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        write_to_console("接收到中斷信號，正在關閉應用程式", "INFO")
    except Exception as e:
        write_to_console(f"應用程式運行時發生錯誤：{str(e)}", "ERROR")
    finally:
        write_to_console("Flask 應用程式已關閉", "INFO")