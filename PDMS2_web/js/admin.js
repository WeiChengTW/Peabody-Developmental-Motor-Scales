(function () {
  // =========================
  // 使用者權限
  // =========================
  let userLevel = 0; // 1: 家長(唯讀), 2: 醫療人員, 3: 主管

  async function ensureAuth() {
    try {
      const r = await fetch('/api/auth/whoami', { credentials: 'include' });
      if (!r.ok) throw new Error('whoami not ok');
      const js = await r.json();
      if (!js.ok || !js.logged_in) {
        location.href = '/html/admin_login.html';
        return false;
      }
      userLevel = Number((js.user && js.user.level) || 0);
    } catch (e) {
      console.warn('whoami 失敗，使用 Demo 權限 (level=3)：', e);
      userLevel = 3;
    }
    if (userLevel !== 3) {
      const addBtn = document.querySelector('[data-action="new"]');
      if (addBtn) addBtn.style.display = 'none';
    }
    return true;
  }

  // =========================
  // 狀態
  // =========================
  const state = {
    rows: [],          // /scores 回來的列：{row_key, uid, name, task_id, test_date, score, result_img_path, ...}
    users: [],         // /users 回來的 uid 列表（或物件）
    sortKey: 'test_date',
    sortAsc: false,
    selUserId: '',
    selLevel: '',
    allLevels: []      // 任務清單（Ch1-t1, Ch1-t2, ...）
  };

  // =========================
  // DOM
  // =========================
  const $tbody = document.querySelector('[data-role="tbody"]');
  const $selName = document.querySelector('[data-role="sel-name"]');
  const $selLvl = document.querySelector('[data-role="sel-level"]');
  const $dialog = document.querySelector('[data-role="dialog"]');
  const $f_userId = $dialog?.querySelector('[data-field="userId"]');
  const $f_name = $dialog?.querySelector('[data-field="name"]');
  const $f_level = $dialog?.querySelector('[data-field="level"]');
  const $f_score = $dialog?.querySelector('[data-field="score"]');
  const $f_scoreText = $dialog?.querySelector('[data-field="score-text"]');
  const $f_testDate = $dialog?.querySelector('[data-field="test_date"]');

  // =========================
  // 小工具：檔案型態與結果欄位
  // =========================
  function isImg(p) {
    return /\.(jpg|jpeg|png|webp)$/i.test(p || '');
  }
  function isVid(p) {
    return /\.(mp4|webm)$/i.test(p || '');
  }
  function makeResultCell(path) {
    if (!path) return '<td>—</td>';
    const clean = String(path).replace(/^\//, '');
    const url = '/artifact/' + clean;
    const thumb = isImg(path) ? url : '/images/video-icon.svg';
    return `
      <td>
        <img class="thumb" src="${thumb}" alt="thumb"
             data-action="preview" data-path="${clean}">
        <button class="btn btn-light" data-action="preview" data-path="${clean}" style="margin-left:8px">
          查看
        </button>
      </td>`;
  }

  // =========================
  // 載入資料
  // =========================
  async function reloadScores() {
    try {
      const r = await fetch('/scores', { credentials: 'include' });
      if (!r.ok) {
        console.warn('/scores 不是 200：', r.status, r.statusText);
        state.rows = [];
        return;
      }
      const rows = await r.json();
      state.rows = Array.isArray(rows) ? rows : [];

      // 若後端沒給 tasks，就從 rows 自己蒐集 task_id
      if (!Array.isArray(state.allLevels) || state.allLevels.length === 0) {
        const set = new Set();
        for (const x of state.rows) {
          if (x && x.task_id) set.add(String(x.task_id));
        }
        state.allLevels = Array.from(set).sort();
      }
    } catch (e) {
      console.warn('reloadScores 失敗：', e);
      state.rows = [];
    }
  }

  async function reloadUsers() {
    try {
      const r = await fetch('/users', { credentials: 'include' });
      if (!r.ok) {
        state.users = [];
        return;
      }
      const js = await r.json();
      // 允許 /users 回傳 ["U1","U2"] 或 [{uid:"U1",name:"小明"}, ...]
      state.users = (js && js.ok && Array.isArray(js.users)) ? js.users : [];
    } catch {
      state.users = [];
    }
  }

  // =========================
  // 重建下拉選單
  // =========================
  function rebuildSelects() {
    // ---- 姓名/uid 下拉 ----
    const optsU = ['<option value="">（請選擇姓名）</option>'];

    if (Array.isArray(state.users) && state.users.length > 0) {
      state.users.forEach(u => {
        let uid = '';
        let name = '';
        if (typeof u === 'string') {
          uid = u;
        } else if (u && typeof u === 'object') {
          uid = u.uid || '';
          name = u.name || '';
        }
        if (!uid) return;
        const label = name ? `${name} (${uid})` : uid;
        optsU.push(`<option value="${uid}">${label}</option>`);
      });
    } else {
      // 如果 /users 沒資料，就從 score_list rows 蒐集 uid
      const seen = new Set();
      (state.rows || []).forEach(r => {
        const uid = String(r.uid || '').trim();
        if (!uid || seen.has(uid)) return;
        seen.add(uid);
        const label = r.name ? `${r.name} (${uid})` : uid;
        optsU.push(`<option value="${uid}">${label}</option>`);
      });
    }

    if ($selName) {
      $selName.innerHTML = optsU.join('');
      if (state.selUserId) $selName.value = state.selUserId;
    }

    // ---- 關卡下拉 ----
    const preferredOrder = [
      'Ch1-t1', 'Ch1-t2', 'Ch1-t3',
      'Ch2-t1', 'Ch2-t2', 'Ch2-t3', 'Ch2-t4', 'Ch2-t5', 'Ch2-t6',
      'Ch3-t1', 'Ch3-t2', 'Ch4-t1', 'Ch4-t2', 'Ch5-t1'
    ];
    let levels = [];
    if (Array.isArray(state.allLevels) && state.allLevels.length > 0) {
      const setAll = new Set(state.allLevels);
      levels = preferredOrder.filter(x => setAll.has(x));
      state.allLevels.forEach(x => { if (!levels.includes(x)) levels.push(x); });
    } else {
      levels = preferredOrder.slice();
    }

    const optsL = ['<option value="">（請選擇關卡）</option>'];
    levels.forEach(lvl => optsL.push(`<option value="${lvl}">${lvl}</option>`));
    if ($selLvl) {
      $selLvl.innerHTML = optsL.join('');
      if (state.selLevel) $selLvl.value = state.selLevel;
    }
  }

  // =========================
  // 資料整理與排序
  // =========================
  function buildRows() {
    let rows = state.rows.slice();

    if (state.selUserId) {
      rows = rows.filter(r => r.uid === state.selUserId);
    }
    if (state.selLevel) {
      rows = rows.filter(r => r.task_id === state.selLevel);
    }

    rows.sort((a, b) => {
      const A = a[state.sortKey];
      const B = b[state.sortKey];
      let cmp = 0;
      if (state.sortKey === 'score') {
        cmp = Number(A || 0) - Number(B || 0);
      } else {
        cmp = String(A || '').localeCompare(String(B || ''));
      }
      return state.sortAsc ? cmp : -cmp;
    });
    return rows;
  }

  // =========================
  // 渲染表格
  // =========================
  function render() {
    const rows = buildRows();
    if (!rows.length) {
      const msg = '沒有符合條件的資料';
      $tbody.innerHTML = `<tr><td colspan="7" class="empty">${msg}</td></tr>`;
      return;
    }

    $tbody.innerHTML = rows.map(r => {
      const opTd = (userLevel === 3)
        ? `<button class="btn btn-edit" data-action="edit">✏️ 修改</button>
           <button class="btn btn-delete" data-action="del">🗑️ 刪除</button>`
        : '';

      return `
        <tr data-key="${r.row_key}">
          <td class="ta-right">${opTd}</td>
          <td>${r.uid}</td>
          <td>${r.name || ''}</td>
          <td>${r.task_id}</td>
          <td>${r.score ?? ''}</td>
          <td>${r.test_date ?? ''}</td>
          ${makeResultCell(r.result_img_path)}
        </tr>`;
    }).join('');
  }

  // =========================
  // 篩選事件
  // =========================
  if ($selName) {
    $selName.addEventListener('change', () => {
      state.selUserId = $selName.value || '';
      render();
    });
  }
  if ($selLvl) {
    $selLvl.addEventListener('change', () => {
      state.selLevel = $selLvl.value || '';
      render();
    });
  }

  // =========================
  // 工具列事件（清空 / 新增）
  // =========================
  document.addEventListener('click', (e) => {
    const a = e.target.closest?.('[data-action]');
    if (!a) return;
    const act = a.dataset.action;

    if (act === 'reset') {
      if (userLevel < 2) {
        alert('⚠️ 只有醫療人員可操作此功能');
        return;
      }
      state.selUserId = '';
      state.selLevel = '';
      if ($selName) $selName.value = '';
      if ($selLvl) $selLvl.value = '';
      if ($tbody) $tbody.innerHTML = '<tr><td colspan="7" class="empty">（已清空）</td></tr>';
    }

    if (act === 'new') {
      if (userLevel < 3) return;
      if (!$dialog) return;

      if ($f_userId) $f_userId.value = '';
      if ($f_name) $f_name.value = '';
      if ($f_level) $f_level.value = '';
      if ($f_score) $f_score.value = '0';
      if ($f_scoreText) $f_scoreText.textContent = '0';
      if ($f_testDate) $f_testDate.value = '';

      $dialog.showModal();
      $dialog.dataset.mode = 'create';
      $dialog.dataset.key = '';
    }
  });

  // =========================
  // 表格內 Edit / Delete
  // =========================
  const $tbodyEl = document.querySelector('[data-role="tbody"]');
  if ($tbodyEl) {
    $tbodyEl.addEventListener('click', async (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;

      const tr = e.target.closest('tr');
      const rowKey = tr?.getAttribute('data-key');
      if (!rowKey) return;

      // ---- 刪除 ----
      if (btn.dataset.action === 'del') {
        if (userLevel < 2) return;
        if (!confirm('確定要刪除此分數紀錄？')) return;

        try {
          const r = await fetch(`/scores?row_key=${encodeURIComponent(rowKey)}`, {
            method: 'DELETE',
            credentials: 'include'
          });
          const js = await r.json().catch(() => ({}));
          if (!r.ok || !js.ok) {
            alert(js.msg || `刪除失敗 (HTTP ${r.status})`);
            return;
          }
          await reloadScores();
          rebuildSelects();
          render();
        } catch (err) {
          alert('刪除失敗：' + err);
        }
        return;
      }

      // ---- 編輯 ----
      if (btn.dataset.action === 'edit') {
        if (userLevel < 3) return;
        if (!$dialog) return;

        const row = state.rows.find(x => x.row_key === rowKey);
        if (!row) return;

        if ($f_userId) $f_userId.value = row.uid || '';
        if ($f_name)  $f_name.value  = row.name || '';
        if ($f_level) $f_level.value = row.task_id || '';
        if ($f_score) $f_score.value = String(row.score ?? 0);
        if ($f_scoreText) $f_scoreText.textContent = String(row.score ?? 0);
        if ($f_testDate) $f_testDate.value = (row.test_date || '').slice(0, 10);

        $dialog.showModal();
        $dialog.dataset.mode = 'edit';
        $dialog.dataset.key = rowKey;
      }
    });
  }

  // =========================
  // 縮圖 / 查看預覽
  // =========================
  document.addEventListener('click', (e) => {
    const el = e.target.closest?.('[data-action="preview"]');
    if (!el) return;

    let path = el.dataset.path || '';
    if (!path) return;

    // 清理路徑
    path = path.replace(/^[\\/]+/, '').replace(/\\/g, '/');

    const win = window.open('', '_blank');
    if (!win) return;

    // ch1-t2 / ch1-t3：side / top 四張圖
    const mST = path.match(/^(.*)-(side|top)(\.[^.]+)$/i);
    let boxesHtml = '';

    if (mST) {
      const base = mST[1];   // "kid/cc22/ch1-t3"
      const ext  = mST[3];   // ".jpg"

      const sideOrig = '/artifact/' + `${base}-side${ext}`;
      const sideRes  = '/artifact/' + `${base}-side_result${ext}`;
      const topOrig  = '/artifact/' + `${base}-top${ext}`;
      const topRes   = '/artifact/' + `${base}-top_result${ext}`;

      boxesHtml = `
        <div class="box">
          <h2>Side 原始圖</h2>
          <img src="${sideOrig}" alt="side-original">
        </div>
        <div class="box">
          <h2>Side 結果圖</h2>
          <img src="${sideRes}" alt="side-result">
        </div>
        <div class="box">
          <h2>Top 原始圖</h2>
          <img src="${topOrig}" alt="top-original">
        </div>
        <div class="box">
          <h2>Top 結果圖</h2>
          <img src="${topRes}" alt="top-result">
        </div>`;
    } else {
      // 一般任務：原圖 + 結果圖
      const origUrl = '/artifact/' + path;
      let resultPath;
      const m = path.match(/^(.*)(\.[^.]+)$/);
      if (m) {
        resultPath = m[1] + '_result' + m[2];
      } else {
        resultPath = path + '_result';
      }
      const resultUrl = '/artifact/' + resultPath;

      boxesHtml = `
        <div class="box">
          <h2>原始圖</h2>
          <img src="${origUrl}" alt="original">
        </div>
        <div class="box">
          <h2>結果圖</h2>
          <img src="${resultUrl}" alt="result">
        </div>`;
    }

    const html = `
<!doctype html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <title>結果預覽</title>
  <style>
    body {
      margin: 0;
      padding: 20px;
      background: #000;
      color: #fff;
      font-family: Arial, Helvetica, sans-serif;
      text-align: center;
    }
    .container {
      width: 100%;
      display: flex;
      justify-content: space-evenly;
      align-items: flex-start;
      gap: 20px;
      flex-wrap: wrap;
    }
    .box {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 45%;
      margin-bottom: 20px;
    }
    .box h2 {
      margin-bottom: 10px;
      font-size: 18px;
      font-weight: 500;
    }
    img {
      width: 100%;
      max-height: 80vh;
      object-fit: contain;
      background: #222;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <div class="container">
    ${boxesHtml}
  </div>
</body>
</html>`;
    win.document.write(html);
    win.document.close();
  });

  // =========================
  // Range 文字同步
  // =========================
  if ($f_score) {
    $f_score.addEventListener('input', () => {
      if ($f_scoreText) $f_scoreText.textContent = $f_score.value;
    });
  }

  // =========================
  // 儲存（新增 / 修改）
  // =========================
  const $saveBtn = $dialog?.querySelector('[data-action="save"]');
  if ($saveBtn) {
    $saveBtn.addEventListener('click', async (ev) => {
      ev.preventDefault();
      if (userLevel < 3) return;

      const uid = ($f_userId?.value || '').trim();
      const name = ($f_name?.value || '').trim();
      const task_id = ($f_level?.value || '').trim();
      const score = Number($f_score?.value || 0);
      const test_date = ($f_testDate?.value || '').trim();

      if (!uid || !task_id || !test_date) {
        alert('請填寫 uid / 關卡 / 測試日期');
        return;
      }

      try {
        const payload = { uid, task_id, score, test_date };
        const mode = $dialog?.dataset.mode || 'create';
        const oldKey = $dialog?.dataset.key || '';
        if (mode === 'edit' && oldKey) {
          // 若後端要知道原來的 composite key，可用 row_key_old
          payload.row_key_old = oldKey;
        }

        const r = await fetch('/scores/upsert', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(payload)
        });
        const js = await r.json().catch(() => ({}));
        if (!r.ok || !js.ok) {
          alert(js.msg || `儲存失敗 (HTTP ${r.status})`);
          return;
        }

        await reloadScores();
        // 若填了 name，就寫回快取資料（user_list 也可以另外更新）
        if (name) {
          state.rows
            .filter(x => x.uid === uid)
            .forEach(x => { if (!x.name) x.name = name; });
        }
        rebuildSelects();
        render();
        $dialog.close();
      } catch (e) {
        alert('儲存失敗：' + e);
      }
    });
  }

  // =========================
  // 排序
  // =========================
  document.querySelectorAll('[data-sort]').forEach(el => {
    el.addEventListener('click', () => {
      const key = el.dataset.sort; // uid | name | task_id | score | test_date
      if (state.sortKey === key) {
        state.sortAsc = !state.sortAsc;
      } else {
        state.sortKey = key;
        state.sortAsc = !(key === 'test_date'); // 預設 test_date 由新到舊
      }
      render();
    });
  });

  // =========================
  // 初始化
  // =========================
  (async function init() {
    const ok = await ensureAuth();
    if (!ok) return;

    // 先拿任務列表（若有）
    try {
      const r = await fetch('/tasks', { credentials: 'include' });
      if (r.ok) {
        const js = await r.json();
        if (js.ok && Array.isArray(js.tasks)) state.allLevels = js.tasks;
      }
    } catch {
      // ignore
    }

    await reloadUsers();
    await reloadScores();

    rebuildSelects();
    render();
  })();

})();
