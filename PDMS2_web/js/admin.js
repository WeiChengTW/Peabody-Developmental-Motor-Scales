(function () {
  const LS_KEY = 'admin-demo-records';

  // ===== 使用者權限 =====
  let userLevel = 0; // 1: 家長(唯讀), 2: 醫療人員, 3: 主管

  async function ensureAuth() {
    try {
      const r = await fetch('/api/auth/whoami');
      const js = await r.json();
      if (!js.ok || !js.logged_in) {
        location.href = '/html/login.html';
        return false;
      }
      userLevel = Number(js.user.level || 0);

      // Level 1：家長 → 隱藏「新增」按鈕
      if (userLevel === 1) {
        const addBtn = document.querySelector('[data-action="new"]');
        if (addBtn) addBtn.style.display = 'none';
      }
      return true;
    } catch (e) {
      console.error('whoami error:', e);
      location.href = '/html/login.html';
      return false;
    }
  }

  // ============ 資料轉換 ============
  function toCanonical(input) {
    const arr = Array.isArray(input)
      ? input
      : (input && typeof input === 'object'
        ? (input.data || input.results || input.records || [input])
        : []);
    return arr.map((raw, idx) => {
      const entries = Object.entries(raw || {});
      const norm = {};
      for (const [k, v] of entries) {
        const k2 = String(k).trim();
        const k3 = k2.toLowerCase();
        norm[k2] = v;
        norm[k3] = v;
      }
      const id =
        norm['userId'] || norm['userid'] || norm['uid'] || norm['user_id'] ||
        norm['id'] || `u-${idx + 1}`;
      const name =
        norm['name'] || norm['username'] || norm['nickname'] ||
        `使用者${idx + 1}`;
      const tasks = {};
      for (const [origKey, val] of entries) {
        const keyTrim = String(origKey).trim();
        const keyLow = keyTrim.toLowerCase();
        const isMetaKey =
          keyLow === 'name' ||
          keyLow === 'userid' || keyLow === 'user_id' || keyLow === 'uid' ||
          keyLow === 'id';
        if (isMetaKey) continue;
        if (val && typeof val === 'object' && ('score' in val)) {
          tasks[keyTrim] = {
            score: Number(val.score ?? 0),
            updatedAt: val.updatedAt || val.time || null
          };
        } else if (typeof val === 'number') {
          tasks[keyTrim] = { score: Number(val), updatedAt: null };
        }
      }
      return { id: String(id), userId: String(id), name: String(name), tasks };
    });
  }

  function fromCanonical(list) {
    return list.map(u => {
      const obj = { name: u.name, userId: u.userId };
      for (const [lvl, info] of Object.entries(u.tasks)) {
        obj[lvl] = { score: info.score, updatedAt: info.updatedAt || new Date().toISOString() };
      }
      return obj;
    });
  }

  async function loadInitial() {
    try {
      const ls = localStorage.getItem(LS_KEY);
      if (ls) return JSON.parse(ls);
    } catch { }
    try {
      const res = await fetch('./data/result_copy.json');
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const j = await res.json();
      return toCanonical(j);
    } catch (err) {
      console.warn('[load] 無法載入 result copy.json：', err);
      const $tbody = document.querySelector('[data-role="tbody"]');
      if ($tbody) $tbody.innerHTML =
        `<tr><td colspan="5" class="empty">讀取失敗：${String(err).replace(/[<>&]/g, '')}</td></tr>`;
      return [];
    }
  }
  function saveLS(list) { try { localStorage.setItem(LS_KEY, JSON.stringify(list)); } catch { } }

  // 狀態與 DOM
  const state = { list: [], sortKey: 'userId', sortAsc: true, selUserId: '', selLevel: '' };

  const $tbody = document.querySelector('[data-role="tbody"]');
  const $selName = document.querySelector('[data-role="sel-name"]');
  const $selLvl = document.querySelector('[data-role="sel-level"]');
  const $dialog = document.querySelector('[data-role="dialog"]');
  const $f_userId = $dialog.querySelector('[data-field="userId"]');
  const $f_name = $dialog.querySelector('[data-field="name"]');
  const $f_level = $dialog.querySelector('[data-field="level"]');
  const $f_score = $dialog.querySelector('[data-field="score"]');
  const $f_scoreText = $dialog.querySelector('[data-field="score-text"]');

  function rebuildSelects() {
    const seenU = new Set();
    const optsU = ['<option value="">（請選擇姓名）</option>'];
    state.list.forEach(u => {
      if (seenU.has(u.userId)) return;
      seenU.add(u.userId);
      optsU.push(`<option value="${u.userId}">${u.userId} - ${u.name}</option>`);
    });
    $selName.innerHTML = optsU.join('');
    if (state.selUserId) $selName.value = state.selUserId;

    const seenL = new Set();
    const optsL = ['<option value="">（請選擇關卡）</option>'];
    state.list.forEach(u => {
      Object.keys(u.tasks).forEach(lvl => {
        if (!seenL.has(lvl)) { seenL.add(lvl); optsL.push(`<option value="${lvl}">${lvl}</option>`); }
      });
    });
    $selLvl.innerHTML = optsL.join('');
    if (state.selLevel) $selLvl.value = state.selLevel;
  }

  function buildRows() {
    const rows = [];
    state.list.forEach(u => {
      if (state.selUserId && u.userId !== state.selUserId) return;
      for (const [lvl, info] of Object.entries(u.tasks)) {
        if (state.selLevel && lvl !== state.selLevel) continue;
        rows.push({
          rowId: `${u.userId}__${lvl}`,
          userId: u.userId,
          name: u.name,
          level: lvl,
          score: Number(info.score ?? 0)
        });
      }
    });
    rows.sort((a, b) => {
      const A = a[state.sortKey], B = b[state.sortKey];
      let cmp = 0;
      if (typeof A === 'number' && typeof B === 'number') cmp = A - B;
      else cmp = String(A).localeCompare(String(B));
      return state.sortAsc ? cmp : -cmp;
    });
    return rows;
  }

  function render() {
    const rows = buildRows();
    if (rows.length === 0) {
      $tbody.innerHTML = '<tr><td colspan="5" class="empty">沒有符合條件的資料</td></tr>';
      return;
    }
    $tbody.innerHTML = rows.map(r => {
      // Level 1 沒有編輯/刪除按鈕
      const opTd = (userLevel >= 2)
        ? `<button class="btn btn-edit" data-action="edit">✏️ 修改</button>
           <button class="btn btn-delete" data-action="del">🗑️ 刪除</button>`
        : '';
      return `
        <tr data-key="${r.rowId}">
          <td class="ta-right">${opTd}</td>
          <td>${r.userId}</td>
          <td>${r.name}</td>
          <td>${r.level}</td>
          <td>${r.score}</td>
        </tr>`;
    }).join('');
  }

  $selName.addEventListener('change', () => { state.selUserId = $selName.value || ''; render(); });
  $selLvl.addEventListener('change', () => { state.selLevel = $selLvl.value || ''; render(); });

  document.addEventListener('click', (e) => {
    const a = e.target.closest('[data-action]'); if (!a) return;
    const act = a.dataset.action;

    if (act === 'reset') {
      if (userLevel < 2) {
        alert('⚠️ 只有醫療人員可操作此功能');
        return;
      }

      state.selUserId = '';
      state.selLevel = '';
      $selName.value = '';
      $selLvl.value = '';
      $tbody.innerHTML = '<tr><td colspan="5" class="empty">（已清空）</td></tr>';
    }


    if (act === 'new') {
      if (userLevel < 2) return; // 前端再保護一次
      $f_userId.value = ''; $f_name.value = ''; $f_level.value = ''; $f_score.value = '0'; $f_scoreText.textContent = '0';
      $dialog.showModal(); $dialog.dataset.mode = 'create'; $dialog.dataset.key = '';
    }
  });

  const $tbodyEl = document.querySelector('[data-role="tbody"]');
  $tbodyEl.addEventListener('click', (e) => {
    const btn = e.target.closest('button'); if (!btn) return;
    const tr = e.target.closest('tr'); const key = tr?.getAttribute('data-key'); if (!key) return;
    const [userId, level] = key.split('__');
    const user = state.list.find(u => u.userId === userId);

    if (btn.dataset.action === 'del') {
      if (userLevel < 2) return;
      if (!confirm('確定要刪除此分數紀錄？')) return;
      if (user && user.tasks[level]) { delete user.tasks[level]; saveLS(state.list); rebuildSelects(); render(); }
      return;
    }

    if (btn.dataset.action === 'edit') {
      if (userLevel < 2) return;
      $f_userId.value = userId;
      $f_name.value = user?.name || '';
      $f_level.value = level;
      const score = user?.tasks?.[level]?.score ?? 0;
      $f_score.value = String(score);
      $f_scoreText.textContent = String(score);
      $dialog.showModal(); $dialog.dataset.mode = 'edit'; $dialog.dataset.key = key;
    }
  });

  $f_score.addEventListener('input', () => { $f_scoreText.textContent = $f_score.value; });

  $dialog.querySelector('[data-action="save"]').addEventListener('click', (ev) => {
    ev.preventDefault();
    if (userLevel < 2) return;

    const userId = $f_userId.value.trim();
    const name = $f_name.value.trim();
    const level = $f_level.value.trim();
    const score = Number($f_score.value);
    if (!userId || !name || !level) { alert('請填寫 uid / 姓名 / 關卡'); return; }

    let user = state.list.find(u => u.userId === userId);
    if (!user) {
      user = { id: userId, userId, name, tasks: {} };
      state.list.push(user);
    } else {
      user.name = name;
    }
    user.tasks[level] = { score, updatedAt: new Date().toISOString() };

    saveLS(state.list); rebuildSelects(); render(); $dialog.close();
  });

  document.querySelectorAll('[data-sort]').forEach(el => {
    el.addEventListener('click', () => {
      const key = el.dataset.sort;
      if (state.sortKey === key) { state.sortAsc = !state.sortAsc; }
      else { state.sortKey = key; state.sortAsc = (key === 'userId'); }
      render();
    });
  });

  (async function init() {
    const ok = await ensureAuth(); // 先確認登入 & 取得 userLevel
    if (!ok) return;

    state.list = await loadInitial();
    saveLS(state.list);
    rebuildSelects();
    render();
  })();
})();
