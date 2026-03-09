(function () {
  let userLevel = 0; 

  async function ensureAuth() {
    try {
      const r = await fetch('/api/auth/whoami', { credentials: 'include' });
      const js = await r.json().catch(() => ({})); 
      if (!r.ok || !js.logged_in) {
        location.href = '/html/admin_login.html';
        return false;
      }
      userLevel = Number((js.user && js.user.level) || 0);
      
      const roleDisplay = document.getElementById('user-role-display');
      if (roleDisplay) {
        const userName = js.user.name || js.user.account;
        const labels = { 1: "家長身分", 2: "醫療人員", 3: "系統主管" };
        roleDisplay.innerHTML = `當前身分：${labels[userLevel] || '未確認'} ｜ 登入帳號：${userName} <button class="text-link" style="margin-left: 12px; font-size: 13px;" data-action="edit-profile">✏️ 修改帳密</button>`;
      }
    } catch (e) {
      location.href = '/html/admin_login.html';
      return false;
    }
    
    if (userLevel < 2) {
      const addBtn = document.getElementById('btn-add-record');
      if (addBtn) addBtn.style.display = 'none';
    }
    return true;
  }

  document.getElementById('btn-logout')?.addEventListener('click', async () => {
    if (confirm('確定要登出系統嗎？')) {
      await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
      location.href = '/html/admin_login.html';
    }
  });

  const state = { rows: [], users: [], sortKey: 'test_date', sortAsc: false, selUserId: '', selLevel: '' };
  const $tbody = document.querySelector('[data-role="tbody"]');
  const $selName = document.querySelector('[data-role="sel-name"]');
  const $selLvl = document.querySelector('[data-role="sel-level"]');
  const $dialog = document.getElementById('record-dialog');
  const $imgDialog = document.getElementById('image-dialog');
  
  const $f_userId = $dialog?.querySelector('[data-field="userId"]');
  const $f_name = $dialog?.querySelector('[data-field="name"]');
  const $f_birthday = $dialog?.querySelector('[data-field="birthday"]');
  const $f_level = $dialog?.querySelector('[data-field="level"]');
  const $f_score = $dialog?.querySelector('[data-field="score"]');
  const $f_testDate = $dialog?.querySelector('[data-field="test_date"]');
  const $scoreDisplay = document.getElementById('score-display');

  if ($f_score && $scoreDisplay) {
    $f_score.addEventListener('input', () => { $scoreDisplay.textContent = $f_score.value; });
  }

  async function reloadScores() {
    try {
      const r = await fetch('/scores', { credentials: 'include' });
      state.rows = (await r.json()) || [];
      render();
    } catch (e) { console.error("資料載入失敗", e); }
  }

  async function reloadUsers() {
    try {
      const r = await fetch('/users', { credentials: 'include' });
      const js = await r.json();
      state.users = (js && js.ok) ? js.users : [];
      rebuildSelects();
    } catch { state.users = []; }
  }

  function rebuildSelects() {
    const optsU = ['<option value="">（請選擇受測者）</option>'];
    state.users.forEach(uid => optsU.push(`<option value="${uid}">${uid}</option>`));
    if ($selName) $selName.innerHTML = optsU.join('');

    const preferredOrder = ['Ch1-t1', 'Ch1-t2', 'Ch1-t3', 'Ch1-t4', 'Ch2-t1', 'Ch2-t2', 'Ch2-t3', 'Ch2-t4', 'Ch2-t5', 'Ch2-t6', 'Ch3-t1', 'Ch3-t2', 'Ch4-t1', 'Ch4-t2', 'Ch5-t1'];
    const taskNames = { "Ch1-t1": "串珠子", "Ch1-t2": "蓋金字塔", "Ch1-t3": "蓋階梯", "Ch1-t4": "疊牆壁", "Ch2-t1": "畫圓", "Ch2-t2": "畫正方形", "Ch2-t3": "畫十字", "Ch2-t4": "畫直線", "Ch2-t5": "著色", "Ch2-t6": "連連看", "Ch3-t1": "剪圓形", "Ch3-t2": "剪正方形", "Ch4-t1": "對摺一次", "Ch4-t2": "對摺兩次", "Ch5-t1": "撿葡萄乾" };
    
    if ($selLvl) {
      const optsL = ['<option value="">（請選擇測驗項目）</option>'];
      preferredOrder.forEach(lvl => optsL.push(`<option value="${lvl}">${lvl}</option>`));
      $selLvl.innerHTML = optsL.join('');
    }

    if ($f_level) {
      const optsDialog = ['<option value="">（僅建立基本資料請留空）</option>'];
      preferredOrder.forEach(lvl => optsDialog.push(`<option value="${lvl}">${lvl} (${taskNames[lvl] || ''})</option>`));
      $f_level.innerHTML = optsDialog.join('');
    }
  }

  function render() {
    if (!$tbody) return;
    const filtered = state.rows.filter(r => 
      (!state.selUserId || r.uid === state.selUserId) && 
      (!state.selLevel || r.task_id === state.selLevel)
    );

    if (filtered.length === 0) {
      $tbody.innerHTML = '<tr><td colspan="7" class="empty">目前沒有符合條件的測驗紀錄</td></tr>';
      return;
    }
    
    $tbody.innerHTML = filtered.map(r => {
      const opTd = (userLevel === 3) 
        ? `<div style="display:flex; gap:8px;">
             <button class="btn btn-sm" data-action="edit">編輯</button>
             <button class="btn btn-sm danger" data-action="del">刪除</button>
           </div>` 
        : '<span style="color:#CCC;">無操作權限</span>';
        
      const imgBtn = `<button class="btn btn-sm" style="background-color: #E3F2FD; color: #1565C0; border-color: #BBDEFB;" 
                        data-action="view-images" 
                        data-raw="${r.raw_image_url || ''}" 
                        data-result="${r.result_image_url || ''}" 
                        data-title="${r.uid} - ${r.task_id}">
                        👀 查看結果圖
                      </button>`;

      return `
        <tr data-key="${r.row_key}">
          <td>${opTd}</td>
          <td style="font-weight:700; color:#5C4E4E;">${r.uid}</td>
          <td>${r.name || '未填寫'}</td>
          <td>${r.task_id || ''}</td>
          <td style="font-weight:700; color:#48CAE4;">${r.score ?? ''}</td>
          <td>${imgBtn}</td>
          <td style="color:var(--text-muted);">${r.test_date ?? ''}</td>
        </tr>`;
    }).join('');
  }

  // 統一處理畫面上的點擊事件
  document.addEventListener('click', async (e) => {
    
    // 🆕 修改密碼的眼睛圖示功能 (防呆強化版)
    if (e.target.id === 'toggle-profile-pwd') {
      const pwdInput = document.getElementById('profile-pwd');
      if (pwdInput) {
        const type = pwdInput.getAttribute('type') === 'password' ? 'text' : 'password';
        pwdInput.setAttribute('type', type);
        e.target.textContent = type === 'password' ? '👁️' : '🙈';
      }
      return;
    }

    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const act = btn.dataset.action;

    if (act === 'view-images') {
      const rawUrl = btn.dataset.raw;
      const resultUrl = btn.dataset.result;
      
      document.getElementById('img-dialog-title').textContent = btn.dataset.title;
      
      const rawImg = document.getElementById('raw-image');
      const resultImg = document.getElementById('result-image');
      const rawErr = document.getElementById('raw-image-error');
      const resultErr = document.getElementById('result-image-error');

      rawImg.style.display = 'inline-block';
      resultImg.style.display = 'inline-block';
      rawErr.style.display = 'none';
      resultErr.style.display = 'none';

      rawImg.src = rawUrl;
      resultImg.src = resultUrl;

      rawImg.onerror = () => { rawImg.style.display = 'none'; rawErr.style.display = 'block'; };
      resultImg.onerror = () => { resultImg.style.display = 'none'; resultErr.style.display = 'block'; };

      const $imgDialog = document.getElementById('image-dialog');
      if($imgDialog) $imgDialog.showModal();
    }

    // 🆕 呼叫修改密碼對話框 (防呆強化版)
    if (act === 'edit-profile') {
      const $profileDialog = document.getElementById('profile-dialog');
      
      // 如果找不到 HTML，直接彈出警告！
      if (!$profileDialog) {
        alert('❌ 系統提示：找不到彈出視窗的程式碼！請確認是否有將 <dialog id="profile-dialog"> 的區塊存進 admin.html 中，並且已經強制重新整理網頁。');
        return;
      }

      const accInput = document.getElementById('profile-acc');
      const pwdInput = document.getElementById('profile-pwd');
      const eyeBtn = document.getElementById('toggle-profile-pwd');

      if (accInput) accInput.value = '';
      if (pwdInput) {
        pwdInput.value = '';
        pwdInput.setAttribute('type', 'password');
      }
      if (eyeBtn) eyeBtn.textContent = '👁️';
      
      $profileDialog.showModal();
    }

    if (act === 'save-profile') {
      e.preventDefault();
      const newAcc = document.getElementById('profile-acc').value.trim();
      const newPwd = document.getElementById('profile-pwd').value.trim();
      
      if (!newAcc || !newPwd) return alert('系統提示：新帳號與新密碼不可為空');

      const btnSave = e.target;
      const originalText = btnSave.textContent;
      btnSave.textContent = '處理中...';
      btnSave.disabled = true;

      try {
        const r = await fetch('/api/auth/update_profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_account: newAcc, new_password: newPwd })
        });
        const js = await r.json().catch(() => ({}));
        
        if (js.ok) {
          alert('✅ 修改成功！請使用新帳號密碼重新登入。');
          location.href = '/html/admin_login.html';
        } else {
          alert('❌ 修改失敗：' + (js.msg || '未知錯誤'));
        }
      } catch (err) {
        alert('網路連線異常，請檢查網路狀態後再試。');
      } finally {
        btnSave.textContent = originalText;
        btnSave.disabled = false;
      }
    }

    if (act === 'new') {
      const taskFields = document.getElementById('task-fields');
      const bdayGroup = document.getElementById('group-birthday');
      const title = document.getElementById('dialog-title');
      
      if (bdayGroup) bdayGroup.style.display = 'block'; 
      
      const isL2 = (userLevel === 2);
      if (taskFields) taskFields.style.display = isL2 ? 'none' : 'block';
      if (title) title.textContent = isL2 ? '新增受測者資料' : '新增測驗紀錄';

      $f_userId.value = ''; $f_name.value = ''; $f_userId.disabled = false;
      if ($f_birthday) $f_birthday.value = '';
      if ($f_level) $f_level.value = '';
      if ($f_score) { $f_score.value = '0'; $scoreDisplay.textContent = '0'; }
      if ($f_testDate) $f_testDate.value = new Date().toISOString().split('T')[0];

      $dialog.showModal();
      $dialog.dataset.mode = 'create';
    }

    if (act === 'save') {
      e.preventDefault();
      const uid = $f_userId.value.trim();
      const task_id = $f_level ? $f_level.value : '';
      if (!uid) return alert('系統提示：請務必填寫受測者編號 (UID)');

      const btnSave = e.target;
      const originalText = btnSave.textContent;
      btnSave.textContent = '資料處理中...';
      btnSave.disabled = true;

      try {
        let js;
        if (userLevel === 2 || (userLevel === 3 && !task_id)) {
          const r = await fetch('/api/user/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, name: $f_name.value, birthday: $f_birthday?.value || '' })
          });
          js = await r.json().catch(() => ({}));
        } 
        else if (userLevel === 3 && task_id) {
          if ($dialog.dataset.mode === 'create' && $f_birthday?.value) {
            await fetch('/api/user/add', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ uid, name: $f_name.value, birthday: $f_birthday.value })
            });
          }
          const payload = { uid, task_id, score: Number($f_score.value), test_date: $f_testDate.value };
          if ($dialog.dataset.mode === 'edit') payload.row_key_old = $dialog.dataset.key;
          
          const r = await fetch('/scores/upsert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          js = await r.json().catch(() => ({}));
        }
        
        if (js.ok) {
          $dialog.close();
          await reloadUsers(); await reloadScores();
        } else {
          alert('作業無法完成：' + (js.msg || js.err || '未知的系統狀態'));
        }
      } catch (err) { 
        alert('網路連線異常，請檢查網路狀態後再試。'); 
      } finally {
        btnSave.textContent = originalText;
        btnSave.disabled = false;
      }
    }

    if (act === 'edit' && userLevel === 3) {
      const row = state.rows.find(x => x.row_key === btn.closest('tr').dataset.key);
      if (row) {
        document.getElementById('task-fields').style.display = 'block';
        const bdayGroup = document.getElementById('group-birthday');
        if (bdayGroup) bdayGroup.style.display = 'none';

        $f_userId.value = row.uid; $f_userId.disabled = true;
        $f_name.value = row.name || '';
        $f_level.value = row.task_id || '';
        $f_score.value = row.score ?? 0; $scoreDisplay.textContent = row.score ?? 0;
        $f_testDate.value = (row.test_date || '').slice(0, 10);
        
        document.getElementById('dialog-title').textContent = '編輯測驗紀錄';
        $dialog.showModal(); 
        $dialog.dataset.mode = 'edit'; 
        $dialog.dataset.key = row.row_key;
      }
    }
    
    if (act === 'del' && userLevel === 3) {
      if (confirm('重要提醒：確定要刪除這筆測驗紀錄嗎？刪除後將無法復原。')) {
        const r = await fetch(`/scores?row_key=${btn.closest('tr').dataset.key}`, { method: 'DELETE' });
        if (r.ok) reloadScores();
      }
    }

    if (act === 'clear-filter') {
      state.selUserId = ''; state.selLevel = '';
      if ($selName) $selName.value = '';
      if ($selLvl) $selLvl.value = '';
      render();
    }
  });

  if ($selName) $selName.addEventListener('change', () => { state.selUserId = $selName.value; render(); });
  if ($selLvl) $selLvl.addEventListener('change', () => { state.selLevel = $selLvl.value; render(); });

  (async function init() {
    if (await ensureAuth()) { 
      await reloadUsers(); 
      await reloadScores(); 
    }
  })();
})();