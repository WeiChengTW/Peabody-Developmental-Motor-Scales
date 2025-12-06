// admin.js (已修復：重置篩選按鈕、管理使用者按鈕、新增紀錄按鈕)
(function () {
  const state = {
    userLevel: 0, 
    currentUid: '',
    currentName: '',
    rows: [],
    usersList: [],
    sortKey: 'test_date',
    sortAsc: false,
    selUserId: '',
    selLevel: '',
    allLevels: []
  };

  const $tbody = document.getElementById('tbody');
  const $selName = document.getElementById('sel-name');
  const $selLvl = document.getElementById('sel-level');
  
  // Dialog 相關元素
  const $dialog = document.getElementById('edit-dialog');
  const $f_userId = $dialog.querySelector('[data-field="userId"]');
  const $f_name = $dialog.querySelector('[data-field="name"]');
  const $f_level = $dialog.querySelector('[data-field="level"]');
  const $f_score = $dialog.querySelector('[data-field="score"]');
  const $f_scoreText = $dialog.querySelector('[data-field="score-text"]');
  const $f_testDate = $dialog.querySelector('[data-field="test_date"]');
  const $scoreSection = document.getElementById('score-fields');
  
  // 記錄目前編輯的 row_key (如果是新增則為 null)
  let editingRowKey = null;

  async function init() {
    try {
      const r = await fetch('/api/auth/whoami');
      const js = await r.json();
      if (!js.ok || !js.logged_in) { alert("請先登入"); window.location.href = '/'; return; }

      const user = js.user;
      state.currentUid = user.account;
      state.userLevel = user.level;
      state.currentName = user.name || user.account;

      const roleName = state.userLevel === 2 ? '管理員' : '家長';
      const infoEl = document.getElementById('current-user-info');
      if(infoEl) infoEl.textContent = `目前登入: ${state.currentName} (${roleName})`;

      if (state.userLevel === 2) document.body.classList.add('is-admin');
      else document.body.classList.remove('is-admin');

      await loadData();
      if (state.userLevel === 2) await loadAllUsers();
      
      rebuildSelects();
      
      if (state.userLevel === 1) {
        state.selUserId = state.currentUid;
        $selName.value = state.currentUid;
        $selName.disabled = true;
      }
      render();

    } catch (e) {
      console.error('Init failed', e);
      alert('初始化失敗，請檢查後端連線');
    }
  }

  async function loadAllUsers() {
      try {
          const r = await fetch('/users');
          const js = await r.json();
          if (js.ok) state.usersList = js.users;
      } catch (e) { console.warn("載入使用者名單失敗", e); }
  }

  async function loadData() {
    const $loading = document.getElementById('tbody');
    if ($loading) $loading.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px">資料讀取中...</td></tr>';

    try {
      let payload = {};
      if (state.userLevel === 1) payload.uid = state.currentUid;

      const r = await fetch('/api/search-scores', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const js = await r.json();
      if (js.success) {
        state.rows = js.data;
        const lvlSet = new Set();
        state.rows.forEach(r => lvlSet.add(r.task_id));
        state.allLevels = Array.from(lvlSet).sort();
      } else {
        alert("資料載入錯誤: " + (js.error || '未知錯誤'));
        state.rows = [];
      }
    } catch (e) {
      console.warn('Load data error', e);
      state.rows = [];
    }
  }

  function rebuildSelects() {
    $selName.innerHTML = '';
    
    if (state.userLevel === 1) {
      const opt = document.createElement('option');
      opt.value = state.currentUid;
      opt.textContent = `${state.currentName} (${state.currentUid})`;
      opt.selected = true;
      $selName.appendChild(opt);
    } else {
      const def = document.createElement('option');
      def.value = '';
      def.textContent = '（所有小朋友）';
      $selName.appendChild(def);

      const sourceList = (state.usersList.length > 0) ? state.usersList : [];
      if (sourceList.length === 0) {
          const uidMap = new Map();
          state.rows.forEach(r => { if (!uidMap.has(r.uid)) uidMap.set(r.uid, r.name || r.uid); });
          Array.from(uidMap.keys()).sort().forEach(uid => {
             const opt = document.createElement('option');
             opt.value = uid;
             opt.textContent = uidMap.get(uid);
             $selName.appendChild(opt);
          });
      } else {
          sourceList.forEach(u => {
             const opt = document.createElement('option');
             opt.value = u.uid;
             opt.textContent = u.name;
             $selName.appendChild(opt);
          });
      }
    }

    $selLvl.innerHTML = '<option value="">（所有關卡）</option>';
    const standardLevels = ['Ch1-t1', 'Ch1-t2', 'Ch1-t3', 'Ch1-t4', 'Ch2-t1', 'Ch2-t2', 'Ch2-t3', 'Ch2-t4', 'Ch2-t5', 'Ch2-t6', 'Ch3-t1', 'Ch3-t2', 'Ch3-t3', 'Ch3-t4', 'Ch4-t1', 'Ch4-t2', 'Ch5-t1'];
    const levelsToShow = new Set([...standardLevels, ...state.allLevels]);
    Array.from(levelsToShow).sort().forEach(lvl => {
      const opt = document.createElement('option');
      opt.value = lvl;
      opt.textContent = lvl;
      $selLvl.appendChild(opt);
    });
  }

  function render() {
    let displayRows = state.rows.filter(r => {
      const matchUser = !state.selUserId || r.uid === state.selUserId;
      const matchLevel = !state.selLevel || r.task_id === state.selLevel;
      return matchUser && matchLevel;
    });

    displayRows.sort((a, b) => {
      let valA = a[state.sortKey] || '';
      let valB = b[state.sortKey] || '';
      if (state.sortKey === 'score') {
        valA = parseFloat(valA) || 0;
        valB = parseFloat(valB) || 0;
      }
      if (valA < valB) return state.sortAsc ? -1 : 1;
      if (valA > valB) return state.sortAsc ? 1 : -1;
      return 0;
    });

    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.sort === state.sortKey) {
            th.classList.add(state.sortAsc ? 'sort-asc' : 'sort-desc');
        }
    });

    if (displayRows.length === 0) {
      $tbody.innerHTML = `<tr><td colspan="8" class="empty">沒有符合的資料</td></tr>`;
      return;
    }

    $tbody.innerHTML = displayRows.map(r => {
      let imgCell = '<span class="muted" style="color:#ccc">無圖片</span>';
      if (r.result_img_path) {
          let thumbSrc = r.result_img_path;
          if (!thumbSrc.startsWith('/')) thumbSrc = '/' + thumbSrc;
          const linkUrl = r.compare_url || thumbSrc;
          imgCell = `
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:40px; height:40px; background:#eee; border-radius:4px; overflow:hidden;">
                    <img src="${thumbSrc}" style="width:100%; height:100%; object-fit:cover;" onerror="this.style.display='none'">
                </div>
                <a href="${linkUrl}" target="_blank" class="btn btn-sm" style="text-decoration:none;">檢視</a>
            </div>
          `;
      }

      let adminBtns = '';
      if (state.userLevel === 2) {
        adminBtns = `
          <td class="admin-col">
             <button class="btn btn-sm btn-delete" onclick="deleteScore('${r.row_key}')">刪除</button>
          </td>
        `;
      } else {
         adminBtns = `<td class="admin-col"></td>`;
      }

      return `
        <tr>
          <td>${r.uid}</td>
          <td>${r.name || '—'}</td>
          <td>${r.task_name || r.task_id}</td>
          <td><span class="badge ${r.score === null ? 'gray' : ''}">${r.score !== null ? r.score : '—'}</span></td>
          <td>${r.test_date}</td>
          <td>${r.time || ''}</td>
          <td>${imgCell}</td>
          ${adminBtns}
        </tr>
      `;
    }).join('');
  }

  // ★★★ 全域事件監聽 (解決按鈕失效問題) ★★★
  document.addEventListener('click', e => {
      // 找到被點擊的按鈕 (處理 icon 點擊的情況)
      const target = e.target.closest('button');
      if (!target) return;
      const action = target.dataset.action;

      // 1. 重置篩選
      if (action === 'reset') {
          if (state.userLevel === 1) {
              state.selLevel = '';
              $selLvl.value = '';
          } else {
              state.selUserId = '';
              $selName.value = '';
              state.selLevel = '';
              $selLvl.value = '';
          }
          render();
      }
      
      // 2. 新增紀錄 (開啟 Dialog)
      else if (action === 'new') {
          editingRowKey = null; // 標記為新增
          $dialog.querySelector('.dlg-title').textContent = "新增測試紀錄";
          $scoreSection.style.display = 'block'; // 顯示分數欄位
          
          // 清空欄位
          $f_userId.value = state.selUserId || ''; // 如果有篩選，自動帶入
          $f_name.value = '';
          $f_level.value = '';
          $f_score.value = 0;
          $f_scoreText.textContent = 0;
          $f_testDate.valueAsDate = new Date();
          
          $dialog.showModal();
      }

      // 3. 管理使用者 (新增/修改學生資料)
      else if (action === 'manage-users') {
          // 這裡我們用一個簡單的 prompt 流程，或者也可以重用 dialog 但隱藏分數欄位
          // 為了簡單直覺，這裡重用 dialog，但隱藏「分數」相關欄位，只留 UID/姓名/生日
          
          editingRowKey = 'USER_MODE'; // 特殊標記
          $dialog.querySelector('.dlg-title').textContent = "管理使用者 (新增/修改)";
          $scoreSection.style.display = 'none'; // 隱藏分數、關卡等欄位，只留基本資料
          
          $f_userId.value = '';
          $f_name.value = '';
          // 生日欄位借用 test_date 欄位，或者我們提示使用者在姓名欄備註？
          // 更好的做法是：我們用 prompt 快速實作，避免修改 HTML 結構太複雜
          $dialog.close(); // 關閉 dialog 避免干擾
          
          // 使用簡單的輸入流程
          const uid = prompt("請輸入使用者的 UID (必填):", "");
          if (!uid) return;
          const name = prompt("請輸入使用者姓名:", "");
          const birth = prompt("請輸入生日 (格式: YYYY-MM-DD) 作為密碼:", "2020-01-01");
          
          if (uid && birth) {
              saveUser(uid, name, birth);
          }
      }

      // 4. Dialog 內的儲存按鈕
      else if (action === 'save') {
          e.preventDefault(); // 阻止 form submit
          
          // 收集資料
          const uid = $f_userId.value.trim();
          const name = $f_name.value.trim();
          
          if (!uid) { alert("UID 為必填"); return; }
          
          if (editingRowKey === null) {
              // === 儲存新成績 ===
              const level = $f_level.value.trim();
              const score = $f_score.value;
              const date = $f_testDate.value;
              
              if (!level) { alert("關卡代號為必填"); return; }
              
              saveScore(uid, name, level, score, date);
          }
      }
  });

  // Dialog 取消按鈕 (預設 behavior="cancel" 會自動關閉，但為了保險可以加)
  $dialog.addEventListener('close', () => {
      // 重置 dialog 狀態
  });

  // 分數滑桿連動
  $f_score.addEventListener('input', e => {
      $f_scoreText.textContent = e.target.value;
  });

  // --- API 呼叫 ---

  async function saveUser(uid, name, birthday) {
      try {
          const r = await fetch('/api/user/upsert', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ uid, name, birthday })
          });
          const js = await r.json();
          if (js.ok) {
              alert("使用者儲存成功！");
              await loadAllUsers(); // 重新載入名單
              rebuildSelects();     // 更新下拉選單
          } else {
              alert("失敗: " + js.msg);
          }
      } catch (e) {
          alert("連線錯誤");
      }
  }

  async function saveScore(uid, name, task_id, score, test_date) {
      try {
          // 如果有填姓名，順便更新使用者資料
          if (name) {
              await fetch('/api/user/upsert', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({ uid, name, birthday: null }) // 不更新生日
              });
          }

          const r = await fetch('/scores/upsert', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ uid, task_id, score, test_date })
          });
          
          const js = await r.json();
          if (js.ok) {
              alert("紀錄新增成功！");
              $dialog.close();
              await loadData(); // 重整表格
              render();
          } else {
              alert("失敗: " + js.msg);
          }
      } catch (e) {
          console.error(e);
          alert("儲存失敗");
      }
  }

  // 篩選與排序監聽
  $selName.addEventListener('change', e => { state.selUserId = e.target.value; render(); });
  $selLvl.addEventListener('change', e => { state.selLevel = e.target.value; render(); });
  
  document.querySelectorAll('th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortAsc = !state.sortAsc;
      else { state.sortKey = key; state.sortAsc = true; }
      render();
    });
  });

  document.getElementById('btn-logout')?.addEventListener('click', async () => {
      if(confirm('確定要登出嗎？')) { await fetch('/api/auth/logout', { method: 'POST' }); window.location.href = '/'; }
  });

  window.deleteScore = async function(rowKey) {
      if(!confirm('確定要刪除這筆紀錄嗎？此動作無法復原。')) return;
      try {
          const r = await fetch(`/scores?row_key=${encodeURIComponent(rowKey)}`, { method: 'DELETE' });
          const js = await r.json();
          if(js.ok) { alert('刪除成功'); loadData().then(rebuildSelects).then(render); } else { alert('刪除失敗: ' + js.msg); }
      } catch(e) { console.error(e); alert('刪除失敗'); }
  };

  init();
})();