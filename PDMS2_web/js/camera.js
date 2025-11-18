// camera.js (修改版：Ch5-t1 自動開始遊戲)

const KEY = "kid-quest-progress-v1";
const TOP = 0; 
const SIDE = 1; 
const waittime = 3;

// 遊戲狀態輪詢變數
let gameStateInterval = null;

function getId(){
  const u = new URL(location.href);
  return u.searchParams.get("id");
}

async function getUid(){
  try {
    const response = await fetch('/session/get-uid');
    if (response.ok) {
      const result = await response.json();
      return result.uid;
    } else {
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      return st.currentUid || null;
    }
  } catch (error) {
    console.error('獲取 UID 時發生錯誤:', error);
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    return st.currentUid || null;
  }
}

function isImagePath(s){ return typeof s==="string" && (s.startsWith("/images/") || /\.(png|jpe?g|svg|webp|gif)$/i.test(s)); }
function setIcon(el, src){
  if(!el) return;
  el.innerHTML = isImagePath(src) ? `<img class="icon-img" src="${src}" alt="">` : "";
}

const ID_TO_META = {
  "ch1-t1": {icon:"/images/bridge.jpg",  title:"串積木：做成一條橋"},
  "ch1-t2": {icon:"/images/tower.jpg",   title:"疊城堡：蓋瞭望塔"},
  "ch1-t3": {icon:"/images/stairs.jpg",  title:"疊階梯：翻過高牆"},
  "ch2-t1": {icon:"/images/circle.jpg",  title:"畫圓：大圓圓魔法陣"},
  "ch2-t2": {icon:"/images/square.jpg",  title:"畫方：守護盾"},
  "ch2-t3": {icon:"/images/cross.jpg",   title:"畫十字：啟動魔法"},
  "ch2-t4": {icon:"/images/line.jpg",    title:"描水平線：打敗恐龍"},
  "ch2-t5": {icon:"/images/fill.jpg",    title:"兩水平線中塗色：提升威力"},
  "ch2-t6": {icon:"/images/connect.png", title:"兩點連線：開門"},
  "ch3-t1": {icon:"/images/circle_win.jpg", title:"剪圓：做圓形窗戶"},
  "ch3-t2": {icon:"/images/square_door.jpg", title:"剪方：做方方正正的門"},
  "ch4-t1": {icon:"/images/fold1.jpg", title:"摺紙一摺：變出小飛毯"},
  "ch4-t2": {icon:"/images/fold2.jpg", title:"摺紙兩摺：更結實的飛毯"},
  "ch5-t1": {icon:"/images/beans.jpg", title:"豆豆裝罐子：完成任務"},
};

const els = {
  taskIcon: document.getElementById("taskIcon"),
  taskTitle: document.getElementById("taskTitle"),
  cameraStream: document.getElementById("cameraStream"),
  placeholderText: document.getElementById("placeholderText"),
  statusInfo: document.getElementById("statusInfo"),
  shotBtn: document.getElementById("shotBtn"), 
  stopBtn: document.getElementById("stopBtn"),
  // Ch5-t1 專用元素
  gameInfo: document.getElementById("gameInfo"),
  beanCount: document.getElementById("beanCount"),
  timeRemaining: document.getElementById("timeRemaining"),
};

let cameraActive = false;
let streamInterval = null;
const id = getId();

function updateStatus(message, type = 'info') {
  els.statusInfo.textContent = message;
  els.statusInfo.className = `status-info ${type}`;
}

// 初始化標題
(function initHeader(){
  const meta = ID_TO_META[id] || {icon:"", title:"拍照存證"};
  setIcon(els.taskIcon, meta.icon);
  if(meta.title) els.taskTitle.textContent = meta.title;
  
  if (els.stopBtn) {
     els.stopBtn.style.display = 'none';
  }
  
  if (id === "ch5-t1") {
    // Ch5-t1 隱藏拍照按鈕（因為會自動開始）
    els.shotBtn.style.display = 'none';
    
    // 顯示遊戲資訊區塊
    if (els.gameInfo) {
      els.gameInfo.style.display = 'block';
    }
    // 隱藏相機預覽區域
    if (els.cameraStream) {
      els.cameraStream.style.display = 'none';
    }
    if (els.placeholderText) {
      els.placeholderText.textContent = '遊戲準備中...';
    }
  } else {
     els.shotBtn.textContent = "🎞️ 拍照、存檔並回主頁"; 
  }
})();

// 串流預覽
function startVideoStream() {
  if (streamInterval) clearInterval(streamInterval);
  streamInterval = setInterval(async () => {
    try {
      const response = await fetch('/opencv-camera/frame');
      if (!response.ok) return;
      const data = await response.json();
      if (data.success) {
        els.cameraStream.src = "data:image/jpeg;base64," + data.image;
      }
    } catch (err) {
      console.error("獲取幀錯誤:", err);
    }
  }, 30);
}

// 開啟相機
async function openCamera() {
  // Ch5-t1 不需要開啟相機預覽，直接自動開始遊戲
  if (id === "ch5-t1") {
    updateStatus('準備自動開始遊戲...', 'info');
    // 延遲1秒後自動開始遊戲
    setTimeout(() => {
      autoStartGame();
    }, 1000);
    return;
  }
  
  try {
    updateStatus('正在開啟相機...', 'loading');
    let CAM_INDEX = TOP;
    if(["ch1-t2", "ch1-t3"].includes(id)) CAM_INDEX = SIDE;
    
    const response = await fetch('/opencv-camera/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: id, camera_index: CAM_INDEX })
    });
    if (!response.ok) throw new Error('無法開啟相機');
    const result = await response.json();
    if (result.success) {
      cameraActive = true;
      updateStatus('相機預覽已開啟，請準備！', 'success');
      els.placeholderText.style.display = 'none';
      els.cameraStream.style.display = 'block';
      els.shotBtn.disabled = false;
      startVideoStream();
    } else {
      throw new Error(result.error || '開啟相機失敗');
    }
  } catch (error) {
    console.error('開啟相機錯誤:', error);
    updateStatus(`開啟相機失敗: ${error.message}`, 'error');
  }
}

// 輪詢遊戲狀態（Ch5-t1 專用）
async function pollGameState(uid) {
  console.log('[遊戲狀態] 開始輪詢，UID:', uid);
  
  if (gameStateInterval) {
    clearInterval(gameStateInterval);
  }
  
  gameStateInterval = setInterval(async () => {
    try {
      const response = await fetch(`/game-state/${uid}`);
      if (!response.ok) {
        console.warn('[遊戲狀態] 無法取得遊戲狀態，狀態碼:', response.status);
        return;
      }
      
      const data = await response.json();
      console.log('[遊戲狀態] 收到資料:', data);
      
      if (data.success && data.state) {
        const state = data.state;
        
        // 更新顯示
        if (els.beanCount) {
          els.beanCount.textContent = state.bean_count;
          console.log('[遊戲狀態] 更新豆豆數量:', state.bean_count);
        }
        if (els.timeRemaining) {
          els.timeRemaining.textContent = state.remaining_time;
          console.log('[遊戲狀態] 更新剩餘時間:', state.remaining_time);
        }
        
        // 警告提示
        if (state.warning) {
          updateStatus('⚠️ 注意：檢測到作弊行為！', 'error');
        } else if (state.running) {
          updateStatus(`遊戲進行中... 豆豆：${state.bean_count} | 剩餘：${state.remaining_time}秒`, 'loading');
        }
        
        // 遊戲結束
        if (state.game_over) {
          console.log('[遊戲狀態] 遊戲結束，分數:', state.score);
          clearInterval(gameStateInterval);
          gameStateInterval = null;
          
          let resultMsg = '';
          if (state.score === 2) {
            resultMsg = '🎉 完美完成！';
          } else if (state.score === 1) {
            resultMsg = '👍 完成任務！';
          } else {
            resultMsg = '👍 完成任務！';
          }
          
          updateStatus(`遊戲結束！${resultMsg}`, 'success');
          
          // 3秒後跳轉
          setTimeout(() => {
            redirectToNextTask(id);
          }, 3000);
        }
      }
    } catch (error) {
      console.error('[遊戲狀態] 輪詢錯誤:', error);
    }
  }, 500); // 每0.5秒更新一次
}

async function captureWithCamera(cameraIndex, fullTaskId, uid) {
  try {
    const switchResponse = await fetch('/opencv-camera/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index: cameraIndex })
    });
    if (!switchResponse.ok) {
      throw new Error('切換相機失敗');
    }
    await new Promise(r => setTimeout(r, 500));
    
    const captureResponse = await fetch('/opencv-camera/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        task_id: fullTaskId,
        uid: uid 
      })
    });
    if (!captureResponse.ok) {
      throw new Error('拍照失敗');
    }
    return await captureResponse.json();
  } catch (error) {
    console.error('拍照錯誤:', error);
    throw error;
  }
}

async function triggerBackgroundAnalysis(taskId, uid) {
  try {
    let body_data = { 
        id: taskId,
        uid: uid 
    };
    
    if (taskId === "ch5-t1") {
        body_data.cam_index = SIDE; 
    } else {
        body_data.cam_index = TOP; 
    }

    const response = await fetch('/run-python', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body_data)
    });
    
    if (!response.ok) {
      console.warn('分析請求失敗');
      return;
    }
    
    const result = await response.json();
    console.log('分析已觸發:', result);
    return result.task_id;
    
  } catch (error) {
    console.warn('觸發分析時發生錯誤:', error);
  }
}

function redirectToNextTask(currentId) {
  const TASK_IDS = Object.keys(ID_TO_META);
  const idx = TASK_IDS.indexOf(currentId);
  const nextTaskId = (idx >= 0 && idx < TASK_IDS.length - 1) ? TASK_IDS[idx + 1] : null;
  if (nextTaskId){
    location.href = `/html/task.html?id=${nextTaskId}`;
  } else {
    location.href = "/html/index.html";
  }
}

// 清空遊戲狀態的函數
async function clearGameState(uid) {
  try {
    const response = await fetch('/clear-game-state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uid: uid })
    });
    
    if (!response.ok) {
      console.warn('清空遊戲狀態失敗');
      return false;
    }
    
    const result = await response.json();
    console.log('遊戲狀態已清空:', result);
    return true;
  } catch (error) {
    console.error('清空遊戲狀態錯誤:', error);
    return false;
  }
}

// 自動開始遊戲（Ch5-t1 專用）
async function autoStartGame() {
  try {
    console.log('[自動開始] 準備開始遊戲...');
    const currentUid = await getUid() || 'default';
    
    updateStatus('遊戲即將開始...', 'loading');
    
    // 清空遊戲狀態
    await clearGameState(currentUid);
    console.log('[自動開始] 狀態已清空');
    
    // 短暫延遲確保狀態已寫入
    await new Promise(r => setTimeout(r, 300));
    
    updateStatus('遊戲開始！請開始收集豆豆...', 'loading');
    
    // 觸發後端遊戲程式
    await triggerBackgroundAnalysis(id, currentUid);
    console.log('[自動開始] 遊戲已啟動');
    
    // 開始輪詢遊戲狀態
    pollGameState(currentUid);
    
  } catch (error) {
    console.error('[自動開始] 錯誤:', error);
    updateStatus(`自動開始失敗: ${error.message}`, 'error');
  }
}

// 主函數（保留給其他任務使用）
async function takeShot() {
  try {
    const currentUid = await getUid() || 'default';
    els.shotBtn.disabled = true;
    
    if (id === "ch5-t1") {
      // Ch5-t1 不再使用此函數，改用 autoStartGame
      console.log('[takeShot] Ch5-t1 應該使用自動開始，不應執行到此處');
      return;
      
    } else if (["ch1-t2", "ch1-t3"].includes(id)) {
      await countdown(waittime);
      await closeCamera();
      
      updateStatus('正在拍攝側面鏡頭...', 'loading');
      await captureWithCamera(SIDE, `${id}-side`, currentUid);
      
      updateStatus('側面完成，切換上方鏡頭...', 'loading');
      await captureWithCamera(TOP, `${id}-top`, currentUid);

      updateStatus('照片拍攝完成！背景分析已啟動，準備跳轉...', 'success');
      await new Promise(r => setTimeout(r, 800));
      redirectToNextTask(id);
      
    } else {
      await countdown(waittime);
      await closeCamera();
      
      updateStatus('正在拍照（上方鏡頭）...', 'loading');
      await captureWithCamera(TOP, id, currentUid);
      
      updateStatus('照片拍攝完成！背景分析已啟動，準備跳轉...', 'success');
      await new Promise(r => setTimeout(r, 800));
      redirectToNextTask(id);
    }
    
  } catch (error) {
    console.error('操作錯誤:', error);
    updateStatus(`操作失敗: ${error.message}`, 'error');
    els.shotBtn.disabled = false;
    if (id !== "ch5-t1") {
      await openCamera();
    }
  }
}

async function countdown(seconds) {
  for (let i = seconds; i > 0; i--) {
    updateStatus(`準備拍照... ${i}`, 'loading');
    await new Promise(r => setTimeout(r, 1000));
  }
}

async function closeCamera() {
  try {
    if (streamInterval) {
      clearInterval(streamInterval);
      streamInterval = null;
    }
    const response = await fetch('/opencv-camera/stop', { method: 'POST' });
    if (!response.ok) console.warn('無法通知後端關閉相機');
    cameraActive = false;
    els.shotBtn.disabled = true;
    els.placeholderText.style.display = 'block';
    els.cameraStream.style.display = 'none';
    els.cameraStream.src = "";
    updateStatus('相機已關閉', 'info');
  } catch (err) {
    console.error('關閉相機失敗:', err);
  }
}

function shotBtnClickHandler() {
    takeShot(); 
}

els.shotBtn.addEventListener("click", shotBtnClickHandler);

document.addEventListener("DOMContentLoaded", async () => {
  await openCamera();
});

window.addEventListener("beforeunload", () => {
  if (streamInterval) clearInterval(streamInterval);
  if (gameStateInterval) clearInterval(gameStateInterval);
  if (cameraActive) closeCamera();
});