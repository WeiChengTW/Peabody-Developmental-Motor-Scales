// 共用 KEY；與你的其他頁一致
const KEY = "kid-quest-progress-v1";

// ======相機參數 (對應 app.py) =====
const TOP = 1; 
const SIDE = 2; 
// ===================================

// 讀 id（如 ch2-t3）
function getId(){
  const u = new URL(location.href);
  return u.searchParams.get("id");
}

// 改用後端 session 獲取 UID
async function getUid(){
  try {
    const response = await fetch('/session/get-uid');
    if (response.ok) {
      const result = await response.json();
      return result.uid;
    } else {
      // 降級到 localStorage
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      return st.currentUid || null;
    }
  } catch (error) {
    console.error('獲取 UID 時發生錯誤:', error);
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    return st.currentUid || null;
  }
}

// 設置圖標
function isImagePath(s){ return typeof s==="string" && (s.startsWith("/images/") || /\.(png|jpe?g|svg|webp|gif)$/i.test(s)); }
function setIcon(el, src){
  if(!el) return;
  el.innerHTML = isImagePath(src) ? `<img class="icon-img" src="${src}" alt="">` : "";
}

// 任務元資料
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

// DOM 元素
const els = {
  taskIcon: document.getElementById("taskIcon"),
  taskTitle: document.getElementById("taskTitle"),
  cameraStream: document.getElementById("cameraStream"),
  placeholderText: document.getElementById("placeholderText"),
  statusInfo: document.getElementById("statusInfo"),
  shotBtn: document.getElementById("shotBtn"),
};

// 狀態變量
let cameraActive = false;
let streamInterval = null;
const id = getId();

// 更新狀態信息
function updateStatus(message, type = 'info') {
  els.statusInfo.textContent = message;
  els.statusInfo.className = `status-info ${type}`;
}

// 初始化標題與小圖
(function initHeader(){
  const meta = ID_TO_META[id] || {icon:"", title:"拍照存證"};
  setIcon(els.taskIcon, meta.icon);
  if(meta.title) els.taskTitle.textContent = meta.title;
  
  if (id === "ch5-t1") {
    els.shotBtn.style.display = 'none'; // Ch5-t1 不顯示按鈕
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
      } else {
        console.warn("相機畫面失敗:", data.error);
      }
    } catch (err) {
      console.error("獲取幀錯誤:", err);
    }
  }, 120); // 約 60ms 更新一次
}

// 開啟 OpenCV 相機（靜態拍照用）
async function openCamera() {
  try {
    updateStatus('正在開啟相機...', 'loading');

    if(["ch1-t2", "ch1-t3"].includes(id))CAM_INDEX = SIDE;
    else CAM_INDEX = TOP;

    // 靜態拍照預覽一律用 TOP 鏡頭
    const response = await fetch('/opencv-camera/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: id, camera_index: CAM_INDEX }) // 使用 CAM_INDEX
    });

    if (!response.ok) throw new Error('無法開啟相機');

    const result = await response.json();

    if (result.success) {
      cameraActive = true;
      updateStatus('相機已開啟，請對準目標後按鈕', 'success');

      // 開始串流
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


async function captureWithCamera(cameraIndex, fullTaskId, uid) {
  try {
    // 1. 切換相機
    const switchResponse = await fetch('/opencv-camera/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index: cameraIndex })
    });

    if (!switchResponse.ok) {
      throw new Error('切換相機失敗');
    }

    // 2. 等待相機穩定
    await new Promise(r => setTimeout(r, 500));

    // 3. 拍照（傳入完整的 task_id，例如 "ch1-t2-side"）
    const captureResponse = await fetch('/opencv-camera/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        task_id: fullTaskId,  // ✅ 完整檔名
        uid: uid 
      })
    });

    if (!captureResponse.ok) {
      throw new Error('拍照失敗');
    }

    const result = await captureResponse.json();
    return result.analysis_task_id; // 回傳分析任務 ID

  } catch (error) {
    console.error('拍照錯誤:', error);
    throw error;
  }
}


// === [新增] 輔助函式 (輪詢任務狀態) ===
async function pollTaskStatus(taskId) {
  const msg = id === "ch5-t1"?'正在啟動遊戲...請稍後...':'分析中...請稍候...'
  updateStatus(msg, 'loading');
  const startTime = Date.now();
  
  // 輪詢 30 秒 (靜態分析) 或 150 秒 (遊戲)
  const timeout = (id === "ch5-t1") ? 150000 : 30000; 

  while (Date.now() - startTime < timeout) {
    try {
      const res = await fetch(`/check-task/${taskId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'completed') {
          const score = data.result.returncode;
          updateStatus(`準備跳轉...`, 'success');
          await new Promise(r => setTimeout(r, 1500)); // 顯示成功訊息
          return; // 成功
        } else if (data.status === 'error') {
          throw new Error(data.error || '分析失敗');
        }
        // else: 狀態是 'running' 或 'pending'，繼續輪詢
      }
    } catch (e) {
      // 忽略 fetch 錯誤，繼續嘗試
      console.warn("Poll error:", e.message);
    }
    await new Promise(r => setTimeout(r, 2000)); // 每 2 秒檢查一次
  }
  
  // 逾時
  updateStatus('分析逾時，將在背景繼續。準備跳轉...', 'info');
  await new Promise(r => setTimeout(r, 1500));
}

// === [新增] 輔助函式 (跳轉) ===
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

async function takeShot() {
  try {
    const currentUid = await getUid() || 'default';
    els.shotBtn.disabled = true;

    if (["ch1-t2", "ch1-t3"].includes(id)) {
      // ✅ 拍攝側面鏡頭，檔名帶 -side
      updateStatus('正在拍攝側面鏡頭...', 'loading');
      await captureWithCamera(SIDE, `${id}-side`, currentUid);

      // ✅ 拍攝上方鏡頭，檔名帶 -top
      updateStatus('側面完成，切換上方鏡頭...', 'loading');
      await captureWithCamera(TOP, `${id}-top`, currentUid);

    } else {
      // ✅ 單張照片（不加後綴）
      updateStatus('正在拍照（上方鏡頭）...', 'loading');
      await captureWithCamera(TOP, id, currentUid);
    }

    updateStatus('照片拍攝完成，開始分析...', 'success');

    // ✅ 所有照片都拍完後，才呼叫 /run-python
    const analysisResponse = await fetch('/run-python', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        id: id,  // ⚠️ 注意：這裡傳原始的 id（不帶 -side/-top）
        uid: currentUid 
      })
    });

    if (!analysisResponse.ok) {
      throw new Error('分析請求失敗');
    }

    const analysisResult = await analysisResponse.json();

    if (analysisResult.task_id) {
      await pollTaskStatus(analysisResult.task_id);
    } else {
      updateStatus('照片已儲存（無分析腳本），準備跳轉...', 'info');
      await new Promise(r => setTimeout(r, 1000));
    }

    await closeCamera();
    redirectToNextTask(id);

  } catch (error) {
    console.error('拍照錯誤:', error);
    updateStatus(`拍照失敗: ${error.message}`, 'error');
    els.shotBtn.disabled = false;
  }
}


// === [新增] Ch5-t1 遊戲自動執行函數 ===
async function runGame() {
  try {
    updateStatus('正在啟動遊戲...請稍候...', 'loading');
    els.placeholderText.style.display = 'block';
    els.cameraStream.style.display = 'none';

    const currentUid = await getUid() || 'default';

    // 直接呼叫 /run-python 啟動 Ch5-t1
    const analysisResponse = await fetch('/run-python', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ id: id, uid: currentUid })
    });
    
    if (!analysisResponse.ok) throw new Error('啟動遊戲失敗 (伺服器錯誤)');
    
    const analysisResult = await analysisResponse.json();
    if (!analysisResult.success) throw new Error(analysisResult.error || '啟動遊戲失敗');
    
    updateStatus('遊戲執行中...請查看彈出視窗。完成後將自動跳轉。', 'info');
    
    // 輪詢遊戲任務，直到它結束
    await pollTaskStatus(analysisResult.task_id);

    // 遊戲結束 (或逾時)，跳轉
    redirectToNextTask(id);

  } catch (error) {
    console.error('遊戲執行錯誤:', error);
    updateStatus(`遊戲失敗: ${error.message}`, 'error');
  }
}


// 關閉相機
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

// 綁定事件
els.shotBtn.addEventListener("click", takeShot);

// === [修正] 進入畫面時，根據 ID 決定行為 ===
document.addEventListener("DOMContentLoaded", async () => {
  if (id === "ch5-t1") {
    // Ch5-t1: 自動執行遊戲，不開網頁相機
    await runGame();
  } else {
    // 其他任務: 開啟網頁相機預覽
    await openCamera();
  }
});

// 頁面卸載時清理資源
window.addEventListener("beforeunload", () => {
  if (streamInterval) clearInterval(streamInterval);
  if (cameraActive) closeCamera();
});