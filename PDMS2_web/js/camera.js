// 共用 KEY；與你的其他頁一致
const KEY = "kid-quest-progress-v1";
// ======相機參數 (對應 app.py) =====
const TOP = 2; 
const SIDE = 1; 
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
  }, 120);
}
// 開啟 OpenCV 相機（靜態拍照用）
async function openCamera() {
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
      updateStatus('相機已開啟，請對準目標後按鈕', 'success');
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
    // 3. 拍照（只拍照，不分析）
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

// === [修改] 背景觸發分析（不等待結果）===
async function triggerBackgroundAnalysis(taskId, uid) {
  try {
    // 發送分析請求，但不等待結果
    fetch('/run-python', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        id: taskId,
        uid: uid 
      })
    }).catch(err => {
      console.warn('背景分析請求失敗:', err);
    });
  } catch (error) {
    console.warn('觸發背景分析時發生錯誤:', error);
  }
}

// === [簡化] 跳轉函數 ===
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

// === [重構] 拍照主函數：拍完就跳，背景運算 ===
async function takeShot() {
  try {
    const currentUid = await getUid() || 'default';
    els.shotBtn.disabled = true;
    
    if (["ch1-t2", "ch1-t3"].includes(id)) {
      // 雙鏡頭任務
      updateStatus('正在拍攝側面鏡頭...', 'loading');
      await captureWithCamera(SIDE, `${id}-side`, currentUid);
      
      updateStatus('側面完成，切換上方鏡頭...', 'loading');
      await captureWithCamera(TOP, `${id}-top`, currentUid);
    } else {
      // 單鏡頭任務
      updateStatus('正在拍照（上方鏡頭）...', 'loading');
      await captureWithCamera(TOP, id, currentUid);
    }
    
    updateStatus('照片拍攝完成！準備跳轉...', 'success');
    
    // ✅ 關鍵改動：先關閉相機
    await closeCamera();
    
    // ✅ 觸發背景分析（不等待）
    triggerBackgroundAnalysis(id, currentUid);
    
    // ✅ 短暫延遲後直接跳轉
    await new Promise(r => setTimeout(r, 800));
    redirectToNextTask(id);
    
  } catch (error) {
    console.error('拍照錯誤:', error);
    updateStatus(`拍照失敗: ${error.message}`, 'error');
    els.shotBtn.disabled = false;
  }
}

// === [保留] Ch5-t1 遊戲需要等待完成 ===
async function runGame() {
  try {
    updateStatus('正在啟動遊戲...請稍候...', 'loading');
    els.placeholderText.style.display = 'block';
    els.cameraStream.style.display = 'none';
    const currentUid = await getUid() || 'default';
    
    const analysisResponse = await fetch('/run-python', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ id: id, uid: currentUid })
    });
    
    if (!analysisResponse.ok) throw new Error('啟動遊戲失敗');
    
    const analysisResult = await analysisResponse.json();
    if (!analysisResult.success) throw new Error(analysisResult.error || '啟動遊戲失敗');
    
    updateStatus('遊戲執行中...請查看彈出視窗。完成後將自動跳轉。', 'info');
    
    // ✅ 遊戲需要等待完成
    await pollTaskStatus(analysisResult.task_id);
    redirectToNextTask(id);
    
  } catch (error) {
    console.error('遊戲執行錯誤:', error);
    updateStatus(`遊戲失敗: ${error.message}`, 'error');
  }
}

// === [保留] 遊戲用的輪詢函數 ===
async function pollTaskStatus(taskId) {
  updateStatus('遊戲執行中...請查看彈出視窗。', 'loading');
  const startTime = Date.now();
  const timeout = 150000; // 150 秒
  
  while (Date.now() - startTime < timeout) {
    try {
      const res = await fetch(`/check-task/${taskId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'completed') {
          updateStatus(`遊戲完成！準備跳轉...`, 'success');
          await new Promise(r => setTimeout(r, 1500));
          return;
        } else if (data.status === 'error') {
          throw new Error(data.error || '遊戲執行失敗');
        }
      }
    } catch (e) {
      console.warn("Poll error:", e.message);
    }
    await new Promise(r => setTimeout(r, 2000));
  }
  
  updateStatus('遊戲逾時，將在背景繼續。準備跳轉...', 'info');
  await new Promise(r => setTimeout(r, 1500));
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

// 進入畫面時的初始化
document.addEventListener("DOMContentLoaded", async () => {
  if (id === "ch5-t1") {
    // Ch5-t1: 自動執行遊戲（需要等待）
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