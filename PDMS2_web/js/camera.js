// camera.js (修改版：移除錄影邏輯，Ch5-t1 視為靜態任務執行)

// 共用 KEY；與你的其他頁一致
const KEY = "kid-quest-progress-v1";
// ======相機參數 (對應 app.py) =====
const TOP = 2; 
const SIDE = 3; 
const waittime = 3;



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
  "ch1-t1": {icon:"/images/bridge.jpg",  title:"串積木：做成一條橋"},
  "ch1-t2": {icon:"/images/tower.jpg",   title:"疊城堡：蓋瞭望塔"},
  "ch1-t3": {icon:"/images/stairs.jpg",  title:"疊階梯：翻過高牆"},
  "ch2-t1": {icon:"/images/circle.jpg",  title:"畫圓：大圓圓魔法陣"},
  "ch2-t2": {icon:"/images/square.jpg",  title:"畫方：守護盾"},
  "ch2-t3": {icon:"/images/cross.jpg",   title:"畫十字：啟動魔法"},
  "ch2-t4": {icon:"/images/line.jpg",    title:"描水平線：打敗恐龍"},
  "ch2-t5": {icon:"/images/fill.jpg",    title:"兩水平線中塗色：提升威力"},
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
  stopBtn: document.getElementById("stopBtn"), // 雖然用不到，但保留 DOM 引用
};
// 狀態變量
let cameraActive = false;
let streamInterval = null;
// 移除 isRecording, recordingTimer, currentRecordingUid
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
  
  if (els.stopBtn) {
     els.stopBtn.style.display = 'none'; // 永遠隱藏停止按鈕
  }
  
  if (id === "ch5-t1") {
    // Ch5-t1 現在是啟動 main.py 遊戲
    els.shotBtn.textContent = "開始遊戲並錄影"; 
    els.shotBtn.title = "點擊後將在獨立視窗中啟動遊戲和錄影。";
  } else {
     // 靜態任務
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
      } else {
        console.warn("相機畫面失敗:", data.error);
      }
    } catch (err) {
      console.error("獲取幀錯誤:", err);
    }
  }, 30);
}
// 開啟 OpenCV 相機
async function openCamera() {
  try {
    updateStatus('正在開啟相機...', 'loading');
    let CAM_INDEX = TOP;
    // Ch5-t1 預覽 SIDE 鏡頭
    if(["ch1-t2", "ch1-t3", 'ch5-t1'].includes(id)) CAM_INDEX = SIDE;
    
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

// === 背景觸發分析（用於 Ch5-t1 啟動遊戲）===
async function triggerBackgroundAnalysis(taskId, uid) {
  try {

    // 發送分析請求到後端
    let body_data = { 
        id: taskId,
        uid: uid 
    };
    
    // 僅針對 Ch5-t1 傳遞相機索引
    if (taskId === "ch5-t1") {
        body_data.cam_index = SIDE; 
    }else {
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
    // 返回任務 ID 以便追蹤
    return result.task_id;
    
  } catch (error) {
    console.warn('觸發分析時發生錯誤:', error);
  }
}

// === 跳轉函數 ===
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

// === 拍照/開始遊戲主函數 ===
async function takeShot() {
  try {
    const currentUid = await getUid() || 'default';
    els.shotBtn.disabled = true;
    
    await closeCamera(); // 拍照或遊戲開始前，先關閉預覽相機
    
    if (id === "ch5-t1") {
      // Ch5-t1：直接啟動 main.py，遊戲/錄影在獨立視窗中進行
      updateStatus('正在啟動遊戲視窗...', 'loading');
      
      // 觸發 run-python (Ch5-t1)
      await triggerBackgroundAnalysis(id, currentUid);
      
      updateStatus('遊戲視窗已開啟，請在獨立視窗中操作。完成後自動跳轉...', 'success');
      
      // 不等待遊戲結束，直接跳轉，讓使用者在遊戲結束後手動跳轉或等待後台完成
      // 這裡採用直接跳轉，並假設 main.py 會在背景運行完畢。
      
    } else if (["ch1-t2", "ch1-t3"].includes(id)) {
        await countdown(waittime);

      // 雙鏡頭任務：拍照 -> 跳轉
      updateStatus('正在拍攝側面鏡頭...', 'loading');
      await captureWithCamera(SIDE, `${id}-side`, currentUid);
      
      updateStatus('側面完成，切換上方鏡頭...', 'loading');
      await captureWithCamera(TOP, `${id}-top`, currentUid);

      updateStatus('照片拍攝完成！背景分析已啟動，準備跳轉...', 'success');
        } else {
        await countdown(waittime);
          // 單鏡頭任務：拍照 -> 跳轉
          updateStatus('正在拍照（上方鏡頭）...', 'loading');
          await captureWithCamera(TOP, id, currentUid);
      
      updateStatus('照片拍攝完成！背景分析已啟動，準備跳轉...', 'success');
    }

    // 短暫延遲後直接跳轉 (Ch5-t1 遊戲視窗需要使用者自己關閉/結束)
    if (id !== "ch5-t1") {
        await new Promise(r => setTimeout(r, 800));
        redirectToNextTask(id);
    } else {
    
        els.shotBtn.disabled = false; 
        els.shotBtn.textContent = "遊戲已啟動，點此跳轉下一任務";
        els.shotBtn.removeEventListener("click", shotBtnClickHandler);
        els.shotBtn.addEventListener("click", () => redirectToNextTask(id));

    }
    
  } catch (error) {
    console.error('操作錯誤:', error);
    updateStatus(`操作失敗: ${error.message}`, 'error');
    els.shotBtn.disabled = false;
    await openCamera(); // 失敗時重新開啟預覽
  }
}

// 倒數計時函數
async function countdown(seconds) {
  for (let i = seconds; i > 0; i--) {
    updateStatus(`準備拍照... ${i}`, 'loading');
    await new Promise(r => setTimeout(r, 1000));
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

// === [修正] 按鈕點擊事件處理函數 (用於移除/新增監聽) ===
function shotBtnClickHandler() {
    // Ch5-t1 和其他任務都呼叫 takeShot，邏輯在 takeShot 內部分流
    takeShot(); 
}

// === 按鈕點擊事件及移除停止按鈕事件 ===
els.shotBtn.addEventListener("click", shotBtnClickHandler);

// 移除停止按鈕點擊事件

// 進入畫面時的初始化
document.addEventListener("DOMContentLoaded", async () => {
  await openCamera();
});

// 頁面卸載時清理資源
window.addEventListener("beforeunload", () => {
  if (streamInterval) clearInterval(streamInterval);
  // 移除 recordingTimer 清理
  if (cameraActive) closeCamera();
});