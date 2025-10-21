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
let isRecording = false;
let recordingTimer = null;
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
    els.shotBtn.textContent = "開始錄影"; // 改按鈕文字
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

// === [新增] 開始錄影 1 分鐘 ===
async function startRecording() {
  try {
    const currentUid = await getUid() || 'default';
    els.shotBtn.disabled = true;
    isRecording = true;
    
    updateStatus('正在開始錄影...', 'loading');
    
    // 發送錄影開始請求到後端
    const startResponse = await fetch('/opencv-camera/record-start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        task_id: id,
        uid: currentUid,
        duration: 60
      })
    });
    
    if (!startResponse.ok) {
      throw new Error('錄影啟動失敗');
    }
    
    const startResult = await startResponse.json();
    if (!startResult.success) {
      throw new Error(startResult.error || '錄影啟動失敗');
    }
    
    updateStatus('錄影中... 60 秒', 'success');
    
    // 倒計時 60 秒
    let remainingSeconds = 60;
    recordingTimer = setInterval(() => {
      remainingSeconds--;
      if (remainingSeconds > 0) {
        updateStatus(`錄影中... ${remainingSeconds} 秒`, 'success');
      } else {
        clearInterval(recordingTimer);
        recordingTimer = null;
        stopRecording(currentUid);
      }
    }, 1000);
    
  } catch (error) {
    console.error('錄影錯誤:', error);
    updateStatus(`錄影失敗: ${error.message}`, 'error');
    isRecording = false;
    els.shotBtn.disabled = false;
  }
}

// === [新增] 結束錄影 ===
async function stopRecording(uid) {
  try {
    if (recordingTimer) {
      clearInterval(recordingTimer);
      recordingTimer = null;
    }
    
    updateStatus('正在結束錄影並保存...', 'loading');
    
    // 發送錄影停止請求到後端
    const stopResponse = await fetch('/opencv-camera/record-stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        task_id: id,
        uid: uid,
        save_to_file: true
      })
    });
    
    if (!stopResponse.ok) {
      throw new Error('結束錄影失敗');
    }
    
    const stopResult = await stopResponse.json();
    if (!stopResult.success) {
      throw new Error(stopResult.error || '結束錄影失敗');
    }
    
    // 取得影片保存信息
    const videoPath = stopResult.video_path || '影片已保存';
    updateStatus(`錄影完成！${videoPath}`, 'success');
    isRecording = false;
    
    // 關閉相機
    await closeCamera();
    
    // 錄完影之後執行 run-python 分析（等待結果）
    updateStatus('正在分析錄影內容...', 'loading');
    try {
      const analysisResponse = await fetch('/run-python', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          id: id,
          uid: uid,
          video_path: stopResult.video_path
        })
      });
      
      if (analysisResponse.ok) {
        const analysisResult = await analysisResponse.json();
        console.log('分析完成:', analysisResult);
        updateStatus('分析完成！準備跳轉...', 'success');
      } else {
        console.warn('分析失敗');
        updateStatus('分析完成，準備跳轉...', 'success');
      }
    } catch (error) {
      console.warn('分析錯誤:', error);
      updateStatus('分析完成，準備跳轉...', 'success');
    }
    
    // 短暫延遲後直接跳轉
    await new Promise(r => setTimeout(r, 800));
    redirectToNextTask(id);
    
  } catch (error) {
    console.error('停止錄影錯誤:', error);
    updateStatus(`停止錄影失敗: ${error.message}`, 'error');
    isRecording = false;
    els.shotBtn.disabled = false;
  }
}

// === [修改] 背景觸發分析（不等待結果）===
async function triggerBackgroundAnalysis(taskId, uid) {
  try {
    // 發送分析請求到後端
    const response = await fetch('/run-python', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        id: taskId,
        uid: uid 
      })
    });
    
    if (!response.ok) {
      console.warn('分析請求失敗');
      return;
    }
    
    const result = await response.json();
    console.log('分析已觸發:', result);
    
  } catch (error) {
    console.warn('觸發分析時發生錯誤:', error);
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

// === [修改] 按鈕點擊事件 ===
els.shotBtn.addEventListener("click", () => {
  if (id === "ch5-t1") {
    startRecording(); // Ch5-t1 觸發錄影
  } else {
    takeShot(); // 其他任務觸發拍照
  }
});

// 進入畫面時的初始化
document.addEventListener("DOMContentLoaded", async () => {
  // Ch5-t1 也開啟相機預覽（不自動錄影，等待按鈕）
  await openCamera();
});

// 頁面卸載時清理資源
window.addEventListener("beforeunload", () => {
  if (streamInterval) clearInterval(streamInterval);
  if (recordingTimer) clearInterval(recordingTimer);
  if (cameraActive) closeCamera();
});