// 共用 KEY；與你的其他頁一致
const KEY = "kid-quest-progress-v1";

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
let currentPhoto = null;
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
  }, 60); // 約 60ms 更新一次
}

// 開啟 OpenCV 相機（預覽一律先用鏡頭 1）
async function openCamera() {
  try {
    updateStatus('正在開啟相機...', 'loading');

    const response = await fetch('/opencv-camera/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: id, camera_index: 1 })
    });

    if (!response.ok) throw new Error('無法開啟相機');

    const result = await response.json();

    if (result.success) {
      cameraActive = true;
      updateStatus('相機已開啟，正在串流...', 'success');

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

// === 新增：輔助函式（開指定鏡頭並拍一張，以 suffix 當檔名尾巴）===
async function captureWithCamera(cameraIndex, fileSuffix, uid) {
  // 切換鏡頭（或確保指定鏡頭已開）
  const startRes = await fetch('/opencv-camera/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ task_id: id, camera_index: cameraIndex })
  });
  if (!startRes.ok) throw new Error('相機切換失敗');

  // 稍等讓鏡頭穩定
  await new Promise(r => setTimeout(r, 300));

  // 依照後端規則：會用 task_id 直接當檔名 (task_id.jpg)
  // 所以我們把 task_id 傳成 `${id}-side` / `${id}-top`（或 just id）
  const capRes = await fetch('/opencv-camera/capture', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ task_id: fileSuffix, uid: uid })
  });

  if (!capRes.ok) throw new Error('拍照失敗');
  const result = await capRes.json();
  if (!result.success) throw new Error(result.error || '拍照失敗');

  console.log(`已拍攝鏡頭 ${cameraIndex}: ${result.filename}`);
}



// === 修改後：只有 ch1-t2 / ch1-t3 拍兩張（1=side → 0=top），其他任務拍一張（1）===
async function takeShot() {
  try {
    const currentUid = await getUid() || 'default';

    //SIDE = 側面攝影機號碼 TOP = 上面攝影機號碼
    SIDE = 1;
    TOP = 0;
    //===================================//
    if (["ch1-t2", "ch1-t3"].includes(id)) {
      updateStatus('正在拍攝側面鏡頭...', 'loading');
      await captureWithCamera(SIDE, `${id}-side`, currentUid);

      updateStatus('側面完成，切換上方鏡頭...', 'loading');
      await captureWithCamera(TOP, `${id}-top`, currentUid);
    } else {
      updateStatus('正在拍照（鏡頭 1）...', 'loading');
      await captureWithCamera(SIDE, id, currentUid);
    }

    updateStatus('照片拍攝完成，開始分析...', 'success');

    // 背景分析
    const analysisResponse = await fetch('/run-python', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ id: id })
    });
    const analysisResult = await analysisResponse.json();
    if (analysisResult.success) {
      console.log(`分析任務已在背景執行，task_id: ${analysisResult.task_id}`);
    }

    // 關閉相機
    await closeCamera();

    // 跳轉到下一個任務
    const TASK_IDS = Object.keys(ID_TO_META);
    const idx = TASK_IDS.indexOf(id);
    const nextTaskId = (idx >= 0 && idx < TASK_IDS.length - 1) ? TASK_IDS[idx + 1] : null;

    if (nextTaskId){
      location.href = `/html/task.html?id=${nextTaskId}`;
    } else {
      location.href = "/html/index.html";
    }

  } catch (error) {
    console.error('拍照錯誤:', error);
    updateStatus(`拍照失敗: ${error.message}`, 'error');
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

// 進入畫面自動開鏡頭（先開 1 當預覽）
document.addEventListener("DOMContentLoaded", async () => {
  await openCamera();
});

// 頁面卸載時清理資源
window.addEventListener("beforeunload", () => {
  if (streamInterval) clearInterval(streamInterval);
  if (cameraActive) closeCamera();
});
