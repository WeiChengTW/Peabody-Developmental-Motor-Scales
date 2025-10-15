// camera.js v1.6 — ch5-t1 走即時評分（強制重啟），一般任務走相機預覽 + 拍照

const KEY = "kid-quest-progress-v1";

function getId() {
  const u = new URL(location.href);
  return (u.searchParams.get("id") || "").toLowerCase();
}

async function getUid() {
  try {
    const r = await fetch("/session/get-uid");
    if (r.ok) {
      const j = await r.json();
      return j.uid;
    }
  } catch {}
  const st = JSON.parse(localStorage.getItem(KEY) || "{}");
  return st.currentUid || null;
}

function isImagePath(s){
  return typeof s==="string" && (s.startsWith("/images/") || /\.(png|jpe?g|svg|webp|gif)$/i.test(s));
}
function setIcon(el, src){
  if (el) el.innerHTML = isImagePath(src) ? `<img class="icon-img" src="${src}" alt="">` : "";
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
  "ch4-t1": {icon:"/images/fold1.jpg",   title:"摺紙一摺：變出小飛毯"},
  "ch4-t2": {icon:"/images/fold2.jpg",   title:"摺紙兩摺：更結實的飛毯"},
  "ch5-t1": {icon:"/images/beans.jpg",   title:"豆豆裝罐子：完成任務"},
};

const els = {
  taskIcon: document.getElementById("taskIcon"),
  taskTitle: document.getElementById("taskTitle"),
  statusInfo: document.getElementById("statusInfo"),

  // 一般相機
  cameraPanel: document.getElementById("cameraPanel"),
  cameraStream: document.getElementById("cameraStream"),
  placeholderText: document.getElementById("placeholderText"),
  shotBtn: document.getElementById("shotBtn"),

  // 撿葡萄乾
  beansPanel: document.getElementById("beansPanel"),
  beansBtn: document.getElementById("beansBtn"),
  beansStream: document.getElementById("beansStream"),
  beansStatus: document.getElementById("beansStatus"),
};

let cameraActive = false;
let streamInterval = null;
let beansInterval = null;
const id = getId();

function updateStatus(msg, type="info"){
  if (!els.statusInfo) return;
  els.statusInfo.textContent = msg;
  els.statusInfo.className = `status-info ${type}`;
}

// 標題小圖
(function initHeader(){
  const meta = ID_TO_META[id] || {icon:"", title:"拍照存證"};
  setIcon(els.taskIcon, meta.icon);
  if (meta.title) els.taskTitle.textContent = meta.title;
})();

/* ========== 一般相機預覽 ========== */
function startVideoStream(){
  if (streamInterval) clearInterval(streamInterval);
  streamInterval = setInterval(() => {
    els.cameraStream.src = `/opencv-camera/frame?ts=${Date.now()}`;
  }, 150);
}

async function openCameraNormal(){
  try{
    updateStatus("正在開啟相機...", "loading");
    const r = await fetch("/opencv-camera/start", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ task_id: id, camera_index: 0 })
    });
    const j = await r.json().catch(()=>({}));
    if (!r.ok || !j.success) throw new Error(j.error || "無法開啟相機");

    cameraActive = true;
    if (els.placeholderText) els.placeholderText.style.display = "none";
    if (els.cameraStream) {
      els.cameraStream.style.display = "block";
      els.cameraStream.src = ""; // 先清空避免殘影
    }
    if (els.shotBtn) els.shotBtn.disabled = false;
    startVideoStream();
    updateStatus("相機已開啟，正在串流...", "success");
  }catch(e){
    console.error(e);
    updateStatus(`開啟相機失敗：${e.message}`, "error");
  }
}

async function closeCamera(){
  try{
    if (streamInterval) { clearInterval(streamInterval); streamInterval=null; }
    await fetch("/opencv-camera/stop", {method:"POST"});
  }finally{
    cameraActive = false;
    if (els.placeholderText) els.placeholderText.style.display = "block";
    if (els.cameraStream){
      els.cameraStream.style.display = "none";
      els.cameraStream.src = "";
    }
    if (els.shotBtn) els.shotBtn.disabled = true;
    updateStatus("相機已關閉", "info");
  }
}

async function captureWithCamera(cameraIndex, fileSuffix, uid){
  const startRes = await fetch("/opencv-camera/start", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ task_id: id, camera_index: cameraIndex })
  });
  if (!startRes.ok) throw new Error(`相機切換失敗：${startRes.status}`);
  await new Promise(r => setTimeout(r, 300));

  const capRes = await fetch("/opencv-camera/capture", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ task_id: fileSuffix, uid })
  });
  const txt = await capRes.text();
  if (!capRes.ok) throw new Error(`拍照失敗：${capRes.status} ${txt}`);
  const j = (()=>{ try{return JSON.parse(txt)}catch{return{}} })();
  if (!j.success) throw new Error(j.error || "拍照失敗");
}

/* ========== 撿葡萄乾（即時）========== */
async function startBeansFlow(){
  try{
    const uid = await getUid() || "default";

    // 1) 先嘗試停舊任務（best-effort，不阻塞）
    await fetch("/beans/stop", { method: "POST" });   // 等停完再開新任務
    await new Promise(r => setTimeout(r, 200));       // 等 0.2 秒確保 thread 結束
    updateStatus("正在啟動撿葡萄乾...", "loading");

    // 2) 強制重啟新的即時任務
    const res = await fetch("/beans/start?force=true", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ uid })
    });
    const js = await res.json().catch(()=>({}));
    if (!res.ok || !js.success) throw new Error(js.error || js.message || "啟動失敗");

    // 3) 切到 beans 面板與串流
    if (els.cameraPanel) els.cameraPanel.style.display = "none";
    if (els.beansPanel)  els.beansPanel.style.display  = "block";

    // 先清 src 再掛新串流，避免快取殘影
    if (els.beansStream){
      els.beansStream.src = "";
      els.beansStream.src = "/beans/stream";
    }
    if (els.beansStatus) els.beansStatus.textContent = "任務進行中…";
    updateStatus("任務進行中…", "success");

    // 4) 輪詢狀態直到結束
    if (beansInterval) clearInterval(beansInterval);
    beansInterval = setInterval(async () => {
      try{
        const s = await fetch("/beans/status");
        const st = await s.json();

        // 相機開啟失敗/執行錯誤時，顯示錯誤並收尾
        if (st.error && !st.running) {
          clearInterval(beansInterval);
          updateStatus(st.error === "camera_open_failed" ? "相機開啟失敗" : "任務執行錯誤", "error");
          if (els.beansStatus) els.beansStatus.textContent = "發生錯誤，請返回上一頁重試";
          if (els.beansStream) els.beansStream.src = "";
          return;
        }

        // 正常完成
        if (!st.running && st.score != null){
          clearInterval(beansInterval);
          if (els.beansStatus) els.beansStatus.textContent = `完成！分數：${st.score}`;
          updateStatus(`完成！分數：${st.score}`, "success");

          // 停止串流畫面（DB 已在後端寫完）
          if (els.beansStream) els.beansStream.src = "";

          // 自動跳轉下一關
          const TASK_IDS = Object.keys(ID_TO_META);
          const idx = TASK_IDS.indexOf(id);
          const nextTaskId = (idx>=0 && idx < TASK_IDS.length-1) ? TASK_IDS[idx+1] : null;
          setTimeout(()=>{
            location.href = nextTaskId ? `/html/task.html?id=${nextTaskId}` : "/html/index.html";
          }, 1200);
        }
      }catch(e){
        console.warn("beans/status 失敗", e);
      }
    }, 600);

  }catch(e){
    console.error("撿葡萄乾啟動失敗:", e);
    updateStatus(`撿葡萄乾啟動失敗：${e.message}`, "error");
  }
}

async function stopBeans(){
  try{
    if (beansInterval) { clearInterval(beansInterval); beansInterval = null; }
    await fetch("/beans/stop", { method:"POST", keepalive: true });
  }catch(e){
    // 忽略停止失敗
  }finally{
    if (els.beansStream) els.beansStream.src = "";
  }
}

/* ========== 一般任務：拍照 → 分析 → 跳下一關 ========== */
async function takeShot(){
  try{
    const uid = await getUid() || "default";
    const SIDE = 0, TOP = 1;

    if (["ch1-t2","ch1-t3"].includes(id)){
      updateStatus("正在拍攝側面鏡頭...", "loading");
      await captureWithCamera(SIDE, `${id}-side`, uid);
      updateStatus("側面完成，切換上方鏡頭...", "loading");
      await captureWithCamera(TOP, `${id}-top`, uid);
    }else{
      updateStatus("正在拍照（側面鏡頭）...", "loading");
      await captureWithCamera(SIDE, id, uid);
    }

    updateStatus("照片拍攝完成，開始分析...", "success");

    const analysisResponse = await fetch("/run-python", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ id })
    });
    const txt = await analysisResponse.text();
    try{
      const j = JSON.parse(txt);
      if (!(analysisResponse.ok && j.success)){
        console.warn("分析 API 回應：", analysisResponse.status, txt);
      }
    }catch{
      console.warn("分析回應非 JSON：", txt);
    }

    await closeCamera();

    // 下一題
    const TASK_IDS = Object.keys(ID_TO_META);
    const idx = TASK_IDS.indexOf(id);
    const nextTaskId = (idx>=0 && idx < TASK_IDS.length-1) ? TASK_IDS[idx+1] : null;
    location.href = nextTaskId ? `/html/task.html?id=${nextTaskId}` : "/html/index.html";

  }catch(e){
    console.error("拍照錯誤:", e);
    updateStatus(`拍照失敗: ${e.message}`, "error");
  }
}

/* ========== 初始化：大家都調 openCamera()，但 ch5-t1 會轉去即時評分 ========== */
async function openCamera(){
  if (id === "ch5-t1"){
    // ch5-t1：隱藏一般相機按鈕（避免誤按），直接開即時任務
    if (els.shotBtn) els.shotBtn.style.display = "none";
    await startBeansFlow();
  }else{
    await openCameraNormal();
  }
}

/* 綁事件 + 開機 */
document.addEventListener("DOMContentLoaded", async () => {
  if (els.shotBtn)  els.shotBtn.addEventListener("click", takeShot);
  if (els.beansBtn) els.beansBtn.addEventListener("click", startBeansFlow); // 保留手動開的按鈕
  await openCamera(); // 進入頁面就「開相機」；ch5-t1 會自動轉即時評分
});

window.addEventListener("beforeunload", () => {
  if (streamInterval) { clearInterval(streamInterval); streamInterval = null; }
  if (beansInterval)  { clearInterval(beansInterval);  beansInterval  = null; }
  if (cameraActive) closeCamera();
  if (id === "ch5-t1") stopBeans(); // ★ 確保離開時停止 beans 任務（避免倒數延續）
});
