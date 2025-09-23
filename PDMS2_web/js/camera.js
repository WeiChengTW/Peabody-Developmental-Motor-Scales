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

// 你在 task.js / script.js 裡的 setIcon 寫法是把 <img> 塞進容器；這裡做個最小版
function isImagePath(s){ return typeof s==="string" && (s.startsWith("/images/") || /\.(png|jpe?g|svg|webp|gif)$/i.test(s)); }
function setIcon(el, src){
  if(!el) return;
  el.innerHTML = isImagePath(src) ? `<img class="icon-img" src="${src}" alt="">` : "";
}

// 從 localStorage 取出 TASK_MAP 對應的小圖/標題資料（簡化：從 id 推回你已有的映射）
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
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  openBtn: document.getElementById("openBtn"),
  shotBtn: document.getElementById("shotBtn"),
  retakeBtn: document.getElementById("retakeBtn"),
  saveBtn: document.getElementById("saveBtn"),
};

let stream = null;
let lastDataURL = null;
const id = getId();

// 初始化標題與小圖
(function initHeader(){
  const meta = ID_TO_META[id] || {icon:"", title:"拍照存證"};
  setIcon(els.taskIcon, meta.icon);
  if(meta.title) els.taskTitle.textContent = meta.title;
})();

// 啟動相機：偏好後鏡頭
async function openCamera(){
  try{
    // 先停舊串流
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
    els.canvas.style.display = "none";
    els.video.style.display = "";

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 }, height: { ideal: 720 }
      }
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    els.video.srcObject = stream;
    await els.video.play();

    els.shotBtn.disabled = false;
    els.retakeBtn.disabled = true;
    els.saveBtn.disabled = true;
  }catch(err){
    alert("無法開啟相機，請確認已允許相機權限。\n" + err);
  }
}

// 拍照：把 video 畫面繪製到 canvas
function takeShot(){
  const vw = els.video.videoWidth || 1280;
  const vh = els.video.videoHeight || 720;
  els.canvas.width = vw;
  els.canvas.height = vh;
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0, vw, vh);

  lastDataURL = els.canvas.toDataURL("image/jpeg", 0.92);
  els.video.style.display = "none";
  els.canvas.style.display = "";
  els.retakeBtn.disabled = false;
  els.saveBtn.disabled = false;
}

// 重拍
function retake(){
  els.canvas.style.display = "none";
  els.video.style.display = "";
  els.saveBtn.disabled = true;
}

// 存檔到本地（下載）＋ 存到 localStorage ＋ 背景執行分析
async function saveAndBack(){
  if(!lastDataURL){
    alert("請先拍一張照片喔！");
    return;
  }

  try {
    // 下載一份
    const a = document.createElement("a");
    a.href = lastDataURL;
    a.download = `${id}.jpg`;
    a.click();
    
    // 獲取當前的 uid
    const currentUid = await getUid() || "default";
    
    // 等待下載完成
    await new Promise(resolve => {
      a.onload = resolve;
      a.onerror = resolve;
      setTimeout(resolve, 1000); // 縮短等待時間
    });

    // 呼叫 move-photos API 把照片移到當前 uid 的資料夾下
    const moveResponse = await fetch('/move-photos', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        photos: [`${id}.jpg`] // 指定要移動的照片
      })
    });
    
    if (moveResponse.ok) {
      const moveResult = await moveResponse.json();
      console.log('照片移動結果:', moveResult);
      if (moveResult.success) {
        console.log(`成功移動 ${moveResult.moved_count} 個檔案到 ${currentUid} 資料夾`);
      } else {
        console.error('照片移動失敗:', moveResult.error);
      }
    } else {
      console.error('照片移動 API 請求失敗');
    }

    // 存進 localStorage（可供之後在首頁或管理模式顯示）
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    if(!st.photos) st.photos = {};
    st.photos[id] = lastDataURL; // 以任務 id 當索引
    localStorage.setItem(KEY, JSON.stringify(st));

    // 背景執行 Python 腳本進行分析（不等待結果）
    fetch('/run-python', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        id: id 
      })
    }).then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Python 腳本執行 API 請求失敗');
      }
    }).then(result => {
      console.log('Python 腳本啟動結果:', result);
      if (result.success && result.task_id) {
        console.log(`分析任務已在背景執行，task_id: ${result.task_id}`);
      } else {
        console.error('背景分析啟動失敗:', result.error);
      }
    }).catch(error => {
      console.error('背景分析執行錯誤:', error);
    });
    
    // 關掉相機
    if(stream){ 
      stream.getTracks().forEach(t=>t.stop()); 
      stream = null; 
    }
    
    // 立即回主頁
    location.href = "/html/index.html";
    
  } catch (error) {
    console.error('處理過程中發生錯誤:', error);
    alert('處理失敗：' + error.message);
  }
}

// 綁定
els.openBtn.addEventListener("click", openCamera);
els.shotBtn.addEventListener("click", takeShot);
els.retakeBtn.addEventListener("click", retake);
els.saveBtn.addEventListener("click", saveAndBack);

// 頁面卸載時清理資源
window.addEventListener("beforeunload", () => {
  if(stream){ 
    stream.getTracks().forEach(t=>t.stop()); 
    stream = null; 
  }
});

// 一些瀏覽器（iOS Safari）需要使用者互動才允許 getUserMedia，
// 因此不自動開；但如果你想嘗試自動開啟，可在 DOMContentLoaded 呼叫 openCamera()。
// window.addEventListener("DOMContentLoaded", openCamera);