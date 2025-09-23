// 與主站共用的 localStorage key
const KEY = "kid-quest-progress-v1";

// 獲取當前的 uid（改用後端 session）
async function getCurrentUid() {
  try {
    const response = await fetch('/session/get-uid');
    if (response.ok) {
      const result = await response.json();
      return result.uid;
    } else {
      // 如果 session 中沒有 UID，嘗試從 localStorage 獲取
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      return st.currentUid || null;
    }
  } catch (error) {
    console.error('獲取 UID 時發生錯誤:', error);
    // 降級到 localStorage
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    return st.currentUid || null;
  }
}

// 任務內容（依 PDF）：ID → 顯示資料
// 圖片檔名放 /images 下（自行準備），沒有就用 emoji 顯示
const TASK_MAP = {
  // 第一關
  "ch1-t1": { emoji:"🧱", title:"串積木：做成一條橋",
    desc:"把魔法積木一顆顆串起來，讓我們過河",
    img:"/images/bridge.jpg",
    steps:[
      "把繩子從積木洞穿過去",
      "一顆接一顆串緊，不要鬆掉",
      "拉直看看，像橋一樣穩固就成功！"
    ]
  },
  "ch1-t2": { emoji:"🏰", title:"疊城堡：蓋瞭望塔",
    desc:"把魔法石頭一層一層疊高，找到前進方向。",
    img:"/video/toy.mp4",
    steps:[
      "從下面往上疊起來",
      "每疊一層就輕輕壓一下",
      "疊到和圖一樣就完成"
    ]
  },
  "ch1-t3": { emoji:"🪜", title:"疊階梯：翻過高牆",
    desc:"把方塊疊成樓梯，繼續前往魔法王國。",
    img:"/images/stairs.jpg",
    steps:[ "排出一階一階的形狀", "確認每格都踩得到", "小心地走上去！" ]
  },

  // 第二關
  "ch2-t1": { emoji:"⭕", title:"畫圓：大圓圓魔法陣",
    desc:"在紙上畫一個大圓圈。",
    img:"/images/circle.jpg",
    steps:[ "跟著指示圖，照著畫圓", "閉合成完整的圓" ]
  },
  "ch2-t2": { emoji:"🟦", title:"畫方：守護盾",
    desc:"畫一個正正方方的盾牌。",
    img:"/images/square.jpg",
    steps:[ "畫一條水平線", "畫一條垂直線", "連成四個直角的方形" ]
  },
  "ch2-t3": { emoji:"➕", title:"畫十字：啟動魔法",
    desc:"畫出十字星，讓魔法運作起來。",
    img:"/video/cross.mp4",
    steps:[ "先畫一條直線", "再畫一條與之垂直的直線", "兩線交叉在中心" ]
  },
  "ch2-t4": { emoji:"📏", title:"描水平線：打敗恐龍",
    desc:"先用一條直線攻擊牠。",
    img:"/images/line.jpg",
    steps:[ "把尺壓穩", "沿著尺邊畫一條直線", "確認線是水平的" ]
  },
  "ch2-t5": { emoji:"🖍️", title:"兩水平線中塗色：提升威力",
    desc:"把兩條水平線之間塗滿顏色！",
    img:"/images/fill.jpg",
    steps:[ "畫第二條平行線", "找到兩線之間的空間", "把空間均勻塗滿" ]
  },
  "ch2-t6": { emoji:"✨", title:"兩點連線：開門",
    desc:"把兩顆星星連起來，打開門！",
    img:"/video/line.mp4",
    steps:[ "找到兩個點", "直直地畫線連起來", "檢查有沒有超出" ]
  },

  // 第三關
  "ch3-t1": { emoji:"✂️", title:"剪圓：做圓形窗戶",
    desc:"幫小精靈剪出一個圓窗。",
    img:"/video/circle.mp4",
    steps:[ "沿著畫好的圓慢慢剪", "手要轉，剪刀慢慢剪", "合上看看圓不圓" ]
  },
  "ch3-t2": { emoji:"📐", title:"剪方：做方方正正的門",
    desc:"幫小精靈剪出正方形的門。",
    img:"/images/square_door.jpg",
    steps:[ "剪直線四邊", "角角對齊成直角", "把邊修整整齊" ]
  },

  // 第四關
  "ch4-t1": { emoji:"🗞️", title:"摺紙一摺：變出小飛毯",
    desc:"把紙對摺一次。",
    img:"/images/fold1.jpg",
    steps:[ "邊對邊", "對齊後再摺", "把摺痕壓緊" ]
  },
  "ch4-t2": { emoji:"🧺", title:"摺紙兩摺：更結實的飛毯",
    desc:"再摺一次，就能起飛！",
    img:"/images/fold2.jpg",
    steps:[ "再對摺一次", "壓出清楚摺痕", "展開檢查是否對齊" ]
  },

  // 第五關
  "ch5-t1": { emoji:"🫘", title:"豆豆裝罐子：完成任務",
    desc:"把彩色豆豆一顆一顆裝進罐子裡。",
    img:"/images/beans.jpg",
    steps:[ "打開罐子", "一顆一顆放進去", "蓋緊蓋子" ]
  }
};

// 讀取 query string
function getId(){
  const u = new URL(location.href);
  return u.searchParams.get("id");
}

// 載入任務資料
function render(){
  const id = getId();
  const data = TASK_MAP[id];
  if(!data){ location.replace("index.html"); return; }

  document.title = `${data.title}｜任務操作`;
  document.getElementById("emoji").textContent = data.emoji;
  document.getElementById("title").textContent = data.title;
  document.getElementById("desc").textContent  = data.desc;

  const img = document.getElementById("img");
  const video = document.getElementById("video");
  const imgEmoji = document.getElementById("imgEmoji");
  if(data.img){
    if(data.img.endsWith('.mp4')){
      // 顯示影片
      if(video){
        video.src = data.img;
        video.style.display = "block";
        video.controls = true;
      }
      if(img) img.style.display = "none";
      imgEmoji.style.display = "none";
    }else{
      // 顯示圖片
      if(img){
        img.src = data.img;
        img.onload = ()=>{ img.style.display="block"; imgEmoji.style.display="none"; if(video) video.style.display="none"; };
        img.onerror = ()=>{ img.style.display="none"; imgEmoji.style.display="block"; if(video) video.style.display="none"; };
      }
      if(video) video.style.display = "none";
    }
  }

  const ol = document.getElementById("steps");
  ol.innerHTML = "";
  (data.steps||[]).forEach(s=>{
    const li = document.createElement("li");
    li.textContent = s;
    ol.appendChild(li);
  });

  // 讀指示
  document.getElementById("readBtn").onclick = ()=>{
    const text = `${data.title}。${data.desc}。步驟：` + (data.steps||[]).join("、");
    const u = new SpeechSynthesisUtterance(text);
    u.lang="zh-TW"; speechSynthesis.cancel(); 
    speechSynthesis.speak(u);
  };
  
  // 停止朗讀
  document.getElementById("stopBtn").onclick = ()=>{
    speechSynthesis.cancel();   // 馬上停止
  };

  // 完成 → 寫回主頁進度並返回
  document.getElementById("doneBtn").onclick = ()=>{
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    // 解析 chX-tY
    const [ch,t] = id.split("-").map(s=>parseInt(s.replace(/\D/g,""),10));
    if(!st.done) st.done = {};
    const chKey = `ch${ch}`;
    if(!Array.isArray(st.done[chKey])) st.done[chKey] = [];
    st.done[chKey][t-1] = true;
    localStorage.setItem(KEY, JSON.stringify(st));

    // 小煙火 + 返回
    celebrate();
    setTimeout(()=>{
      location.href = `/html/camera.html?id=${encodeURIComponent(id)}`;
    }, 800);
  };
}

// 簡易彩紙
function celebrate(){
  const box = document.getElementById("confetti");
  box.innerHTML="";
  const pieces = "🎉🎈✨💫⭐🍬".split("");
  for(let i=0;i<24;i++){
    const s = document.createElement("span");
    s.textContent = pieces[Math.floor(Math.random()*pieces.length)];
    s.style.left = Math.random()*100 + "vw";
    s.style.top = "-10vh";
    s.style.transform = `translateY(0) rotate(${Math.random()*90}deg)`;
    box.appendChild(s);
  }
  box.classList.add("active");
  setTimeout(()=>box.classList.remove("active"), 700);
}

render();

// ========= 安全返回處理 =========
const HOME = "/html/index.html";
function safeBack(e){
  if(e) e.preventDefault();
  // 有前一頁且同一網域，就回上一頁；否則回首頁
  if (document.referrer) {
    try {
      const prev = new URL(document.referrer);
      if (prev.origin === location.origin) { 
        history.back(); 
        return; 
      }
    } catch (err) {}
  }
  location.href = HOME;
}

// 綁定兩顆返回按鈕
const topBtn = document.getElementById("backBtn");
const bottomBtn = document.getElementById("backBtnBottom");
if (topBtn) topBtn.addEventListener("click", safeBack);
if (bottomBtn) bottomBtn.addEventListener("click", safeBack);