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
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      return st.currentUid || null;
    }
  } catch (error) {
    console.error('獲取 UID 時發生錯誤:', error);
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    return st.currentUid || null;
  }
}

/* ========= SVG 圖示庫 ========= */
const SVG_ICONS = {
  bridge: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="60" width="80" height="8" fill="#8B4513"/>
    <rect x="10" y="50" width="8" height="20" fill="#654321"/>
    <rect x="42" y="50" width="8" height="20" fill="#654321"/>
    <rect x="82" y="50" width="8" height="20" fill="#654321"/>
    <path d="M 15 60 Q 30 40, 45 60" stroke="#654321" stroke-width="2" fill="none"/>
    <path d="M 45 60 Q 60 40, 75 60" stroke="#654321" stroke-width="2" fill="none"/>
  </svg>`,
  
  castle: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="30" y="40" width="40" height="50" fill="#A9A9A9"/>
    <rect x="25" y="30" width="10" height="15" fill="#808080"/>
    <rect x="42" y="30" width="10" height="15" fill="#808080"/>
    <rect x="65" y="30" width="10" height="15" fill="#808080"/>
    <rect x="42" y="60" width="16" height="30" fill="#654321"/>
    <polygon points="50,20 40,35 60,35" fill="#DC143C"/>
  </svg>`,
  
  stairs: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="70" width="20" height="20" fill="#8B4513"/>
    <rect x="30" y="55" width="20" height="35" fill="#A0522D"/>
    <rect x="50" y="40" width="20" height="50" fill="#8B4513"/>
    <rect x="70" y="25" width="20" height="65" fill="#A0522D"/>
  </svg>`,
  
  wall: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="20" width="35" height="15" fill="#B22222" stroke="#8B0000" stroke-width="1"/>
    <rect x="55" y="20" width="35" height="15" fill="#B22222" stroke="#8B0000" stroke-width="1"/>
    <rect x="10" y="40" width="35" height="15" fill="#CD5C5C" stroke="#8B0000" stroke-width="1"/>
    <rect x="55" y="40" width="35" height="15" fill="#CD5C5C" stroke="#8B0000" stroke-width="1"/>
    <rect x="10" y="60" width="35" height="15" fill="#B22222" stroke="#8B0000" stroke-width="1"/>
    <rect x="55" y="60" width="35" height="15" fill="#B22222" stroke="#8B0000" stroke-width="1"/>
  </svg>`,
  
  circle: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="35" fill="none" stroke="#FF69B4" stroke-width="4"/>
  </svg>`,
  
  square: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="20" y="20" width="60" height="60" fill="none" stroke="#4169E1" stroke-width="4"/>
  </svg>`,
  
  cross: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <line x1="50" y1="15" x2="50" y2="85" stroke="#FF6347" stroke-width="6" stroke-linecap="round"/>
    <line x1="15" y1="50" x2="85" y2="50" stroke="#FF6347" stroke-width="6" stroke-linecap="round"/>
  </svg>`,
  
  line: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <line x1="10" y1="50" x2="90" y2="50" stroke="#32CD32" stroke-width="4" stroke-linecap="round"/>
  </svg>`,
  
  paint: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <line x1="10" y1="35" x2="90" y2="35" stroke="#000" stroke-width="2"/>
    <line x1="10" y1="65" x2="90" y2="65" stroke="#000" stroke-width="2"/>
    <rect x="10" y="37" width="80" height="26" fill="#FFD700" opacity="0.7"/>
  </svg>`,
  
  connect: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="25" cy="50" r="8" fill="#FFD700"/>
    <circle cx="75" cy="50" r="8" fill="#FFD700"/>
    <line x1="25" y1="50" x2="75" y2="50" stroke="#4169E1" stroke-width="3"/>
  </svg>`,
  
  scissorsCircle: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="30" fill="none" stroke="#4169E1" stroke-width="2" stroke-dasharray="5,3"/>
    <path d="M 20 30 L 35 45 M 80 30 L 65 45" stroke="#DC143C" stroke-width="3" stroke-linecap="round"/>
    <circle cx="20" cy="25" r="5" fill="#DC143C"/>
    <circle cx="80" cy="25" r="5" fill="#DC143C"/>
  </svg>`,
  
  scissorsSquare: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="30" y="30" width="40" height="40" fill="none" stroke="#4169E1" stroke-width="2" stroke-dasharray="5,3"/>
    <path d="M 20 25 L 35 40 M 80 25 L 65 40" stroke="#DC143C" stroke-width="3" stroke-linecap="round"/>
    <circle cx="20" cy="20" r="5" fill="#DC143C"/>
    <circle cx="80" cy="20" r="5" fill="#DC143C"/>
  </svg>`,
  
  paper: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="30,20 70,20 70,80 30,80" fill="#FFF8DC" stroke="#DAA520" stroke-width="2"/>
    <line x1="40" y1="35" x2="60" y2="35" stroke="#DAA520" stroke-width="1"/>
    <line x1="40" y1="45" x2="60" y2="45" stroke="#DAA520" stroke-width="1"/>
    <line x1="40" y1="55" x2="60" y2="55" stroke="#DAA520" stroke-width="1"/>
  </svg>`,
  
  foldOnce: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="25,30 50,30 50,70 25,70" fill="#FFF8DC" stroke="#DAA520" stroke-width="2"/>
    <polygon points="50,30 75,30 75,70 50,70" fill="#F5DEB3" stroke="#DAA520" stroke-width="2"/>
    <line x1="50" y1="30" x2="50" y2="70" stroke="#DAA520" stroke-width="2" stroke-dasharray="3,3"/>
  </svg>`,
  
  foldTwice: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="20,35 35,35 35,65 20,65" fill="#FFF8DC" stroke="#DAA520" stroke-width="1.5"/>
    <polygon points="35,35 50,35 50,65 35,65" fill="#F5DEB3" stroke="#DAA520" stroke-width="1.5"/>
    <polygon points="50,35 65,35 65,65 50,65" fill="#DEB887" stroke="#DAA520" stroke-width="1.5"/>
    <polygon points="65,35 80,35 80,65 65,65" fill="#D2B48C" stroke="#DAA520" stroke-width="1.5"/>
  </svg>`,
  
  beans: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="35" cy="45" rx="8" ry="12" fill="#FF69B4"/>
    <ellipse cx="50" cy="50" rx="8" ry="12" fill="#4169E1"/>
    <ellipse cx="65" cy="48" rx="8" ry="12" fill="#32CD32"/>
    <ellipse cx="42" cy="62" rx="8" ry="12" fill="#FFD700"/>
    <ellipse cx="58" cy="65" rx="8" ry="12" fill="#FF6347"/>
  </svg>`
};

/* ========= 慶祝彩紙 SVG 圖示 ========= */
const CELEBRATION_SVG = {
  party: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="70" r="8" fill="#FF69B4"/>
    <rect x="48" y="30" width="4" height="40" fill="#FFD700"/>
    <path d="M 35 25 L 50 30 L 45 15 Z" fill="#FF6347"/>
    <path d="M 65 25 L 50 30 L 55 15 Z" fill="#4169E1"/>
    <path d="M 40 35 L 50 30 L 38 20 Z" fill="#32CD32"/>
    <path d="M 60 35 L 50 30 L 62 20 Z" fill="#FF1493"/>
  </svg>`,
  
  balloon: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="50" cy="45" rx="18" ry="22" fill="#FF69B4"/>
    <ellipse cx="45" cy="38" rx="6" ry="8" fill="#FFB6C1" opacity="0.6"/>
    <path d="M 50 67 Q 48 75, 50 82" stroke="#666" stroke-width="2" fill="none"/>
    <path d="M 48 82 L 50 82 L 52 82 L 50 88 Z" fill="#DC143C"/>
  </svg>`,
  
  sparkle: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="50,20 55,45 80,50 55,55 50,80 45,55 20,50 45,45" fill="#FFD700"/>
    <polygon points="50,30 52,45 65,50 52,55 50,70 48,55 35,50 48,45" fill="#FFF"/>
  </svg>`,
  
  star: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="50,15 61,45 92,45 67,63 78,93 50,75 22,93 33,63 8,45 39,45" fill="#FFD700"/>
    <polygon points="50,25 57,45 75,45 62,55 68,73 50,63 32,73 38,55 25,45 43,45" fill="#FFF8DC"/>
  </svg>`,
  
  twinkle: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <line x1="50" y1="20" x2="50" y2="80" stroke="#87CEEB" stroke-width="3" stroke-linecap="round"/>
    <line x1="20" y1="50" x2="80" y2="50" stroke="#87CEEB" stroke-width="3" stroke-linecap="round"/>
    <line x1="30" y1="30" x2="70" y2="70" stroke="#ADD8E6" stroke-width="2" stroke-linecap="round"/>
    <line x1="70" y1="30" x2="30" y2="70" stroke="#ADD8E6" stroke-width="2" stroke-linecap="round"/>
    <circle cx="50" cy="50" r="8" fill="#FFF" opacity="0.8"/>
  </svg>`,
  
  candy: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="50" cy="50" rx="15" ry="18" fill="#FF69B4"/>
    <ellipse cx="45" cy="50" rx="7" ry="18" fill="#FF1493"/>
    <ellipse cx="55" cy="50" rx="7" ry="18" fill="#FFF"/>
    <rect x="48" y="20" width="4" height="15" fill="#DC143C"/>
    <rect x="48" y="65" width="4" height="15" fill="#DC143C"/>
    <path d="M 48 20 Q 45 18, 43 20 Q 41 22, 43 24 Q 45 26, 48 24" fill="#FFB6C1"/>
    <path d="M 52 76 Q 55 78, 57 76 Q 59 74, 57 72 Q 55 70, 52 72" fill="#FFB6C1"/>
  </svg>`
};

// 任務內容（依 PDF）：ID → 顯示資料
const TASK_MAP = {
  // 第一關
  "ch1-t1": { emoji:"bridge", title:"串積木：做成一條橋",
    desc:"把魔法積木一顆顆串起來，讓我們過河",
    img:"/video/ch1-t1.mp4",
    steps:[
      "把繩子從積木洞穿過去",
      "一顆接一顆串緊，不要鬆掉",
      "拉直看看，像橋一樣穩固就成功！"
    ]
  },
  "ch1-t2": { emoji:"castle", title:"疊城堡：蓋瞭望塔",
    desc:"把魔法石頭一層一層疊高，找到前進方向。",
    img:"/video/ch1-t2.mp4",
    steps:[
      "從下面往上疊起來",
      "每疊一層就輕輕壓一下",
      "疊到和圖一樣就完成"
    ]
  },
  "ch1-t3": { emoji:"stairs", title:"疊階梯：翻過高牆",
    desc:"把方塊疊成樓梯，繼續前往魔法王國。",
    img: () => Math.random() < 0.5 ? "/video/ch1-t3-L.mp4" : "/video/ch1-t3-R.mp4",
    steps:[ "排出一階一階的形狀", "確認每格都踩得到", "小心地走上去！" ]
  },
  "ch1-t4": { emoji:"bridge", title:"疊高牆：蓋出傳送門",
    desc:"把方塊推成一面大牆，變出傳送門。",
    img: "/video/ch1-t4.mp4",
    steps:[ "排出兩層大牆", "確認牆壁間不要有空隙", "慢慢的疊起來！" ]
  },

  // 第二關
  "ch2-t1": { emoji:"circle", title:"畫圓：大圓圓魔法陣",
    desc:"在紙上畫一個大圓圈。",
    img:"/video/ch2-t1.mp4",
    steps:[ "跟著指示圖，照著畫圓", "閉合成完整的圓" ]
  },
  "ch2-t2": { emoji:"square", title:"畫方：守護盾",
    desc:"畫一個正正方方的盾牌。",
    img:"/video/ch2-t2.mp4",
    steps:[ "畫一條水平線", "畫一條垂直線", "連成四個直角的方形" ]
  },
  "ch2-t3": { emoji:"cross", title:"畫十字：啟動魔法",
    desc:"畫出十字星，讓魔法運作起來。",
    img:"/video/ch2-t3.mp4",
    steps:[ "先畫一條直線", "再畫一條與之垂直的直線", "兩線交叉在中心" ]
  },
  "ch2-t4": { emoji:"line", title:"描水平線：打敗恐龍",
    desc:"先用一條直線攻擊牠。",
    img:"/video/ch2-t4.mp4",
    steps:[ "把尺壓穩", "沿著尺邊畫一條直線", "確認線是水平的" ]
  },
  "ch2-t5": { emoji:"paint", title:"兩水平線中塗色：提升威力",
    desc:"把兩條水平線之間塗滿顏色！",
    img:"/video/ch2-t5.mp4",
    steps:[ "畫第二條平行線", "找到兩線之間的空間", "把空間均勻塗滿" ]
  },
  "ch2-t6": { emoji:"connect", title:"兩點連線：開門",
    desc:"把兩顆星星連起來，打開門！",
    img:"/video/ch2-t6.mp4",
    steps:[ "找到兩個點", "直直地畫線連起來", "檢查有沒有超出" ]
  },

  // 第三關
  "ch3-t1": { emoji:"scissorsCircle", title:"剪圓：做圓形窗戶",
    desc:"幫小精靈剪出一個圓窗。",
    img:"/video/ch3-t1.mp4",
    steps:[ "沿著畫好的圓慢慢剪", "手要轉，剪刀慢慢剪", "合上看看圓不圓" ]
  },
  "ch3-t2": { emoji:"scissorsSquare", title:"剪方：做方方正正的門",
    desc:"幫小精靈剪出正方形的門。",
    img:"/video/ch3-t2.mp4",
    steps:[ "剪直線四邊", "角角對齊成直角", "把邊修整整齊" ]
  },

  // 第四關
  "ch4-t1": { emoji:"foldOnce", title:"摺紙一摺：變出小飛毯",
    desc:"把紙對摺一次。",
    img:"/video/ch4-t1.mp4",
    steps:[ "邊對邊", "對齊後再摺", "把摺痕壓緊" ]
  },
  "ch4-t2": { emoji:"foldTwice", title:"摺紙兩摺：更結實的飛毯",
    desc:"再摺一次，就能起飛！",
    img:"/video/ch4-t2.mp4",
    steps:[ "再對摺一次", "壓出清楚摺痕", "展開檢查是否對齊" ]
  },

  // 第五關
  "ch5-t1": { emoji:"beans", title:"豆豆裝罐子：完成任務",
    desc:"把彩色豆豆一顆一顆裝進罐子裡。",
    img:"/video/ch5-t1.mp4",
    steps:[ "打開罐子", "一顆一顆放進去", "蓋緊蓋子" ]
  }
};

// 讀取 query string
function getId(){
  const u = new URL(location.href);
  return u.searchParams.get("id");
}

function render(){
  const id = getId();
  const data = TASK_MAP[id];
  if(!data){ location.replace("index.html"); return; }

  document.title = `${data.title}｜任務操作`;
  document.getElementById("emoji").innerHTML = SVG_ICONS[data.emoji];
  document.getElementById("title").textContent = data.title;
  document.getElementById("desc").textContent  = data.desc;

  const img = document.getElementById("img");
  const video = document.getElementById("video");
  const imgEmoji = document.getElementById("imgEmoji");
  
  // 如果 img 是函數就呼叫它，否則直接用
  const imgSrc = typeof data.img === 'function' ? data.img() : data.img;
  
  if(imgSrc){
    if(imgSrc.endsWith('.mp4')){
      // 顯示影片
      if(video){
        video.src = imgSrc;
        video.style.display = "block";
        video.controls = true;
      }
      if(img) img.style.display = "none";
      imgEmoji.style.display = "none";
      
      // 如果是 ch1-t3，送給 Flask
      if(id === "ch1-t3"){
        const stair_type = imgSrc.includes("-L.mp4") ? "L" : "R";
        fetch("/save-stair-type", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ stair_type: stair_type })
        }).catch(err => console.error("save-stair-type 失敗:", err));
      }
    }else{
      // 顯示圖片
      if(img){
        img.src = imgSrc;
        img.onload = ()=>{ img.style.display="block"; imgEmoji.style.display="none"; if(video) video.style.display="none"; };
        img.onerror = ()=>{ img.style.display="none"; imgEmoji.style.display="block"; if(video) video.style.display="none"; };
      }
      if(video) video.style.display = "none";
    }
  }

  // 步驟清單
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
    speechSynthesis.cancel();
  };

  // 完成按鈕
  document.getElementById("doneBtn").onclick = ()=>{
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    // 解析 chX-tY
    const [ch,t] = id.split("-").map(s=>parseInt(s.replace(/\D/g,""),10));
    if(!st.done) st.done = {};
    const chKey = `ch${ch}`;
    if(!Array.isArray(st.done[chKey])) st.done[chKey] = [];
    st.done[chKey][t-1] = true;
    localStorage.setItem(KEY, JSON.stringify(st));

    // 小煙火 + 返回相機頁
    celebrate();
    setTimeout(()=>{
      location.href = `/html/camera.html?id=${encodeURIComponent(id)}`;
    }, 800);
  };
}

// 簡易彩紙（使用 SVG）
function celebrate(){
  const box = document.getElementById("confetti");
  box.innerHTML="";
  const pieces = ["party", "balloon", "sparkle", "star", "twinkle", "candy"];
  
  for(let i=0; i<24; i++){
    const wrapper = document.createElement("div");
    wrapper.style.position = "absolute";
    wrapper.style.width = "40px";
    wrapper.style.height = "40px";
    wrapper.style.left = Math.random()*100 + "vw";
    wrapper.style.top = "-50px";
    wrapper.style.transform = `rotate(${Math.random()*360}deg)`;
    wrapper.style.transition = "all 0.7s ease-out";
    
    const svgKey = pieces[Math.floor(Math.random()*pieces.length)];
    wrapper.innerHTML = CELEBRATION_SVG[svgKey];
    
    box.appendChild(wrapper);
    
    // 觸發動畫
    setTimeout(() => {
      wrapper.style.top = "100vh";
      wrapper.style.transform = `translateY(0) rotate(${Math.random()*720}deg)`;
    }, 50);
  }
  
  box.classList.add("active");
  setTimeout(()=>{
    box.classList.remove("active");
    box.innerHTML = "";
  }, 700);
}

render();

// 安全返回處理
const HOME = "/html/index.html";
function safeBack(e){
  if(e) e.preventDefault();
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

// 綁定返回按鈕
const topBtn = document.getElementById("backBtn");
const bottomBtn = document.getElementById("backBtnBottom");
if (topBtn) topBtn.addEventListener("click", safeBack);
if (bottomBtn) bottomBtn.addEventListener("click", safeBack);