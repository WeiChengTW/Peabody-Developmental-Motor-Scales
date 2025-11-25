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
  
  maze: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="40" fill="none" stroke="#4169E1" stroke-width="3"/>
    <path d="M 50 10 L 50 30 M 50 70 L 50 90 M 10 50 L 30 50 M 70 50 L 90 50" stroke="#4169E1" stroke-width="3"/>
    <circle cx="50" cy="50" r="15" fill="none" stroke="#FFD700" stroke-width="2"/>
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
  
  house: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="50,20 20,50 80,50" fill="#DC143C"/>
    <rect x="30" y="50" width="40" height="40" fill="#8B4513"/>
    <rect x="42" y="65" width="16" height="25" fill="#654321"/>
    <rect x="55" y="58" width="12" height="12" fill="#87CEEB"/>
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
  
  treasure: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="25" y="45" width="50" height="35" fill="#8B4513" stroke="#654321" stroke-width="2"/>
    <path d="M 25 45 Q 50 30, 75 45" fill="#DAA520" stroke="#B8860B" stroke-width="2"/>
    <rect x="47" y="55" width="6" height="15" fill="#FFD700"/>
    <circle cx="50" cy="70" r="2" fill="#000"/>
  </svg>`,
  
  beans: `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="35" cy="45" rx="8" ry="12" fill="#FF69B4"/>
    <ellipse cx="50" cy="50" rx="8" ry="12" fill="#4169E1"/>
    <ellipse cx="65" cy="48" rx="8" ry="12" fill="#32CD32"/>
    <ellipse cx="42" cy="62" rx="8" ry="12" fill="#FFD700"/>
    <ellipse cx="58" cy="65" rx="8" ry="12" fill="#FF6347"/>
  </svg>`
};

/* ========= 故事資料（使用 SVG） ========= */
const STORY = [
  {
    key:"ch1", emoji:"bridge",
    title:"第一關：建造魔法道路",
    intro:"小河被颱風沖壞了！把零件找齊，做出能過河的道路吧。",
    tasks:[
      {icon:"bridge", title:"串積木：做成一條橋", note:"把魔法積木一顆顆串起來，讓我們過河。"},
      {icon:"castle", title:"疊城堡：蓋瞭望塔", note:"把魔法石頭一層一層疊高，找到前進方向。"},
      {icon:"stairs", title:"疊階梯：翻過高牆", note:"把方塊疊成樓梯，繼續前往魔法王國。"},
      {icon:"wall", title:"疊高牆：蓋出傳送門", note:"把方塊推成一面大牆，變出傳送門。"}
    ]
  },
  {
    key:"ch2", emoji:"maze",
    title:"第二關：神秘圖案迷宮",
    intro:"巫師教我們用圖形魔法通過迷宮！",
    tasks:[
      {icon:"circle", title:"畫圓：大圓圓魔法陣", note:"在紙上畫一個大圓圈。"},
      {icon:"square", title:"畫方：守護盾", note:"畫一個正正方方的盾牌。"},
      {icon:"cross", title:"畫十字：啟動魔法", note:"畫出十字星，讓魔法運作起來。"},
      {icon:"line", title:"描水平線：打敗恐龍", note:"先用一條直線攻擊牠。"},
      {icon:"paint", title:"兩水平線中塗色：提升威力", note:"把兩條水平線之間塗滿顏色！"},
      {icon:"connect", title:"兩點連線：開門", note:"把兩顆星星連起來，打開門！"}
    ]
  },
  {
    key:"ch3", emoji:"house",
    title:"第三關：精靈小屋",
    intro:"幫助精靈修好小屋，他會給我們魔法紙作為回報。",
    tasks:[
      {icon:"scissorsCircle", title:"剪圓：做圓形窗戶", note:"幫小精靈剪出一個圓窗。"},
      {icon:"scissorsSquare", title:"剪方：做方方正正的門", note:"幫小精靈剪出正方形的門。"}
    ]
  },
  {
    key:"ch4", emoji:"paper",
    title:"第四關：摺紙飛毯",
    intro:"用魔法紙摺出會飛的飛毯！",
    tasks:[
      {icon:"foldOnce", title:"摺紙一摺：變出小飛毯", note:"把紙對摺一次。"},
      {icon:"foldTwice", title:"摺紙兩摺：更結實的飛毯", note:"再摺一次，就能起飛！"}
    ]
  },
  {
    key:"ch5", emoji:"treasure",
    title:"第五關：寶藏大發現",
    intro:"到寶藏洞窟把魔法豆豆裝進罐子，回到魔法王國！",
    tasks:[
      {icon:"beans", title:"豆豆裝罐子：完成任務", note:"把彩色豆豆一顆一顆裝進罐子!"}
    ]
  }
];

/* ========= 狀態儲存 ========= */
const KEY = "kid-quest-progress-v1";

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

async function setCurrentUid(uid) {
  try {
    const response = await fetch('/session/set-uid', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ uid: uid })
    });
    
    if (response.ok) {
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      st.currentUid = uid;
      localStorage.setItem(KEY, JSON.stringify(st));
      return true;
    } else {
      console.error('設置 UID 到 session 失敗');
      return false;
    }
  } catch (error) {
    console.error('設置 UID 時發生錯誤:', error);
    return false;
  }
}

const state = {
  name:"",
  chapterIndex:0,
  done: {}
};

function loadState(){
  try{ Object.assign(state, JSON.parse(localStorage.getItem(KEY))||{}); }catch{}
  for(const ch of STORY){
    if(!Array.isArray(state.done[ch.key])) state.done[ch.key]=new Array(ch.tasks.length).fill(false);
    else if(state.done[ch.key].length!==ch.tasks.length){
      const copy = new Array(ch.tasks.length).fill(false);
      for(let i=0;i<Math.min(copy.length, state.done[ch.key].length);i++) copy[i]=!!state.done[ch.key][i];
      state.done[ch.key]=copy;
    }
  }
}
function saveState(){ localStorage.setItem(KEY, JSON.stringify(state)); }

/* ========= DOM 快捷 ========= */
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));

function makeTaskId(chIdx, tIdx){
  return `ch${chIdx+1}-t${tIdx+1}`;
}

/* ========= 元件注入（使用 SVG） ========= */
function renderStickers(){
  const rail = $("#stickerRail");
  rail.innerHTML="";
  STORY.forEach((ch,idx)=>{
    const btn = document.createElement("button");
    btn.className="sticker";
    btn.setAttribute("aria-label", ch.title);
    btn.innerHTML = `<div class="emoji">${SVG_ICONS[ch.emoji]}</div><div class="caption">${idx+1}關</div>`;
    if(idx===state.chapterIndex) btn.classList.add("active");
    btn.addEventListener("click",()=>{ state.chapterIndex=idx; saveState(); renderAll(); });
    rail.appendChild(btn);
  });
}

function renderStory(){
  const ch = STORY[state.chapterIndex];
  $("#storyEmoji").innerHTML = SVG_ICONS[ch.emoji];
  $("#chapterTitle").textContent = ch.title;
  $("#chapterIntro").textContent = personalize(ch.intro);
  $("#prevBtn").disabled = state.chapterIndex===0;
  $("#nextBtn").disabled = state.chapterIndex===STORY.length-1;
}

function renderTasks(){
  const ch = STORY[state.chapterIndex];
  const grid = $("#tasksGrid");
  grid.innerHTML="";
  ch.tasks.forEach((t,i)=>{
    const tpl = $("#taskTpl").content.cloneNode(true);
    const card = tpl.querySelector(".task-card");
    tpl.querySelector(".task-icon").innerHTML = SVG_ICONS[t.icon];
    tpl.querySelector(".task-title").textContent = t.title;
    tpl.querySelector(".task-note").textContent = t.note||"";

    const startBtn = tpl.querySelector(".start-btn");
    const doneBtn  = tpl.querySelector(".done-btn");

    if(state.done[ch.key][i]) card.classList.add("is-done");

    startBtn.addEventListener("click", ()=>{
      const id = makeTaskId(state.chapterIndex, i);
      window.location.href = `task.html?id=${encodeURIComponent(id)}`;
    });

    doneBtn.addEventListener("click", ()=>{
      state.done[ch.key][i] = !state.done[ch.key][i];
      saveState(); renderAll();
      if(state.done[ch.key][i]) celebrate();
    });

    grid.appendChild(tpl);
  });
}

/* ========= 小老師模式 ========= */
function renderAdmin(){
  const list = $("#chapterList");
  list.innerHTML="";
  STORY.forEach((ch,idx)=>{
    const item = document.createElement("div");
    item.className="admin-item";
    const done = (state.done[ch.key]||[]).filter(Boolean).length;
    item.innerHTML = `
      <span>${idx+1}. ${ch.title}</span>
      <span class="mini">${done}/${ch.tasks.length}</span>
    `;
    item.addEventListener("click",()=>{ state.chapterIndex=idx; saveState(); renderAll(); });
    list.appendChild(item);
  });
  $("#childName").value = state.name||"";
}

/* ========= 星星進度與彩紙 ========= */
function renderStars(){
  let total=0, done=0;
  for(const ch of STORY){ total += ch.tasks.length; done += (state.done[ch.key]||[]).filter(Boolean).filter(Boolean).length; }
  const pct = total? done/total : 0;
  const stars = [$("#star1"),$("#star2"),$("#star3"),$("#star4"),$("#star5")];
  stars.forEach(s=>s.classList.remove("lit"));
  const lit = Math.round(pct*5);
  for(let i=0;i<lit;i++) stars[i].classList.add("lit");
}

function celebrate(){
  const box = $("#confetti");
  box.innerHTML="";
  const pieces = "🎉🎈✨💫⭐🍬".split("");
  for(let i=0;i<30;i++){
    const s = document.createElement("span");
    s.textContent = pieces[Math.floor(Math.random()*pieces.length)];
    s.style.left = Math.random()*100 + "vw";
    s.style.top = "-10vh";
    s.style.transform = `translateY(0) rotate(${Math.random()*90}deg)`;
    box.appendChild(s);
  }
  box.classList.add("active");
  setTimeout(()=>box.classList.remove("active"), 900);
}

/* ========= 旁白 ========= */
function speakStory(){
  const ch = STORY[state.chapterIndex];
  const text = `${ch.title}。${personalize(ch.intro)}。` + ch.tasks.map(t=>t.title).join("、");
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "zh-TW"; u.rate = 1; u.pitch = 1.05;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}

/* ========= 工具 ========= */
function personalize(text){
  const name = (state.name||"").trim();
  if(!name) return text;
  return text.replaceAll("我們", `${name}和我們`).replaceAll("巫師", `巫師（${name}的好朋友）`);
}

function toast(msg){
  const n = document.createElement("div");
  n.className = "btn ghost pill";
  n.style.position="fixed"; n.style.left="50%"; n.style.bottom="18px"; n.style.transform="translateX(-50%)";
  n.style.zIndex=3; n.textContent = msg;
  document.body.appendChild(n);
  setTimeout(()=>n.remove(), 1800);
}

/* ========= 綁定事件 ========= */
function bindEvents(){
  $("#prevBtn").addEventListener("click", ()=>{ if(state.chapterIndex>0){ state.chapterIndex--; saveState(); renderAll(); }});
  $("#nextBtn").addEventListener("click", ()=>{ if(state.chapterIndex<STORY.length-1){ state.chapterIndex++; saveState(); renderAll(); }});

  $("#toggleAdmin").addEventListener("click", (e)=>{
    const panel = $("#adminPanel");
    const now = panel.hasAttribute("hidden");
    if(now) panel.removeAttribute("hidden"); else panel.setAttribute("hidden","");
    e.currentTarget.setAttribute("aria-expanded", now? "true":"false");
  });
  $("#closeAdmin").addEventListener("click", ()=> $("#adminPanel").setAttribute("hidden",""));

  $("#resetBtn").addEventListener("click", ()=>{
    if(confirm("要把所有進度清空嗎？")){
      localStorage.removeItem(KEY); loadState(); renderAll();
    }
  });

  $("#childName").addEventListener("input", (e)=>{
    state.name = e.target.value; saveState(); renderAll();
  });

  $("#ttsBtn").addEventListener("click", speakStory);
}

/* ========= 啟動 ========= */
function renderAll(){
  renderStickers();
  renderStory();
  renderTasks();
  renderAdmin();
  renderStars();
}

loadState();
window.addEventListener("DOMContentLoaded", ()=>{ 
  bindEvents(); 
  renderAll(); 
});