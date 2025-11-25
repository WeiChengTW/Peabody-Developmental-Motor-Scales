/* ========= 故事資料（依 PDF 串接） ========= */
const STORY = [
  {
    key:"ch1", emoji:"🌉",
    title:"第一關：建造魔法道路",
    intro:"小河被颱風沖壞了！把零件找齊，做出能過河的道路吧。",
    tasks:[
      {icon:"🧱", title:"串積木：做成一條橋", note:"把魔法積木一顆顆串起來，讓我們過河。"},
      {icon:"🏰", title:"疊城堡：蓋瞭望塔", note:"把魔法石頭一層一層疊高，找到前進方向。"},
      {icon:"🪜", title:"疊階梯：翻過高牆", note:"把方塊疊成樓梯，繼續前往魔法王國。"},
      {icon:"🧱", title:"疊高牆：蓋出傳送門", note:"把方塊推成一面大牆，變出傳送門。"}
    ]
  },
  {
    key:"ch2", emoji:"🌀",
    title:"第二關：神秘圖案迷宮",
    intro:"巫師教我們用圖形魔法通過迷宮！",
    tasks:[
      {icon:"⭕", title:"畫圓：大圓圓魔法陣", note:"在紙上畫一個大圓圈。"},
      {icon:"🟦", title:"畫方：守護盾", note:"畫一個正正方方的盾牌。"},
      {icon:"➕", title:"畫十字：啟動魔法", note:"畫出十字星，讓魔法運作起來。"},
      {icon:"📏", title:"描水平線：打敗恐龍", note:"先用一條直線攻擊牠。"},
      {icon:"🖍️", title:"兩水平線中塗色：提升威力", note:"把兩條水平線之間塗滿顏色！"},
      {icon:"✨", title:"兩點連線：開門", note:"把兩顆星星連起來，打開門！"}
    ]
  },
  {
    key:"ch3", emoji:"🏡",
    title:"第三關：精靈小屋",
    intro:"幫助精靈修好小屋，他會給我們魔法紙作為回報。",
    tasks:[
      {icon:"✂️", title:"剪圓：做圓形窗戶", note:"幫小精靈剪出一個圓窗。"},
      {icon:"📐", title:"剪方：做方方正正的門", note:"幫小精靈剪出正方形的門。"},
      {icon:"📐", title:"延直線剪：把窗戶剪開", note:"幫小精靈把剛剛的窗戶沿線剪開。"},
      {icon:"✂️", title:"平分紙張：剪窗簾", note:"幫小精靈的窗戶剪出一個窗簾。"}
    ]
  },
  {
    key:"ch4", emoji:"🪄",
    title:"第四關：摺紙飛毯",
    intro:"用魔法紙摺出會飛的飛毯！",
    tasks:[
      {icon:"🗞️", title:"摺紙一摺：變出小飛毯", note:"把紙對摺一次。"},
      {icon:"🧺", title:"摺紙兩摺：更結實的飛毯", note:"再摺一次，就能起飛！"}
    ]
  },
  {
    key:"ch5", emoji:"💎",
    title:"第五關：寶藏大發現",
    intro:"到寶藏洞窟把魔法豆豆裝進罐子，回到魔法王國！",
    tasks:[
      {icon:"🫘", title:"豆豆裝罐子：完成任務", note:"把彩色豆豆一顆一顆裝進罐子。"}
    ]
  }
];

/* ========= 狀態儲存 ========= */
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

// 設置 UID 到後端 session
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
      // 同時保存到 localStorage 作為備份
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
  done: {} // e.g. { ch1: [true,false,...] }
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

// 依關卡與任務序號產生任務頁用的 ID：ch{關卡}-t{任務}
function makeTaskId(chIdx, tIdx){
  return `ch${chIdx+1}-t${tIdx+1}`;
}

/* ========= 元件注入 ========= */
function renderStickers(){
  const rail = $("#stickerRail");
  rail.innerHTML="";
  STORY.forEach((ch,idx)=>{
    const btn = document.createElement("button");
    btn.className="sticker";
    btn.setAttribute("aria-label", ch.title);
    btn.innerHTML = `<div class="emoji">${ch.emoji}</div><div class="caption">${idx+1}關</div>`;
    if(idx===state.chapterIndex) btn.classList.add("active");
    btn.addEventListener("click",()=>{ state.chapterIndex=idx; saveState(); renderAll(); });
    rail.appendChild(btn);
  });
}

function renderStory(){
  const ch = STORY[state.chapterIndex];
  $("#storyEmoji").textContent = ch.emoji;
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
    tpl.querySelector(".task-icon").textContent = t.icon;
    tpl.querySelector(".task-title").textContent = t.title;
    tpl.querySelector(".task-note").textContent = t.note||"";

    const startBtn = tpl.querySelector(".start-btn");
    const doneBtn  = tpl.querySelector(".done-btn");

    if(state.done[ch.key][i]) card.classList.add("is-done");

    startBtn.addEventListener("click", ()=>{
      const id = makeTaskId(state.chapterIndex, i);  // 例如 ch2-t4
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
  // 五顆星：按關卡完成度點亮
  let total=0, done=0;
  for(const ch of STORY){ total += ch.tasks.length; done += (state.done[ch.key]||[]).filter(Boolean).length; }
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

/* ========= 旁白（SpeechSynthesis） ========= */
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
  // 簡易提示
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
