/* ========= 故事大綱（魔法屋 / PDF 連動） ========= */
const STORY = [
  {
    key: "ch1", emoji: "🏠",
    title: "走進魔法屋：暖身與探索",
    intro: "我們先做簡單有趣的暖身任務，熟悉規則與操作，讓小幫手準備好進入魔法屋冒險！",
    tasks: [
      { icon: "🧸", title: "認識道具與規則", note: "看看有哪些道具、按鈕怎麼用，知道任務怎麼開始與結束。" },
      { icon: "🧭", title: "跟著路徑走迷宮", note: "沿著路線慢慢走，不急不躁，練習手眼協調與專注力。" },
      { icon: "🖍️", title: "沿線塗色不越線", note: "在區域內上色、盡量不超出邊界，建立手部控制的基本感覺。" }
    ]
  },
  {
    key: "ch2", emoji: "🖊️",
    title: "幾何小畫家：手部精細控制",
    intro: "進入幾何任務區！透過畫圓、描線、連線、剪紙與堆疊，觀察手部精細動作的表現。",
    tasks: [
      { icon: "⭕", title: "畫圓與圈圈", note: "盡量畫出圓滑的圓形，注意起筆、收筆與連續性。" },
      { icon: "➕", title: "描十字/方形直線", note: "沿著直線慢慢描，保持穩定不偏離線道。" },
      { icon: "📏", title: "描直線與平行線", note: "從起點到終點，均速前進；練習筆壓與方向控制。" },
      { icon: "🔗", title: "連連看：點到點", note: "依序把點連起來，觀察轉折與定位的準確度。" },
      { icon: "🧱", title: "疊金字塔（積木）", note: "照範例堆出穩固的金字塔，考驗手部穩定與空間概念。" },
      { icon: "✂️", title: "剪紙沿線走", note: "拿剪刀沿著線剪，注意安全與持剪姿勢（成人需在旁協助）。" }
    ]
  },
  {
    key: "ch3", emoji: "🏗️",
    title: "巧手小建築：組合與空間",
    intro: "用積木與形狀做出目標圖樣，練習組合、排序與空間理解。",
    tasks: [
      { icon: "🧩", title: "拼出指定圖樣", note: "依範例把形狀拼好，觀察對位與順序安排。" },
      { icon: "📦", title: "分類與收納", note: "把不同形狀分類放好，建立秩序感與規劃能力。" }
    ]
  },
  {
    key: "ch4", emoji: "🪡",
    title: "穩定控制與雙手協調",
    intro: "更進一步挑戰：需要手眼協調與雙手合作的任務！",
    tasks: [
      { icon: "🧵", title: "穿線/扣鈕練習", note: "把線穿過洞、把鈕扣好，手指分工與雙手配合要穩。" },
      { icon: "✂️", title: "沿曲線剪裁", note: "沿著彎彎的線剪下來，維持速度與路徑的連續性。" }
    ]
  },
  {
    key: "ch5", emoji: "🧰",
    title: "生活小幫手：功能性動作",
    intro: "把技巧帶進生活場景！試試轉開瓶蓋、旋鈕等日常精細動作。",
    tasks: [
      { icon: "🔧", title: "旋轉與開關", note: "轉開/關上瓶蓋與旋鈕，注意拇指、食指與手腕的配合。" }
    ]
  }
];

/* ========= 進度儲存（本機 + 後端 Session） ========= */
const KEY = "kid-quest-progress-v1";

// 嘗試從後端 session 取得 uid；失敗則回退 localStorage
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
    console.error('取得 UID 時發生錯誤:', error);
    const st = JSON.parse(localStorage.getItem(KEY) || "{}");
    return st.currentUid || null;
  }
}

// 同步 UID 到後端 session，並備份到 localStorage
async function setCurrentUid(uid) {
  try {
    const response = await fetch('/session/set-uid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uid })
    });
    if (response.ok) {
      const st = JSON.parse(localStorage.getItem(KEY) || "{}");
      st.currentUid = uid;
      localStorage.setItem(KEY, JSON.stringify(st));
      return true;
    } else {
      console.error('設定 UID 到 session 失敗');
      return false;
    }
  } catch (error) {
    console.error('設定 UID 時發生錯誤:', error);
    return false;
  }
}

const state = {
  name: "",
  chapterIndex: 0,
  done: {} // e.g. { ch1: [true,false,...] }
};

function loadState() {
  try { Object.assign(state, JSON.parse(localStorage.getItem(KEY)) || {}); } catch {}
  for (const ch of STORY) {
    if (!Array.isArray(state.done[ch.key])) state.done[ch.key] = new Array(ch.tasks.length).fill(false);
    else if (state.done[ch.key].length !== ch.tasks.length) {
      const copy = new Array(ch.tasks.length).fill(false);
      for (let i = 0; i < Math.min(copy.length, state.done[ch.key].length); i++) copy[i] = !!state.done[ch.key][i];
      state.done[ch.key] = copy;
    }
  }
}
function saveState() { localStorage.setItem(KEY, JSON.stringify(state)); }

/* ========= DOM 工具 ========= */
const $  = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));

// 產生任務 ID：ch{章}-t{第幾個任務}
function makeTaskId(chIdx, tIdx) {
  return `ch${chIdx+1}-t${tIdx+1}`;
}

/* ========= 章節貼紙（分頁） ========= */
function renderStickers() {
  const rail = $("#stickerRail");
  rail.innerHTML = "";
  STORY.forEach((ch, idx) => {
    const btn = document.createElement("button");
    btn.className = "sticker";
    btn.setAttribute("aria-label", ch.title);
    btn.innerHTML = `<div class="emoji">${ch.emoji}</div><div class="caption">第${idx+1}章</div>`;
    if (idx === state.chapterIndex) btn.classList.add("active");
    btn.addEventListener("click", () => { state.chapterIndex = idx; saveState(); renderAll(); });
    rail.appendChild(btn);
  });
}

function renderStory() {
  const ch = STORY[state.chapterIndex];
  $("#storyEmoji").textContent   = ch.emoji;
  $("#chapterTitle").textContent = ch.title;
  $("#chapterIntro").textContent = personalize(ch.intro);
  $("#prevBtn").disabled = state.chapterIndex === 0;
  $("#nextBtn").disabled = state.chapterIndex === STORY.length - 1;
}

function renderTasks() {
  const ch = STORY[state.chapterIndex];
  const grid = $("#tasksGrid");
  grid.innerHTML = "";
  ch.tasks.forEach((t, i) => {
    const tpl  = $("#taskTpl").content.cloneNode(true);
    const card = tpl.querySelector(".task-card");
    tpl.querySelector(".task-icon").textContent  = t.icon;
    tpl.querySelector(".task-title").textContent = t.title;
    tpl.querySelector(".task-note").textContent  = t.note || "";

    const startBtn = tpl.querySelector(".start-btn");
    const doneBtn  = tpl.querySelector(".done-btn");

    if (state.done[ch.key][i]) card.classList.add("is-done");

    startBtn.addEventListener("click", () => {
      const id = makeTaskId(state.chapterIndex, i); // e.g., ch2-t4
      window.location.href = `task.html?id=${encodeURIComponent(id)}`;
    });

    doneBtn.addEventListener("click", () => {
      state.done[ch.key][i] = !state.done[ch.key][i];
      saveState(); renderAll();
      if (state.done[ch.key][i]) celebrate();
    });

    grid.appendChild(tpl);
  });
}

/* ========= 右側管理區（快速切換/統計） ========= */
function renderAdmin() {
  const list = $("#chapterList");
  list.innerHTML = "";
  STORY.forEach((ch, idx) => {
    const item = document.createElement("div");
    item.className = "admin-item";
    const done = (state.done[ch.key] || []).filter(Boolean).length;
    item.innerHTML = `
      <span>${idx+1}. ${ch.title}</span>
      <span class="mini">${done}/${ch.tasks.length}</span>
    `;
    item.addEventListener("click", () => { state.chapterIndex = idx; saveState(); renderAll(); });
    list.appendChild(item);
  });
  $("#childName").value = state.name || "";
}

/* ========= 星等進度（5 星） ========= */
function renderStars() {
  let total = 0, done = 0;
  for (const ch of STORY) { total += ch.tasks.length; done += (state.done[ch.key] || []).filter(Boolean).length; }
  const pct = total ? done / total : 0;
  const stars = [$("#star1"), $("#star2"), $("#star3"), $("#star4"), $("#star5")];
  stars.forEach(s => s.classList.remove("lit"));
  const lit = Math.round(pct * 5);
  for (let i = 0; i < lit; i++) stars[i].classList.add("lit");
}

function celebrate() {
  const box = $("#confetti");
  box.innerHTML = "";
  const pieces = "🎉✨⭐🎈🎊🍭🍬".split("");
  for (let i = 0; i < 30; i++) {
    const s = document.createElement("span");
    s.textContent = pieces[Math.floor(Math.random() * pieces.length)];
    s.style.left = Math.random() * 100 + "vw";
    s.style.top  = "-10vh";
    s.style.transform = `translateY(0) rotate(${Math.random()*90}deg)`;
    box.appendChild(s);
  }
  box.classList.add("active");
  setTimeout(() => box.classList.remove("active"), 900);
}

/* ========= 文字轉語音（SpeechSynthesis） ========= */
function speakStory() {
  const ch = STORY[state.chapterIndex];
  const text = `${ch.title}。${personalize(ch.intro)}。` + ch.tasks.map(t => t.title).join("。");
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "zh-TW"; u.rate = 1; u.pitch = 1.05;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}

/* ========= 個人化（插入名字） ========= */
function personalize(text) {
  const name = (state.name || "").trim();
  if (!name) return text;
  // 將「小幫手」替換為「{name} 小幫手」
  return text.replaceAll("小幫手", `${name} 小幫手`);
}

function toast(msg) {
  const n = document.createElement("div");
  n.className = "btn ghost pill";
  n.style.position="fixed"; n.style.left="50%"; n.style.bottom="18px"; n.style.transform="translateX(-50%)";
  n.style.zIndex=3; n.textContent = msg;
  document.body.appendChild(n);
  setTimeout(() => n.remove(), 1800);
}

/* ========= 事件綁定 ========= */
function bindEvents() {
  $("#prevBtn").addEventListener("click", () => { if (state.chapterIndex > 0) { state.chapterIndex--; saveState(); renderAll(); }});
  $("#nextBtn").addEventListener("click", () => { if (state.chapterIndex < STORY.length - 1) { state.chapterIndex++; saveState(); renderAll(); }});

  $("#toggleAdmin").addEventListener("click", (e) => {
    const panel = $("#adminPanel");
    const now = panel.hasAttribute("hidden");
    if (now) panel.removeAttribute("hidden"); else panel.setAttribute("hidden", "");
    e.currentTarget.setAttribute("aria-expanded", now ? "true" : "false");
  });
  $("#closeAdmin").addEventListener("click", () => $("#adminPanel").setAttribute("hidden", ""));

  $("#resetBtn").addEventListener("click", () => {
    if (confirm("確定要清除目前的進度與星等嗎？")) {
      localStorage.removeItem(KEY); loadState(); renderAll();
    }
  });

  $("#childName").addEventListener("input", (e) => {
    state.name = e.target.value; saveState(); renderAll();
  });

  $("#ttsBtn").addEventListener("click", speakStory);
}

/* ========= 啟動 ========= */
function renderAll() {
  renderStickers();
  renderStory();
  renderTasks();
  renderAdmin();
  renderStars();
}
loadState();
window.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  renderAll();
});
