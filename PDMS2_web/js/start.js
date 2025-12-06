// 目標故事首頁；若你的檔案不在 /html/index.html，改這行即可
const STORY_HOME = "/html/index.html";

(function () {
  const btn = document.getElementById("enterBtn");
  const uidInput = document.getElementById("uid");
  if (!btn || !uidInput) return;

  const go = async () => {
    const uid = uidInput.value.trim();
    
    // 檢查UID是否為空
    if (!uid) {
      alert("請輸入有效的UID");
      uidInput.focus();
      return;
    }

    // 點擊的微動畫
    btn.style.transform = "scale(0.98)";
    btn.disabled = true;
    btn.textContent = "創建中...";

    try {
      // 調用API創建資料夾
      const response = await fetch("/create-uid-folder", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      body: JSON.stringify({ uid: uid }),
      });

      const result = await response.json();

      if (result.success) {
        // UID 已由後端存入 session，同時備份到 localStorage
        // UID 已由後端存入 session，同時備份到 localStorage（只存目前是誰）
        const META_KEY = "kid-quest-progress-v1-meta";
        const st = JSON.parse(localStorage.getItem(META_KEY) || "{}");
        st.currentUid = uid;
        localStorage.setItem(META_KEY, JSON.stringify(st));

        
        // 成功創建資料夾，跳轉到故事首頁（不需要 URL 參數）
        setTimeout(() => { 
          window.location.href = STORY_HOME; 
        }, 180);
      } else {
        // 顯示錯誤訊息
        alert(`錯誤: ${result.error}`);
        btn.disabled = false;
        btn.textContent = "進入故事";
        btn.style.transform = "";
      }
    } catch (error) {
      console.error("創建資料夾時發生錯誤:", error);
      alert("創建資料夾時發生錯誤，請稍後再試");
      btn.disabled = false;
      btn.textContent = "進入故事";
      btn.style.transform = "";
    }
  };

  btn.addEventListener("click", go);
  
  // 按Enter鍵也能觸發
  uidInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      go();
    }
  });

  // 全域按Enter或空白鍵
  window.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { 
      e.preventDefault(); 
      go(); 
    }
  });
})();
