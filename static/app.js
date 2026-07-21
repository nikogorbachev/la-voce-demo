// static/app.js
const ossList = document.getElementById("ossList");
const paidList = document.getElementById("paidList");
const detTitle = document.getElementById("detTitle");
const detBadge = document.getElementById("detBadge");
const detDesc = document.getElementById("detDesc");
const detPros = document.getElementById("detPros");
const detCons = document.getElementById("detCons");
const vitsTools = document.getElementById("vitsTools");
const punctToggle = document.getElementById("punctToggle");
const notice = document.getElementById("notice");

const textEl = document.getElementById("text");
const readBtn = document.getElementById("readBtn");
const readIcon = document.getElementById("readIcon");
const normalizeBtn = document.getElementById("normalizeBtn");
const normPanel = document.getElementById("normPanel");
const normResult = document.getElementById("normResult");
const closeNorm = document.getElementById("closeNorm");
const useNorm = document.getElementById("useNorm");
const playerWrap = document.getElementById("playerWrap");
const player = document.getElementById("player");

let selectedId = MODELS[0].id;

function renderList(container, group) {
  container.innerHTML = "";
  MODELS.filter(m => m.group === group).forEach(m => {
    const card = document.createElement("div");
    card.className = "model-card" + (m.id === selectedId ? " selected" : "");
    card.dataset.id = m.id;
    card.innerHTML = `
      <div class="row1">
        <div class="name">${m.name}</div>
        <span class="badge ${m.group === 'oss' ? 'oss' : 'paid'}">${m.group === 'oss' ? 'Free' : 'Paid'}</span>
      </div>
      <div class="short">${m.short}</div>
    `;
    card.addEventListener("click", () => selectModel(m.id));
    container.appendChild(card);
  });
}

function selectModel(id) {
  selectedId = id;
  renderList(ossList, "oss");
  renderList(paidList, "paid");
  renderDetail(id);
  hideNotice();
  playerWrap.classList.add("hidden");
}

function renderDetail(id) {
  const m = MODELS.find(x => x.id === id);
  detTitle.textContent = m.name;
  detBadge.textContent = m.group === "oss" ? "Free / self-hosted" : "Provider a pagamento";
  detBadge.className = "badge " + (m.group === "oss" ? "oss" : "paid");
  detDesc.textContent = m.desc;
  detPros.innerHTML = m.pros.map(p => `<div>&bull; ${p}</div>`).join("");
  detCons.innerHTML = m.cons.map(c => `<div>&bull; ${c}</div>`).join("");

  vitsTools.classList.toggle("hidden", id !== "vits");
}

function showNotice(msg) {
  notice.textContent = msg;
  notice.classList.remove("hidden");
}
function hideNotice() {
  notice.classList.add("hidden");
}

// ---- normalize (VITS-only preview) ----
normalizeBtn.addEventListener("click", async () => {
  const text = textEl.value;
  if (!text.trim()) { alert("Inserisci del testo prima."); return; }

  normalizeBtn.disabled = true;
  try {
    const res = await fetch("/normalize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, apply_editorial_punctuation: selectedId === "vits" && punctToggle.checked }),
    });
    if (!res.ok) throw new Error("Errore normalize");
    const data = await res.json();
    normResult.textContent = `Originale:\n${data.original}\n\nNormalizzato:\n${data.normalized}`;
    normPanel.classList.remove("hidden");
  } catch (err) {
    alert("Errore nella normalizzazione: " + err.message);
  } finally {
    normalizeBtn.disabled = false;
  }
});

closeNorm.addEventListener("click", () => normPanel.classList.add("hidden"));
useNorm.addEventListener("click", () => {
  const parts = normResult.textContent.split("\n\nNormalizzato:\n");
  if (parts.length === 2) textEl.value = parts[1];
  normPanel.classList.add("hidden");
});

// ---- read / synthesize ----
readBtn.addEventListener("click", async () => {
  const text = textEl.value;
  if (!text.trim()) { alert("Inserisci del testo prima."); return; }

  hideNotice();
  readIcon.classList.remove("fa-play");
  readIcon.classList.add("fa-spinner", "loading");
  readBtn.disabled = true;

  try {
    const res = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model_id: selectedId,
        apply_editorial_punctuation: selectedId === "vits" && punctToggle.checked,
      }),
    });

    if (res.status === 501) {
      const err = await res.json();
      showNotice(err.detail || "Questo provider non è ancora collegato a una chiave API in questo ambiente demo.");
      return;
    }
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "TTS failed");
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    player.src = url;
    playerWrap.classList.remove("hidden");
    await player.play();
  } catch (err) {
    alert("Errore TTS: " + err.message);
  } finally {
    readIcon.classList.remove("fa-spinner", "loading");
    readIcon.classList.add("fa-play");
    readBtn.disabled = false;
  }
});

// init
renderList(ossList, "oss");
renderList(paidList, "paid");
renderDetail(selectedId);
