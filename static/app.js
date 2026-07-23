const ossList = document.getElementById("ossList");
const paidList = document.getElementById("paidList");
const detTitle = document.getElementById("detTitle");
const detBadge = document.getElementById("detBadge");
const detDesc = document.getElementById("detDesc");
const detPros = document.getElementById("detPros");
const detCons = document.getElementById("detCons");

const normToggle = document.getElementById("normToggle");
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

const samplePills = document.querySelectorAll(".sample-pill");
const charCount = document.getElementById("charCount");
const clearTextBtn = document.getElementById("clearTextBtn");

let selectedId = MODELS[0].id;

// Memory cache for active browser session blobs
const generatedSnippets = {};

// Map of models that have an existing saved_snippets/<model>_latest.wav on disk
const availableSnippets = {};

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
  
  // Instantly display existing snippet if available for this model tab
  loadModelSnippet(id);
}

function loadModelSnippet(modelId) {
  // 1. In-session generated audio takes top priority
  if (generatedSnippets[modelId]) {
    player.src = generatedSnippets[modelId];
    playerWrap.classList.remove("hidden");
    return;
  }

  // 2. Load /saved_snippets/<modelId>_latest.wav if confirmed present on startup scan
  if (availableSnippets[modelId]) {
    player.src = `/snippets/${modelId}?t=${Date.now()}`;
    playerWrap.classList.remove("hidden");
  } else {
    player.src = "";
    playerWrap.classList.add("hidden");
  }
}

function renderDetail(id) {
  const m = MODELS.find(x => x.id === id);
  detTitle.textContent = m.name;
  detBadge.textContent = m.group === "oss" ? "Free / self-hosted" : "Provider a pagamento";
  detBadge.className = "badge " + (m.group === "oss" ? "oss" : "paid");
  detDesc.textContent = m.desc;
  detPros.innerHTML = m.pros.map(p => `<div>&bull; ${p}</div>`).join("");
  detCons.innerHTML = m.cons.map(c => `<div>&bull; ${c}</div>`).join("");

  normToggle.checked = (m.group === "oss");
  punctToggle.checked = (id === "vits");
}

function showNotice(msg) {
  notice.textContent = msg;
  notice.classList.remove("hidden");
}

function hideNotice() {
  notice.classList.add("hidden");
}

// ---- Scan disk on startup for existing /saved_snippets/<model>_latest.wav files ----
async function scanExistingSnippets() {
  // Only scan local self-hosted OSS models that save snippets to disk
  const localModels = MODELS.filter(m => m.group === "oss").map(m => m.id);
  
  await Promise.all(
    localModels.map(async (modelId) => {
      try {
        const res = await fetch(`/snippets/${modelId}`, { method: "HEAD" });
        availableSnippets[modelId] = res.ok;
      } catch (err) {
        availableSnippets[modelId] = false;
      }
    })
  );

  // Load preview snippet after scan finishes
  loadModelSnippet(selectedId);
}

// ---- Preview Normalized Text ----
normalizeBtn.addEventListener("click", async () => {
  const text = textEl.value;
  if (!text.trim()) { alert("Inserisci del testo prima."); return; }

  normalizeBtn.disabled = true;
  try {
    const res = await fetch("/normalize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        text, 
        normalize: normToggle.checked,
        punctuation: punctToggle.checked
      }),
    });
    if (!res.ok) throw new Error("Errore durante la normalizzazione");
    const data = await res.json();
    normResult.textContent = data.normalized;
    normPanel.classList.remove("hidden");
  } catch (err) {
    alert("Errore: " + err.message);
  } finally {
    normalizeBtn.disabled = false;
  }
});

closeNorm.addEventListener("click", () => normPanel.classList.add("hidden"));
useNorm.addEventListener("click", () => {
  textEl.value = normResult.textContent;
  normPanel.classList.add("hidden");
});

// ---- Read / Synthesize ----
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
        normalize: normToggle.checked,
        punctuation: punctToggle.checked
      }),
    });

    if (res.status === 501 || res.status === 503) {
      const err = await res.json();
      showNotice(err.detail);
      return;
    }
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Sintesi vocale fallita");
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    
    // Store generated snippet URL in memory and mark disk snippet available
    generatedSnippets[selectedId] = url;
    availableSnippets[selectedId] = true;

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

// Sample text presets
const SAMPLES = {
  intro: "Buongiorno. Sono la sintesi vocale di Quotidiano Nazionale. Ecco le ultime notizie.",
  breaking: "A New York ha vinto le elezioni Zohran Mamdani, scatenando polemiche.",
  economy: "Contratto enti locali 25-27, ecco di quanto aumentano gli stipendi. Tutte le novità della bozza di accordo."
};

function updateCharCount() {
  const len = textEl.value.length;
  charCount.textContent = `${len} caratteri`;
}

textEl.addEventListener("input", updateCharCount);

samplePills.forEach(pill => {
  pill.addEventListener("click", () => {
    samplePills.forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    
    const key = pill.dataset.sample;
    if (SAMPLES[key]) {
      textEl.value = SAMPLES[key];
      updateCharCount();
    }
  });
});

clearTextBtn.addEventListener("click", () => {
  textEl.value = "";
  updateCharCount();
  samplePills.forEach(p => p.classList.remove("active"));
  textEl.focus();
});

// Aggiungi questo all'avvio in app.js
async function applyEnvironmentSettings() {
  try {
    const res = await fetch("/config");
    if (!res.ok) return;
    const config = await res.json();

    if (config.is_read_only) {
      const readBtn = document.getElementById("readBtn");
      if (readBtn) {
        // Disabilita il pulsante
        readBtn.disabled = true;
        readBtn.classList.add("btn-disabled-cloud");

        // Messaggio in italiano corretto e naturale
        const tooltipText = "La generazione audio in tempo reale è disponibile solo in ambiente locale. Consulta il README per le istruzioni di esecuzione.";
        
        // Imposta l'attributo title nativo per il tooltip
        readBtn.title = tooltipText;

        // Se preferisci aggiornare anche il testo del bottone:
        readBtn.innerHTML = `<i class="fas fa-ban"></i> Generazione disabilitata in Cloud`;
      }
    }
  } catch (err) {
    console.error("Impossibile recuperare la configurazione dell'ambiente:", err);
  }
}

// Chiamata durante la fase di inizializzazione
applyEnvironmentSettings();

// Initial startup execution
renderList(ossList, "oss");
renderList(paidList, "paid");
renderDetail(selectedId);
updateCharCount();

// Scan for pre-existing .wav files in saved_snippets/
scanExistingSnippets();