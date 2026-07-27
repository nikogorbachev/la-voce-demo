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

// Stato iniziale dell'applicazione
let selectedId = MODELS[0].id;
let selectedSample = "intro"; // Campioni supportati: "intro", "breaking", "economy"

// Testi predefiniti associati ai 3 pulsanti
const SAMPLES = {
  intro: "Buongiorno. Sono la sintesi vocale di Quotidiano Nazionale. Ecco le ultime notizie.",
  breaking: "A New York ha vinto le elezioni Zohran Mamdani, scatenando polemiche.",
  economy: "Contratto enti locali 25-27, ecco di quanto aumentano gli stipendi. Tutte le novità della bozza di accordo."
};

// Cache dei blob generati durante la sessione corrente in memoria
const generatedSnippets = {};

// Mappa dei file disponibili su disco (es. key = "vits_intro")
const availableSnippets = {};

function getSnippetKey(modelId, sampleId = selectedSample) {
  return `${modelId}_${sampleId}`;
}

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
  
  // Carica lo snippet audio corrispondente al modello e al campione attivo
  loadModelSnippet(id, selectedSample);
}

function loadModelSnippet(modelId, sampleId = selectedSample) {
  const key = getSnippetKey(modelId, sampleId);

  // 1. Priorità all'audio generato nella sessione di lavoro corrente
  if (generatedSnippets[key]) {
    player.src = generatedSnippets[key];
    playerWrap.classList.remove("hidden");
    return;
  }

  // 2. Se presente su disco, carica /snippets/<modelId>?sample_id=<sampleId>
  if (availableSnippets[key]) {
    player.src = `/snippets/${modelId}?sample_id=${sampleId}&t=${Date.now()}`;
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

// ---- Controllo all'avvio per tutte le combinazioni di modello + campione ----
async function scanExistingSnippets() {
  const allModels = MODELS.map(m => m.id);
  const sampleKeys = ["intro", "breaking", "economy"];
  const scanPromises = [];

  allModels.forEach(modelId => {
    sampleKeys.forEach(sampleId => {
      const key = getSnippetKey(modelId, sampleId);
      scanPromises.push(
        fetch(`/snippets/${modelId}?sample_id=${sampleId}`, { method: "HEAD" })
          .then(res => { availableSnippets[key] = res.ok; })
          .catch(() => { availableSnippets[key] = false; })
      );
    });
  });

  await Promise.all(scanPromises);
  
  // Una volta completata la scansione, carica lo snippet di default ("intro")
  loadModelSnippet(selectedId, selectedSample);
}

// ---- Gestione dei pulsanti dei campioni (Pills) ----
samplePills.forEach(pill => {
  pill.addEventListener("click", () => {
    samplePills.forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    
    const sampleKey = pill.dataset.sample;
    selectedSample = sampleKey;

    // Aggiorna il testo nella textarea
    if (SAMPLES[sampleKey]) {
      textEl.value = SAMPLES[sampleKey];
      updateCharCount();
    }

    // Carica immediatamente lo snippet audio locale per la combinazione scelta
    loadModelSnippet(selectedId, selectedSample);
  });
});

// ---- Anteprima del testo normalizzato ----
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

// ---- Generazione TTS / Lettura ----
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
        sample_id: selectedSample,
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
    
    const key = getSnippetKey(selectedId, selectedSample);
    generatedSnippets[key] = url;
    availableSnippets[key] = true;

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

function updateCharCount() {
  const len = textEl.value.length;
  charCount.textContent = `${len} caratteri`;
}

textEl.addEventListener("input", updateCharCount);

clearTextBtn.addEventListener("click", () => {
  textEl.value = "";
  updateCharCount();
  samplePills.forEach(p => p.classList.remove("active"));
  textEl.focus();
});

async function applyEnvironmentSettings() {
  try {
    const res = await fetch("/config");
    if (!res.ok) return;
    const config = await res.json();

    if (config.is_read_only) {
      const readBtn = document.getElementById("readBtn");
      if (readBtn) {
        readBtn.disabled = true;
        readBtn.classList.add("btn-disabled-cloud");
        readBtn.title = "La generazione audio in tempo reale è disponibile solo in ambiente locale. Consulta il README per le istruzioni di esecuzione.";
        readBtn.innerHTML = `<i class="fas fa-ban"></i> Generazione disabilitata in Cloud`;
      }
    }
  } catch (err) {
    console.error("Impossibile recuperare la configurazione dell'ambiente:", err);
  }
}

// Inizializzazione dell'interfaccia utente
applyEnvironmentSettings();
renderList(ossList, "oss");
renderList(paidList, "paid");
renderDetail(selectedId);
updateCharCount();

// Avvia la scansione degli snippet salvati
scanExistingSnippets();