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

// Initial state
let selectedId = MODELS[0].id;
let selectedSample = "intro";

const SAMPLES = {
  intro: "Buongiorno. Sono la sintesi vocale di Quotidiano Nazionale. Ecco le ultime notizie.",
  breaking: "“Impossibile da controllare”. Enorme incendio vicino a Madrid: le foto e i video. \n\nCarlos Novillo, responsabile della gestione delle emergenze del governo regionale: “Ha raggiunto il suo momento più critico, e attualmente supera la capacità dei vigili del fuoco di contenerlo”. Più di 100 mila evacuati in Francia.",
  economy: "Carburanti, 17 centesimi in meno sul diesel fino al 6 agosto. Salta la tassa sulle sigarette. \n\nNessun intervento sulla benzina e soprattutto nessun aumento delle accise sui tabacchi, dopo il no polemico di Lega e Forza Italia che ha aperto uno scontro nella maggioranza.",
  weather: "Fiammata africana, dopo i nubifragi torna il super caldo: attesi 40 gradi in Emilia Romagna, ecco quando. \n\nSi apre un periodo di alta pressione che potrebbe proseguire per minimo 7-10 giorni, con un graduale aumento termico fino al prossimo weekend, la svolta ad agosto. Ad agosto tempo stabile e soleggiato con temperature decisamente superiori alla norma."
};

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
  
  // Instantly point player src to the active model + sample snippet
  loadModelSnippet(id, selectedSample);
}

function loadModelSnippet(modelId, sampleId = selectedSample) {
  // Directly point src to snippet endpoint (or relative static file saved_snippets/${modelId}_${sampleId}.wav)
  player.src = `/saved_snippets/${modelId}_${sampleId}.wav`;
  
  // Ensure player card is always visible
  if (playerWrap) {
    playerWrap.classList.remove("hidden");
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

function hideNotice() {
  if (notice) notice.classList.add("hidden");
}

// Sample Pill Clicks
samplePills.forEach(pill => {
  pill.addEventListener("click", () => {
    samplePills.forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    
    const sampleKey = pill.dataset.sample;
    selectedSample = sampleKey;

    if (SAMPLES[sampleKey]) {
      textEl.value = SAMPLES[sampleKey];
      updateCharCount();
    }

    // Instantly load new snippet for selected pill
    loadModelSnippet(selectedId, selectedSample);
  });
});

// Normalized text preview panel logic
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

// Initialize UI
renderList(ossList, "oss");
renderList(paidList, "paid");
renderDetail(selectedId);
updateCharCount();

// Instantly load audio snippet for initial model (VITS) + initial sample (Intro QN)
loadModelSnippet(selectedId, selectedSample);