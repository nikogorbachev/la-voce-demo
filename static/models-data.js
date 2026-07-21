// static/models-data.js
// Single source of truth for model cards on the dashboard.
// id "vits" is special-cased in app.js to enable the punctuation-annotation toggle
// and to hit the real local /tts endpoint. All other ids call /tts with model_id
// and are expected to return 501 until an API key is wired up in main.py.

const MODELS = [
  {
    id: "vits",
    group: "oss",
    name: "VITS (in-house, finetuned)",
    short: "Il nostro checkpoint, ~46k step",
    desc: "Modello VITS finetunato internamente sulla nostra voce editoriale. Qualità in plateau dopo circa 46k step — prossimo miglioramento richiede più dati, non più training. Nessun costo di inferenza oltre al compute GCP.",
    pros: ["Dati mai fuori dal nostro cloud", "Costo di inferenza ~zero", "Pieno controllo su voce e pipeline"],
    cons: ["Qualità non ancora a livello dei provider premium", "Serve un nuovo ciclo di raccolta dati per migliorare"],
  },
  {
    id: "kokoro",
    group: "oss",
    name: "Kokoro-82M",
    short: "Apache 2.0 · leggero · niente cloning",
    desc: "82M parametri, licenza Apache 2.0 — uso commerciale libero. Supporta l'italiano. Molto veloce ed economico da ospitare, ma non supporta il clonaggio vocale.",
    pros: ["Licenza commerciale pulita (Apache 2.0)", "Girato su GPU modeste, inferenza rapida", "Self-hosted su GCP a costo di solo compute"],
    cons: ["Nessun voice cloning", "Meno espressivo dei modelli con cloning"],
  },
  {
    id: "chatterbox",
    group: "oss",
    name: "Chatterbox / Chatterbox-Turbo",
    short: "MIT · cloning zero-shot",
    desc: "Modello di Resemble AI, licenza MIT. Cloning zero-shot da un breve campione audio. In test interni dell'azienda produttrice, preferito a ElevenLabs nel 65% dei casi (dato vendor, da verificare con test propri).",
    pros: ["Licenza MIT: uso commerciale e finetuning senza vincoli", "Supporta voice cloning", "Candidato diretto a sostituire/affiancare VITS"],
    cons: ["Qualità italiana da validare con test propri", "Richiede comunque GPU per inferenza in tempi ragionevoli"],
  },
  {
    id: "elevenlabs",
    group: "paid",
    name: "ElevenLabs",
    short: "Top qualità · il più costoso",
    desc: "Il provider più maturo per espressività e voice cloning. Prezzo tra i più alti del mercato: adatto a contenuti ad alta visibilità, meno a un uso quotidiano su tutto il flusso editoriale.",
    pros: ["Qualità ed espressività tra le migliori disponibili", "Voice cloning molto curato", "UI e API mature"],
    cons: ["Il più caro del confronto (~100$/M caratteri)", "I dati passano dal loro cloud", "Poco controllo se serve personalizzazione profonda"],
  },
  {
    id: "voxtral",
    group: "paid",
    name: "Mistral (Voxtral)",
    short: "Economico · cloning parziale",
    desc: "Backend di Mistral, tra i più economici dei provider a pagamento. Buona opzione per un uso quotidiano su notizie a media priorità se non si vuole gestire l'infrastruttura in-house.",
    pros: ["Prezzo competitivo tra i provider paid", "Nessuna manutenzione infrastrutturale", "Voice cloning disponibile"],
    cons: ["Dati fuori dal nostro cloud", "Controllo limitato su voce/pipeline rispetto a in-house"],
  },
  {
    id: "gemini",
    group: "paid",
    name: "Gemini Flash TTS",
    short: "Free tier per test · buon rapporto qualità/prezzo",
    desc: "Testabile gratis in Google AI Studio prima di attivare la fatturazione. Benchmark recenti lo avvicinano a ElevenLabs in inglese; qualità sull'italiano ancora da validare internamente.",
    pros: ["Free tier per prototipare senza carta di credito", "Prezzo per carattere basso rispetto a ElevenLabs", "Benchmark in rapido miglioramento"],
    cons: ["Nessun voice cloning", "Qualità italiana non ancora testata da noi"],
  },
  {
    id: "cartesia",
    group: "paid",
    name: "Cartesia (Sonic)",
    short: "Bassa latenza · self-host solo su licenza",
    desc: "Il più veloce in latenza, buon compromesso qualità/costo. Offre anche deployment on-prem, ma resta a pagamento: l'on-prem sposta solo il costo di calcolo, non elimina la licenza.",
    pros: ["Latenza molto bassa", "Free tier generoso per prototipare", "Opzione on-prem per requisiti di residenza dati"],
    cons: ["On-prem non significa gratuito: licenza comunque dovuta", "Voice cloning meno curato di ElevenLabs/Chatterbox"],
  },
];
