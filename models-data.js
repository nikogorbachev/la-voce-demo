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
    short: "Finetuning nostro · più controllo · leggero",
    desc: "Modello VITS finetunato internamente sulla voce della direttrice Agnese Pini. Qualità in plateau dopo circa 66k step di finetuning. Il prossimo miglioramento richiede la raccolta di più dati e un ulteriore addestramento. Nessun costo di inferenza oltre al compute GCP.",
    pros: ["Dati mai fuori dal nostro cloud", "Costo di inferenza pari al solo compute GCP", "Pieno controllo su voce e pipeline"],
    cons: ["Qualità non ancora al livello dei provider premium", "Richiede un nuovo ciclo di lavoro per migliorare la resa"],
  },
  {
    id: "kokoro",
    group: "oss",
    name: "Kokoro-82M",
    short: "Voce generica · poco controllo · leggero",
    desc: "Licenza Apache 2.0 con uso commerciale libero. Supporta l'italiano. Veloce ed economico da ospitare, ma non supporta né il clonaggio vocale né il controllo sulla voce.",
    pros: ["Licenza permessiva per uso commerciale", "Gira su GPU modeste con inferenza rapida", "Self-hosted su GCP al solo costo di compute"],
    cons: ["Nessun voice cloning", "Meno espressivo dei modelli con cloning"],
  },
  {
    id: "chatterbox",
    group: "oss",
    name: "Chatterbox",
    short: "Qualità elevata · controllo medio · pesante/costoso",
    desc: "Modello di Resemble AI con licenza MIT. Supporta il controllo sul timbro tramite voice cloning da un campione audio. Non supporta il controllo della prosodia o dello stile di lettura.",
    pros: ["Licenza MIT: uso commerciale e finetuning senza vincoli", "Supporta il voice cloning"],
    cons: ["Qualità in italiano da validare con test dedicati", "Richiede GPU per un'inferenza in tempi ragionevoli", "Poco controllo su stile e prosodia"],
  },
  // {
  //   id: "parler",
  //   group: "oss",
  //   name: "Parler-TTS Mini (Multilingual)",
  //   short: "Voce generica · controllo medio · pesante/costoso",
  //   desc: "Genera la voce in italiano controllando stile, tono ed emozione tramite descrizione testuale (in questa demo usiamo il prompt seguente: 'Julia's voice is clear and expressive with a slightly warm tone, moderate pace, very high audio quality, close-mic recording, like a news narrator').",
  //   pros: ["Controllo dello stile via prompt testuale", "Supporta la lingua italiana"],
  //   cons: ["Livello effettivo di controllo stilistico da verificare"],
  // },
  {
    id: "elevenlabs",
    group: "paid",
    name: "ElevenLabs",
    short: "Top qualità · il più costoso",
    desc: "Il provider più maturo per espressività e voice cloning. Prezzo tra i più alti del mercato: adatto a contenuti ad alta visibilità, meno a un uso quotidiano su tutto il flusso editoriale.",
    pros: ["Qualità ed espressività tra le migliori disponibili", "Voice cloning estremamente curato", "UI e API mature"],
    cons: ["Il più caro del confronto (~100$/1M caratteri)", "I dati passano dal loro cloud", "Poco controllo se serve personalizzazione profonda"],
  },
  {
    id: "voxtral",
    group: "paid",
    name: "Mistral (Voxtral)",
    short: "Economico · controllo parziale",
    desc: "Backend di Mistral, tra i più economici dei provider a pagamento. Buona opzione per un uso quotidiano su notizie a media priorità se non si vuole gestire l'infrastruttura in-house.",
    pros: ["Prezzo competitivo tra i provider a pagamento", "Nessuna manutenzione infrastrutturale", "Voice cloning disponibile"],
    cons: ["Dati gestiti fuori dal nostro cloud", "Controllo limitato su voce e pipeline rispetto ad opzioni in-house"],
  },
  {
    id: "gemini",
    group: "paid",
    name: "Gemini Flash TTS",
    short: "Free tier per test · buon rapporto qualità/prezzo",
    desc: "Testabile gratuitamente in Google AI Studio prima di attivare la fatturazione. Qualità sulla lingua italiana ancora da validare internamente.",
    pros: ["Free tier per prototipare senza carta di credito", "Prezzo per carattere molto basso rispetto a ElevenLabs"],
    cons: ["Nessun voice cloning", "Voce generica/robotica", "Qualità in italiano da testare approfonditamente"],
  },
  {
    id: "cartesia",
    group: "paid",
    name: "Cartesia (Sonic)",
    short: "Bassa latenza · buon rapporto qualità/prezzo",
    desc: "Il più veloce come latenza e un ottimo compromesso tra qualità e costo. Offre anche deployment on-premise, ma resta a pagamento: l'on-premise sposta solo il costo di calcolo, senza eliminare la licenza.",
    pros: ["Latenza di risposta estremamente bassa", "Free tier disponibile per prototipare", "Opzione on-premise per requisiti di residenza dati"],
    cons: ["L'on-premise richiede comunque la licenza software", "Voice cloning meno curato rispetto a ElevenLabs"],
  },
];
