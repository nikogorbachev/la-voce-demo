from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import io
import re
import os
import asyncio
import threading
import sys
import types
import numpy as np
import torch
import torchaudio
import transformers.pytorch_utils
import transformers.utils.import_utils
from packaging import version

from dotenv import load_dotenv

# Carica le variabili definite nel file .env
load_dotenv()

# --- Patch 1: MPS Friendly check for Transformers 5.x ---
transformers.pytorch_utils.isin_mps_friendly = torch.isin

# --- Patch 2: Missing function for Coqui TTS compatibility ---
if not hasattr(transformers.utils.import_utils, "is_torch_greater_or_equal"):
    def is_torch_greater_or_equal(target_version: str) -> bool:
        return version.parse(torch.__version__) >= version.parse(target_version)
    transformers.utils.import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal

if not hasattr(transformers.utils.import_utils, "is_torchcodec_available"):
    def is_torchcodec_available() -> bool:
        return False
    transformers.utils.import_utils.is_torchcodec_available = is_torchcodec_available

# --- Mock Perth Watermarker to bypass Chatterbox watermarking requirement ---
class DummyWatermarker:
    def __init__(self, *args, **kwargs):
        pass

    def apply_watermark(self, wav, *args, **kwargs):
        return wav

    def embed_watermark(self, wav, *args, **kwargs):
        return wav

mock_perth = types.ModuleType("perth")
mock_perth.PerthImplicitWatermarker = DummyWatermarker
sys.modules["perth"] = mock_perth

# --- Safe Imports AFTER Monkey-Patches ---
import soundfile as sf
from TTS.utils.synthesizer import Synthesizer

# --- Optional Model Libraries ---
try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except Exception:
    HAS_NUM2WORDS = False

try:
    from kokoro import KPipeline
    HAS_KOKORO = True
except Exception:
    HAS_KOKORO = False

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    HAS_CHATTERBOX = True
except Exception:
    HAS_CHATTERBOX = False

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    HAS_PARLER = True
except Exception:
    HAS_PARLER = False

try:
    from f5_tts.model import DiT, CFM
    from f5_tts.infer.utils_infer import infer_process, load_vocoder
    from f5_tts.model.utils import get_tokenizer 
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    HAS_F5 = True
except Exception:
    HAS_F5 = False

torch.set_num_threads(4)

app = FastAPI()

# --- Global Environment & Snippet Folder Setup ---
ENV = os.getenv("ENVIRONMENT", os.getenv("ENV", "DEVELOPMENT")).upper()
IS_READ_ONLY = ENV in ["STAGING", "PRODUCTION", "DEMO"]

SNIPPETS_DIR = os.path.join(os.path.dirname(__file__), "saved_snippets")
os.makedirs(SNIPPETS_DIR, exist_ok=True)

if IS_READ_ONLY:
    print(f"Running in {ENV} mode: Real-time generation disabled. Pre-recorded audio snippets served.")

ABBREV = {
    r'\bprof\.?\b': 'professore',
    r'\bdott\.?\b': 'dottore',
    r'\bsig\.?\b': 'signore',
    r'\bsigra\.?\b': 'signora',
}

PROPER_NAMES_DICT = [
    r'\bZohran Mamdani\b',
    r'\bWashington\b',
    r'\bCEO\b',
]

PHONETIC_LEXICON = {
    r'\bNew York\b': 'niuu ioorch',
    r'\bZohran Mamdani\b': 'zooran mamdaani',
    r'\bWashington\b': 'uoosh-ing-ton',
    r'\bCEO\b': 'si-i-o',
}

UNCONFIGURED_PROVIDERS = {"elevenlabs", "voxtral", "gemini"}
synth_lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting TTS Engine on device: {device}")

# --- 1. In-House VITS Setup ---
vits_checkpoint_path = "vits_it_female.pth"
vits_config_path = "config_it_female.json"

vits_synthesizer = None
if not IS_READ_ONLY and os.path.exists(vits_checkpoint_path) and os.path.exists(vits_config_path):
    print("Loading In-House VITS Checkpoint...")
    vits_synthesizer = Synthesizer(
        tts_checkpoint=vits_checkpoint_path,
        tts_config_path=vits_config_path,
        use_cuda=(device == "cuda"),
    )

# --- 2. Lazy Loaded Setup Functions ---
kokoro_pipeline = None
def get_kokoro():
    global kokoro_pipeline
    if kokoro_pipeline is None:
        if not HAS_KOKORO:
            raise HTTPException(status_code=500, detail="kokoro library not installed.")
        print("Initializing Kokoro-82M Italian Pipeline...")
        kokoro_pipeline = KPipeline(lang_code='i')
    return kokoro_pipeline

chatterbox_model = None
def get_chatterbox():
    global chatterbox_model
    if chatterbox_model is None:
        if not HAS_CHATTERBOX:
            raise HTTPException(status_code=500, detail="chatterbox-tts library not installed.")
        print("Initializing Chatterbox Multilingual on CPU...")
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return chatterbox_model

parler_model = None
parler_tokenizer = None
parler_description_tokenizer = None

def get_parler():
    global parler_model, parler_tokenizer, parler_description_tokenizer
    if parler_model is None:
        if not HAS_PARLER:
            raise HTTPException(status_code=500, detail="parler-tts library not installed.")
        print("Initializing Parler-TTS Mini Multilingual...")
        model_id = "parler-tts/parler-tts-mini-multilingual-v1.1"
        parler_model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to("cpu")
        parler_tokenizer = AutoTokenizer.from_pretrained(model_id)
        parler_description_tokenizer = AutoTokenizer.from_pretrained(
            parler_model.config.text_encoder._name_or_path
        )
    return parler_model, parler_tokenizer, parler_description_tokenizer

f5_model = None
vocos_vocoder = None

def get_f5():
    global f5_model, vocos_vocoder
    if f5_model is None:
        if not HAS_F5:
            raise HTTPException(status_code=500, detail="f5-tts library not installed.")
        print("Downloading & Initializing F5-TTS (Italian Checkpoint) and Vocos Vocoder...")
        ckpt_local_path = hf_hub_download(repo_id="alien79/F5-TTS-italian", filename="model_159600.safetensors")
        vocab_local_path = hf_hub_download(repo_id="alien79/F5-TTS-italian", filename="vocab.txt")

        vocab_char_map, vocab_size = get_tokenizer(vocab_local_path, "custom")

        transformer = DiT(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4, text_num_embeds=vocab_size,
        )

        cfm_model = CFM(
            transformer=transformer,
            odeint_kwargs=dict(method="euler"),
            audio_drop_prob=0.0,
            cond_drop_prob=0.0,
            vocab_char_map=vocab_char_map,
        ).to("cpu")
        
        state_dict = load_file(ckpt_local_path)
        if "ema_model_state_dict" in state_dict:
            state_dict = state_dict["ema_model_state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        cfm_model.load_state_dict(state_dict, strict=False)
        cfm_model.eval()
        
        f5_model = cfm_model
        vocos_vocoder = load_vocoder(vocoder_name="vocos", device="cpu")
        print("F5-TTS Italian Model and Vocos loaded successfully!")

    return f5_model, vocos_vocoder


# Third-party Providers 

try:
    from cartesia import AsyncCartesia
    HAS_CARTESIA = True
except Exception:
    HAS_CARTESIA = False

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", None)
cartesia_client = None

def get_cartesia_client():
    global cartesia_client
    if cartesia_client is None:
        if not HAS_CARTESIA:
            raise HTTPException(
                status_code=500, 
                detail="Libreria 'cartesia' non installata. Esegui 'pip install cartesia'."
            )
        if not CARTESIA_API_KEY:
            raise HTTPException(
                status_code=501, 
                detail="La chiave 'CARTESIA_API_KEY' non è stata configurata nelle variabili d'ambiente."
            )
        cartesia_client = AsyncCartesia(api_key=CARTESIA_API_KEY)
    return cartesia_client




# --- Text Processing Helpers ---
def _expand_numbers_and_symbols(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r'%', ' percento', t)

    def _repl_decimal(match):
        int_part, dec_part = match.group(1), match.group(2)
        if HAS_NUM2WORDS:
            try:
                int_str = num2words(int(int_part), lang='it')
                dec_str = num2words(int(dec_part), lang='it')
                return f"{int_str} virgola {dec_str}"
            except Exception:
                return f"{int_part} virgola {dec_part}"
        return f"{int_part} virgola {dec_part}"

    t = re.sub(r'\b(\d+)[.,](\d+)\b', _repl_decimal, t)

    def _repl_int(match):
        s = match.group(0)
        if HAS_NUM2WORDS:
            try:
                return num2words(int(s), lang='it')
            except Exception:
                return s
        return s

    t = re.sub(r'\b\d+\b', _repl_int, t)
    return t

def apply_editorial_punctuation(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r'\s*,\s*', ' — ', t)
    t = re.sub(r'\s*\.\s*', '; — ', t)

    for pattern in PROPER_NAMES_DICT:
        t = re.sub(pattern, lambda m: f", {m.group(0)} ,", t, flags=re.IGNORECASE)

    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'(?:;\s*—|—|;)\s*$', '', t).strip()
    return f", {t}; ."

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    for pattern, replacement in PHONETIC_LEXICON.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

    t = t.lower()
    for pat, repl in ABBREV.items():
        t = re.sub(pat, repl, t)

    t = re.sub(r"cha", "cia", t)
    t = re.sub(r"cho", "cio", t)
    t = re.sub(r"chu", "ciu", t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def process_text_pipeline(text: str, do_normalize: bool, do_punct: bool) -> str:
    out = _expand_numbers_and_symbols(text)
    if do_punct:
        out = apply_editorial_punctuation(out)
    if do_normalize:
        out = normalize_text(out)
    return out

def save_and_wrap_audio(wav_data, samplerate: int, model_id: str) -> io.BytesIO:
    """Saves generated WAV bytes locally to disk for static demo playback."""
    out_path = os.path.join(SNIPPETS_DIR, f"{model_id}_latest.wav")
    sf.write(out_path, wav_data, samplerate, format='WAV')
    
    buf = io.BytesIO()
    sf.write(buf, wav_data, samplerate, format='WAV')
    buf.seek(0)
    return buf

# --- Model Inference Synthesizers ---
async def synthesize_vits(text_in: str) -> io.BytesIO:
    if vits_synthesizer is None:
        raise HTTPException(status_code=500, detail="VITS checkpoint not loaded.")
    
    def _do_synth():
        with synth_lock:
            wav = vits_synthesizer.tts(text=text_in, language_name='it', length_scale=1.15)
        return save_and_wrap_audio(wav, vits_synthesizer.output_sample_rate, "vits")

    return await asyncio.to_thread(_do_synth)

async def synthesize_kokoro(text_in: str) -> io.BytesIO:
    k_pipe = get_kokoro()

    def _do_synth():
        with synth_lock:
            generator = k_pipe(text_in, voice="if_sara", speed=1.0)
            chunks = [audio for _, _, audio in generator if audio is not None]

            if not chunks:
                raise ValueError("Kokoro produced no audio output.")

            if isinstance(chunks[0], torch.Tensor):
                wav_np = torch.cat(chunks).cpu().numpy()
            else:
                wav_np = np.concatenate(chunks)

        return save_and_wrap_audio(wav_np, 24000, "kokoro")

    return await asyncio.to_thread(_do_synth)

async def synthesize_chatterbox(text_in: str, ref_audio_path: str = None) -> io.BytesIO:
    cb_model = get_chatterbox()

    def _do_synth():
        with synth_lock:
            with torch.no_grad():
                ref_path = ref_audio_path if (ref_audio_path and os.path.exists(ref_audio_path)) else "example.wav"
                wav_tensor = cb_model.generate(
                    text_in, 
                    language_id="it", 
                    audio_prompt_path=ref_path if os.path.exists(ref_path) else None
                )
                wav_np = wav_tensor.squeeze().cpu().numpy()

        return save_and_wrap_audio(wav_np, cb_model.sr, "chatterbox")

    return await asyncio.to_thread(_do_synth)

async def synthesize_parler(text_in: str, description: str = None) -> io.BytesIO:
    model, tokenizer, desc_tokenizer = get_parler()
    if not description:
        description = (
            "Julia's voice is clear and expressive with a slightly warm tone, moderate pace, "
            "very high audio quality, close-mic recording, like a news narrator"
        )

    def _do_synth():
        with synth_lock:
            input_ids = desc_tokenizer(description, return_tensors="pt").input_ids
            prompt_input_ids = tokenizer(text_in, return_tensors="pt").input_ids
            
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )
            audio_arr = generation.cpu().numpy().squeeze()

        return save_and_wrap_audio(audio_arr, model.config.sampling_rate, "parler")

    return await asyncio.to_thread(_do_synth)

async def synthesize_f5(text_in: str, ref_audio_path: str = None) -> io.BytesIO:
    f5, vocoder = get_f5()
    
    def _do_synth():
        with synth_lock:
            with torch.no_grad():
                ref_path = ref_audio_path if (ref_audio_path and os.path.exists(ref_audio_path)) else "example.wav"
                wav_np, sr, _ = infer_process(
                    ref_audio_path=ref_path,
                    ref_text="I lettori, le persone erano — arrabbiate con noi, gli dicevano ma eeh che fate? E avevano ragione; Allora-, eh, io avevo iniziato a fare la direttrice, era proprio il primo anno che mi sono trovata dentro il caos del Covid e non sapevo, che pesci pigliare",
                    gen_text=text_in,
                    model_obj=f5,
                    vocoder=vocoder,
                    device="cpu",
                    show_info=print,
                    nfe_step=16, 
                )
                
        return save_and_wrap_audio(wav_np, sr, "f5")

    return await asyncio.to_thread(_do_synth)






async def synthesize_cartesia(text_in: str) -> io.BytesIO:
    client = get_cartesia_client()

    try:
        # Obtain async generator for raw audio bytes
        audio_stream = await client.tts.bytes(
            model_id="sonic-3",
            transcript=text_in,
            voice={"mode": "id", "id": "30ab9d55-a5f5-4113-849a-dbb29b7dad62"},  # Inserisci il tuo Voice ID
            output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100},
            language="it",
        )

        # Consume the async generator into a bytearray
        audio_data = bytearray()
        async for chunk in audio_stream:
            audio_data.extend(chunk)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore API Cartesia: {str(e)}")

    # Save complete audio snippet to local storage
    out_path = os.path.join(SNIPPETS_DIR, "cartesia_latest.wav")
    with open(out_path, "wb") as f:
        f.write(audio_data)

    buf = io.BytesIO(audio_data)
    buf.seek(0)
    return buf

# --- API Routes ---

@app.get("/config")
async def config_endpoint():
    """Returns environment status and available audio snippets to the UI."""
    existing_snippets = {}
    for model_id in ["vits", "kokoro", "chatterbox", "parler", "f5"]:
        path = os.path.join(SNIPPETS_DIR, f"{model_id}_latest.wav")
        existing_snippets[model_id] = os.path.exists(path)

    return JSONResponse({
        "environment": ENV,
        "is_read_only": IS_READ_ONLY, 
        # "is_read_only": True, 
        "snippets": existing_snippets
    })

@app.api_route("/snippets/{model_id}", methods=["GET", "HEAD"])
async def get_snippet_endpoint(model_id: str):
    """Serves the saved snippet for a given model if present."""
    snippet_path = os.path.join(SNIPPETS_DIR, f"{model_id}_latest.wav")
    
    if not os.path.exists(snippet_path):
        raise HTTPException(status_code=404, detail=f"No snippet found for '{model_id}'")
        
    return FileResponse(snippet_path, media_type="audio/wav")

@app.post("/normalize")
async def normalize_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    do_norm = bool(data.get("normalize", True))
    do_punct = bool(data.get("punctuation", False))

    processed = process_text_pipeline(text, do_norm, do_punct)
    return JSONResponse({"original": text, "normalized": processed})

@app.post("/tts")
async def tts_endpoint(request: Request):
    if IS_READ_ONLY:
        raise HTTPException(
            status_code=503,
            detail="In quest'ambiente demo (GCP / Staging) la generazione in tempo reale è disabilitata. Ascolta la demo memorizzata."
        )

    data = await request.json()
    text = data.get("text", "")
    model_id = data.get("model_id", "vits")
    do_norm = bool(data.get("normalize", True))
    do_punct = bool(data.get("punctuation", False))
    description = data.get("description", None)

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if model_id in UNCONFIGURED_PROVIDERS:
        raise HTTPException(
            status_code=501,
            detail=f"'{model_id}' non è ancora collegato a una chiave API in questo ambiente demo."
        )

    processed_text = process_text_pipeline(text, do_norm, do_punct)

    if model_id == "vits":
        wav_buf = await synthesize_vits(processed_text)
    elif model_id == "kokoro":
        wav_buf = await synthesize_kokoro(processed_text)
    elif model_id == "chatterbox":
        wav_buf = await synthesize_chatterbox(processed_text, 'example.wav')
    elif model_id == "parler":
        wav_buf = await synthesize_parler(processed_text, description)
    elif model_id == "f5":
        wav_buf = await synthesize_f5(processed_text, 'example.wav')
    elif model_id == "cartesia":
            wav_buf = await synthesize_cartesia(processed_text)
    else:
        raise HTTPException(status_code=400, detail=f"Modello sconosciuto: {model_id}")

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"}
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")