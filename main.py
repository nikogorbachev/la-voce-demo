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

load_dotenv()

# --- Monkey Patches ---
transformers.pytorch_utils.isin_mps_friendly = torch.isin

if not hasattr(transformers.utils.import_utils, "is_torch_greater_or_equal"):
    def is_torch_greater_or_equal(target_version: str) -> bool:
        return version.parse(torch.__version__) >= version.parse(target_version)
    transformers.utils.import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal

if not hasattr(transformers.utils.import_utils, "is_torchcodec_available"):
    def is_torchcodec_available() -> bool:
        return False
    transformers.utils.import_utils.is_torchcodec_available = is_torchcodec_available

class DummyWatermarker:
    def __init__(self, *args, **kwargs): pass
    def apply_watermark(self, wav, *args, **kwargs): return wav
    def embed_watermark(self, wav, *args, **kwargs): return wav

mock_perth = types.ModuleType("perth")
mock_perth.PerthImplicitWatermarker = DummyWatermarker
sys.modules["perth"] = mock_perth

import soundfile as sf
from TTS.utils.synthesizer import Synthesizer

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


torch.set_num_threads(4)

app = FastAPI()

ENV = os.getenv("ENVIRONMENT", os.getenv("ENV", "DEVELOPMENT")).upper()
IS_READ_ONLY = ENV in ["STAGING", "PRODUCTION", "DEMO"]

SNIPPETS_DIR = os.path.join(os.path.dirname(__file__), "saved_snippets")
os.makedirs(SNIPPETS_DIR, exist_ok=True)

SAMPLE_KEYS = ["intro", "breaking", "economy", "weather"]
ALL_MODELS = ["vits", "kokoro", "chatterbox", "parler", "f5", "cartesia", "gemini", "voxtral", "elevenlabs"]

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
    r'\bCentro Studi Conflavoro\b',
    r'\bLega e Forza Italia\b'
]

PHONETIC_LEXICON = {
    r'\bNew York\b': 'niuu ioorch',
    r'\bZohran Mamdani\b': 'zooran mamdaani',
    r'\bWashington\b': 'uoosh-ing-ton',
    r'\bCEO\b': 'si-i-o',
    r'\bafricana\b': 'africaana',
    r'\btemperatur\b': 'temperatuur',
    r'\bweekend\b': 'uiichend',
    r'\bBordeaux\b': 'bordoo',
    r'\bsi valuta\b': 'si vaaluta',
    r'\bMacron\b': 'Macroon',
    r'\bMadrid\b': 'madriid',
    r'\bCarlos Novillo\b': 'caarlos noviillo',
    r'\bdiesel\b': 'diisel',
    r'\bil no\b': 'il noo'



}

UNCONFIGURED_PROVIDERS = set()
synth_lock = threading.Lock()

device = "cpu"

vits_checkpoint_path = "vits_it_female.pth"
vits_config_path = "config_it_female.json"

vits_synthesizer = None
if not IS_READ_ONLY and os.path.exists(vits_checkpoint_path) and os.path.exists(vits_config_path):
    vits_synthesizer = Synthesizer(
        tts_checkpoint=vits_checkpoint_path,
        tts_config_path=vits_config_path,
        use_cuda=(device == "cuda"),
    )

kokoro_pipeline = None
def get_kokoro():
    global kokoro_pipeline
    if kokoro_pipeline is None:
        if not HAS_KOKORO: raise HTTPException(status_code=500, detail="kokoro library not installed.")
        kokoro_pipeline = KPipeline(lang_code='i', device='cpu')
    return kokoro_pipeline

chatterbox_model = None
def get_chatterbox():
    global chatterbox_model
    if chatterbox_model is None:
        if not HAS_CHATTERBOX: raise HTTPException(status_code=500, detail="chatterbox-tts library not installed.")
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return chatterbox_model

parler_model = None
parler_tokenizer = None
parler_description_tokenizer = None


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
        if not HAS_CARTESIA: raise HTTPException(status_code=500, detail="Libreria 'cartesia' non installata.")
        if not CARTESIA_API_KEY: raise HTTPException(status_code=501, detail="CARTESIA_API_KEY non configurata.")
        cartesia_client = AsyncCartesia(api_key=CARTESIA_API_KEY)
    return cartesia_client

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
gemini_client = None

def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        if not HAS_GEMINI: raise HTTPException(status_code=500, detail="Libreria 'google-genai' non installata.")
        if not GEMINI_API_KEY: raise HTTPException(status_code=501, detail="GEMINI_API_KEY non configurata.")
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client

try:
    from mistralai.client import Mistral
    HAS_MISTRAL = True
except Exception:
    HAS_MISTRAL = False

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", None)
VOXTRAL_VOICE_ID = os.getenv("VOXTRAL_VOICE_ID", None)
mistral_client = None

def get_mistral_client():
    global mistral_client
    if mistral_client is None:
        if not HAS_MISTRAL: raise HTTPException(status_code=500, detail="Libreria 'mistralai' non installata.")
        if not MISTRAL_API_KEY: raise HTTPException(status_code=501, detail="MISTRAL_API_KEY non configurata.")
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return mistral_client

try:
    from elevenlabs.client import ElevenLabs
    HAS_ELEVENLABS = True
except Exception:
    HAS_ELEVENLABS = False

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", None)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
elevenlabs_client = None

def get_elevenlabs_client():
    global elevenlabs_client
    if elevenlabs_client is None:
        if not HAS_ELEVENLABS: raise HTTPException(status_code=500, detail="Libreria 'elevenlabs' non installata.")
        if not ELEVENLABS_API_KEY: raise HTTPException(status_code=501, detail="ELEVENLABS_API_KEY non configurata.")
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return elevenlabs_client

# --- Text Processing Helpers ---
def _expand_numbers_and_symbols(text: str) -> str:
    if not text: return ""
    t = re.sub(r'%', ' percento', text)
    def _repl_decimal(match):
        int_part, dec_part = match.group(1), match.group(2)
        if HAS_NUM2WORDS:
            try: return f"{num2words(int(int_part), lang='it')} virgola {num2words(int(dec_part), lang='it')}"
            except Exception: pass
        return f"{int_part} virgola {dec_part}"
    t = re.sub(r'\b(\d+)[.,](\d+)\b', _repl_decimal, t)
    def _repl_int(match):
        s = match.group(0)
        if HAS_NUM2WORDS:
            try: return num2words(int(s), lang='it')
            except Exception: pass
        return s
    return re.sub(r'\b\d+\b', _repl_int, t)

def apply_editorial_punctuation(text: str) -> str:
    if not text: return ""
    t = re.sub(r'\s*,\s*', ' — ', text.strip())
    t = re.sub(r'\s*:\s*', ': — ', t)
    t = re.sub(r'\s*\.\s*', '; — ', t)
    for pattern in PROPER_NAMES_DICT:
        t = re.sub(pattern, lambda m: f", {m.group(0)} ,", t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'(?:;\s*—|—|;)\s*$', '', t).strip()
    return f", {t}; ."

def normalize_text(text: str) -> str:
    if not text: return ""
    t = text.strip()
    for pattern, replacement in PHONETIC_LEXICON.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    t = t.lower()
    for pat, repl in ABBREV.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"cha", "cia", t)
    t = re.sub(r"cho", "cio", t)
    t = re.sub(r"chu", "ciu", t)
    return re.sub(r'\s+', ' ', t).strip()

def process_text_pipeline(text: str, do_normalize: bool, do_punct: bool) -> str:
    out = _expand_numbers_and_symbols(text)
    if do_punct: out = apply_editorial_punctuation(out)
    if do_normalize: out = normalize_text(out)
    return out

def save_and_wrap_audio(wav_data, samplerate: int, model_id: str, sample_id: str = "intro") -> io.BytesIO:
    """Saves generated WAV bytes locally to disk using model_id and sample_id."""
    out_path = os.path.join(SNIPPETS_DIR, f"{model_id}_{sample_id}.wav")
    sf.write(out_path, wav_data, samplerate, format='WAV')
    
    buf = io.BytesIO()
    sf.write(buf, wav_data, samplerate, format='WAV')
    buf.seek(0)
    return buf

# --- Model Synthesizers ---
async def synthesize_vits(text_in: str, sample_id: str = "intro") -> io.BytesIO:
    if vits_synthesizer is None: raise HTTPException(status_code=500, detail="VITS checkpoint not loaded.")
    def _do_synth():
        with synth_lock:
            wav = vits_synthesizer.tts(text=text_in, language_name='it', length_scale=1.15)
        return save_and_wrap_audio(wav, vits_synthesizer.output_sample_rate, "vits", sample_id)
    return await asyncio.to_thread(_do_synth)

async def synthesize_kokoro(text_in: str, sample_id: str = "intro") -> io.BytesIO:
    k_pipe = get_kokoro()
    def _do_synth():
        with synth_lock:
            generator = k_pipe(text_in, voice="if_sara", speed=1.0)
            chunks = [audio for _, _, audio in generator if audio is not None]
            if not chunks: raise ValueError("Kokoro produced no audio output.")
            wav_np = torch.cat(chunks).cpu().numpy() if isinstance(chunks[0], torch.Tensor) else np.concatenate(chunks)
        return save_and_wrap_audio(wav_np, 24000, "kokoro", sample_id)
    return await asyncio.to_thread(_do_synth)

async def synthesize_chatterbox(text_in: str, ref_audio_path: str = None, sample_id: str = "intro") -> io.BytesIO:
    cb_model = get_chatterbox()
    def _do_synth():
        with synth_lock:
            with torch.no_grad():
                ref_path = ref_audio_path if (ref_audio_path and os.path.exists(ref_audio_path)) else "example.wav"
                wav_tensor = cb_model.generate(text_in, language_id="it", audio_prompt_path=ref_path if os.path.exists(ref_path) else None)
                wav_np = wav_tensor.squeeze().cpu().numpy()
        return save_and_wrap_audio(wav_np, cb_model.sr, "chatterbox", sample_id)
    return await asyncio.to_thread(_do_synth)


async def synthesize_cartesia(text_in: str, sample_id: str = "intro") -> io.BytesIO:
    client = get_cartesia_client()
    try:
        audio_stream = await client.tts.bytes(
            model_id="sonic-3",
            transcript=text_in,
            voice={"mode": "id", "id": "30ab9d55-a5f5-4113-849a-dbb29b7dad62"},
            output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100},
            language="it",
        )
        audio_data = bytearray()
        async for chunk in audio_stream:
            audio_data.extend(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore API Cartesia: {str(e)}")

    out_path = os.path.join(SNIPPETS_DIR, f"cartesia_{sample_id}.wav")
    with open(out_path, "wb") as f:
        f.write(audio_data)

    buf = io.BytesIO(audio_data)
    buf.seek(0)
    return buf

import base64
import wave

async def synthesize_gemini(text_in: str, sample_id: str = "intro") -> io.BytesIO:
    client = get_gemini_client()
    def _do_synth():
        try:
            interaction = client.interactions.create(
                model="gemini-3.1-flash-tts-preview",
                input=f"Read the following text out loud in clear Italian: {text_in}",
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": "Despina"}]}
            )
            if not hasattr(interaction, "output_audio") or not interaction.output_audio:
                raise ValueError("Gemini Interactions API did not return output_audio.")
            return base64.b64decode(interaction.output_audio.data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore API Gemini: {str(e)}")

    raw_pcm = await asyncio.to_thread(_do_synth)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(raw_pcm)

    wav_bytes = wav_buf.getvalue()
    out_path = os.path.join(SNIPPETS_DIR, f"gemini_{sample_id}.wav")
    with open(out_path, "wb") as f:
        f.write(wav_bytes)

    buf = io.BytesIO(wav_bytes)
    buf.seek(0)
    return buf

async def synthesize_voxtral(text_in: str, sample_id: str = "intro") -> io.BytesIO:
    client = get_mistral_client()
    if not VOXTRAL_VOICE_ID:
        raise HTTPException(status_code=501, detail="VOXTRAL_VOICE_ID non impostata.")

    def _do_synth():
        try:
            response = client.audio.speech.complete(
                model="voxtral-mini-tts-2603",
                input=text_in,
                voice_id=VOXTRAL_VOICE_ID,
                response_format="mp3"
            )
            if not hasattr(response, "audio_data") or not response.audio_data:
                raise ValueError("L'API Voxtral non ha restituito alcun dato audio.")
            return base64.b64decode(response.audio_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore API Voxtral: {str(e)}")

    mp3_bytes = await asyncio.to_thread(_do_synth)
    out_path = os.path.join(SNIPPETS_DIR, f"voxtral_{sample_id}.wav")
    with open(out_path, "wb") as f:
        f.write(mp3_bytes)

    buf = io.BytesIO(mp3_bytes)
    buf.seek(0)
    return buf

async def synthesize_elevenlabs(text_in: str, sample_id: str = "intro") -> io.BytesIO:
    client = get_elevenlabs_client()
    def _do_synth():
        try:
            audio_generator = client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text_in,
                model_id="eleven_v3",
                output_format="mp3_44100_128",
            )
            return audio_generator if isinstance(audio_generator, (bytes, bytearray)) else b"".join(audio_generator)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore API ElevenLabs: {str(e)}")

    audio_bytes = await asyncio.to_thread(_do_synth)
    out_path = os.path.join(SNIPPETS_DIR, f"elevenlabs_{sample_id}.wav")
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    buf = io.BytesIO(audio_bytes)
    buf.seek(0)
    return buf

# --- API Routes ---

@app.get("/config")
async def config_endpoint():
    existing_snippets = {}
    for model_id in ALL_MODELS:
        existing_snippets[model_id] = {}
        for sample_id in SAMPLE_KEYS:
            path = os.path.join(SNIPPETS_DIR, f"{model_id}_{sample_id}.wav")
            existing_snippets[model_id][sample_id] = os.path.exists(path)

    return JSONResponse({
        "environment": ENV,
        "is_read_only": IS_READ_ONLY, 
        "snippets": existing_snippets
    })

@app.api_route("/snippets/{model_id}", methods=["GET", "HEAD"])
async def get_snippet_endpoint(model_id: str, sample_id: str = "intro"):
    """Serves the saved snippet for a model and sample_id (intro, breaking, economy)."""
    if sample_id not in SAMPLE_KEYS:
        sample_id = "intro"

    snippet_path = os.path.join(SNIPPETS_DIR, f"{model_id}_{sample_id}.wav")
    if not os.path.exists(snippet_path):
        raise HTTPException(status_code=404, detail=f"No snippet found for '{model_id}_{sample_id}'")
        
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
            detail="In quest'ambiente demo (GCP / Staging) la generazione in tempo reale è disabilitata."
        )

    data = await request.json()
    text = data.get("text", "")
    model_id = data.get("model_id", "vits")
    sample_id = data.get("sample_id", "intro")
    if sample_id not in SAMPLE_KEYS:
        sample_id = "intro"

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
        wav_buf = await synthesize_vits(processed_text, sample_id)
    elif model_id == "kokoro":
        wav_buf = await synthesize_kokoro(processed_text, sample_id)
    elif model_id == "chatterbox":
        wav_buf = await synthesize_chatterbox(processed_text, 'example.wav', sample_id)
    elif model_id == "cartesia":
        wav_buf = await synthesize_cartesia(processed_text, sample_id)
    elif model_id == "gemini":
        wav_buf = await synthesize_gemini(processed_text, sample_id)
    elif model_id == "voxtral":
        wav_buf = await synthesize_voxtral(processed_text, sample_id)
    elif model_id == "elevenlabs":
        wav_buf = await synthesize_elevenlabs(processed_text, sample_id)
    else:
        raise HTTPException(status_code=400, detail=f"Modello sconosciuto: {model_id}")

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"}
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")