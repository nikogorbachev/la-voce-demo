from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
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

# Patch missing function in Transformers 5.x for Coqui TTS
transformers.pytorch_utils.isin_mps_friendly = torch.isin

import soundfile as sf

# --- Optional Model Libraries ---
try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except Exception:
    HAS_NUM2WORDS = False

# Official Kokoro package import check
try:
    from kokoro import KPipeline
    HAS_KOKORO = True
except Exception:
    HAS_KOKORO = False

# --- Mock Perth Watermarker to bypass Chatterbox watermarking requirement ---
class DummyWatermarker:
    def __init__(self, *args, **kwargs):
        pass

    def apply_watermark(self, wav, *args, **kwargs):
        return wav  # Return un-watermarked raw audio array/tensor

    def embed_watermark(self, wav, *args, **kwargs):
        return wav

mock_perth = types.ModuleType("perth")
mock_perth.PerthImplicitWatermarker = DummyWatermarker
sys.modules["perth"] = mock_perth

# Now import Chatterbox and Coqui TTS safely
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    HAS_CHATTERBOX = True
except Exception:
    HAS_CHATTERBOX = False

from TTS.utils.synthesizer import Synthesizer

app = FastAPI()

ABBREV = {
    r'\bprof\.?\b': 'professore',
    r'\bdott\.?\b': 'dottore',
    r'\bsig\.?\b': 'signore',
    r'\bsigra\.?\b': 'signora',
}

PROPER_NAMES_DICT = [
    r'\bNew York\b',
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

# Only paid APIs remain unconfigured in this demo
UNCONFIGURED_PROVIDERS = {"elevenlabs", "voxtral", "gemini", "cartesia"}

# Lock for GPU/CPU thread safety across local models
synth_lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting TTS Engine on device: {device}")

# --- 1. In-House VITS Setup ---
vits_checkpoint_path = "vits_it_female.pth"
vits_config_path = "config_it_female.json"

vits_synthesizer = None
if os.path.exists(vits_checkpoint_path) and os.path.exists(vits_config_path):
    print("Loading In-House VITS Checkpoint...")
    vits_synthesizer = Synthesizer(
        tts_checkpoint=vits_checkpoint_path,
        tts_config_path=vits_config_path,
        use_cuda=(device == "cuda"),
    )
else:
    print("Warning: VITS checkpoint not found locally. VITS requests will fail until .pth file is provided.")

# --- 2. Kokoro Setup (Lazy Loaded) ---
kokoro_pipeline = None

def get_kokoro():
    global kokoro_pipeline
    if kokoro_pipeline is None:
        if not HAS_KOKORO:
            raise HTTPException(status_code=500, detail="kokoro library not installed. Run 'pip install kokoro'")
        print("Initializing Kokoro-82M Italian Pipeline...")
        # 'i' specifies the Italian language pipeline in Kokoro
        kokoro_pipeline = KPipeline(lang_code='i')
    return kokoro_pipeline

# --- 3. Chatterbox Setup (Lazy Loaded) ---
chatterbox_model = None

def get_chatterbox():
    global chatterbox_model
    if chatterbox_model is None:
        if not HAS_CHATTERBOX:
            raise HTTPException(
                status_code=500, 
                detail="chatterbox-tts library not installed. Run 'pip install chatterbox-tts'"
            )
        print("Initializing Chatterbox Multilingual on CPU (bypassing low GPU VRAM)...")
        # Force device="cpu" so it loads into system RAM instead of 2GB VRAM
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return chatterbox_model


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


# --- Model Inference Synthesizers ---

async def synthesize_vits(text_in: str) -> io.BytesIO:
    if vits_synthesizer is None:
        raise HTTPException(status_code=500, detail="VITS checkpoint not loaded.")
    
    def _do_synth():
        with synth_lock:
            wav = vits_synthesizer.tts(text=text_in, language_name='it')
        buf = io.BytesIO()
        sf.write(buf, wav, vits_synthesizer.output_sample_rate, format='WAV')
        buf.seek(0)
        return buf

    return await asyncio.to_thread(_do_synth)


async def synthesize_kokoro(text_in: str) -> io.BytesIO:
    k_pipe = get_kokoro()

    def _do_synth():
        with synth_lock:
            # Generate Italian speech using Kokoro's native female Italian voice 'if_sara'
            generator = k_pipe(text_in, voice="if_sara", speed=1.0)
            chunks = []
            for _, _, audio in generator:
                if audio is not None:
                    chunks.append(audio)

            if not chunks:
                raise ValueError("Kokoro produced no audio output.")

            if isinstance(chunks[0], torch.Tensor):
                wav_np = torch.cat(chunks).cpu().numpy()
            else:
                wav_np = np.concatenate(chunks)

        buf = io.BytesIO()
        sf.write(buf, wav_np, 24000, format='WAV')
        buf.seek(0)
        return buf

    return await asyncio.to_thread(_do_synth)


async def synthesize_chatterbox(text_in: str, ref_audio_path: str = None) -> io.BytesIO:
    cb_model = get_chatterbox()

    def _do_synth():
        with synth_lock:
            with torch.no_grad():
                if ref_audio_path and os.path.exists(ref_audio_path):
                    wav_tensor = cb_model.generate(
                        text_in, 
                        language_id="it", 
                        audio_prompt_path=ref_audio_path
                    )
                else:
                    wav_tensor = cb_model.generate(text_in, language_id="it")
                
                wav_np = wav_tensor.squeeze().cpu().numpy()

        buf = io.BytesIO()
        sf.write(buf, wav_np, cb_model.sr, format='WAV')
        buf.seek(0)
        return buf

    return await asyncio.to_thread(_do_synth)


# --- API Routes ---

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
    data = await request.json()
    text = data.get("text", "")
    model_id = data.get("model_id", "vits")
    do_norm = bool(data.get("normalize", True))
    do_punct = bool(data.get("punctuation", False))

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if model_id in UNCONFIGURED_PROVIDERS:
        raise HTTPException(
            status_code=501,
            detail=f"'{model_id}' non è ancora collegato a una chiave API in questo ambiente demo."
        )

    processed_text = process_text_pipeline(text, do_norm, do_punct)

    # Dynamic Model Router
    if model_id == "vits":
        wav_buf = await synthesize_vits(processed_text)
    elif model_id == "kokoro":
        wav_buf = await synthesize_kokoro(processed_text)
    elif model_id == "chatterbox":
        wav_buf = await synthesize_chatterbox(processed_text, 'example_voice_cloning.wav')
    else:
        raise HTTPException(status_code=400, detail=f"Modello sconosciuto: {model_id}")

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"}
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")