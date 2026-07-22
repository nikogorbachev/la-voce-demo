from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import re
import os
import asyncio
import threading

import torch
import transformers.pytorch_utils
transformers.pytorch_utils.isin_mps_friendly = torch.isin

# NOW import Coqui TTS and Chatterbox
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf

try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except Exception:
    HAS_NUM2WORDS = False

app = FastAPI()

ABBREV = {
    r'\bprof\.?\b': 'professore',
    r'\bdott\.?\b': 'dottore',
    r'\bsig\.?\b': 'signore',
    r'\bsigra\.?\b': 'signora',
}

# 1. Explicit list of proper names to isolate with commas for VITS attention
PROPER_NAMES_DICT = [
    r'\bNew York\b',
    r'\bZohran Mamdani\b',
    r'\bWashington\b',
    r'\bCEO\b',
]

# 2. Phonetic transliteration dictionary for foreign proper names
PHONETIC_LEXICON = {
    r'\bNew York\b': 'niuu ioorch',
    r'\bZohran Mamdani\b': 'zooran mamdaani',
    r'\bWashington\b': 'uoosh-ing-ton',
    r'\bCEO\b': 'si-i-o',
}

UNCONFIGURED_PROVIDERS = {"elevenlabs", "voxtral", "gemini", "cartesia"}
UNCONFIGURED_OSS_MODELS = {"kokoro", "chatterbox"}


def _expand_numbers_and_symbols(text: str) -> str:
    """
    Expands percentage signs, decimal numbers (e.g. 0.75 -> zero virgola settantacinque),
    and standalone integers into Italian words before sentence punctuation processing.
    """
    if not text:
        return ""
    t = text

    # 1. Replace percentage signs
    t = re.sub(r'%', ' percento', t)

    # 2. Expand decimal numbers (e.g., 0.75 or 0,75)
    def _repl_decimal(match):
        int_part = match.group(1)
        dec_part = match.group(2)
        if HAS_NUM2WORDS:
            try:
                int_str = num2words(int(int_part), lang='it')
                dec_str = num2words(int(dec_part), lang='it')
                return f"{int_str} virgola {dec_str}"
            except Exception:
                return f"{int_part} virgola {dec_part}"
        return f"{int_part} virgola {dec_part}"

    t = re.sub(r'\b(\d+)[.,](\d+)\b', _repl_decimal, t)

    # 3. Expand remaining integers
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
    """
    VITS-specific newsroom punctuation pass:
    1. Replaces input commas (,) with em-dashes (—).
    2. Replaces sentence-ending periods (.) with semicolon + em-dash (; —).
    3. Surrounds ONLY explicit proper names in PROPER_NAMES_DICT with commas.
    4. Ends the passage cleanly with '; .'.
    """
    if not text:
        return ""
    t = text.strip()

    # Step 1: Turn all input commas into em-dashes
    t = re.sub(r'\s*,\s*', ' — ', t)

    # Step 2: Turn all sentence-ending periods into semicolon + em-dash
    t = re.sub(r'\s*\.\s*', '; — ', t)

    # Step 3: Annotate ONLY explicit proper names from dictionary with commas
    for pattern in PROPER_NAMES_DICT:
        t = re.sub(pattern, lambda m: f", {m.group(0)} ,", t, flags=re.IGNORECASE)

    # Step 4: Clean up multiple spaces
    t = re.sub(r'\s+', ' ', t).strip()

    # Step 5: Remove trailing em-dash or semicolon before appending final '; .'
    t = re.sub(r'(?:;\s*—|—|;)\s*$', '', t).strip()

    # Step 6: Wrap string with leading comma and trailing '; .'
    return f", {t}; ."


def normalize_text(text: str) -> str:
    """
    Standard normalization pass:
    - Transliterates English proper names using PHONETIC_LEXICON
    - Lowercase conversion
    - Expands abbreviations
    """
    if not text:
        return ""
    t = text.strip()
    if not t:
        return t

    # Apply phonetic transliteration for English proper names
    for pattern, replacement in PHONETIC_LEXICON.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

    t = t.lower()

    for pat, repl in ABBREV.items():
        t = re.sub(pat, repl, t)

    # General Italian phonetic fixes
    t = re.sub(r"cha", "cia", t)
    t = re.sub(r"cho", "cio", t)
    t = re.sub(r"chu", "ciu", t)

    # Normalize spacing around punctuation marks
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def process_text_pipeline(text: str, do_normalize: bool, do_punct: bool) -> str:
    out = text

    # Pre-process numbers & symbols (% -> percento, 0.75 -> zero virgola settantacinque)
    # so decimal dots aren't mistaken for sentence periods
    out = _expand_numbers_and_symbols(out)

    # Run punctuation pass
    if do_punct:
        out = apply_editorial_punctuation(out)

    # Run lowercasing & abbreviation expansion
    if do_normalize:
        out = normalize_text(out)

    return out


model_checkpoint_path = "vits_it_female.pth"
config_path = "config_it_female.json"

device = "cpu"
print(f"Starting TTS on device: {device}")

synthesizer = Synthesizer(
    tts_checkpoint=model_checkpoint_path,
    tts_config_path=config_path,
    use_cuda=False,
)

SAMPLE_RATE = synthesizer.output_sample_rate
synth_lock = threading.Lock()


async def synthesize_wav_bytes(text_in: str) -> io.BytesIO:
    def _do_synth(raw_txt):
        with synth_lock:
            wav = synthesizer.tts(text=raw_txt, language_name='it')
        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format='WAV')
        buf.seek(0)
        return buf

    buf = await asyncio.to_thread(_do_synth, text_in)
    return buf


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
    if model_id in UNCONFIGURED_OSS_MODELS:
        raise HTTPException(
            status_code=501,
            detail=f"'{model_id}' non è ancora deployato in questo ambiente demo."
        )

    processed_text = process_text_pipeline(text, do_norm, do_punct)
    wav_buf = await synthesize_wav_bytes(processed_text)

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"}
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")