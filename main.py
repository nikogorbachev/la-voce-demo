from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import re
import os
import asyncio
import threading


import torch
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

# Third-party providers that aren't wired to a live API key in this environment yet.
# Add the actual API call for a provider here once you have a key, and remove its
# id from this set so /tts routes real requests to it instead of returning 501.
UNCONFIGURED_PROVIDERS = {"elevenlabs", "voxtral", "gemini", "cartesia"}

# Providers that are real local/open-source models but not yet wired into this
# demo's inference code (e.g. Kokoro, Chatterbox checkpoints not yet deployed
# alongside the VITS one). Same 501 behavior as third-party providers for now.
UNCONFIGURED_OSS_MODELS = {"kokoro", "chatterbox"}


def _expand_number(match):
    s = match.group(0)
    if HAS_NUM2WORDS:
        try:
            return num2words(int(s), lang='it')
        except Exception:
            return s
    return s


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    if t == "":
        return t

    t = t.lower()
    for pat, repl in ABBREV.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r'\d+', _expand_number, t)
    t = re.sub(r'\s+', ' ', t)
    t = t.replace(" ,", ",")
    t = t.replace(": ", ", ")

    # a bit of hardcoded english spelling parsing
    t = re.sub(r"cha", "cia", t)
    t = re.sub(r"cho", "cio", t)
    t = re.sub(r"chu", "ciu", t)
    return t


def apply_editorial_punctuation(text: str) -> str:
    """
    Placeholder for the optional LLM-assisted pass discussed in planning:
    call an LLM (e.g. via the Anthropic or Mistral API) to insert
    newsroom-style pauses/em dashes and phonetic respelling for foreign
    names (e.g. "Zohran Mamdani" -> "— zooh-ran mam-daani —"), plus doubled
    vowels to mark stressed syllables.

    This is intentionally a no-op passthrough until an API key + prompt are
    wired in — it's here so the frontend toggle and /normalize route already
    have somewhere to plug the real call in without touching app.js again.
    """
    # TODO: replace with a real call, e.g.:
    #   response = anthropic_client.messages.create(
    #       model="claude-...",
    #       messages=[{"role": "user", "content": PUNCTUATION_PROMPT.format(text=text)}],
    #   )
    #   return response.content[0].text
    return text


model_checkpoint_path = "best_model.pth"
config_path = "config.json"

device = "cpu"
print(f"Starting TTS on device: {device}")

synthesizer = Synthesizer(
    tts_checkpoint=model_checkpoint_path,
    tts_config_path=config_path,
    use_cuda=False,
)

SAMPLE_RATE = synthesizer.output_sample_rate

synth_lock = threading.Lock()


async def synthesize_wav_bytes(normalized_text: str) -> io.BytesIO:
    """Run blocking synthesizer in a thread and return a BytesIO WAV buffer."""
    def _do_synth(text_in):
        with synth_lock:
            wav = synthesizer.tts(text=text_in, language_name='it')
        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format='WAV')
        buf.seek(0)
        return buf

    buf = await asyncio.to_thread(_do_synth, normalized_text)
    return buf


@app.post("/normalize")
async def normalize_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    apply_punct = bool(data.get("apply_editorial_punctuation", False))

    normalized = normalize_text(text)
    if apply_punct:
        normalized = apply_editorial_punctuation(normalized)

    return JSONResponse({"original": text, "normalized": normalized})


@app.post("/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    model_id = data.get("model_id", "vits")
    apply_punct = bool(data.get("apply_editorial_punctuation", False))

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if model_id in UNCONFIGURED_PROVIDERS:
        raise HTTPException(
            status_code=501,
            detail=f"'{model_id}' non è ancora collegato a una chiave API in questo ambiente demo. "
                    "Aggiungi la chiamata reale in main.py (UNCONFIGURED_PROVIDERS) per attivarlo.",
        )
    if model_id in UNCONFIGURED_OSS_MODELS:
        raise HTTPException(
            status_code=501,
            detail=f"'{model_id}' non è ancora deployato in questo ambiente demo. "
                    "Aggiungi il checkpoint/pipeline in main.py (UNCONFIGURED_OSS_MODELS) per attivarlo.",
        )

    # Only the in-house VITS path is live end-to-end in this demo.
    normalized = normalize_text(text)
    if apply_punct:
        normalized = apply_editorial_punctuation(normalized)

    wav_buf = await synthesize_wav_bytes(normalized)

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
