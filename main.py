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
import transformers.utils.import_utils
from packaging import version

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

# Official Kokoro package import check
try:
    from kokoro import KPipeline
    HAS_KOKORO = True
except Exception:
    HAS_KOKORO = False


# Now import Chatterbox and Coqui TTS safely
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
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_model, infer_process
    HAS_F5 = True
except Exception:
    HAS_F5 = False

from TTS.utils.synthesizer import Synthesizer
torch.set_num_threads(4)

app = FastAPI()

ABBREV = {
    r'\bprof\.?\b': 'professore',
    r'\bdott\.?\b': 'dottore',
    r'\bsig\.?\b': 'signore',
    r'\bsigra\.?\b': 'signora',
}

PROPER_NAMES_DICT = [
    # r'\bNew York\b',
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



from transformers import AutoTokenizer

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
        
        # 1. Main model
        parler_model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to("cpu")
        
        # 2. Italian text prompt tokenizer
        parler_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 3. Voice description prompt tokenizer (Flan-T5 text encoder)
        parler_description_tokenizer = AutoTokenizer.from_pretrained(
            parler_model.config.text_encoder._name_or_path
        )
        
    return parler_model, parler_tokenizer, parler_description_tokenizer


from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import infer_process

from f5_tts.infer.utils_infer import infer_process, load_vocoder
from f5_tts.model.utils import get_tokenizer 
f5_model = None
vocos_vocoder = None

def get_f5():
    global f5_model, vocos_vocoder
    if f5_model is None:
        if not HAS_F5:
            raise HTTPException(status_code=500, detail="f5-tts library not installed.")
        print("Downloading & Initializing F5-TTS (Italian Checkpoint) and Vocos Vocoder...")
        
        # 1. Download model checkpoint file from Hugging Face
        ckpt_local_path = hf_hub_download(
            repo_id="alien79/F5-TTS-italian", 
            filename="model_159600.safetensors"
        )
        
        vocab_local_path = hf_hub_download(
            repo_id="alien79/F5-TTS-italian",
            filename="vocab.txt"
        )

        vocab_char_map, vocab_size = get_tokenizer(vocab_local_path, "custom")

        # 2. Pass vocab_char_map into DiT so it uses char-level embeddings, not byte fallback
        transformer = DiT(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4,
            text_num_embeds=vocab_size,
        )

        # 3. CFM also needs the vocab map — this is what list_str_to_tensor checks
        cfm_model = CFM(
            transformer=transformer,
            odeint_kwargs=dict(method="euler"),
            audio_drop_prob=0.0,
            cond_drop_prob=0.0,
            vocab_char_map=vocab_char_map,
        ).to("cpu")
        
        # 4. Load safetensors weights
        state_dict = load_file(ckpt_local_path)
        if "ema_model_state_dict" in state_dict:
            state_dict = state_dict["ema_model_state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        cfm_model.load_state_dict(state_dict, strict=False)
        cfm_model.eval()
        
        f5_model = cfm_model
        
        # 5. Load Vocos vocoder (valid parameters: vocoder_name, device)
        vocos_vocoder = load_vocoder(vocoder_name="vocos", device="cpu")
        print("F5-TTS Italian Model and Vocos loaded successfully!")

    return f5_model, vocos_vocoder



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
            wav = vits_synthesizer.tts(text=text_in, language_name='it', length_scale=1.15)
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



async def synthesize_parler(text_in: str, description: str = None) -> io.BytesIO:
    model, tokenizer, desc_tokenizer = get_parler()
    
    # Use trained Italian speaker 'Julia' to guarantee a female voice output
    if not description:
        description = (
            "Julia's voice is clear and expressive with a slightly warm tone, moderate pace, "
            "very high audio quality, close-mic recording, like a news narrator"
        )

    def _do_synth():
        with synth_lock:
            # Tokenize voice prompt using description_tokenizer
            input_ids = desc_tokenizer(description, return_tensors="pt").input_ids
            # Tokenize Italian spoken text using main tokenizer
            prompt_input_ids = tokenizer(text_in, return_tensors="pt").input_ids
            
            # Generate audio using both tokenized inputs
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )
            audio_arr = generation.cpu().numpy().squeeze()

        buf = io.BytesIO()
        sf.write(buf, audio_arr, model.config.sampling_rate, format='WAV')
        buf.seek(0)
        return buf

    return await asyncio.to_thread(_do_synth)


async def synthesize_f5(text_in: str, ref_audio_path: str = None) -> io.BytesIO:
    f5, vocoder = get_f5()
    
    def _do_synth():
        with synth_lock:
            with torch.no_grad():
                ref_path = ref_audio_path if (ref_audio_path and os.path.exists(ref_audio_path)) else "example.wav"
                
                # Use kwargs explicitly to prevent positional misalignment in batch processing
                wav_np, sr, _ = infer_process(
                    ref_audio=ref_path,
                    ref_text="I lettori, le persone erano — arrabbiate con noi, gli dicevano ma eeh che fate? E avevano ragione; Allora-, eh, io avevo iniziato a fare la direttrice, era proprio il primo anno che mi sono trovata dentro il caos del Covid e non sapevo, che pesci pigliare",
                    gen_text=text_in,
                    model_obj=f5,
                    vocoder=vocoder,
                    device="cpu",
                    show_info=print,
                    nfe_step=16, 
                )
                
        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format='WAV')
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
    description = data.get("description", None)  # For Parler prompt

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
        wav_buf = await synthesize_chatterbox(processed_text, 'example.wav')
    elif model_id == "parler":
        wav_buf = await synthesize_parler(processed_text, description)
    elif model_id == "f5":
        wav_buf = await synthesize_f5(processed_text, 'example.wav')
    else:
        raise HTTPException(status_code=400, detail=f"Modello sconosciuto: {model_id}")

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"}
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")