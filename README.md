# La Voce

A minimal **fullstack newsreader app** that converts Italian text into speech using a **finetuned VITS model** on Italian news.  

Built as a prototype for experimenting with **text normalization**, **TTS synthesis**, and a simple retro/newspaper-inspired web interface

Inspired by and based on [vits-finetuning-news-it](https://github.com/n1kg0r/vits-finetuning-news-it)


---

## Features

- Paste Italian text and listen in the finetuned voice  
- Automatic **text normalization** (lowercase, numbers-to-words, punctuation fixes, optional substitutions like *cha → cia*)  
- **Interactive web interface** with:
  - Play button with loading animation  
  - Panel to view normalized text  
  - Minimal, newspaper-style design  

---

## Project Structure

```bash
newsreader-app/
│── main.py            # FastAPI backend + routes
│── requirements.txt   # dependencies
│── static/            # frontend
│    ├── index.html
│    ├── style.css
│    └── app.js
```

## Model Download

Before running the app, download the pretrained model (not included in this repository due to size):

```bash
gdown 1Uro2gsqQ8SrcWHwx4BK8FCrBu-Oc0rPt --output vits_it_male.pth
```

Or:
```bash
gdown 1KGcKJnwkxluqS0khS53oNcvpxFZkFzI5 --output vits_it_female.pth
```


## Usage

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the server:**

```bash 
uvicorn main:app --reload
```

3. **Open the frontend:**

* Open `static/index.html` in a browser or serve via FastAPI at `http://127.0.0.1:8000`

4. **Try it out:**

* Paste text, click play, and optionally view the normalized text.



## Notes

* The app loads the finetuned **VITS model (~1GB)** into memory; avoid multiple large models simultaneously

## References

[VITS Italian news finetuning repository](https://github.com/n1kg0r/vits-finetuning-news-it)