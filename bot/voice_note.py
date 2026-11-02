"""Voice-note helpers: Waxum download -> Groq Whisper -> edge-tts -> ogg/opus PTT send."""
from __future__ import annotations
import asyncio, base64, json, os, re, subprocess, tempfile
from pathlib import Path

STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3")
STT_LANG = os.environ.get("STT_LANG", "hi")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-IN-PrabhatNeural")
STT_PROMPT = ("Hinglish baatchit, Hindi aur English mix. Jaise: haan bhai, kya haal hai, "
              "mujhe batao, thik hai, kaise karu, sardi zukaam ka gharelu ilaaj.")

def download(waxum, session: str, media: dict) -> bytes:
    body = {k: media[k] for k in ("direct_path", "media_key", "file_sha256", "file_enc_sha256", "file_length")}
    body["media_type"] = "audio"
    res = waxum("POST", f"/api/v1/sessions/{session}/media/download", body)
    return base64.b64decode(res["data"])

def transcribe(audio: bytes) -> str:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.ogg"; wav = Path(td) / "in.wav"
        src.write_bytes(audio)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)], check=True, timeout=60)
        cmd = ["curl", "-sS", "https://api.groq.com/openai/v1/audio/transcriptions",
               "-H", "Authorization: Bearer " + os.environ["GROQ_API_KEY"],
               "-F", f"file=@{wav}", "-F", f"model={STT_MODEL}", "-F", "temperature=0",
               "-F", "response_format=json", "-F", f"prompt={STT_PROMPT}"]
        if STT_LANG:
            cmd += ["-F", f"language={STT_LANG}"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        data = json.loads(p.stdout or "{}")
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        return (data.get("text") or "").strip()

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

def clean_reply(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"[*_#`>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def tts_ogg(text: str) -> tuple[bytes, int]:
    import edge_tts
    with tempfile.TemporaryDirectory() as td:
        mp3 = Path(td) / "o.mp3"; ogg = Path(td) / "o.ogg"
        async def go():
            await edge_tts.Communicate(text, TTS_VOICE, rate="+5%").save(str(mp3))
        asyncio.run(go())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", "48000",
                        "-c:a", "libopus", "-b:a", "32k", "-application", "voip", str(ogg)], check=True, timeout=60)
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(ogg)],
                             capture_output=True, text=True).stdout.strip()
        return ogg.read_bytes(), max(1, round(float(dur or 1)))

def send_voice(waxum, session: str, to: str, ogg: bytes, secs: int, reply_to: str | None = None) -> dict:
    body = {"to": to, "ptt": True, "seconds": secs,
            "audio": {"data": base64.b64encode(ogg).decode(), "mimetype": "audio/ogg; codecs=opus"}}
    if reply_to:
        body["reply_to"] = reply_to
    return waxum("POST", f"/api/v1/sessions/{session}/messages/audio", body)
