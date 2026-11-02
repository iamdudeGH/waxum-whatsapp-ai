#!/usr/bin/env python3
"""Two-way WhatsApp voice agent with utterance VAD + echo lock."""
from __future__ import annotations

import array
import asyncio
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent
PERSONA = ROOT / "persona.txt"
LOG = ROOT / "voice.log"
TOKEN = Path(os.environ.get("WAXUM_TOKEN_FILE", str(ROOT.parent / "superadmin.token"))).read_text().strip()
SESSION = os.environ.get("WAXUM_SESSION", "main")
MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
VOICE = os.environ.get("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_RATE = os.environ.get("TTS_RATE", "-5%")
STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3")
STT_LANG = os.environ.get("STT_LANG", "hi")  # hi keeps Hinglish from becoming Greek/PT
ECHO_TAIL_S = float(os.environ.get("ECHO_TAIL_S", "0.8"))
BARGE_RMS = int(os.environ.get("BARGE_RMS", "2200"))
BARGE_GRACE_S = float(os.environ.get("BARGE_GRACE_S", "1.5"))
BARGE_FRAMES = int(os.environ.get("BARGE_FRAMES", "4"))
SAMPLE_RATE = 16000
FRAME_SAMPLES = 960
FRAME_BYTES = FRAME_SAMPLES * 2
SILENCE_FRAME = b"\x00" * FRAME_BYTES
ANSWER_GRACE_S = float(os.environ.get("ANSWER_GRACE_S", "7"))
SPEECH_RMS = int(os.environ.get("SPEECH_RMS", "300"))
END_SILENCE_MS = int(os.environ.get("END_SILENCE_MS", "800"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "400"))
MAX_SPEECH_MS = int(os.environ.get("MAX_SPEECH_MS", "7000"))


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def rms(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0
    acc = sum(s * s for s in samples)
    return int((acc / len(samples)) ** 0.5)


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def to_roman_hinglish(text: str) -> str:
    """Drop Devanagari / non-latin junk so Edge en-IN TTS never crashes."""
    text = (text or "").strip()
    if not text:
        return ""
    # keep latin letters, digits, basic punct and spaces
    kept = []
    for ch in text:
        o = ord(ch)
        if ch.isalnum() and o < 128:
            kept.append(ch)
        elif ch in " .,!?'-+":
            kept.append(ch)
        elif ch.isspace():
            kept.append(" ")
    out = " ".join("".join(kept).split())
    return out


def clean_reply(text: str) -> str:
    text = (text or "").replace("**", "").replace("*", "").replace("`", "").strip()
    # drop model reasoning leaks
    low = text.lower()
    if low.startswith(("user says", "the user", "we need", "developer", "possibly", "that seems")):
        return ""
    text = to_roman_hinglish(text)
    # keep up to 3 short sentences for slightly deeper answers
    parts=[]; buf=text
    for sep in [". ", "! ", "? "]:
        pass
    import re
    sents=re.split(r'(?<=[.!?])\s+', text)
    sents=[s.strip() for s in sents if s.strip()]
    text=" ".join(sents[:3])
    words=text.split()
    if len(words) > 55:
        text=" ".join(words[:55])
    return text.strip()


def groq_chat(messages: list[dict]) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 180}
    proc = subprocess.run(
        [
            "curl", "-sS", "https://api.groq.com/openai/v1/chat/completions",
            "-H", f"Authorization: Bearer {os.environ['GROQ_API_KEY']}",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: Mozilla/5.0",
            "-d", json.dumps(payload),
        ],
        capture_output=True, text=True, timeout=60,
    )
    data = json.loads(proc.stdout)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text:
        reason = (msg.get("reasoning") or "").strip()
        for line in reversed(reason.splitlines()):
            line = line.strip().strip('"')
            if line and not line.lower().startswith(("the user", "we need", "developer", "user says")):
                text = line
                break
    text = clean_reply(text)
    if not text or has_devanagari(text):
        # one forced rewrite into roman Hinglish
        rewrite = messages + [
            {"role": "assistant", "content": text or "(empty)"},
            {"role": "user", "content": "Rewrite your last reply in roman Hinglish only. One short sentence. No Hindi script. No English-only."},
        ]
        payload2 = {"model": MODEL, "messages": rewrite, "temperature": 0.2, "max_tokens": 80}
        proc2 = subprocess.run(
            [
                "curl", "-sS", "https://api.groq.com/openai/v1/chat/completions",
                "-H", f"Authorization: Bearer {os.environ['GROQ_API_KEY']}",
                "-H", "Content-Type: application/json",
                "-H", "User-Agent: Mozilla/5.0",
                "-d", json.dumps(payload2),
            ],
            capture_output=True, text=True, timeout=60,
        )
        data2 = json.loads(proc2.stdout)
        if "error" not in data2:
            text = clean_reply(data2["choices"][0]["message"].get("content") or "")
    if not text:
        text = "Haan, boliye?"
    return text



def normalize_stt(raw: str) -> str:
    """Turn messy Whisper output into likely Hinglish intent."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "You fix WhatsApp call transcripts. Input may be wrong-language Whisper text. "
                "Output ONLY the likely user meaning in short roman Hinglish. "
                "If it is pure noise/unknown, output UNKNOWN. No quotes."
            ),
        },
        {"role": "user", "content": raw},
    ]
    payload = {"model": MODEL, "messages": messages, "temperature": 0.1, "max_tokens": 60}
    proc = subprocess.run(
        [
            "curl", "-sS", "https://api.groq.com/openai/v1/chat/completions",
            "-H", f"Authorization: Bearer {os.environ['GROQ_API_KEY']}",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: Mozilla/5.0",
            "-d", json.dumps(payload),
        ],
        capture_output=True, text=True, timeout=60,
    )
    data = json.loads(proc.stdout)
    if "error" in data:
        return raw
    out = (data["choices"][0]["message"].get("content") or "").strip()
    out = to_roman_hinglish(out)
    if not out or out.upper() == "UNKNOWN":
        return ""
    return out


def transcribe_pcm(pcm: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        cmd = [
            "curl", "-sS", "https://api.groq.com/openai/v1/audio/transcriptions",
            "-H", f"Authorization: Bearer {os.environ['GROQ_API_KEY']}",
            "-H", "User-Agent: Mozilla/5.0",
            "-F", f"file=@{path};type=audio/wav",
            "-F", f"model={STT_MODEL}",
            "-F", "temperature=0",
            "-F", "prompt=Indian Hinglish phone call. Words like haan, theek hai, bilkul, kya haal, meeting, call, okay.",
            "-F", "response_format=json",
        ]
        if STT_LANG.strip():
            cmd.extend(["-F", f"language={STT_LANG.strip()}"])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(proc.stdout or "{}")
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        text = (data.get("text") or "").strip()
        # filter whisper hallucinations / prompt leaks
        bad = {"hinglish phone call.", "thank you.", "thanks for watching.", "subscribe."}
        if text.lower().strip() in bad:
            return ""
        return text
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def synthesize_pcm(text: str) -> bytes:
    text = to_roman_hinglish(text) or "Haan?"
    with tempfile.TemporaryDirectory() as td:
        mp3 = Path(td) / "out.mp3"
        proc = subprocess.run(
            [
                os.environ.get("EDGE_TTS_BIN", "edge-tts"),
                "--voice", VOICE,
                f"--rate={TTS_RATE}",
                "--volume=+20%",
                "--text", text,
                "--write-media", str(mp3),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0 or not mp3.exists():
            raise RuntimeError(proc.stderr or "edge-tts failed")
        pcm = subprocess.check_output(
            [
                "ffmpeg", "-v", "error", "-i", str(mp3),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
            ],
            timeout=60,
        )
        samples = array.array("h")
        samples.frombytes(pcm)
        for i, s in enumerate(samples):
            samples[i] = max(-32767, min(32767, int(s * 1.6)))
        return samples.tobytes()


def pcm_frames(pcm: bytes):
    if len(pcm) % FRAME_BYTES:
        pcm += b"\x00" * (FRAME_BYTES - (len(pcm) % FRAME_BYTES))
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


async def run_call(to: str, greeting: str | None = None):
    persona = PERSONA.read_text(encoding="utf-8").strip()
    persona += (
        " Live phone call mode: answer only what they just said."
        " ALWAYS speak Hinglish in roman letters only, e.g. 'Haan bilkul, samajh gaya.'"
        " Never pure English. Never Devanagari/Hindi script (TTS will break)."
        " Whisper transcript can be wrong language or garbage — still infer Hinglish intent."
        " If nonsense/unclear: say exactly 'Ek baar phir se boliye?'"
        " 2-3 short spoken sentences max when the question needs it (home remedies, how-to). Still no long monologue."
    )
    history: list[dict] = [{"role": "system", "content": persona}]
    url = f"ws://127.0.0.1:3451/api/v1/sessions/{SESSION}/calls/media/ws?to={to}&kind=audio"
    log(f"dialing {to} model={MODEL} voice={VOICE}")

    greeting_pcm = b""
    if greeting:
        greeting_pcm = await asyncio.to_thread(synthesize_pcm, greeting)
        log(f"greeting pcm bytes={len(greeting_pcm)}")

    out_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=900)
    peer_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5000)
    stop = asyncio.Event()
    peer_frames = 0
    speaking_until = 0.0
    speak_started = 0.0

    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {TOKEN}"},
        max_size=8_000_000,
        open_timeout=30,
        ping_interval=20,
    ) as ws:
        meta = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        log(f"meta {meta}")
        if meta.get("type") != "call_started":
            return
        call_id = meta.get("call_id")

        async def sender():
            nonlocal speaking_until
            next_t = time.monotonic()
            try:
                while not stop.is_set():
                    try:
                        frame = out_q.get_nowait()
                        speaking_until = time.time() + ECHO_TAIL_S
                    except asyncio.QueueEmpty:
                        frame = SILENCE_FRAME
                    if frame is None:
                        break
                    await ws.send(frame)
                    next_t += 0.06
                    await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            except Exception as e:
                log(f"sender_end {e}")
                stop.set()

        async def receiver():
            nonlocal peer_frames
            rx = bytearray()
            try:
                async for message in ws:
                    if isinstance(message, str):
                        log(f"text {message[:180]}")
                        continue
                    if peer_frames < 40 or peer_frames % 500 == 0:
                        log(f"rx_msg_bytes={len(message)} buf={len(rx)}")
                    rx.extend(message)
                    while len(rx) >= FRAME_BYTES:
                        fr = bytes(rx[:FRAME_BYTES])
                        del rx[:FRAME_BYTES]
                        peer_frames += 1
                        try:
                            peer_q.put_nowait(fr)
                        except asyncio.QueueFull:
                            try:
                                peer_q.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            peer_q.put_nowait(fr)
            except Exception as e:
                log(f"receiver_end {e}")
                stop.set()

        async def enqueue_pcm(pcm: bytes):
            nonlocal speaking_until, speak_started
            speak_started = time.time()
            speaking_until = time.time() + ECHO_TAIL_S
            for fr in pcm_frames(pcm):
                await out_q.put(fr)

        def stop_playback():
            n = 0
            while True:
                try:
                    out_q.get_nowait(); n += 1
                except asyncio.QueueEmpty:
                    return n

        async def stats():
            while not stop.is_set():
                log(
                    f"stats peer_frames={peer_frames} out_q={out_q.qsize()} "
                    f"peer_q={peer_q.qsize()} speaking={time.time()<speaking_until}"
                )
                await asyncio.sleep(2)

        async def thinker():
            nonlocal speaking_until
            log("waiting for you to pick up...")
            # Wait until peer media actually starts (answered), up to 40s.
            waited = 0.0
            while not stop.is_set() and waited < 45:
                if waited >= 1.5 and peer_frames >= 15:
                    break
                await asyncio.sleep(0.2)
                waited += 0.2
            if stop.is_set():
                return
            if peer_frames < 15:
                log("no pickup detected; hanging up")
                stop.set()
                return
            log(f"pickup detected after {waited:.1f}s peer_frames={peer_frames}")
            await asyncio.sleep(min(1.0, ANSWER_GRACE_S / 6))
            if greeting_pcm:
                await enqueue_pcm(greeting_pcm)
                history.append({"role": "assistant", "content": greeting or ""})
                log(f"greeting enqueued after pickup peer_frames={peer_frames}")
                # brief second cue if first was clipped
                await asyncio.sleep(0.3)

            buf = bytearray()
            speech_ms = 0
            silence_ms = 0
            in_speech = False
            loud = 0

            async def finish():
                nonlocal buf, speech_ms, silence_ms, in_speech
                pcm = bytes(buf)
                buf.clear()
                speech_ms = 0
                silence_ms = 0
                in_speech = False
                if len(pcm) < int(SAMPLE_RATE * 0.35) * 2:
                    return
                level = rms(pcm)
                log(f"utterance_rms={level} bytes={len(pcm)}")
                if level < SPEECH_RMS // 2:
                    return
                try:
                    text = await asyncio.to_thread(transcribe_pcm, pcm)
                    log(f"heard_raw {text!r}")
                    if not text or len(text.strip()) < 2 or text.strip().lower().startswith("phone call in hinglish"):
                        return
                    text = await asyncio.to_thread(normalize_stt, text)
                    log(f"heard {text!r}")
                    if not text:
                        reply = "Ek baar phir se boliye?"
                        history.append({"role": "assistant", "content": reply})
                        out = await asyncio.to_thread(synthesize_pcm, reply)
                        while not peer_q.empty():
                            try:
                                peer_q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        await enqueue_pcm(out)
                        return
                    history.append({"role": "user", "content": text})
                    reply = await asyncio.to_thread(groq_chat, history[-8:])
                    log(f"say {reply!r}")
                    if not reply:
                        return
                    history.append({"role": "assistant", "content": reply})
                    out = await asyncio.to_thread(synthesize_pcm, reply)
                    # drop queued inbound during thinking
                    while not peer_q.empty():
                        try:
                            peer_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    await enqueue_pcm(out)
                except Exception as e:
                    log(f"turn_error {e}")

            while not stop.is_set():
                # Don't listen while we speak / shortly after (echo lock)
                if time.time() < speaking_until or not out_q.empty():
                    speech_ms = 0
                    silence_ms = 0
                    in_speech = False
                    barging = time.time() - speak_started > BARGE_GRACE_S and not out_q.empty()
                    try:
                        while True:
                            fr = peer_q.get_nowait()
                            if barging and rms(fr) >= BARGE_RMS:
                                loud += 1
                                buf.extend(fr)
                            else:
                                loud = 0
                                buf.clear()
                            if loud >= BARGE_FRAMES:
                                n = stop_playback()
                                speaking_until = time.time() + 0.15
                                log(f"barge_in dropped_frames={n} rms={rms(fr)}")
                                in_speech = True
                                speech_ms = loud * 60
                                loud = 0
                                break
                    except asyncio.QueueEmpty:
                        pass
                    if not in_speech:
                        await asyncio.sleep(0.03)
                        continue
                    await asyncio.sleep(0.15)
                    continue
                try:
                    frame = await asyncio.wait_for(peer_q.get(), timeout=0.4)
                except asyncio.TimeoutError:
                    if in_speech and speech_ms >= MIN_SPEECH_MS and silence_ms >= END_SILENCE_MS:
                        await finish()
                    continue
                level = rms(frame)
                if level >= SPEECH_RMS:
                    in_speech = True
                    silence_ms = 0
                    speech_ms += 60
                    buf.extend(frame)
                    if speech_ms >= MAX_SPEECH_MS:
                        await finish()
                elif in_speech:
                    silence_ms += 60
                    buf.extend(frame)
                    if silence_ms >= END_SILENCE_MS and speech_ms >= MIN_SPEECH_MS:
                        await finish()

        tasks = [
            asyncio.create_task(sender(), name="sender"),
            asyncio.create_task(receiver(), name="receiver"),
            asyncio.create_task(thinker(), name="thinker"),
            asyncio.create_task(stats(), name="stats"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            try:
                exc = d.exception()
                log(f"task_done {d.get_name()} err={exc!r}")
            except Exception as e:
                log(f"task_status {e!r}")
        stop.set()
        try:
            out_q.put_nowait(None)
        except Exception:
            pass
        for t in pending:
            t.cancel()
        log(f"call ended {call_id} peer_frames={peer_frames}")


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--to", required=True)
    p.add_argument("--greeting", default="Namaste. Main aapka AI assistant hoon. Boliye.")
    args = p.parse_args()
    await run_call(args.to, args.greeting)


if __name__ == "__main__":
    asyncio.run(main())
