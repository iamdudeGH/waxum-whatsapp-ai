#!/usr/bin/env python3
"""Waxum inbound -> Groq -> WhatsApp auto-reply worker (low latency)."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import voice_note

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state.json"
PERSONA = ROOT / "persona.txt"
LOG = ROOT / "bot.log"

WAXUM = os.environ.get("WAXUM_URL", "http://127.0.0.1:3451").rstrip("/")
SESSION = os.environ.get("WAXUM_SESSION", "main")
TOKEN_FILE = Path(os.environ.get("WAXUM_TOKEN_FILE", "./superadmin.token"))
MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
POLL_SECS = float(os.environ.get("POLL_SECS", "0.4"))
HISTORY = int(os.environ.get("CHAT_HISTORY", "4"))


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def token() -> str:
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def waxum(method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        WAXUM + path,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + token(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"last_id": 0, "seen": []}


def save_state(st: dict) -> None:
    st["seen"] = list(st.get("seen", []))[-500:]
    STATE.write_text(json.dumps(st), encoding="utf-8")


def mark_seen(st: dict, msg: dict) -> None:
    mid = msg.get("id")
    message_id = msg.get("message_id")
    if mid is not None:
        st["last_id"] = max(st.get("last_id", 0), mid)
    if message_id:
        st.setdefault("seen", []).append(message_id)
    save_state(st)


def groq_chat(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": int(os.environ.get("MAX_TOKENS", "160")),
    }
    proc = subprocess.run(
        [
            "curl", "-sS", "https://api.groq.com/openai/v1/chat/completions",
            "-H", "Authorization: Bearer " + os.environ["GROQ_API_KEY"],
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: Mozilla/5.0",
            "-d", json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or "curl failed")
    data = json.loads(proc.stdout)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        # some models put the answer in reasoning / refuse empty content
        reason = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        for line in reversed(reason.splitlines()):
            line=line.strip().strip('"')
            if len(line) > 2 and not line.lower().startswith(("the user","we need","developer","user asks")):
                content=line
                break
    return content


def recent_chat(chat_jid: str) -> list[dict]:
    data = waxum("GET", f"/api/v1/sessions/{SESSION}/messages?limit=30")
    msgs = [m for m in data.get("messages", []) if m.get("chat_jid") == chat_jid]
    msgs = list(reversed(msgs[:HISTORY]))
    out = []
    for m in msgs:
        role = "assistant" if m.get("direction") == "out" else "user"
        body = (m.get("body") or "").strip()
        if body:
            out.append({"role": role, "content": body})
    return out


def send_text(to: str, text: str) -> dict:
    return waxum(
        "POST",
        f"/api/v1/sessions/{SESSION}/messages/text",
        {"to": to, "text": text},
    )


ALLOWLIST_FILE = ROOT / "allowlist.txt"

def load_allowlist() -> set[str]:
    if not ALLOWLIST_FILE.exists():
        return set()
    out=set()
    for line in ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines():
        line=line.split("#",1)[0].strip()
        digits="".join(ch for ch in line if ch.isdigit())
        if digits:
            out.add(digits)
    return out

_lid_phone_cache: dict[str, str] = {}

def resolve_phone(msg: dict) -> str:
    """Map WhatsApp LID/chat JID to E.164 digits for allowlist checks."""
    direct = msg.get("from_phone") or msg.get("phone_number") or msg.get("phone")
    if direct:
        digits = "".join(ch for ch in str(direct) if ch.isdigit())
        if digits:
            return digits
    jid = str(msg.get("sender_jid") or msg.get("chat_jid") or "")
    if not jid:
        return ""
    if jid.endswith("@s.whatsapp.net") or jid.endswith("@c.us"):
        return "".join(ch for ch in jid.split("@", 1)[0] if ch.isdigit())
    if jid in _lid_phone_cache:
        return _lid_phone_cache[jid]
    try:
        data = waxum(
            "GET",
            f"/api/v1/sessions/{SESSION}/contacts/{urllib.parse.quote(jid, safe='')}/lid",
        )
        phone = "".join(ch for ch in str(data.get("phone_number") or "") if ch.isdigit())
        if phone:
            _lid_phone_cache[jid] = phone
            return phone
    except Exception as e:
        log(f"lid_resolve_error jid={jid}: {e}")
    return "".join(ch for ch in jid.split("@", 1)[0] if ch.isdigit())

def phone_digits(msg: dict) -> str:
    return resolve_phone(msg)

def handle_inbound(msg: dict, st: dict) -> None:
    mid = msg.get("id")
    message_id = msg.get("message_id")
    if mid is not None and mid <= st.get("last_id", 0):
        return
    if message_id and message_id in st.get("seen", []):
        return
    if msg.get("direction") != "in":
        mark_seen(st, msg)
        return
    mtype = msg.get("msg_type")
    is_voice = mtype in ("audio", "ptt", "voice") and bool(msg.get("media"))
    if mtype not in (None, "text", "extended_text") and not is_voice:
        mark_seen(st, msg)
        return
    body = "[voice]" if is_voice else (msg.get("body") or "").strip()
    if not body and not is_voice:
        mark_seen(st, msg)
        return
    chat = msg.get("chat_jid") or ""
    if chat.endswith("@g.us") or chat.endswith("@broadcast"):
        mark_seen(st, msg)
        return
    allow = load_allowlist()
    phone = resolve_phone(msg)
    if not allow:
        mark_seen(st, msg)
        return
    if phone not in allow and not any(phone.endswith(a) or a.endswith(phone) for a in allow):
        log(f"skip_not_allowlisted id={mid} phone={phone or '?'} jid={chat}")
        mark_seen(st, msg)
        return
    to = msg.get("from_phone") or (f"{phone}@s.whatsapp.net" if phone else chat)
    t0 = time.time()
    if is_voice:
        try:
            audio = voice_note.download(waxum, SESSION, msg["media"])
            body = voice_note.transcribe(audio)
            log(f"voice_in id={mid} bytes={len(audio)} stt_ms={int((time.time()-t0)*1000)} text={body[:120]!r}")
            if not body:
                send_text(to, "Ek baar phir se boliye?")
                mark_seen(st, msg)
                return
        except Exception as e:
            log(f"voice_in_error id={mid}: {e}")
            mark_seen(st, msg)
            return
    persona = PERSONA.read_text(encoding="utf-8").strip()
    respectful = {
        "".join(c for c in x if c.isdigit())
        for x in os.environ.get("RESPECTFUL_NUMBERS", "").split(",")
        if x.strip()
    }
    if phone and any(phone.endswith(n) or n.endswith(phone) for n in respectful):
        persona += " With this person be respectful while casual: polite soft Hinglish, no rude slang."

    if is_voice:
        persona += (
            "\nUser ne voice note bheja; upar transcript hai (galtiyan ho sakti hain). "
            "Roman Hinglish mein 1-3 short sentences mein jawab do."
        )
    history = recent_chat(chat)
    if not history or history[-1].get("content") != body:
        history.append({"role": "user", "content": body})
    messages = [{"role": "system", "content": persona}] + history
    try:
        raw = groq_chat(messages)
        reply = voice_note.clean_reply(raw)
        if not reply:
            log(f"empty_raw id={mid} raw={raw[:160]!r}")
            raw = groq_chat(messages + [{"role": "user", "content": "Ab short Hinglish mein jawab do."}])
            reply = voice_note.clean_reply(raw)
    except Exception as e:
        log(f"groq_error id={mid}: {e}")
        mark_seen(st, msg)
        return
    if not reply:
        reply = "Haan, boliye?"
        log(f"empty_reply_fallback id={mid}")
    if is_voice:
        try:
            ogg, secs = voice_note.tts_ogg(reply)
            res = voice_note.send_voice(waxum, SESSION, to, ogg, secs, msg.get("message_id"))
            log(
                f"voice_replied id={mid} to={to} secs={secs} ms={int((time.time()-t0)*1000)} "
                f"msg={res.get('message_id')} text={reply[:80]!r}"
            )
            mark_seen(st, msg)
            return
        except Exception as e:
            log(f"voice_send_error id={mid}: {e} -> text fallback")
    try:
        res = send_text(to, reply)
        log(
            f"replied id={mid} to={to} ms={int((time.time()-t0)*1000)} "
            f"msg={res.get('message_id')} text={reply[:80]!r}"
        )
    except Exception as e:
        log(f"send_error id={mid}: {e}")
        return
    mark_seen(st, msg)


def bootstrap_cursor(st: dict) -> dict:
    if st.get("last_id"):
        return st
    try:
        data = waxum("GET", f"/api/v1/sessions/{SESSION}/messages?limit=1")
        msgs = data.get("messages") or []
        if msgs and msgs[0].get("id") is not None:
            st["last_id"] = msgs[0]["id"]
            save_state(st)
            log(f"bootstrap last_id={st['last_id']}")
    except Exception as e:
        log(f"bootstrap_error: {e}")
    return st


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing")
    st = bootstrap_cursor(load_state())
    log(f"started model={MODEL} session={SESSION} poll={POLL_SECS}s")
    while True:
        try:
            data = waxum("GET", f"/api/v1/sessions/{SESSION}/messages?limit=20")
            msgs = list(reversed(data.get("messages") or []))
            for msg in msgs:
                handle_inbound(msg, st)
        except Exception as e:
            log(f"loop_error: {e}")
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
