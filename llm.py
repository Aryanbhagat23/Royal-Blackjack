"""
Chat support: a local model (Ollama) or Google Gemini
----------------------------------------------------
Adds a Professor you can talk to in plain language. Two ways to power it:

* **Ollama** — a model running on your own computer. Free, private, and needs no key, but it only
  works on that machine, so it cannot work on the deployed link.
* **Gemini** — Google's hosted API, which has a free tier. This one does work on the deployed app.
  The key is read from Streamlit secrets or the GEMINI_API_KEY environment variable and is never
  written into the code.

Whichever is available is used; if both are, the app lets you choose. If neither is set up, the chat
hides itself and the rest of the app is unaffected.

The point of doing it this way: the model is never asked to figure out Blackjack by itself. Every
question is sent with the real numbers attached — the agent's learned values for the exact situation,
the true count, the odds from the shoe — so the LLM explains what the reinforcement learning agent
decided, rather than inventing its own advice. If a question mentions a hand ("16 vs 10", "soft 18
against a 9", "pair of 8s vs 6"), that hand is looked up in the agent's table and attached too.

Setup (one time):
    1. Install Ollama from ollama.com
    2. ollama pull llama3.2
    3. Start the app — the chat appears automatically
"""

import json
import re
import urllib.error
import urllib.request

import blackjack_rl as rl

HOST = "http://127.0.0.1:11434"          # Ollama, on this machine
GEMINI_HOST = "https://generativelanguage.googleapis.com"
TIMEOUT = 1.5

# Ollama is on this machine, so never go through a system proxy: on a campus or office network
# urllib would otherwise send localhost requests to the proxy and hang instead of failing fast.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# Gemini is on the internet, so it must use the system proxy if there is one.
_web = urllib.request.build_opener()
PREFERRED = ["llama3.2", "llama3.1", "llama3", "mistral", "phi3", "qwen2.5", "gemma2"]

# Known-good text chat models, best first. Anything else the key can use is listed after these.
GEMINI_PREFERRED = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite",
                    "gemini-flash-lite-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
# Model families that answer generateContent but are not text chat models (images, speech, robots...).
_NOT_CHAT = ("embedding", "aqa", "vision", "image", "tts", "audio", "live", "gemma", "robotics",
             "computer-use", "imagen", "veo", "learnlm", "banana", "thinking-exp")
HISTORY_TURNS = 12                       # how many past messages are sent with each question


class LLMError(Exception):
    """A problem worth showing the user, already phrased in plain language."""


SYSTEM_PROMPT = """You are "the Professor", a friendly Blackjack and reinforcement learning tutor built into a \
student project called Royal Blackjack.

The app contains a reinforcement learning agent that taught itself Blackjack by playing 30 million hands against \
itself using Monte Carlo control. It matches published basic strategy on 96.7% of decisions. A second agent counts \
cards and beats the house edge.

Rules for your answers:
- The numbers in CONTEXT come from the trained agent and from simulations. Treat them as the truth and build your \
answer around them. Never contradict them and never invent different numbers.
- If the context gives an expected value, quote it and say what it means in money (e.g. -0.12 = losing about \
12 cents per dollar bet on average).
- Lead with the direct answer, then the reason.
- If you are asked something the context does not cover, say what you do know and be honest about the rest.
- Never encourage real gambling. If asked about betting real money, mention that the house edge means the player \
loses over time, and that counting is the only exception and gets people barred from casinos."""


def _get(path, timeout=TIMEOUT):
    with _opener.open(f"{HOST}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def available():
    """(is_running, [model names]). Never raises: the app must work without Ollama."""
    try:
        data = _get("/api/tags")
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


def pick_model(models):
    """Prefer a small, fast, well-known model; otherwise take whatever is installed."""
    for want in PREFERRED:
        for m in models:
            if m.split(":")[0] == want:
                return m
    return models[0] if models else None


def clean_history(messages, limit=HISTORY_TURNS):
    """Make a chat history every API accepts: no empty turns, strict user/assistant alternation,
    starting with the user, and only the last `limit` messages."""
    out = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if not text:
            continue
        role = "assistant" if m.get("role") == "assistant" else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    out = out[-limit:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _system(context, style="", examples=()):
    parts = [SYSTEM_PROMPT]
    if style:
        parts.append("ANSWER STYLE: " + style)
    if examples:
        shots = "\n\n".join(f"Student: {q}\nProfessor: {a}" for q, a in examples)
        parts.append("Answers students rated helpful before (copy the tone and clarity, NOT the numbers):\n" + shots)
    if context:
        parts.append("CONTEXT (real numbers from this app):\n" + context)
    return "\n\n".join(parts)


def chat_stream(model, messages, context="", temperature=0.4, style="", examples=()):
    """Yield the reply piece by piece, so it types out live in the app. Raises LLMError."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": _system(context, style, examples)}] + clean_history(messages),
        "stream": True,
        "options": {"temperature": temperature, "num_predict": 600},
    }
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=120) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise LLMError(f"The local model returned an error: {chunk['error']}")
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return
    except LLMError:
        raise
    except urllib.error.URLError:
        raise LLMError("Lost contact with Ollama. Is it still running?")
    except Exception as e:
        raise LLMError(f"The local model returned an error: {e}")


# ---------------------------------------------------------
# Google Gemini (works on the deployed app)
# ---------------------------------------------------------

def api_key():
    """The Gemini key from Streamlit secrets or the environment. Never hard-code it."""
    import os
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    return key.strip() if key else None


def _is_chat_model(name):
    return name.startswith("gemini") and not any(bad in name for bad in _NOT_CHAT)


def rank_gemini(names):
    """Known-good models first (in GEMINI_PREFERRED order), then stable flash, then the rest."""
    def rank(n):
        if n in GEMINI_PREFERRED:
            return (0, GEMINI_PREFERRED.index(n), len(n))
        return (1, 0 if "flash" in n else 1, 1 if ("preview" in n or "exp" in n) else 0, len(n))
    return sorted(set(names), key=rank)


def gemini_models(key):
    """Ask Google which models this key can actually use, so retired names are never offered.
    Returns chat-capable ones, best first. If the list can't be fetched (network hiccup, bad key),
    fall back to the known-good names; the chat then reports the real problem when it's used."""
    req = urllib.request.Request(f"{GEMINI_HOST}/v1beta/models?pageSize=1000", headers={"x-goog-api-key": key})
    try:
        with _web.open(req, timeout=8) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return list(GEMINI_PREFERRED[:3])
    names = []
    for m in data.get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        name = m.get("name", "").split("/")[-1]
        if name and _is_chat_model(name):
            names.append(name)
    return rank_gemini(names) or list(GEMINI_PREFERRED[:3])


def _gemini_error(e):
    """Turn an HTTP error from Google into (message, worth_trying_another_model)."""
    detail = ""
    try:
        detail = json.loads(e.read().decode()).get("error", {}).get("message", "")
    except Exception:
        pass
    low = detail.lower()
    if e.code == 400 and ("api key" in low or "api_key" in low):
        return "Google rejected the API key. Check GEMINI_API_KEY in your Streamlit secrets.", False
    if e.code in (401, 403):
        return (f"This API key isn't allowed to use Gemini ({detail or 'permission denied'}). Make sure the "
                "Generative Language API is enabled for the key's project."), False
    if e.code == 429:
        return ("Gemini's free-tier limit was reached (a few requests per minute and a daily cap per model). "
                "Wait a minute and try again, or pick a different model."), True
    if e.code == 404:
        return f"That model is no longer available ({detail or 'not found'}). Pick another model.", True
    if e.code >= 500:
        return f"Google's servers are busy right now (error {e.code}). Try again in a moment.", True
    return f"Gemini returned an error {e.code}: {detail}", e.code == 400


def _gemini_once(model, messages, system, key, temperature):
    history = clean_history(messages)
    if not history:
        raise LLMError("There's no question to answer yet.")
    config = {"temperature": temperature, "maxOutputTokens": 2048}
    if "2.5-flash" in model:
        # 2.5 Flash "thinks" before answering and those hidden tokens count against maxOutputTokens,
        # which used to leave answers cut off or empty. Short tutoring answers don't need it.
        config["thinkingConfig"] = {"thinkingBudget": 0}
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]} for m in history],
        "generationConfig": config,
    }
    url = f"{GEMINI_HOST}/v1beta/models/{model}:streamGenerateContent?alt=sse"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    got_text, finish = False, None
    with _web.open(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            block = chunk.get("promptFeedback", {}).get("blockReason")
            if block:
                raise LLMError(f"Gemini declined to answer that question ({block}). Try rephrasing it.")
            for cand in chunk.get("candidates", []):
                finish = cand.get("finishReason") or finish
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        got_text = True
                        yield part["text"]
    if not got_text:
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"):
            raise LLMError(f"Gemini declined to answer that question ({finish}). Try rephrasing it.")
        raise LLMError(f"Gemini sent back an empty answer (finish reason: {finish or 'unknown'}). Try again.")


def gemini_stream(model, messages, context, key, temperature=0.4, style="", examples=(), fallbacks=()):
    """Stream a Gemini reply. If the chosen model is rate limited or retired before it has said anything,
    quietly try the next model in `fallbacks`. Raises LLMError with a plain-language message."""
    if not key:
        raise LLMError("No GEMINI_API_KEY found in Streamlit secrets or the environment.")
    system = _system(context, style, examples)
    tried = [model] + [m for m in fallbacks if m != model][:2]
    last = None
    for m in tried:
        started = False
        try:
            for piece in _gemini_once(m, messages, system, key, temperature):
                started = True
                yield piece
            return
        except urllib.error.HTTPError as e:
            msg, retry = _gemini_error(e)
            last = LLMError(msg)
            if started or not retry:
                raise last
        except LLMError:
            raise
        except Exception as e:
            if started:
                raise LLMError(f"The connection to Gemini dropped mid-answer ({e}).")
            last = LLMError(f"Could not reach Gemini ({e}). Check the internet connection.")
    raise last


def providers():
    """What can power the chat right now: [(id, label, [models]), ...]."""
    out = []
    ok, models = available()
    if ok and models:
        out.append(("ollama", "Ollama (on this computer)", models))
    key = api_key()
    if key:
        out.append(("gemini", "Google Gemini (works online)", gemini_models(key)))
    return out


def default_model(provider, models):
    return pick_model(models) if provider == "ollama" else models[0]


def stream(provider, model, messages, context="", style="", examples=(), fallbacks=(), raise_errors=False):
    """One entry point the pages use, whichever provider is selected.

    With raise_errors=False (the default) a failure is shown as a short note inside the answer, which is
    right for one-off questions. The chat page passes True so a failed reply is never saved into the
    conversation and re-sent to the model."""
    if provider == "gemini":
        gen = gemini_stream(model, messages, context, api_key(), style=style, examples=examples,
                            fallbacks=fallbacks)
    else:
        gen = chat_stream(model, messages, context, style=style, examples=examples)
    if raise_errors:
        return gen
    return _soften(gen)


def _soften(gen):
    try:
        yield from gen
    except LLMError as e:
        yield f"\n\n⚠️ {e}"


# ---------------------------------------------------------
# Building the context that keeps answers grounded
# ---------------------------------------------------------

def project_context(grade=None, baselines=None, race=None):
    """Facts about the project itself, for general questions."""
    lines = ["The agent learns with Monte Carlo control: after each hand every (state, action) it used is moved "
             "toward the money that hand actually won, with step size 1/N so values are running averages.",
             "State = (player total, dealer up card, usable ace, can double, pair value). Actions = hit, stand, "
             "double, split. Reward = money per unit bet, given only at the end of a hand.",
             "Exploration: one random exploring move per hand, and only decisions from that move onward are "
             "learned from."]
    if grade:
        lines.append(f"The expert agent matches the published basic strategy chart on {grade['matches']}/"
                     f"{grade['total']} two-card decisions ({grade['matches'] / grade['total']:.1%}), and every "
                     f"difference is a statistical tie (no real mistakes).")
    if baselines:
        lines.append("Money per $100 bet: " + ", ".join(f"{k} {v:+.2f}" for k, v in baselines.items())
                     + ". The trained agent is about -0.43 and the card-counting agent is about +0.60 per $100 "
                       "wagered.")
    if race:
        parts = []
        for name, rows in race.items():
            if rows:
                parts.append(f"{name} {rows[-1]['match']:.0%}")
        if parts:
            lines.append("Algorithm comparison (match with the strategy chart): " + ", ".join(parts)
                         + ". Monte Carlo wins because hands are short and the reward is terminal, so waiting for "
                           "the true result is unbiased; the neural network approximates a table that only needs a "
                           "few hundred entries and trains ~50x slower per hand.")
    return "\n".join(lines)


def hand_context(cards, dealer_card, values, best, bust=None, dealer_bust=None, true_count=None, bankroll=None):
    """Facts about the hand being played right now."""
    hand = " + ".join(c["rank"] for c in cards)
    lines = [f"The player's hand is {hand}, and the dealer shows {dealer_card['rank']}.",
             f"The trained agent recommends: {best.upper()}.",
             "Expected value per $1 bet for each legal move: "
             + ", ".join(f"{a} {v:+.3f}" for a, v in values.items()) + "."]
    if bust is not None:
        lines.append(f"Chance of busting if the player hits: {bust:.0%}.")
    if dealer_bust is not None:
        lines.append(f"Chance the dealer busts from here: {dealer_bust:.0%}.")
    if true_count is not None:
        lines.append(f"Current Hi-Lo true count: {true_count:+.1f}.")
    if bankroll is not None:
        lines.append(f"The player's bankroll is ${bankroll:,.0f}.")
    return "\n".join(lines)


# ---- hands mentioned in a question ("16 vs 10", "soft 18 against a 9", "pair of 8s vs 6", "A,7 vs 3") ----

_RANK = r"(a|ace|aces|10|[2-9]|j|q|k|jack|queen|king|t)"
_VS = r"\s*(?:vs\.?|versus|v\.?|against|and|with|to|on)\s*(?:a\s+|an\s+|the\s+)?(?:dealer(?:'s)?\s*)?(?:up\s*card\s*(?:of\s*)?)?(?:showing\s+)?"
_DEALER = r"(a|ace|10|[2-9]|j|q|k|jack|queen|king|t)\b"
_PAIR_RE = re.compile(r"(?:pair\s+of\s+|two\s+)" + _RANK + r"'?s?" + _VS + _DEALER)
_TWO_RE = re.compile(r"\b" + _RANK + r"\s*[,+/&-]\s*" + _RANK + r"\b" + _VS + _DEALER)
_TOTAL_RE = re.compile(r"\b(hard|soft)?\s*(1[2-9]|2[01]|[5-9]|1[01])\b" + _VS + _DEALER)


def _rank(tok):
    tok = tok.lower()
    if tok.startswith("a"):
        return "A"
    if tok in ("t", "j", "jack", "q", "queen", "k", "king", "10"):
        return "10"
    return tok


def _card(rank):
    return {"rank": rank, "value": rl.CARD_VALUES[rank]}


def _cards_for_total(total, soft):
    if soft:
        if not 13 <= total <= 21:
            return None
        return [_card("A"), _card(str(total - 11))]
    if not 5 <= total <= 20:
        return None
    hi = min(10, total - 2)
    if total - hi == hi and total < 20:          # avoid accidentally making a pair (e.g. 8 = 4+4)
        hi -= 1
    return [_card(str(hi)), _card(str(total - hi))]


def parse_hands(question, limit=2):
    """Find hands like '16 vs 10' in a question. Returns [(cards, dealer_card), ...]."""
    q = question.lower()
    found = []
    for m in _PAIR_RE.finditer(q):
        r = _rank(m.group(1))
        found.append((m.start(), [_card(r), _card(r)], _card(_rank(m.group(2)))))
    for m in _TWO_RE.finditer(q):
        found.append((m.start(), [_card(_rank(m.group(1))), _card(_rank(m.group(2)))], _card(_rank(m.group(3)))))
    for m in _TOTAL_RE.finditer(q):
        if any(abs(m.start() - s) < 6 for s, _, _ in found):
            continue
        cards = _cards_for_total(int(m.group(2)), m.group(1) == "soft")
        if cards:
            found.append((m.start(), cards, _card(_rank(m.group(3)))))
    found.sort(key=lambda f: f[0])
    out, seen = [], set()
    for _, cards, up in found:
        key = (tuple(c["rank"] for c in cards), up["rank"])
        if key not in seen:
            seen.add(key)
            out.append((cards, up))
    return out[:limit]


def question_context(question, Q, hit_soft17=False):
    """Look up any hand the question mentions in the agent's own table, plus the textbook chart."""
    lines = []
    names = {"H": "hit", "S": "stand", "D": "double", "P": "split"}
    for cards, up in parse_hands(question):
        pair = rl.is_pair(cards)
        best, values = rl.best_action(Q, cards, up, can_double=True, can_split=pair)
        total, soft = rl.hand_info(cards)
        up_name = "Ace" if up["rank"] == "A" else up["rank"]
        kind = "pair" if pair else ("soft" if soft else "hard")
        desc = f"pair of {cards[0]['rank']}s" if pair else f"{kind} {total} ({' + '.join(c['rank'] for c in cards)})"
        line = (f"For a {desc} against a dealer {up_name}: the trained agent picks {best.upper()}. Expected value "
                "per $1 bet: " + ", ".join(f"{a} {v:+.3f}" for a, v in values.items()) + ".")
        try:
            letter = rl.chart_letter(kind, cards[0]["value"] if pair else total, up["value"], hit_soft17)
            line += f" The published basic strategy chart says {names[letter].upper()}."
        except (KeyError, ValueError):
            pass                                   # totals the chart doesn't list (soft 21)
        lines.append(line)
    return "\n".join(lines)


SUGGESTED_QUESTIONS = [
    "Why is hitting 16 vs 10 better than standing?",
    "Explain Monte Carlo control like I'm new to RL.",
    "Why did the neural network lose to a lookup table?",
    "What is a true count and why does it matter?",
    "Why does the agent still lose money with perfect play?",
]
