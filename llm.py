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
decided, rather than inventing its own advice.

Setup (one time):
    1. Install Ollama from ollama.com
    2. ollama pull llama3.2
    3. Start the app — the chat appears automatically
"""

import json
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:11434"          # Ollama, on this machine
GEMINI_HOST = "https://generativelanguage.googleapis.com"
TIMEOUT = 1.5

# Ollama is on this machine, so never go through a system proxy: on a campus or office network
# urllib would otherwise send localhost requests to the proxy and hang instead of failing fast.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PREFERRED = ["llama3.2", "llama3.1", "llama3", "mistral", "phi3", "qwen2.5", "gemma2"]

SYSTEM_PROMPT = """You are "the Professor", a friendly Blackjack and reinforcement learning tutor built into a \
student project called Royal Blackjack.

The app contains a reinforcement learning agent that taught itself Blackjack by playing 30 million hands against \
itself using Monte Carlo control. It matches published basic strategy on 96.7% of decisions. A second agent counts \
cards and beats the house edge.

Rules for your answers:
- The numbers in CONTEXT come from the trained agent and from simulations. Treat them as the truth and build your \
answer around them. Never contradict them and never invent different numbers.
- If the context gives an expected value, quote it.
- Keep answers to 2-4 sentences unless asked for more. Plain language, no bullet lists unless asked.
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


def chat_stream(model, messages, context="", temperature=0.4):
    """Yield the reply piece by piece, so it types out live in the app."""
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": SYSTEM_PROMPT}]
                     + ([{"role": "system", "content": "CONTEXT (real numbers from this app):\n" + context}]
                        if context else [])
                     + messages),
        "stream": True,
        "options": {"temperature": temperature, "num_predict": 320},
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
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return
    except urllib.error.URLError:
        yield "\n\n(Lost contact with Ollama. Is it still running?)"
    except Exception as e:
        yield f"\n\n(The local model returned an error: {e})"


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
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def gemini_models(key):
    """Ask Google which models this key can actually use, so no model name is hard-coded
    (they are renamed and retired often). Returns the chat-capable ones, fastest first."""
    req = urllib.request.Request(f"{GEMINI_HOST}/v1beta/models", headers={"x-goog-api-key": key})
    try:
        with _opener.open(req, timeout=6) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return []
    names = []
    for m in data.get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", ["generateContent"]):
            continue
        name = m.get("name", "").split("/")[-1]
        if not name or "embedding" in name or "aqa" in name or "vision" in name:
            continue
        names.append(name)
    def rank(n):
        return (0 if "flash" in n else 1,            # flash models are the free, fast ones
                1 if "preview" in n or "exp" in n else 0,
                -len(n))
    return sorted(names, key=rank)


def gemini_stream(model, messages, context, key, temperature=0.4):
    """Stream a Gemini reply. Same output shape as chat_stream, so pages don't care which is used."""
    system = SYSTEM_PROMPT + (("\n\nCONTEXT (real numbers from this app):\n" + context) if context else "")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]} for m in messages],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 400},
    }
    url = f"{GEMINI_HOST}/v1beta/models/{model}:streamGenerateContent?alt=sse"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with _opener.open(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for cand in chunk.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text"):
                            yield part["text"]
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode()).get("error", {}).get("message", "")
        except Exception:
            pass
        yield f"\n\n(Gemini returned an error: {e.code}. {detail})"
    except Exception as e:
        yield f"\n\n(Could not reach Gemini: {e})"


def providers():
    """What can power the chat right now: [(id, label, [models]), ...]."""
    out = []
    ok, models = available()
    if ok and models:
        out.append(("ollama", "Ollama (on this computer)", models))
    key = api_key()
    if key:
        gm = gemini_models(key)
        if gm:
            out.append(("gemini", "Google Gemini (works online)", gm))
    return out


def stream(provider, model, messages, context=""):
    """One entry point the pages use, whichever provider is selected."""
    if provider == "gemini":
        return gemini_stream(model, messages, context, api_key())
    return chat_stream(model, messages, context)


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


SUGGESTED_QUESTIONS = [
    "Why is hitting better than standing here?",
    "Explain Monte Carlo control like I'm new to RL.",
    "Why did the neural network lose to a lookup table?",
    "What is a true count and why does it matter?",
    "Why does the agent still lose money with perfect play?",
]