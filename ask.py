"""Ask the Professor — chat with a local LLM (Ollama) that is fed the agent's real numbers."""

import streamlit as st

import algorithms as alg
import blackjack_rl as rl
import llm
from shared import DEALERS, baselines, get_bundles, page_header

page_header("💬 Ask the Professor",
            "Ask anything about Blackjack or the reinforcement learning behind it, in plain language.")


@st.cache_data(ttl=60, show_spinner=False)
def status():
    return llm.providers()


provs = status()

if not provs:
    st.info("**The chat needs a language model, and none is set up.** There are two ways to add one.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 💻 A model on your own computer")
        st.markdown("Free and private, but it only works on that machine, so not on the public link.")
        st.code("# install Ollama from https://ollama.com\nollama pull llama3.2\nollama serve", language="bash")
    with c2:
        st.markdown("#### ☁️ Google Gemini")
        st.markdown("Has a free tier and works on the deployed app too. Get a key at "
                    "aistudio.google.com, then add it to `.streamlit/secrets.toml`:")
        st.code('GEMINI_API_KEY = "your-key-here"', language="toml")
        st.caption("On Streamlit Cloud, paste the same line under Settings → Secrets. Never commit the key.")
    st.caption("Everything else works without this — the Professor's advice comes from the trained agent, "
               "not from a language model.")
    if st.button("🔄 Check again", width="stretch"):
        status.clear()
        st.rerun()
    st.stop()

if len(provs) > 1:
    labels = [p[1] for p in provs]
    chosen = provs[labels.index(st.radio("Powered by", labels, horizontal=True))]
else:
    chosen = provs[0]
    st.caption(f"Powered by {chosen[1]}")
provider, _, models = chosen
default = llm.pick_model(models) if provider == "ollama" else models[0]
model = st.selectbox("Model", models, index=models.index(default))

# ---- facts handed to the model so it can't invent numbers ----
bundle = get_bundles()[("expert", "S17")]
grade = rl.grade(bundle["Q"], bundle["N"], False)
try:
    race = alg.load_race()
except (OSError, ValueError):
    race = None
context = llm.project_context(grade=grade, baselines=baselines(False), race=race)

with st.expander("🔍 What the model is told before it answers"):
    st.caption("The language model is never asked to work Blackjack out for itself. These measured facts are "
               "attached to every question, so it explains the trained agent's results instead of inventing "
               "its own.")
    st.code(context, language=None)

st.session_state.setdefault("ask_msgs", [])

if not st.session_state.ask_msgs:
    st.markdown("**Try one of these:**")
    cols = st.columns(len(llm.SUGGESTED_QUESTIONS))
    for col, q in zip(cols, llm.SUGGESTED_QUESTIONS):
        if col.button(q, key=f"sugg_{q[:14]}", width="stretch"):
            st.session_state.ask_msgs.append({"role": "user", "content": q})
            st.rerun()

for m in st.session_state.ask_msgs:
    with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else None):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about the game, the agent, or reinforcement learning…")
if prompt:
    st.session_state.ask_msgs.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.ask_msgs and st.session_state.ask_msgs[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="🎓"):
        reply = st.write_stream(llm.stream(provider, model, st.session_state.ask_msgs, context))
    st.session_state.ask_msgs.append({"role": "assistant", "content": reply})
    st.rerun()

if st.session_state.ask_msgs and st.button("🗑️ Clear conversation"):
    st.session_state.ask_msgs = []
    st.rerun()
