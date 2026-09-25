"""Ask the Professor — chat with an LLM (Ollama or Gemini) that is fed the agent's real numbers,
and that learns from 👍 / 👎 which teaching style works best (see professor_rl.py)."""

import streamlit as st

import algorithms as alg
import llm
import professor_rl as prl
from shared import exact_baselines, exact_grade, get_agents, page_header

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
model = st.selectbox("Model", models, index=models.index(llm.default_model(provider, models)))

# ---- facts handed to the model so it can't invent numbers ----
grade = exact_grade("expert", "S17")
try:
    race = alg.load_race()
except (OSError, ValueError):
    race = None
context = llm.project_context(grade=grade, baselines=exact_baselines("S17"), race=race)
expert_q = get_agents()[("expert", "S17")]

with st.expander("🔍 What the model is told before it answers"):
    st.caption("The language model is never asked to work Blackjack out for itself. These measured facts are "
               "attached to every question, so it explains the trained agent's results instead of inventing "
               "its own.")
    st.code(context, language=None)

ss = st.session_state
ss.setdefault("ask_msgs", [])
ss.setdefault("ask_error", None)


def ask(question):
    ss.ask_msgs.append({"role": "user", "content": question})
    ss.ask_error = None


def rate(i):
    """👍 / 👎 is the reward signal for the style bandit."""
    m = ss.ask_msgs[i]
    value = ss.get(f"fb_{i}")
    if value is None or m.get("rated"):
        return
    prl.record(m.get("style"), value == 1, m.get("q", ""), m["content"])
    m["rated"] = True


if not ss.ask_msgs:
    st.markdown("**Try one of these:**")
    cols = st.columns(len(llm.SUGGESTED_QUESTIONS))
    for col, q in zip(cols, llm.SUGGESTED_QUESTIONS):
        if col.button(q, key=f"sugg_{q[:14]}", width="stretch"):
            ask(q)
            st.rerun()

for i, m in enumerate(ss.ask_msgs):
    with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("style"):
            st.feedback("thumbs", key=f"fb_{i}", on_change=rate, args=(i,), disabled=m.get("rated", False))

if ss.ask_error:
    question, message = ss.ask_error
    st.error(f"**The Professor couldn't answer.** {message}")
    if st.button("🔁 Try again", key="retry"):
        ask(question)
        st.rerun()

prompt = st.chat_input("Ask about the game, the agent, or reinforcement learning…")
if prompt:
    ask(prompt)
    st.rerun()

if ss.ask_msgs and ss.ask_msgs[-1]["role"] == "user":
    question = ss.ask_msgs[-1]["content"]
    style = prl.choose_style()
    extra = llm.question_context(question, expert_q)
    full_context = context + ("\n\nHANDS IN THIS QUESTION (looked up in the agent's table):\n" + extra
                              if extra else "")
    failure = []

    def answer():
        try:
            yield from llm.stream(provider, model, ss.ask_msgs, full_context, style=prl.style_prompt(style),
                                  examples=prl.examples(), fallbacks=models, raise_errors=True)
        except llm.LLMError as e:
            failure.append(str(e))

    with st.chat_message("assistant", avatar="🎓"):
        reply = st.write_stream(answer())
    if failure or not reply:
        ss.ask_msgs.pop()                      # never keep a failed turn in the history sent to the model
        ss.ask_error = (question, failure[0] if failure else "The model sent back an empty answer.")
    else:
        ss.ask_msgs.append({"role": "assistant", "content": reply, "style": style, "q": question})
    st.rerun()

if ss.ask_msgs and st.button("🗑️ Clear conversation"):
    ss.ask_msgs = []
    ss.ask_error = None
    st.rerun()

with st.expander("📈 How the Professor is learning from your ratings"):
    st.caption("Every answer is written in one of a few teaching styles, picked by Thompson sampling, a "
               "multi-armed bandit. Each 👍 or 👎 is a reward: styles students like get picked more often, "
               "while less-tried ones still get the occasional chance. Liked answers are also shown to the "
               "model as examples of a helpful reply.")
    st.dataframe(
        [{"Style": label, "👍": up, "👎": down, "Estimated 👍 rate": f"{rate_:.0%}"}
         for label, up, down, rate_ in prl.stats()],
        hide_index=True, width="stretch")
