"""
The Professor learns from feedback (reinforcement learning on the chat itself)
------------------------------------------------------------------------------
The language model's weights can't be retrained from a Streamlit app, but *how it is asked* can be.
Each answer is written in one of a few teaching styles, and students rate answers with 👍 / 👎.
Picking the style is a **multi-armed bandit** — the same idea the counting agent uses to size bets:

* each style is an arm; a 👍 is reward 1, a 👎 is reward 0
* the chance each style gets a 👍 is modelled as Beta(1 + likes, 1 + dislikes)
* **Thompson sampling**: draw one sample per style and use the highest. Styles with little data still
  get tried (exploration); styles students like get used more and more (exploitation).

Answers that got a 👍 are also kept and shown to the model as examples of what "helpful" looks like,
so good answers shape later ones (a lightweight version of learning from human feedback).

Ratings are saved to professor_feedback.json next to the app. On Streamlit Community Cloud that file
resets when the app restarts; point FEEDBACK_PATH at persistent storage to keep it forever.
"""

import json
import os
import random
import threading
import time

FEEDBACK_PATH = os.environ.get(
    "PROFESSOR_FEEDBACK_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "professor_feedback.json"))
MAX_EXAMPLES = 30                       # liked answers kept on file
SHOWN_EXAMPLES = 2                      # liked answers included with each question

STYLES = {
    "direct": ("Straight to the point",
               "Give the direct answer in the first sentence, then one or two sentences of reasoning that quote "
               "the key number from CONTEXT. 2-4 sentences total, no lists."),
    "numbers": ("Walk through the numbers",
                "Compare the options using the exact expected values or percentages from CONTEXT, one short line "
                "per option, then finish with a one-sentence takeaway. Under 120 words."),
    "intuition": ("Intuition first",
                  "Start with a plain-language intuition or an everyday analogy a beginner would get, then back "
                  "it up with one real number from CONTEXT. 3-5 sentences, no jargon without explaining it."),
    "coach": ("Coach with a check-in",
              "Answer clearly in 2-4 sentences using the numbers from CONTEXT, then end with one short question "
              "that checks the student understood (for example: 'So what would you do with 15 vs 10?')."),
}

_lock = threading.Lock()


def _empty():
    return {"arms": {k: {"likes": 0, "dislikes": 0} for k in STYLES}, "examples": []}


def load():
    try:
        with open(FEEDBACK_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _empty()
    base = _empty()
    for k in STYLES:                        # tolerate files from older versions with other styles
        base["arms"][k].update(data.get("arms", {}).get(k, {}))
    base["examples"] = data.get("examples", [])[-MAX_EXAMPLES:]
    return base


def _save(data):
    tmp = FEEDBACK_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, FEEDBACK_PATH)
    except OSError:
        pass                                # read-only disk: learning just won't persist


def choose_style(rng=random):
    """Thompson sampling: one Beta draw per style, highest wins."""
    arms = load()["arms"]
    draws = {k: rng.betavariate(1 + a["likes"], 1 + a["dislikes"]) for k, a in arms.items()}
    return max(draws, key=draws.get)


def style_prompt(style):
    return STYLES.get(style, STYLES["direct"])[1]


def examples(n=SHOWN_EXAMPLES):
    """The most recent liked (question, answer) pairs, to show the model what worked."""
    return [(e["q"], e["a"]) for e in load()["examples"][-n:]]


def record(style, liked, question="", answer=""):
    """Reward the style that wrote an answer. liked: True for 👍, False for 👎."""
    if style not in STYLES:
        return
    with _lock:
        data = load()
        data["arms"][style]["likes" if liked else "dislikes"] += 1
        if liked and question and answer and len(answer) < 1500 and "⚠️" not in answer:
            data["examples"].append({"q": question[:300], "a": answer, "style": style, "t": int(time.time())})
            data["examples"] = data["examples"][-MAX_EXAMPLES:]
        _save(data)


def stats():
    """[(label, likes, dislikes, estimated 👍 rate)] for showing what has been learned."""
    rows = []
    for k, a in load()["arms"].items():
        mean = (1 + a["likes"]) / (2 + a["likes"] + a["dislikes"])
        rows.append((STYLES[k][0], a["likes"], a["dislikes"], mean))
    return sorted(rows, key=lambda r: -r[3])
