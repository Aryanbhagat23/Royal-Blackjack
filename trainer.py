"""Strategy Trainer — drills scored against the exact solver, with spaced repetition of mistakes."""

import random

import pandas as pd
import streamlit as st

import blackjack_rl as rl
import solver
from shared import DEALERS, dealer_bust_chance, explain, get_agents, make_card, page_header, value_bars

page_header("🎯 Strategy Trainer",
            "Learn perfect play one decision at a time. Every answer is checked against the exact solver, "
            "and the hands you miss come back until you get them right.")

SUITS = [("♠", False), ("♥", True), ("♦", True), ("♣", False)]
TIE = 0.001                     # moves within 0.1¢ per $1 of perfect count as correct (a genuine tie)
REVIEW_AFTER = 3                # a missed hand returns after this many other hands
ICONS = {"hit": "🃏 Hit", "stand": "✋ Stand", "double": "⏬ Double", "split": "✂️ Split"}
ss = st.session_state
ss.setdefault("tr_history", [])
ss.setdefault("tr_review", [])  # [(due_at_hand_number, state)]
ss.setdefault("tr_current", None)
ss.setdefault("tr_answer", None)

st.markdown("""
<style>
.tr-felt { background: radial-gradient(ellipse at center, #1f6f45, #0d3b24); border: 8px solid #5a3b1c; border-radius: 22px;
  padding: 18px 10px; display: flex; gap: 48px; justify-content: center; flex-wrap: wrap; margin: 6px 0 14px; }
.tr-spot { text-align: center; } .tr-spot b { display: block; color: #f3efe2 !important; letter-spacing: 2px; margin-bottom: 8px; }
.tr-row { display: flex; gap: 8px; justify-content: center; }
.tr-card { width: 70px; height: 100px; background: #fbfaf5; border-radius: 9px; box-shadow: 2px 4px 10px rgba(0,0,0,.5);
  display: flex; flex-direction: column; align-items: center; justify-content: center; font: 800 26px Georgia; color: #111 !important; }
.tr-card span { font-size: 22px; color: inherit !important; } .tr-card.red { color: #c0262d !important; }
.tr-verdict { border-radius: 16px; padding: 14px 18px; margin: 4px 0 10px; font-size: 16px; }
.tr-good { background: #133d26; border: 1px solid #1baf7a; } .tr-bad { background: #45191a; border: 1px solid #e34948; }
</style>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])
rule = c1.segmented_control("Dealer", list(DEALERS), default="S17", key="tr_rule",
                            format_func=lambda k: DEALERS[k]["rule"]) or "S17"
focus = c2.segmented_control("Practice", ["Real table", "Tough spots", "Soft hands", "Pairs", "Doubles"],
                             default="Real table", key="tr_focus") or "Real table"
RULES = solver.AGENT_RULES[rule]


@st.cache_data(show_spinner=False)
def decision_pool(rule_key):
    """[(state, kind, frequency)] for every first decision."""
    w = solver.decision_weights(solver.AGENT_RULES[rule_key])
    return [(state, kind, w.get(state, 0.0)) for kind, _, state in solver.two_card_states()]


def in_focus(state, kind):
    t, up, soft, _, pair = state
    if focus == "Soft hands":
        return kind == "soft"
    if focus == "Pairs":
        return kind == "pair"
    if focus == "Doubles":
        return kind != "pair" and ((not soft and 9 <= t <= 11) or (soft and 13 <= t <= 19))
    if focus == "Tough spots":
        return kind == "pair" or soft or 12 <= t <= 16
    return True


def cards_for(state, kind):
    t, up, soft, _, pair = state
    if kind == "pair":
        ranks = ["A", "A"] if pair == 11 else [random.choice(["10", "J", "Q", "K"])] * 2 if pair == 10 else [str(pair)] * 2
    elif kind == "soft":
        ranks = ["A", str(t - 11) if t - 11 < 10 else "10"]
    else:
        options = [(a, t - a) for a in range(2, 11) if 2 <= t - a <= 10 and a != t - a]
        a, b = random.choice(options)
        ranks = [random.choice(["10", "J", "Q", "K"]) if v == 10 else str(v) for v in (a, b)]
    random.shuffle(ranks)
    up_rank = "A" if up == 11 else random.choice(["10", "J", "Q", "K"]) if up == 10 else str(up)
    return ranks, up_rank


def deal():
    hand_no = len(ss.tr_history)
    due = [item for item in ss.tr_review if item[0] <= hand_no]
    if due:
        ss.tr_review.remove(due[0])
        state, kind = due[0][1], due[0][2]
        review = True
    else:
        pool = [(s, k, w) for s, k, w in decision_pool(rule) if in_focus(s, k)]
        state, kind, _ = random.choices(pool, weights=[max(w, 1e-4) for _, _, w in pool])[0]
        review = False
    ranks, up_rank = cards_for(state, kind)
    ss.tr_current = {"state": state, "kind": kind, "ranks": ranks, "up": up_rank, "review": review,
                     "suits": [random.choice(SUITS) for _ in range(3)], "rule": rule}
    ss.tr_answer = None


if ss.tr_current is None or ss.tr_current["rule"] != rule:
    deal()
cur = ss.tr_current
state, kind = cur["state"], cur["kind"]
legal = rl.legal_actions(True, bool(state[4]))
values = {solver.NAMES[a]: v for a, v in solver.action_values(state, RULES).items() if a in legal}
best = max(values, key=values.get)


def card_div(rank, suit):
    sym, red = suit
    return f'<div class="tr-card{" red" if red else ""}">{rank}<span>{sym}</span></div>'


total, soft = state[0], state[2]
label = (f"pair of {cur['ranks'][0]}s" if kind == "pair" else f"{'soft' if soft else 'hard'} {total}")
st.markdown(
    f'<div class="tr-felt"><div class="tr-spot"><b>DEALER</b><div class="tr-row">{card_div(cur["up"], cur["suits"][2])}'
    f'</div></div><div class="tr-spot"><b>YOU · {label.upper()}</b><div class="tr-row">'
    f'{card_div(cur["ranks"][0], cur["suits"][0])}{card_div(cur["ranks"][1], cur["suits"][1])}</div></div></div>',
    unsafe_allow_html=True)
if cur["review"]:
    st.caption("🔁 Review: you missed this one earlier.")


def answer(choice):
    cost = values[best] - values[choice]
    ok = cost <= TIE
    ss.tr_answer = {"choice": choice, "cost": cost, "ok": ok}
    ss.tr_history.append({"hand": label, "dealer": cur["up"], "kind": kind, "choice": choice, "best": best,
                          "cost": cost, "ok": ok, "review": cur["review"]})
    if not ok:
        ss.tr_review.append((len(ss.tr_history) + REVIEW_AFTER, state, kind))


if ss.tr_answer is None:
    cols = st.columns(len(legal))
    for col, a in zip(cols, legal):
        name = rl.ACTIONS[a]
        col.button(ICONS[name], key=f"tr_{name}", width="stretch", on_click=answer, args=(name,))
else:
    res = ss.tr_answer
    if res["ok"]:
        extra = "" if res["choice"] == best else f" (a near-tie with {best}: only {res['cost'] * 100:.2f}¢ per $1 apart)"
        st.markdown(f'<div class="tr-verdict tr-good">✅ <b>{res["choice"].capitalize()}</b> is right{extra}.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="tr-verdict tr-bad">❌ <b>{best.capitalize()}</b> is best. '
                    f'{res["choice"].capitalize()} costs <b>{res["cost"] * 100:.1f}¢</b> per $1 bet every time '
                    f'you play it here.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        cards = [make_card(r if r in rl.CARD_VALUES else "10") for r in cur["ranks"]]
        up_card = make_card(cur["up"] if cur["up"] in rl.CARD_VALUES else "10")
        bust = sum(rl.score(cards + [make_card(r)]) > 21 for r in rl.CARD_VALUES) / 13
        dbust = dealer_bust_chance(up_card["rank"], RULES.hit_soft17)
        st.markdown("**Why**")
        st.markdown(explain(best, cards, up_card, bust, dbust))
        agent_move, _ = rl.best_action(get_agents()[("expert", rule)], cards, up_card, True, kind == "pair")
        st.caption(f"The reinforcement learning agent plays **{agent_move}** here"
                   + (" too." if agent_move == best else " (a near-tie it hasn't fully separated)."))
    with c2:
        st.markdown("**Exact value of each move** (per $1 bet)")
        st.markdown(value_bars(values, best), unsafe_allow_html=True)
    st.button("Next hand ➡️", key="tr_next", width="stretch", on_click=deal)

# ---------------- Progress ----------------
hist = ss.tr_history
if hist:
    st.divider()
    n = len(hist)
    right = sum(h["ok"] for h in hist)
    lost = sum(h["cost"] for h in hist)
    streak = 0
    for h in reversed(hist):
        if not h["ok"]:
            break
        streak += 1
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hands", n)
    m2.metric("Accuracy", f"{right / n:.0%}")
    m3.metric("Cost of mistakes", f"{lost / n * 100:.2f}¢", "per $1 bet, on average", delta_color="off")
    m4.metric("Streak", f"🔥 {streak}" if streak >= 3 else streak)
    misses = [h for h in hist if not h["ok"]]
    if misses:
        st.markdown("**Your weak spots** (these come back for review)")
        df = pd.DataFrame(misses)
        df["spot"] = df["hand"] + " vs " + df["dealer"].astype(str)
        weak = (df.groupby("spot").agg(Missed=("ok", "size"), Played=("choice", lambda s: ", ".join(sorted(set(s)))),
                                        Best=("best", "first"), Cost=("cost", "sum"))
                .sort_values("Cost", ascending=False).head(8))
        weak["Cost"] = (weak["Cost"] * 100).map(lambda c: f"{c:.1f}¢")
        st.dataframe(weak, width="stretch")
    if st.button("↺ Reset my progress"):
        ss.tr_history, ss.tr_review = [], []
        deal()
        st.rerun()
