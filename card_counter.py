"""Card Counter — the agent that learns to bet with the count, and can beat the house."""

import pandas as pd
import streamlit as st

import blackjack_rl as rl
import counting as C
from shared import DEALERS, get_bundles, page_header, theme

page_header("🧮 Card Counter",
            "The one agent that can beat the casino: it learns what each count is worth, and bets accordingly.")

rule = st.segmented_control("Dealer", list(DEALERS), default="S17", key="cnt_rule",
                            format_func=lambda k: f"{DEALERS[k]['emoji']} {DEALERS[k]['name']} ({DEALERS[k]['rule']})") or "S17"
hs17 = DEALERS[rule]["hit_soft17"]
base_bundle = get_bundles()[("expert", rule)]
BASE_Q = rl.robust_q(base_bundle["Q"], base_bundle["N"])


@st.cache_resource(show_spinner=False)
def counter(rule):
    try:
        return C.load(C.counting_path(rule))
    except (OSError, ValueError, EOFError):
        return None


bundle = counter(rule)

st.markdown("""
<style>
.cardmap { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 14px; }
.cardmap div { border: 1px solid var(--border); border-radius: 10px; padding: 8px 14px; background: var(--surface); text-align: center; }
.cardmap b { display: block; font-size: 18px; color: var(--acc); }
.cardmap span { font-size: 12px; color: var(--muted); }
</style>
""", unsafe_allow_html=True)

if bundle is None:
    st.warning("No counting agent is trained for this dealer yet.")
    if st.button("🧠 Train the counting agent now (about 4 minutes)", width="stretch"):
        bar = st.progress(0.0, text="Playing shoes and learning the value of each count...")
        b = C.train_counting(25_000_000, hs17, BASE_Q,
                             progress=lambda p: bar.progress(p, text=f"Training... {p:.0%}"))
        bar.progress(1.0, text="Measuring which decisions the count should change...")
        b["deviations"] = C.find_deviations(BASE_Q, hs17, tcs=(-2, 0, 2, 4), trials=4000)
        C.save(b, C.counting_path(rule), 25_000_000)
        counter.clear()
        st.rerun()
    st.stop()

# ---------------- How counting works ----------------

st.markdown("### 1. Keeping the count")
st.markdown("Every card that comes out shifts the odds. The **Hi-Lo** system gives each card a value, and the "
            "running count is their sum. Divide by the decks still to be dealt and you get the **true count**.")
st.markdown('<div class="cardmap">'
            '<div><b>2 3 4 5 6</b><span>+1 · small cards gone is good for you</span></div>'
            '<div><b>7 8 9</b><span>0 · neutral</span></div>'
            '<div><b>10 J Q K A</b><span>−1 · big cards gone is bad for you</span></div>'
            '</div>', unsafe_allow_html=True)
st.markdown("A shoe rich in tens and aces means more blackjacks (which pay 3 to 2), more winning doubles, and a "
            "dealer who busts more often when forced to draw. That's the player's edge.")

# ---------------- What the count is worth ----------------

st.markdown("### 2. What each count is worth")
ev = C.ev_by_count(bundle)
df = pd.DataFrame({"True count": [f"{b:+d}" if b else "0" for b in ev],
                   "Profit per $1 bet": [ev[b][0] for b in ev],
                   "Rounds measured": [ev[b][1] for b in ev]}).set_index("True count")
c1, c2 = st.columns([1.4, 1], gap="large")
with c1:
    st.bar_chart(df["Profit per $1 bet"], color=theme()["accent"], height=260)
with c2:
    positive = [b for b in ev if ev[b][0] > 0]
    cross = min(positive) if positive else None
    st.metric("The count where the player takes over",
              f"true count {cross:+d}" if cross is not None else "never", delta_color="off")
    st.markdown(f"At a true count of **{max(ev):+d}** the agent measured **{ev[max(ev)][0] * 100:+.2f}%** per dollar, "
                f"while at **{min(ev):+d}** it measured **{ev[min(ev)][0] * 100:+.2f}%**. Nobody told it this; it "
                "played millions of shoes and recorded what happened at each count.")
st.dataframe(df.style.format({"Profit per $1 bet": "{:+.4f}", "Rounds measured": "{:,.0f}"}), width="stretch")

# ---------------- Bet ramp ----------------

st.markdown("### 3. The bet ramp it learned")
ramp = C.bet_ramp(bundle)
ramp_df = pd.DataFrame({"True count": [f"{b:+d}" if b else "0" for b in ramp],
                        "Bet (units)": [ramp[b] for b in ramp]}).set_index("True count")
c1, c2 = st.columns([1.4, 1], gap="large")
c1.bar_chart(ramp_df, color=theme()["accent2"], height=260)
with c2:
    st.markdown("Each bet size earns *(bet × the value of this count)*, so the agent bets the table minimum while "
                "the shoe is bad and jumps to its maximum once the count turns profitable. This **bet spread** is "
                "where counting actually makes its money, far more than playing decisions.")
    st.caption(f"Bet menu available to the agent: {', '.join(str(b) + '×' for b in C.BETS)}")

# ---------------- Deviations ----------------

st.markdown("### 4. Decisions the count changes")
devs = bundle.get("deviations", [])
if not devs:
    st.info("No playing deviations were measured for this dealer yet.")
else:
    names = {"hard": "Hard", "soft": "Soft", "pair": "Pair of"}
    rows = [{"Hand": f"{names[d['kind']]} {d['total']}", "Dealer shows": "A" if d["up"] == 11 else str(d["up"]),
             "True count": f"{d['tc']:+d}", "Normally": d["base"], "With the count": d["counting"],
             "Gain per $1": d["gain"]} for d in devs]
    st.dataframe(pd.DataFrame(rows).style.format({"Gain per $1": "{:+.3f}"}), hide_index=True, width="stretch")
    st.caption("Measured directly: a shoe is built at that true count, the hand is dealt, each move is forced, and "
               "the results are compared over thousands of rounds. Only differences bigger than the measurement "
               "error are listed.")

# ---------------- Simulation ----------------

st.markdown("### 5. Put it to the test")
st.markdown("Play the same game two ways: **counting** (bet ramp plus deviations) versus **flat betting** with the "
            "expert agent that ignores the count.")
bench = bundle.get("benchmark")
if bench:
    b1, b2 = st.columns(2)
    b1.metric("🧮 Counting agent", f"{bench['counting']:+.2f} per $100 wagered",
              f"measured over {bench['rounds']:,} rounds", delta_color="normal")
    b2.metric("🎩 Flat betting expert", f"{bench['flat']:+.2f} per $100 wagered",
              f"measured over {bench['rounds']:,} rounds", delta_color="inverse")
    st.caption("Those are the reference runs. Try your own below, but note that short runs swing a lot.")
rounds = st.select_slider("Rounds to simulate", [200_000, 500_000, 2_000_000], value=500_000)
if st.button("🎲 Run the simulation", width="stretch", type="primary"):
    with st.spinner("Dealing shoes..."):
        c = C.simulate(bundle, BASE_Q, rounds, hs17, counting=True, track=150)
        f = C.simulate(bundle, BASE_Q, rounds, hs17, counting=False, track=150)
    m1, m2 = st.columns(2)
    m1.metric("🧮 Counting agent", f"{c['per_100_wagered']:+.2f} per $100 wagered",
              f"{c['profit']:+,.0f} units over {rounds:,} rounds", delta_color="normal" if c["profit"] > 0 else "inverse")
    m2.metric("🎩 Flat betting expert", f"{f['per_100_wagered']:+.2f} per $100 wagered",
              f"{f['profit']:+,.0f} units over {rounds:,} rounds", delta_color="normal" if f["profit"] > 0 else "inverse")
    curve = pd.DataFrame({"Counting": c["curve"], "Flat betting": f["curve"]})
    st.markdown("**Bankroll over time (in betting units)**")
    st.line_chart(curve, color=[theme()["accent"], "#6c7a70"], height=280)
    margin = 100 * 2 * 1.15 * (rounds ** 0.5) / (c["wagered"] or 1)
    if c["profit"] > 0:
        st.success(f"The counting agent finished **{c['profit']:+,.0f} units ahead**. It beat the house edge, "
                   "which no basic-strategy player can do.")
    else:
        st.info("This run came out behind. Counting earns only a fraction of a percent per hand, so short runs "
                "swing wildly, exactly as they would for a real counter. Try more rounds.")
    st.caption(f"Margin of error on this run: about ±{margin:.2f} per $100 wagered.")

with st.expander("⚠️ Reality check, before anyone tries this in a casino"):
    st.markdown(
        "- **The edge is tiny.** Around half a percent. It takes tens of thousands of hands for skill to beat luck, "
        "and losing streaks of hundreds of units are normal.\n"
        "- **Counting is not illegal, but casinos hate it.** They can ask you to leave, ban you, shuffle early, or "
        "cut the deck so shallow that counting stops working.\n"
        "- **A 1-to-12 bet spread is obvious.** Real counters use smaller spreads and disguise them, which lowers "
        "the edge further.\n"
        "- **This agent assumes ideal conditions:** 6 decks, 75% of the shoe dealt, no reshuffling when you win, "
        "and no betting limits.\n"
        "- Treat it as a demonstration of reinforcement learning, not a business plan."
    )

st.caption(f"Trained on {bundle.get('rounds', 0):,} shoe rounds against {DEALERS[rule]['name']}.")
