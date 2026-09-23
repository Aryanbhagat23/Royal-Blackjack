"""AI Lab — look inside the agents: strategies, learning curves, Q-values and tournaments."""

import pandas as pd
import streamlit as st

import algorithms as alg
import blackjack_rl as rl
from shared import DEALERS, LEVELS, get_agents, get_bundles, page_header, value_bars

AGENTS = get_agents()
BUNDLES = get_bundles()
LEVEL_NAMES = {"expert": "🎩 Expert (Veteran Vic)", "rookie": "🧢 Rookie (Riley)"}
COLORS = {"H": "background-color:#e8a0a0;color:#111", "S": "background-color:#9fd8a8;color:#111",
          "D": "background-color:#9fb8ea;color:#111", "P": "background-color:#f5dc8a;color:#111"}

page_header("🧪 AI Lab", "Look inside the agents: what they learned, how they learned it, and how they compare.")

rule = st.segmented_control("Dealer rules", list(DEALERS), default="S17", key="lab_rule",
                            format_func=lambda k: f"{DEALERS[k]['emoji']} {DEALERS[k]['name']} ({DEALERS[k]['rule']})") or "S17"
hs17 = DEALERS[rule]["hit_soft17"]

t0, t1, t2, t3, t4, t5 = st.tabs(["📐 Accuracy check", "📋 Strategy charts", "📈 Learning curve",
                                  "🔍 Q-value explorer", "⚔️ Tournament", "🔬 Algorithm race"])


def chart_df(level, kind):
    Q = AGENTS[(level, rule)]
    if kind == "pairs":
        rows, cols = rl.pair_table(Q)
        df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
        df.index.name = "Pair \\ Dealer"
        return df
    soft = kind == "soft"
    rows, cols = rl.strategy_table(Q, usable_ace=soft)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    df.index = [f"A+{t - 11}" if soft else str(t) for t in df.index]
    df.index.name = "You \\ Dealer"
    return df


# ---------------- Accuracy check ----------------
with t0:
    st.markdown("How close did each agent get to **basic strategy**, the mathematically correct way to play that "
                "professionals memorize? Every two-card decision is compared: 150 hard totals, 80 soft totals, and "
                "100 pairs.")
    grades = {lvl: rl.grade(BUNDLES[(lvl, rule)]["Q"], BUNDLES[(lvl, rule)]["N"], hs17) for lvl in LEVELS}
    cols = st.columns(2)
    for col, lvl in zip(cols, ("expert", "rookie")):
        g = grades[lvl]
        ties = sum(d["tie"] for d in g["disagreements"])
        real = len(g["disagreements"]) - ties
        col.metric(f"{LEVEL_NAMES[lvl]} · {LEVELS[lvl]:,} hands", f"{g['matches'] / g['total']:.1%} match",
                   f"{real} real mistakes" if real else "0 real mistakes", delta_color="inverse" if real else "off")
        col.caption(f"{g['matches']} of {g['total']} decisions match the chart. {ties} of the differences are "
                    "statistical ties (moves worth almost exactly the same).")
    exp = grades["expert"]
    if exp["disagreements"]:
        with st.expander(f"🔍 See the Expert's {len(exp['disagreements'])} differences from the chart"):
            names = {"hard": "Hard", "soft": "Soft", "pair": "Pair of"}
            rows = [{"Hand": f"{names[d['kind']]} {('A' if d['total'] == 11 else d['total']) if d['kind'] == 'pair' else d['total']}",
                     "Dealer": "A" if d["up"] == 11 else d["up"], "Agent plays": d["agent"], "Chart says": d["chart"],
                     "Value gap (per $1)": d["gap"], "Verdict": "🤝 statistical tie" if d["tie"] else "❌ mistake"}
                    for d in exp["disagreements"]]
            st.dataframe(pd.DataFrame(rows).style.format({"Value gap (per $1)": "{:+.3f}"}), hide_index=True, width="stretch")
            st.caption("The value gap is how much the agent thinks its choice beats the chart's. Tiny gaps are "
                       "within measurement noise. Some also come from the chart being built for 6 decks, while the "
                       "agent trains on an endless deck.")

    st.markdown("#### 💰 Does it actually play as well?")
    hands = st.select_slider("Hands to simulate", [100_000, 300_000, 1_000_000], value=300_000)
    if st.button("▶️ Compare Expert vs. basic strategy vs. random play", width="stretch"):
        with st.spinner("Simulating..."):
            e = rl.evaluate(AGENTS[("expert", rule)], hands, hs17)
            b = rl.evaluate(None, hands, hs17, policy=rl.basic_strategy_policy(hs17))
            r = rl.evaluate({}, min(hands, 100_000), hs17, random_unseen=True)
        margin = 2 * 1.15 / hands ** 0.5 * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("🎩 Expert agent", f"{e['avg_reward'] * 100:+.2f}", "per $100 bet", delta_color="off")
        c2.metric("📘 Basic strategy chart", f"{b['avg_reward'] * 100:+.2f}", "per $100 bet", delta_color="off")
        c3.metric("🎲 Random play", f"{r['avg_reward'] * 100:+.2f}", "per $100 bet", delta_color="off")
        st.caption(f"Margin of error about ±{margin:.2f} per $100. Even perfect play loses a little: that's the house edge.")


# ---------------- Strategy charts ----------------
with t1:
    kind = {"Hard totals": "hard", "Soft totals (Ace = 11)": "soft", "Pairs": "pairs"}[
        st.radio("Hand type", ["Hard totals", "Soft totals (Ace = 11)", "Pairs"], horizontal=True)]
    exp, rook = chart_df("expert", kind), chart_df("rookie", kind)
    diff = (exp != rook).to_numpy().sum()
    st.markdown(f"The Rookie disagrees with the Expert on **{diff} of {exp.size}** decisions shown. "
                "Those are the mistakes that come from too little experience.")
    c1, c2 = st.columns(2)
    for col, lvl, df in ((c1, "expert", exp), (c2, "rookie", rook)):
        with col:
            st.markdown(f"**{LEVEL_NAMES[lvl]}** · {LEVELS[lvl]:,} hands")
            st.dataframe(df.style.map(lambda v: COLORS.get(v, "")), width="stretch",
                         height={"hard": 600, "soft": 330, "pairs": 400}[kind])
    st.caption("H = Hit · S = Stand · D = Double · P = Split. Nobody programmed these charts; each agent discovered its own.")

# ---------------- Learning curve ----------------
with t2:
    st.markdown("Train a **brand-new agent from zero** and test it after every step, to watch it improve.")
    total = st.select_slider("Training hands", [50_000, 100_000, 200_000, 400_000], value=200_000)


    @st.cache_data(show_spinner=False)
    def curve(total, hs17):
        bar = st.progress(0.0, text="Training and testing...")
        pts = rl.learning_curve(total, 10, 20_000, hs17, progress=lambda p: bar.progress(p, text=f"Training... {p:.0%}"))
        bar.empty()
        return pts


    if st.button("▶️ Run the experiment", width="stretch"):
        ss_key = (total, hs17)
        st.session_state.curve = (ss_key, curve(total, hs17))
    if "curve" in st.session_state and st.session_state.curve[0] == (total, hs17):
        pts = st.session_state.curve[1]
        df = pd.DataFrame(pts, columns=["Hands trained", "Return per $100", "Win rate"])
        df["Return per $100"] *= 100
        df["Win rate"] *= 100
        df = df.set_index("Hands trained")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Money won or lost per $100 bet**")
            st.line_chart(df["Return per $100"], color="#d4af37", height=260)
        with c2:
            st.markdown("**Win rate (%)**")
            st.line_chart(df["Win rate"], color="#5fd07a", height=260)
        first, last = df.iloc[0], df.iloc[-1]
        st.success(f"After {df.index[-1]:,} hands, the agent went from **{first['Return per $100']:+.1f}** to "
                   f"**{last['Return per $100']:+.1f}** per $100, and its win rate from {first['Win rate']:.1f}% "
                   f"to {last['Win rate']:.1f}%.")
        st.caption("At zero hands the agent knows nothing and plays randomly. The line levels off near "
                   "−$2 per $100, the real house edge, which no strategy can beat without counting cards.")

# ---------------- Q-value explorer ----------------
with t3:
    st.markdown("Pick any situation and see exactly what each agent **believes** every move is worth.")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Hand", ["Hard total", "Soft total", "Pair"])
    if kind == "Pair":
        pv = c2.selectbox("Pair of", ["A", "10", "9", "8", "7", "6", "5", "4", "3", "2"], index=3)
        pair = 11 if pv == "A" else int(pv)
        ptotal, usable, can_double = (12 if pair == 11 else 2 * pair), pair == 11, True
        c4.caption("Pairs are always a first decision, so double and split are both allowed.")
    else:
        lo = 13 if kind == "Soft total" else 5
        ptotal = c2.number_input("Your total", lo, 21 if kind == "Soft total" else 20, 16 if kind != "Soft total" else 18)
        pair, usable = 0, kind == "Soft total"
        can_double = c4.toggle("First two cards", value=True)
    up = c3.selectbox("Dealer shows", ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"], index=8)
    up_val = 11 if up == "A" else int(up)
    state = (int(ptotal), up_val, usable, can_double, pair)
    legal = rl.legal_actions(can_double, pair > 0)
    cols = st.columns(2)
    for col, lvl in zip(cols, ("expert", "rookie")):
        with col:
            q = AGENTS[(lvl, rule)].get(state)
            st.markdown(f"**{LEVEL_NAMES[lvl]}**")
            if q is None:
                st.warning("This agent never saw this situation during training, so it has no opinion.")
                continue
            values = {rl.ACTIONS[i]: q[i] for i in legal}
            best = max(values, key=values.get)
            n = BUNDLES[(lvl, rule)]["N"].get(state)
            seen = f" · seen {sum(n):,} times" if n else ""
            st.markdown(f'<div class="panel"><b style="color:var(--acc)">Best: {best.upper()}</b>{seen}<br><br>'
                        f'{value_bars(values, best)}</div>', unsafe_allow_html=True)
    st.caption(f"State fed to the agent: `{state}` → (total, dealer card, soft, can double, pair). "
               "Values are average money won per $1 bet.")

# ---------------- Tournament ----------------
with t4:
    hands = st.select_slider("Hands per agent", [5_000, 20_000, 50_000], value=20_000)
    if st.button("⚔️ Start tournament", width="stretch"):
        rows = []
        bar = st.progress(0.0)
        jobs = [(l, r) for l in LEVELS for r in DEALERS]
        for i, (l, r) in enumerate(jobs):
            e = rl.evaluate(AGENTS[(l, r)], hands, hit_soft17=DEALERS[r]["hit_soft17"])
            rows.append({"Agent": LEVEL_NAMES[l], "Dealer": DEALERS[r]["name"], "Win %": e["win_rate"] * 100,
                         "Loss %": e["loss_rate"] * 100, "Push %": e["tie_rate"] * 100,
                         "Return per $100": e["avg_reward"] * 100})
            bar.progress((i + 1) / len(jobs))
        bar.empty()
        df = pd.DataFrame(rows).sort_values("Return per $100", ascending=False)
        st.dataframe(df.style.format({c: "{:.1f}" for c in ["Win %", "Loss %", "Push %"]} |
                                     {"Return per $100": "${:+.2f}"}), hide_index=True, width="stretch")
        champ = df.iloc[0]
        st.success(f"🏆 Champion: **{champ['Agent']}** against {champ['Dealer']} "
                   f"({champ['Return per $100']:+.2f} per $100).")


# ---------------- Algorithm race ----------------
with t5:
    st.markdown("Blackjack, learned four different ways. Same game, same scoring: how often each agent matches "
                "the official strategy chart, and how much money it makes per $100 bet.")
    cards = [
        ("🎲 Monte Carlo", "This project's method. Waits until the hand ends, then learns from the actual result. "
                           "Unbiased but has to finish a hand before learning anything."),
        ("⚡ Q-learning", "Learns after every single move, using its own best guess about the next situation "
                         "(*bootstrapping*). Off-policy: it learns the best strategy while exploring."),
        ("🐢 SARSA", "Same as Q-learning, but it learns toward the move it actually takes next, including "
                     "exploratory ones. On-policy, so it ends up a bit more cautious."),
        ("🧠 Deep Q-Network", "Q-learning with a small neural network instead of a lookup table, written from "
                              "scratch with numpy: 24 inputs, two hidden layers, replay buffer, target network."),
    ]
    for col, (title, desc) in zip(st.columns(4, gap="medium"), cards):
        col.markdown(f'<div class="feature" style="min-height:190px"><h3 style="font-size:18px">{title}</h3>'
                     f'<p>{desc}</p></div>', unsafe_allow_html=True)

    try:
        race = alg.load_race()
    except (OSError, ValueError):
        race = None
    if race is None:
        st.info("No saved race yet. Run `python algorithms.py` to generate one.")
    else:
        rows = []
        for name, points in race.items():
            for p in points:
                rows.append({"Algorithm": name, "Hands trained": p["hands"],
                             "Chart match %": p["match"] * 100, "Per $100": p["per_100"]})
        rdf = pd.DataFrame(rows)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Accuracy as training goes on**")
            st.line_chart(rdf.pivot(index="Hands trained", columns="Algorithm", values="Chart match %"), height=300)
        with c2:
            st.markdown("**Money won or lost per $100 bet**")
            st.line_chart(rdf.pivot(index="Hands trained", columns="Algorithm", values="Per $100"), height=300)
        final = (rdf.sort_values("Hands trained").groupby("Algorithm").last()
                 .sort_values("Chart match %", ascending=False))
        st.dataframe(final.style.format({"Chart match %": "{:.1f}%", "Per $100": "{:+.2f}",
                                         "Hands trained": "{:,.0f}"}), width="stretch")
        best = final.index[0]
        st.success(f"🏆 **{best}** ends up closest to perfect play on this game.")
        st.markdown(
            "**Why Monte Carlo wins here:** a Blackjack hand is over in a couple of moves and the reward comes "
            "at the very end, so waiting for the true result is both simple and unbiased. Q-learning and SARSA "
            "learn from their own estimates, which adds bias and noise for no benefit when hands are this short. "
            "The neural network has to *approximate* a table that only needs a few hundred entries, so it trades "
            "exactness for a generalisation it doesn't need — and it trains roughly 50× slower per hand. "
            "Neural networks earn their keep when the state space is far too big to tabulate, like Atari or Go."
        )
    st.caption("Note: when a learner splits a pair, the two hands are played out by the trained expert, so all "
               "four are compared on the same terms.")