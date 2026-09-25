"""Research Lab — exact, reproducible evaluation of the reinforcement learning agents."""

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

import algorithms as alg
import blackjack_rl as rl
import experiments as ex
import solver
from shared import BASE_DIR, DEALERS, LEVELS, exact_baselines, exact_grade, get_agents, get_bundles, page_header

page_header("🔬 Research Lab",
            "Ground truth for the reinforcement learning: every number on this page is computed exactly, "
            "not simulated.")

# Reference palette (dark mode), fixed per entity so a filter never repaints a series.
SERIES = {"Monte Carlo": "#3987e5", "MC + precision practice": "#d95926", "Q-learning": "#199e70",
          "SARSA": "#c98500", "Deep Q-Network": "#d55181"}
COST_BINS = ["Perfect", "< 0.1¢", "0.1–1¢", "1–5¢", "> 5¢"]
COST_COLORS = ["#2b2b29", "#184f95", "#256abf", "#3987e5", "#86b6ef"]
LETTER_COLORS = {"H": "background-color:#e8a0a0;color:#111", "S": "background-color:#9fd8a8;color:#111",
                 "D": "background-color:#9fb8ea;color:#111", "P": "background-color:#f5dc8a;color:#111",
                 "R": "background-color:#cfcfcf;color:#111"}
UP_LABEL = {u: ("A" if u == 11 else str(u)) for u in solver.UPCARDS}


def chart_theme(c):
    return (c.configure(background="transparent")
            .configure_axis(labelColor="#c3c2b7", titleColor="#c3c2b7", gridColor="#ffffff14", domainColor="#ffffff33")
            .configure_legend(labelColor="#c3c2b7", titleColor="#c3c2b7")
            .configure_view(stroke=None))


def cost_bin(cost):
    c = cost * 100                                   # cents per $1
    if c <= 1e-10:
        return COST_BINS[0]
    return COST_BINS[1] if c < 0.1 else COST_BINS[2] if c < 1 else COST_BINS[3] if c < 5 else COST_BINS[4]


def hand_label(kind, total):
    if kind == "pair":
        return "A,A" if total == 11 else ("10,10" if total == 10 else f"{total},{total}")
    return f"A+{total - 11}" if kind == "soft" else f"Hard {total}"


rule = st.segmented_control("Dealer rules", list(DEALERS), default="S17", key="res_rule",
                            format_func=lambda k: f"{DEALERS[k]['emoji']} {DEALERS[k]['rule']}") or "S17"
RULES = solver.AGENT_RULES[rule]
bundle = get_bundles()[("expert", rule)]
grade = exact_grade("expert", rule)
rookie = exact_grade("rookie", rule)

tabs = st.tabs(["🎯 Exact grade", "🗺️ Where it errs", "🏁 Algorithm race", "🧮 Rules & house edge",
                "📦 Data & reproducibility"])

# =====================================================================
# Exact grade
# =====================================================================
with tabs[0]:
    wrong = [d for d in grade["decisions"] if d["cost"] > 1e-12]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Perfect play", f"{grade['optimal_ev'] * 100:+.3f}", "per $100 bet (the true house edge)",
              delta_color="off")
    c2.metric("Expert agent", f"{grade['policy_ev'] * 100:+.3f}", "per $100 bet", delta_color="off")
    c3.metric("Gap to perfect", f"{grade['regret'] * 10000:.2f}¢", "per $100 bet", delta_color="off")
    c4.metric("Perfect first decisions", f"{grade['optimal_share']:.1%}",
              f"{len(grade['decisions']) - len(wrong)} of {len(grade['decisions'])}", delta_color="off")

    ref = bundle.get("refined")
    story = (f"The expert learned from **{bundle['hands']:,} hands** of Monte Carlo self-play")
    if ref:
        story += (f", then **{ref['deals']:,} deals of precision practice** "
                  f"({ref['settled']} of {ref['total']} decisions statistically settled)")
    st.markdown(story + ". It never saw the exact solver: the solver is only the examiner.")

    base = exact_baselines(rule)
    rows = [("📏 Simple rule (stand on 17)", base["Simple rule (stand on 17)"]),
            (f"🧢 Rookie agent ({LEVELS['rookie']:,} hands)", rookie["policy_ev"] * 100),
            ("📘 Published 6-deck chart", base["Basic strategy chart (6-deck)"]),
            ("🎩 Expert agent", grade["policy_ev"] * 100),
            ("✨ Perfect play (exact solver)", base["Perfect play"])]
    df = pd.DataFrame([{"Strategy": n, "Per $100 bet": v, "Gap to perfect (¢ per $100)": (base["Perfect play"] - v) * 100}
                       for n, v in rows])
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Per $100 bet": st.column_config.NumberColumn(format="%+.3f"),
                                "Gap to perfect (¢ per $100)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Exact expected values on an infinite deck. A simulation would need hundreds of millions of hands "
               "to measure gaps this small; the solver gets them from one pass over every possible card.")

    with st.expander("📐 Method: how the grading works"):
        st.markdown(f"""
**Exact solver (`solver.py`).** The value of every move is computed by working through every card that can
come next (dynamic programming over an infinite deck), with the same rules the agents play:
{RULES.label()}, dealer peeks for blackjack, one split, split aces get one card.
Values are conditional on no natural blackjack, exactly like the agents' Q-values.

**Exact value of the agent (`solver.policy_ev`).** The agent's strategy is played through the same recursion,
so its long-run result is exact too. *Regret* = perfect EV − agent EV.

**Cost of a decision.** For each of the 330 two-card first decisions: value of the perfect move minus value
of the agent's move, both followed by perfect play. *Frequency* is how often the decision occurs per round.

**Precision practice (`blackjack_rl.refine`).** Random self-play gives rare hands (a pair of 4s against a 3)
about 50× less practice than common ones, and that is where a Monte Carlo agent's mistakes live. Precision
practice is still learning from experience only:
* *exploring starts* — practice hands are dealt straight into decisions that are not settled yet;
* *paired comparisons* — every legal move is played against the **same** upcoming cards, so luck cancels
  out of the comparison (common random numbers);
* *a stopping rule* — a decision is settled once the best move beats every other by more than
  z = {ref['z'] if ref else 3} standard errors, or the others are shown to be within
  {(ref['tol'] if ref else 0.001) * 100:.1f}¢ per $1 of it (a genuine tie).

This is best-arm identification (a bandit problem) inside Monte Carlo control.

**Validation.** `tests/` checks the solver against the game engine by simulation (within 4 standard errors),
checks the Gymnasium environment against the solver, and checks well-known decisions.
""")

# =====================================================================
# Where it errs
# =====================================================================
with tabs[1]:
    st.markdown("Every first decision, coloured by what the agent's choice costs compared with perfect play. "
                "Letters are the agent's move. Hover a cell for the exact values.")
    view = st.radio("Show", ["Expert agent", "Rookie agent"], horizontal=True, key="res_view")
    g = grade if view == "Expert agent" else rookie
    rows = []
    for d in g["decisions"]:
        vals = d["values"]
        rows.append({"kind": d["kind"], "Hand": hand_label(d["kind"], d["total"]), "order": -d["total"],
                     "Dealer": UP_LABEL[d["up"]], "Move": d["agent"][0].upper() if d["agent"] != "split" else "P",
                     "Agent": d["agent"], "Perfect": d["optimal"], "Cost (¢ per $1)": round(d["cost"] * 100, 4),
                     "Band": cost_bin(d["cost"]), "Frequency (% of rounds)": round(d["weight"] * 100, 4),
                     "EVs": ", ".join(f"{k} {v:+.4f}" for k, v in vals.items())})
    hdf = pd.DataFrame(rows)
    dealer_order = [UP_LABEL[u] for u in solver.UPCARDS]

    def heat(kind, height):
        data = hdf[hdf.kind == kind]
        order = list(data.sort_values("order")["Hand"].drop_duplicates())
        base_ = alt.Chart(data).encode(
            x=alt.X("Dealer:O", sort=dealer_order, title="Dealer up card", axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("Hand:O", sort=order, title=None))
        rect = base_.mark_rect(cornerRadius=4, stroke="#1a1a19", strokeWidth=2).encode(
            color=alt.Color("Band:O", scale=alt.Scale(domain=COST_BINS, range=COST_COLORS),
                            legend=alt.Legend(title="Cost vs perfect", orient="bottom")),
            tooltip=["Hand", "Dealer", "Agent", "Perfect", "Cost (¢ per $1)", "Frequency (% of rounds)", "EVs"])
        text = base_.mark_text(fontSize=12, fontWeight="bold").encode(
            text="Move:N",
            color=alt.condition(alt.datum.Band == COST_BINS[4], alt.value("#0b0b0b"), alt.value("#f3efe2")))
        return chart_theme((rect + text).properties(height=height))

    k1, k2, k3 = st.tabs(["Hard totals", "Soft totals", "Pairs"])
    with k1:
        st.altair_chart(heat("hard", 460), width="stretch")
    with k2:
        st.altair_chart(heat("soft", 280), width="stretch")
    with k3:
        st.altair_chart(heat("pair", 330), width="stretch")

    bad = hdf[hdf["Cost (¢ per $1)"] > 0].copy()
    if bad.empty:
        st.success("Every first decision is a perfect move.")
    else:
        bad["Cost per $100 of total play (¢)"] = bad["Cost (¢ per $1)"] * bad["Frequency (% of rounds)"]
        st.markdown(f"**{len(bad)} imperfect decisions**, most expensive first:")
        st.dataframe(bad.sort_values("Cost per $100 of total play (¢)", ascending=False)[
            ["Hand", "Dealer", "Agent", "Perfect", "Cost (¢ per $1)", "Frequency (% of rounds)",
             "Cost per $100 of total play (¢)"]], hide_index=True, width="stretch")
        st.caption("A cost below 0.1¢ per $1 is a statistical tie: the moves are worth almost exactly the same, "
                   "and telling them apart would take hundreds of millions of hands.")

    st.markdown("#### 🔍 Agent's estimate vs the exact value")
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Hand", ["Hard", "Soft", "Pair"], key="res_kind")
    if kind == "Pair":
        pv = c2.selectbox("Pair of", ["A", "10", "9", "8", "7", "6", "5", "4", "3", "2"], index=6, key="res_pair")
        v = 11 if pv == "A" else int(pv)
        state = (12 if v == 11 else 2 * v, 0, v == 11, True, v)
    else:
        lo_, hi_ = (13, 20) if kind == "Soft" else (5, 19)
        tot = c2.number_input("Total", lo_, hi_, 16 if kind == "Hard" else 18, key="res_total")
        state = (int(tot), 0, kind == "Soft", True, 0)
    up = c3.selectbox("Dealer shows", dealer_order, index=8, key="res_up")
    state = (state[0], 11 if up == "A" else int(up), state[2], state[3], state[4])
    Q = get_agents()[("expert", rule)]
    N = bundle["N"].get(state, [0] * 4)
    q = Q.get(state)
    exact = solver.action_values(state, RULES)
    pts = []
    for a, val in exact.items():
        name = solver.NAMES[a]
        est = q[a] if q is not None and q[a] > rl.UNTRIED else None
        se = rl._SD[a] / max(N[a], 1) ** 0.5 if est is not None else None
        pts.append({"Move": name, "Exact": val, "Agent": est, "lo": est - 1.96 * se if est is not None else None,
                    "hi": est + 1.96 * se if est is not None else None, "Practised": N[a]})
    pdf = pd.DataFrame(pts).dropna()
    if pdf.empty:
        st.info("The agent has no learned values for this situation.")
    else:
        pdf["Error"] = pdf["Agent"] - pdf["Exact"]
        pdf["e_lo"] = pdf["lo"] - pdf["Exact"]
        pdf["e_hi"] = pdf["hi"] - pdf["Exact"]
        span = max(0.01, float(pdf[["e_lo", "e_hi"]].abs().max().max()) * 1.15)
        x = alt.X("Error:Q", title="Agent's estimate minus the exact value (per $1 bet)",
                  scale=alt.Scale(domain=[-span, span]))
        zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#d95926", strokeWidth=2).encode(x="z:Q")
        bars = alt.Chart(pdf).mark_rule(strokeWidth=3, color="#3987e5").encode(
            y=alt.Y("Move:N", title=None), x=alt.X("e_lo:Q", scale=alt.Scale(domain=[-span, span])), x2="e_hi:Q")
        dots = alt.Chart(pdf).mark_circle(size=110, color="#3987e5", stroke="#1a1a19", strokeWidth=2, opacity=1).encode(
            y="Move:N", x=x, tooltip=["Move", alt.Tooltip("Agent:Q", format="+.4f", title="Agent's estimate"),
                                      alt.Tooltip("Exact:Q", format="+.4f", title="Exact value"),
                                      alt.Tooltip("Error:Q", format="+.4f"), "Practised"])
        st.altair_chart(chart_theme((zero + bars + dots).properties(height=70 + 44 * len(pdf))), width="stretch")
        best_exact = pdf.loc[pdf["Exact"].idxmax(), "Move"]
        best_agent = pdf.loc[pdf["Agent"].idxmax(), "Move"]
        st.caption(f"🔵 how far the agent's learned value is from the truth, with a 95% confidence interval · 🟧 the exact "
                   f"value. Intervals that cross the orange line mean the agent's estimate is statistically spot on. "
                   f"Perfect play: **{best_exact}** · agent plays: **{best_agent}**.")

# =====================================================================
# Algorithm race
# =====================================================================
with tabs[2]:
    try:
        race = alg.load_race()
        meta = alg.load_race_meta()
    except (OSError, ValueError):
        race, meta = None, {}
    if not race:
        st.info("No race results yet. Run `python experiments.py race`.")
    else:
        st.markdown("The same game learned five ways, each scored **exactly** at every checkpoint. Lines are the "
                    "mean over random seeds, and shaded bands show ±1 standard deviation between seeds.")
        if meta:
            st.caption(f"Seeds {meta.get('seeds')} · {meta.get('budget', 0):,} hands per learner "
                       f"(DQN {meta.get('dqn_budget', 0):,}) · {meta.get('rules', '')} · "
                       f"Python {meta.get('python', '?')} · commit {meta.get('git_commit') or 'n/a'} · "
                       f"{meta.get('date', '')}")
        rows = []
        for name, points in race.items():
            for p in points:
                rows.append({"Algorithm": name, "Hands": p["hands"],
                             "Regret": max(p.get("regret_100", float("nan")) * 100, 0.01),
                             "Regret sd": p.get("regret_100_sd", 0.0) * 100,
                             "Optimal": p.get("optimal", p.get("match", 0)) * 100,
                             "Optimal sd": p.get("optimal_sd", 0.0) * 100})
        rdf = pd.DataFrame(rows)
        rdf["r_lo"] = (rdf["Regret"] - rdf["Regret sd"]).clip(lower=0.01)
        rdf["r_hi"] = rdf["Regret"] + rdf["Regret sd"]
        rdf["o_lo"] = rdf["Optimal"] - rdf["Optimal sd"]
        rdf["o_hi"] = (rdf["Optimal"] + rdf["Optimal sd"]).clip(upper=100)
        names = [n for n in SERIES if n in set(rdf.Algorithm)]
        color = alt.Color("Algorithm:N", scale=alt.Scale(domain=names, range=[SERIES[n] for n in names]),
                          legend=alt.Legend(orient="bottom", title=None, columns=3, labelLimit=260))

        def race_chart(y, lo, hi, title, log=False, fmt=".2f"):
            scale = alt.Scale(type="log") if log else alt.Scale(zero=False)
            base_ = alt.Chart(rdf).encode(x=alt.X("Hands:Q", title="Hands of experience (log scale)",
                                                  scale=alt.Scale(type="log"), axis=alt.Axis(format="~s")),
                                          color=color)
            band = base_.mark_area(opacity=0.15).encode(y=alt.Y(f"{lo}:Q", scale=scale, title=title), y2=f"{hi}:Q")
            line = base_.mark_line(strokeWidth=2).encode(y=f"{y}:Q")
            dots = base_.mark_circle(size=70, stroke="#1a1a19", strokeWidth=2, opacity=1).encode(
                y=f"{y}:Q", tooltip=["Algorithm", alt.Tooltip("Hands:Q", format=","),
                                     alt.Tooltip(f"{y}:Q", format=fmt)])
            return chart_theme((band + line + dots).properties(height=320))

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Gap to perfect play** (¢ per $100 bet, log scale — lower is better)")
            st.altair_chart(race_chart("Regret", "r_lo", "r_hi", "¢ per $100", log=True, fmt=".3f"),
                            width="stretch")
        with c2:
            st.markdown("**Perfect first decisions** (% of 330 — higher is better)")
            st.altair_chart(race_chart("Optimal", "o_lo", "o_hi", "%", fmt=".1f"), width="stretch")
        final = rdf.sort_values("Hands").groupby("Algorithm").last()
        table = pd.DataFrame({
            "Hands": [f"{h:,.0f}" for h in final["Hands"]],
            "Gap to perfect (¢ per $100)": [f"{r:.2f} ± {s:.2f}" for r, s in zip(final["Regret"], final["Regret sd"])],
            "Perfect decisions": [f"{o:.1f}% ± {s:.1f}" for o, s in zip(final["Optimal"], final["Optimal sd"])],
        }, index=final.index).loc[[n for n in names if n in final.index]]
        st.dataframe(table, width="stretch")
        st.markdown(
            "**Reading it.** Monte Carlo beats the temporal-difference methods because a hand lasts a couple of "
            "moves and the reward comes at the end, so learning from the real result is unbiased; Q-learning and "
            "SARSA learn from their own estimates, which adds bias for no benefit here. The neural network has to "
            "approximate a table that only needs a few hundred entries. **Precision practice** spends the same "
            "number of hands where the agent is still unsure and compares moves on identical cards. Once the base "
            "strategy is decent it closes the gap to perfect play much faster; at very small budgets it doesn't "
            "help yet, because it needs a reasonable strategy to play out the rest of each practice hand.")

# =====================================================================
# Rules & house edge
# =====================================================================
with tabs[3]:
    st.markdown("What is a table actually worth? Pick the rules printed on the felt and the solver computes the "
                "exact house edge under perfect play, plus the perfect strategy for those rules.")
    c1, c2, c3, c4 = st.columns(4)
    h17 = c1.toggle("Dealer hits soft 17", value=rule == "H17", key="cal_h17")
    das = c2.toggle("Double after split", value=True, key="cal_das")
    sur = c3.toggle("Late surrender", value=False, key="cal_sur")
    pay = c4.radio("Blackjack pays", ["3:2", "6:5"], horizontal=True, key="cal_pay")
    custom = solver.Rules(hit_soft17=h17, double_after_split=das, surrender=sur,
                          blackjack_pays=1.5 if pay == "3:2" else 1.2)
    edge = solver.house_edge(custom)
    ref_edge = solver.house_edge(solver.Rules())
    m1, m2, m3 = st.columns(3)
    m1.metric("House edge", f"{-edge * 100:.3f}%", "casino's cut of every bet", delta_color="off")
    m2.metric("Player loses", f"${-edge * 100:.2f}", "per $100 bet, playing perfectly", delta_color="off")
    m3.metric("vs. a good table", f"{(edge - ref_edge) * 100:+.3f}", "per $100 (S17, DAS, 3:2)", delta_color="off")
    effects = []
    for label, change in [("Dealer hits soft 17", {"hit_soft17": True}),
                          ("No double after split", {"double_after_split": False}),
                          ("Late surrender allowed", {"surrender": True}),
                          ("Blackjack pays 6:5", {"blackjack_pays": 1.2})]:
        r2 = solver.Rules(**{**solver.Rules().__dict__, **change})
        effects.append({"Rule change (from S17, DAS, no surrender, 3:2)": label,
                        "Effect on the player (per $100)": (solver.house_edge(r2) - ref_edge) * 100})
    st.dataframe(pd.DataFrame(effects), hide_index=True, width="stretch",
                 column_config={"Effect on the player (per $100)": st.column_config.NumberColumn(format="%+.3f")})
    st.caption("Infinite-deck model; a 6-deck shoe is within about 0.1% of these values. 6:5 blackjack is the "
               "single most expensive rule in the casino.")
    st.markdown("**Perfect strategy for these rules** · H hit · S stand · D double · P split · R surrender")
    chart = solver.strategy_letters(custom)
    cols = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    s1, s2, s3 = st.columns([1.1, 1, 1])
    for col, key, title, h in ((s1, "hard", "Hard totals", 560), (s2, "soft", "Soft totals", 330),
                               (s3, "pair", "Pairs", 400)):
        with col:
            st.markdown(f"*{title}*")
            df = pd.DataFrame.from_dict(chart[key], orient="index", columns=cols)
            st.dataframe(df.style.map(lambda v: LETTER_COLORS.get(v, "")), width="stretch", height=h)

# =====================================================================
# Data & reproducibility
# =====================================================================
with tabs[4]:
    st.markdown("Everything needed to check, reuse or extend these results.")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("⬇️ Exact values (CSV)", pd.DataFrame(ex.exact_table_rows(RULES)).to_csv(index=False),
                       file_name=f"exact_values_{rule}.csv", mime="text/csv", width="stretch")
    d2.download_button("⬇️ Agent Q-table (CSV)", pd.DataFrame(ex.q_table_rows(rule)).to_csv(index=False),
                       file_name=f"agent_q_table_{rule}.csv", mime="text/csv", width="stretch")
    dec = pd.DataFrame([{k: v for k, v in d.items() if k not in ("state", "values")} | d["values"]
                        for d in grade["decisions"]])
    d3.download_button("⬇️ Agent decisions graded (CSV)", dec.to_csv(index=False),
                       file_name=f"expert_decisions_{rule}.csv", mime="text/csv", width="stretch")
    race_path = os.path.join(BASE_DIR, "algorithm_race.json")
    if os.path.exists(race_path):
        with open(race_path) as f:
            d4.download_button("⬇️ Algorithm race (JSON)", f.read(), file_name="algorithm_race.json",
                               mime="application/json", width="stretch")

    st.markdown("#### Reproduce every number")
    st.code("""pip install -r requirements.txt -r requirements-dev.txt
pytest                                   # engine, solver, environment and chat tests
python blackjack_rl.py train 30000000    # Monte Carlo self-play (seeded)
python blackjack_rl.py refine 10000000   # precision practice on the saved experts
python experiments.py race --seeds 0 1 2 # multi-seed algorithm comparison, scored exactly
python experiments.py report             # results/report.md with the exact grade
python experiments.py export             # CSV tables in results/""", language="bash")

    st.markdown("#### Use the environment with any RL library")
    st.code("""from blackjack_env import BlackjackEnv   # Gymnasium API, with double and split
import solver

env = BlackjackEnv(hit_soft17=False)
obs, info = env.reset(seed=0)            # obs = (total, dealer card, soft, can double, pair)
action = env.action_space.sample(mask=info["action_mask"])
obs, reward, terminated, truncated, info = env.step(action)

# Grade any strategy exactly:  policy(state, legal_actions) -> action
result = solver.regret(my_policy)
print(result["regret"] * 10000, "cents per $100 from perfect play")""", language="python")

    st.markdown("#### Assumptions and limits")
    st.markdown("""
* **Infinite deck** for the base agents and the solver: every card is equally likely on every draw. A 6-deck
  shoe changes a handful of close decisions and the house edge by less than 0.1%. The card-counting agent
  (Card Counter page) is the one that plays a real 6-deck shoe.
* **Rules modelled:** dealer peeks, double on any two cards and after a split, one split per round, split aces
  get one card. Not modelled: re-splitting, insurance, early surrender.
* **Precision practice** settles decisions to z = 3 standard errors; near-ties within 0.1¢ per $1 are treated as
  ties by design.
""")
    cff = os.path.join(BASE_DIR, "CITATION.cff")
    if os.path.exists(cff):
        with st.expander("📚 How to cite"):
            with open(cff) as f:
                st.code(f.read(), language="yaml")
