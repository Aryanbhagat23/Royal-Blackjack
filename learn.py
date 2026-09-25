"""How the AI Works — the reinforcement learning explained for students and teachers."""

import streamlit as st

import algorithms as alg
from shared import LEVELS, exact_baselines, exact_grade, get_bundles, page_header

page_header("🎓 How the AI Works", "Reinforcement learning, explained with the Blackjack agent in this app.")

st.markdown("### The big idea")
st.markdown(
    "The agent is never told how to play. It plays a hand, sees whether it won or lost, and adjusts its "
    "opinion of every move it made. After hundreds of thousands of hands, those opinions become a strategy."
)

st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontcolor="#111", color="#d4af37", fillcolor="#f6dc8c"];
  edge [color="#d4af37", fontcolor="#f3efe2", fontname="Helvetica"];
  Agent [label="🤖 Agent\\n(Q-table)"];
  Env [label="🃏 Environment\\n(Blackjack game)", fillcolor="#9fd8a8"];
  Agent -> Env [label="  action: hit / stand / double  "];
  Env -> Agent [label="  new state + reward (+1 / −1 / 0)  "];
}
""")

st.markdown("### 1. The problem as a Markov Decision Process")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="feature"><div class="ico">👁️</div><h3>State</h3><p>What the agent sees: '
                '<b>(your total, dealer\'s card, soft hand?, can double?, pair?)</b>. About 800 possible states.</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="feature"><div class="ico">🎮</div><h3>Actions</h3><p><b>Hit</b>, <b>Stand</b>, '
                '<b>Double</b> (first two cards), or <b>Split</b> (a pair becomes two hands).</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="feature"><div class="ico">🏆</div><h3>Reward</h3><p><b>+1</b> win, <b>−1</b> loss, '
                '<b>0</b> push, doubled after a double down. Only given at the end of a hand.</p></div>',
                unsafe_allow_html=True)

st.markdown("### 2. Learning: Monte Carlo control")
st.markdown("After each hand, every (state, action) pair used is moved toward the money it actually won, *G*. "
            "The step size is 1/N, where N is how many times that pair has been tried, so each value is simply "
            "the **running average** of its results and settles on the true value:")
st.latex(r"Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\,\big(G - Q(s,a)\big)")
st.markdown("After a split, the split decision earns the total of both hands. In the code "
            "(`blackjack_rl.py → train()`):")
st.code("for state, action, hand in history[explore_start:]:\n"
        "    g = total_payout if hand == SPLIT else payout[hand]\n"
        "    N[state][action] += 1\n"
        "    Q[state][action] += (g - Q[state][action]) / N[state][action]", language="python")

st.markdown("### 3. Exploring without fooling itself")
st.markdown(
    "The agent plays its best known strategy, except that at one random decision per hand it tries a random "
    "move. It then learns only from that exploring move onward, so every sample measures exactly *\"take this "
    "action, then play my best\"*.\n\n"
    "An early version also learned from hands where no exploring move happened, which caused a subtle "
    "**selection bias**: short, lucky hands (like drawing a 10 on 11) got counted more often, so the agent "
    "overrated hitting. Skipping those hands fixed it. Hitting 11 against a 10 went from a wrongly "
    "estimated +0.22 to +0.11, matching a direct simulation."
)

st.markdown("### 4. Safety for rare situations")
st.markdown(
    "Some situations almost never happen, like a hard 20 right after splitting tens. The agent had tried "
    "standing there just **once** (and lost), so it wrongly believed hitting 20 was better. Two safeguards "
    "fix this without hand-coding a strategy:\n\n"
    "- **Borrowing experience:** with an endless deck, hitting or standing depends only on your total, so a "
    "rare two-card hand borrows values from the same total made with three cards, which it has practiced "
    "thousands of times.\n"
    "- **Guaranteed rules:** hitting a hard 11 or less can never bust you, so the agent never stands there, "
    "and it never hits a hard 20. Doubles or splits tried fewer than 30 times are never recommended."
)

st.markdown("### 5. Proof that it learned: grading against the exact answer")
_g, _rk, _base = exact_grade("expert", "S17"), exact_grade("rookie", "S17"), exact_baselines("S17")
_ref = get_bundles()[("expert", "S17")].get("refined")
st.markdown(
    "Blackjack is small enough that the *true* value of every move can be calculated exactly, by working "
    "through every card that could come next. The agent never sees that calculation. It is only used as the "
    "examiner, so “it plays well” is measured, not guessed. Against a dealer who stands on 17:"
)
st.markdown(
    "| Player | Money per $100 (exact) | What it is |\n|---|---|---|\n"
    "| 🎲 Random play | **about −$43** | picks legal moves at random (simulated) |\n"
    f"| 📏 Simple rule | **{_base['Simple rule (stand on 17)']:+.2f}** | no learning: copy the dealer, hit until 17 |\n"
    f"| 🧢 Rookie agent ({LEVELS['rookie']:,} hands) | **{_rk['policy_ev'] * 100:+.2f}** | too little experience |\n"
    f"| 📘 Published 6-deck chart | **{_base['Basic strategy chart (6-deck)']:+.3f}** | what professionals memorise |\n"
    f"| 🤖 Expert agent | **{_g['policy_ev'] * 100:+.3f}** | learned from wins and losses alone |\n"
    f"| ✨ Perfect play | **{_g['optimal_ev'] * 100:+.3f}** | the mathematical best, i.e. the house edge |\n"
)
st.markdown(
    f"- The Expert picks a provably perfect move in **{_g['optimal_share']:.1%}** of the 330 two-card decisions, "
    f"and gives up only **{_g['regret'] * 10000:.2f}¢ per $100** compared with perfect play.\n"
    "- Earlier versions of this page said the agent beat the strategy chart (−$0.43 vs −$0.48). Those were "
    "simulations with a margin of error of about ±$0.18, so the difference was noise. Exact grading settles it.\n"
    "- Even perfect play loses a little. That is the house edge, which no strategy beats without counting cards."
)

st.markdown("### 6. Precision practice: fixing the last mistakes")
st.markdown(
    "The exact grade showed where the agent's remaining mistakes were: **rare hands**, like a pair of 4s against "
    "a 3. Random self-play gives them about 50× less practice than common hands like 16 against a 10. The fix is "
    "still pure reinforcement learning, just smarter about where to spend experience:\n\n"
    "- **Exploring starts:** practice hands are dealt straight into decisions the agent is unsure about.\n"
    "- **Paired comparisons:** every move is tried against the *same* upcoming cards, so luck cancels out and "
    "small differences show up with far fewer hands.\n"
    "- **Knowing when to stop:** a decision is settled once the best move is ahead by 3 standard errors, or the "
    "moves are proven to be within 0.1¢ of each other (a genuine tie).\n"
)
if _ref:
    st.markdown(f"After {_ref['deals']:,} practice deals, {_ref['settled']} of {_ref['total']} decisions were "
                "settled, and the gap to perfect play fell about 20-fold.")
st.page_link("research.py", label="See every decision graded in the Research Lab", icon="🔬")
st.page_link("card_counter.py", label="Or meet the counting agent that beats the house", icon="🧮")

st.markdown("### 7. The counting agent: beating the house")
st.markdown(
    "The base agent trains on an endless deck, so it can never know which cards are gone. A second agent trains "
    "on a real 6-deck shoe with the **Hi-Lo true count** added to its state, and learns two things:\n\n"
    "- **What each count is worth.** It measured −2.6% per dollar at a true count of −3, rising to +2.4% at +5, "
    "turning profitable around +1.\n"
    "- **How much to bet.** Each bet size earns *(bet × the value of this count)*, so it learned to bet the minimum "
    "in bad shoes and the maximum in good ones. That bet spread is where counting makes its money.\n\n"
    "It also measured **36 playing deviations**, rediscovering famous ones like doubling 9 against a 2 at a "
    "positive count and hitting 13 against a 2 when the count goes negative.\n\n"
    "**Result:** +$0.60 per $100 wagered against a dealer who stands on 17, versus −$0.52 for the same agent "
    "flat betting. It is the only version here that beats the house."
)
st.page_link("card_counter.py", label="Open the Card Counter", icon="🧮")

st.markdown("### 8. Five algorithms, same game")
try:
    _race = alg.load_race()
except (OSError, ValueError):
    _race = None
if _race:
    _rows = ["| Algorithm | Perfect decisions | Gap to perfect (¢ per $100) |", "|---|---|---|"]
    for _name, _pts in _race.items():
        _p = _pts[-1]
        if "regret_100" in _p:
            _rows.append(f"| {_name} ({_p['hands']:,} hands) | {_p['optimal']:.1%} | {_p['regret_100'] * 100:.2f} |")
    st.markdown("The same Blackjack learned five ways, averaged over several random seeds and graded exactly:\n\n"
                + "\n".join(_rows))
st.markdown(
    "Monte Carlo beats Q-learning and SARSA because a Blackjack hand lasts two or three moves and the reward "
    "arrives at the end, so learning from the true result is unbiased and simple; the temporal-difference methods "
    "learn from their own estimates, which adds noise for no gain here. The neural network has to approximate a "
    "table with only a few hundred entries, and trains about 50× slower per hand. **Precision practice** gets the "
    "most out of the same number of hands by spending them where the agent is unsure."
)
st.page_link("research.py", label="See the race in the Research Lab", icon="🔬")

st.markdown("### 9. What isn't reinforcement learning")
st.markdown("This is worth being clear about, because people assume the opposite:")
st.markdown(
    "- **The Professor's explanations** are rule-based sentences written around the agent's decision.\n"
    "- **Bust percentages** come from probability and Monte Carlo simulation of the shoe.\n"
    "- **The Hi-Lo card count** is a standard formula; the base agent doesn't use it.\n"
    "- **The exact solver** is dynamic programming, not learning. It is the examiner, never the teacher.\n"
    "- **The optional chat** (Ask the Professor) is a language model (Ollama or Gemini). It does not decide "
    "anything: it is handed the agent's real learned values and asked to put them in plain words. Its 👍/👎 "
    "ratings do drive a small bandit that learns which teaching style students prefer."
)

with st.expander("❓ Common questions"):
    st.markdown(
        "**Why Monte Carlo instead of Q-learning?** Rewards only come at the end of a short hand, so "
        "learning from the true final result is simple and unbiased.\n\n"
        "**Is it model-free?** Yes. The agent never sees card probabilities or dealer rules; it learns only "
        "from outcomes.\n\n"
        "**Limitations?** The base agents use an endless deck (no card memory), only one split per hand is allowed, "
        "and insurance isn't implemented. A lookup table only works because Blackjack has few states; bigger games "
        "need neural networks.\n\n"
        "**Can I train my own agent?** Yes: `blackjack_env.py` is a Gymnasium environment with double and split, "
        "and `solver.regret()` grades any strategy exactly. See the Research Lab's Data tab."
    )
