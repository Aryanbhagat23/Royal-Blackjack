"""How the AI Works — the reinforcement learning explained for students and teachers."""

import streamlit as st

from shared import LEVELS, page_header

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

st.markdown("### 5. Proof that it learned")
st.markdown("Every result is measured against three baselines, so “it plays well” means something:")
st.markdown(
    "| Player | Money per $100 | What it is |\n|---|---|---|\n"
    "| 🎲 Random play | **−$43** | picks legal moves at random |\n"
    "| 📏 Simple rule | **−$5.54** | no learning at all: copy the dealer, hit until 17 |\n"
    "| 📘 Basic strategy chart | **−$0.48** | the best a non-counting player can do |\n"
    "| 🤖 This agent (30M hands) | **−$0.43** | learned from wins and losses alone |\n"
    "| 🧮 Counting agent | **+$0.60** | per $100 wagered, the only one that beats the house |\n"
)
st.caption("The simple rule is the important one: it shows how much of the gap is closed by learning rather "
           "than by just following an obvious heuristic.")
st.markdown(
    f"- **The Expert** ({LEVELS['expert']:,} hands) matches textbook basic strategy on about 97% of decisions, "
    "and every difference is a statistical tie.\n"
    f"- **The Rookie** ({LEVELS['rookie']:,} hands) makes clear mistakes: more experience means better play.\n"
    "- Agents trained against different dealer rules learned **different strategies**.\n"
    "- The Expert loses only about **$0.42 per $100**, as good as the official strategy chart. Random play "
    "loses about $43. That small loss is the house edge, which no strategy can beat without counting cards."
)
st.page_link("lab.py", label="See all of this live in the AI Lab", icon="🧪")
st.page_link("card_counter.py", label="Or meet the counting agent that beats the house", icon="🧮")

st.markdown("### 6. The counting agent: beating the house")
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

st.markdown("### 7. Four algorithms, same game")
st.markdown(
    "The same Blackjack was learned four ways and scored identically:\n\n"
    "| Algorithm | Matches the chart | Per $100 |\n|---|---|---|\n"
    "| 🎲 Monte Carlo (this project) | **92%** | −0.8 |\n"
    "| ⚡ Q-learning | 82% | −1.5 |\n"
    "| 🐢 SARSA | 82% | −3.0 |\n"
    "| 🧠 Deep Q-Network (numpy, from scratch) | 75% | −6.6 |\n\n"
    "Monte Carlo wins because a Blackjack hand lasts two or three moves and the reward arrives at the end, so "
    "waiting for the true result is unbiased and simple. Q-learning and SARSA learn from their own estimates, "
    "which adds noise for no gain here. The neural network has to approximate a table with only a few hundred "
    "entries, and trains about 50× slower per hand — neural networks pay off when the game is far too big to "
    "tabulate, like Atari or Go."
)
st.page_link("lab.py", label="See the race in the AI Lab", icon="🔬")

st.markdown("### 8. What isn't reinforcement learning")
st.markdown("This is worth being clear about, because people assume the opposite:")
st.markdown(
    "- **The Professor's explanations** are rule-based sentences written around the agent's decision.\n"
    "- **Bust percentages** come from probability and Monte Carlo simulation of the shoe.\n"
    "- **The Hi-Lo card count** is a standard formula; the base agent doesn't use it.\n"
    "- **The optional chat** (Ask the Professor) is a local language model through Ollama. It does not decide "
    "anything: it is handed the agent's real learned values and asked to put them in plain words."
)

with st.expander("❓ Common questions"):
    st.markdown(
        "**Why Monte Carlo instead of Q-learning?** Rewards only come at the end of a short hand, so "
        "learning from the true final result is simple and unbiased.\n\n"
        "**Is it model-free?** Yes. The agent never sees card probabilities or dealer rules; it learns only "
        "from outcomes.\n\n"
        "**Limitations?** Training uses an endless deck (no card memory), only one split per hand is allowed, "
        "insurance and surrender aren't implemented, and a lookup table only works because Blackjack has few states. Bigger games "
        "need neural networks (Deep Q-Learning).\n\n"
        "**Next steps?** Add the true count to the state and learn bet sizing (a counting agent can beat the house), "
        "and compare against Q-learning, SARSA, and a neural network."
    )
