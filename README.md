# ♠ Royal Blackjack ♥
### A casino that taught itself to play: reinforcement learning in action

Royal Blackjack is a full casino Blackjack game built with **Streamlit**, powered by AI agents that learned
to play entirely on their own through **Monte Carlo reinforcement learning**, then graded against an exact solver. No strategy was programmed in:
the agents started knowing nothing, played hundreds of thousands of hands, and rediscovered the
professional "basic strategy" from wins and losses alone.

![Royal Blackjack home page](screenshots/home.png)

<table>
<tr>
<td width="50%"><img src="screenshots/casino-table.png" alt="Casino table with AI players and the Professor"><br><sub><b>Casino table</b> — AI players beside you, the Professor explaining every move, live odds from the shoe.</sub></td>
<td width="50%"><img src="screenshots/card-counter.png" alt="Card counter page"><br><sub><b>Card counter</b> — what each true count is worth, learned by playing 25 million shoe rounds.</sub></td>
</tr>
<tr>
<td width="50%"><img src="screenshots/algorithm-race.png" alt="Algorithm race comparison"><br><sub><b>Algorithm race</b> — Monte Carlo vs Q-learning vs SARSA vs a from-scratch neural network.</sub></td>
<td width="50%"><img src="screenshots/neon-theme.png" alt="Neon Vegas theme"><br><sub><b>Five themes</b> — one click restyles every page, table, card and chip.</sub></td>
</tr>
</table>

---

## ✨ Features

| Page | What it does |
|---|---|
| 🎨 **5 casino themes** | Classic Gold, Neon Vegas, Tokyo Night, Pirate Tavern, and Cyber Casino. One click restyles every page, table, card, chip, and font, each with its own effects (neon lights, falling petals, gold doubloons, terminal scanlines). |
| 🎰 **Casino Table** | Full casino table with chips, a 6-deck shoe, dealing animations, four table styles, and AI players seated beside you. |
| 👥 **Multiplayer** | Online rooms: create a table, share the 5-letter code or invite link, and up to 4 friends play against the same dealer from their own devices, with live turns, a leaderboard, chat, and emoji reactions. |
| 🔍 **What if?** | After every hand, replay each decision: what you chose, what the expert would have chosen, and — using the dealer's real hand — exactly what standing instead would have produced. |
| 🎓 **The Professor** | Live coach that recommends the best move, explains why, shows real bust odds from the shoe, and grades your decisions. |
| 🧭 **Casino Advisor** | Tap your cards and the dealer's up card to get the best move, then print a strategy card generated from what the AI learned. |
| 🧮 **Card Counter** | A second agent trained on a real 6-deck shoe with the Hi-Lo count in its state. It learns what each count is worth, how much to bet, and which decisions the count should change — and it beats the house edge. |
| 💬 **Ask the Professor** (optional) | A chat that explains the agent's decisions in plain language, powered by either a local model (**Ollama**) or **Google Gemini**. It is fed the agent's real learned values (including any hand you mention, like "16 vs 10"), so it explains measured results instead of inventing advice. It also learns from 👍/👎: a Thompson-sampling bandit picks the teaching style students rate best. With neither configured it hides itself, and nothing else changes. |
| 🎯 **Strategy Trainer** | Drills real decisions and checks every answer against the exact solver. Shows what each mistake costs in cents, tracks your weak spots, and brings missed hands back for review (spaced repetition). |
| 🔬 **Research Lab** | Exact grade of the agents against a solver, a heatmap of every decision's cost, agent estimates with confidence intervals, a five-algorithm race over several seeds, a house-edge calculator for any table rules, and CSV/JSON downloads. |
| 🧪 **AI Lab** | Strategy charts, a live learning-curve experiment, a Q-value explorer, and agent tournaments. |
| 📖 **How the AI Works** | The reinforcement learning explained with diagrams, equations, and the actual code. |

**Also included:** two dealers with different rules (stands on 17 vs. hits soft 17), an Expert agent
(30 million hands + precision practice) and a Rookie agent (5,000 hands), Hi-Lo card counting, bankroll tracking, and 3:2 blackjack payouts.

---

## 🧠 How the reinforcement learning works

Blackjack is modeled as a **Markov Decision Process**:

- **State:** `(player_total, dealer_up_card, usable_ace, can_double, pair_value)`
- **Actions:** hit, stand, double, split
- **Reward:** money won per unit bet (+1 win, −1 loss, 0 push; ×2 after doubling; a split earns the total of both hands)

**1. Monte Carlo control with exploring decisions (30 million hands).** Each hand the agent plays its current best
strategy except for one random exploring move, and learns only from that move onward. Values are running averages:

```
Q(s, a) ← Q(s, a) + (G − Q(s, a)) / N(s, a)
```

An earlier version also learned from hands with no exploring move. That caused a selection bias (short, lucky hands
were over-counted) and made the agent overrate hitting; skipping those hands fixed it.

**2. Precision practice (10 million deals).** Exact grading (below) showed that every remaining mistake was in a
*rare* hand: a pair of 4s against a 3 gets about 50× less random practice than 16 against a 10. Precision practice
is still learning from experience only, but spends it where the agent is unsure:

- **exploring starts** — practice hands are dealt straight into decisions that are not settled yet;
- **paired comparisons** — every legal move is played against the *same* upcoming cards (common random numbers),
  so luck cancels out of the comparison;
- **a stopping rule** — a decision is settled when the best move is ahead by 3 standard errors, or the other moves
  are within 0.1¢ per $1 of it (a genuine tie). This is best-arm identification inside Monte Carlo control.

**3. Safety for rare situations.** Hands that almost never occur borrow hit/stand values from the same total made
with more cards (exact for an endless deck), and guaranteed rules (never stand on 11 or less, never hit a hard 20)
act as a final guardrail.

## 🎯 Exact grading: how good is it, really?

Blackjack is small enough to **solve exactly**. `solver.py` computes the true value of every move by working through
every card that can come next (dynamic programming over an infinite deck, same rules as the agents), and the exact
long-run value of *any* strategy. The agents never see the solver; it is only the examiner, so there is no
simulation noise in any number below.

| Dealer stands on 17 | Money per $100 (exact) | Gap to perfect | Perfect first decisions |
|---|---|---|---|
| 📏 Simple rule (hit until 17) | −$5.675 | 510¢ | — |
| 🧢 Rookie agent (5,000 hands) | −$14.505 | 1,393¢ | 45.2% |
| 🤖 Expert, self-play only (30M hands) | −$0.597 | 2.66¢ | 97.3% |
| 📘 Published 6-deck chart | −$0.571 | 0.08¢ | 99.4% |
| 🎯 **Expert + precision practice** | **−$0.572** | **0.13¢** | **99.7%** |
| ✨ Perfect play | −$0.570 | 0 | 100% |

Against a dealer who **hits soft 17**, the expert with precision practice plays **perfectly**: 100% of first decisions,
exact value −$0.789 per $100, identical to perfect play.

> **A correction.** Earlier versions of this README said the agent (−$0.43) beat the chart (−$0.48). Those were
> simulations with a margin of error of about ±$0.18 per $100, so the gap was noise. Exact grading shows the
> self-play agent was 2.7¢ per $100 *behind* the chart, and precision practice closed almost all of that gap.

The remaining S17 difference is soft 18 against a 2 (double vs stand), worth 0.2¢ per $1 on a hand that comes up
in 0.09% of rounds.

### The counting agent (25 million shoe rounds per dealer)

A second agent (`counting.py`) plays a real 6-deck shoe, so it can track the **Hi-Lo true count**:

- **State** gains the true-count bucket, and count-specific values are only trusted after 500 practices; otherwise it defers to the base expert.
- **Bet sizing** is a contextual bandit: each bet size earns *(bet × the value of that count)*.
- **Playing deviations** are measured directly — a shoe is built at a chosen true count, the hand is dealt, each move is forced, and the outcomes are compared over thousands of rounds.

| | Counting agent | Same expert, flat betting |
|---|---|---|
| Dealer stands on 17 | **+$0.60 per $100 wagered** | −$0.52 |
| Dealer hits soft 17 | **+$0.04 per $100 wagered** | −$0.76 |

These counting results are simulations (a finite shoe can't be solved the same way), so treat the second decimal
as noise. It is a demonstration, not a business plan.

### Algorithms compared

The same game learned five ways and scored **exactly** (`python experiments.py race`), mean ± standard deviation
over 3 seeds, dealer stands on 17:

| Algorithm | Hands | Perfect first decisions | Gap to perfect play (per $100) |
|---|---|---|---|
| 🎯 **MC + precision practice** | 6M | **98.0% ± 0.5** | **2.4¢ ± 0.9** |
| 🎲 Monte Carlo | 6M | 92.7% ± 0.9 | 8.8¢ ± 2.9 |
| ⚡ Q-learning | 6M | 83.8% ± 1.7 | 234¢ ± 75 |
| 🐢 SARSA | 6M | 80.4% ± 0.7 | 241¢ ± 88 |
| 🧠 Deep Q-Network (numpy, no PyTorch) | 200k | 72.3% ± 4.4 | 420¢ ± 66 |

With the same number of hands, precision practice (half self-play, half targeted practice) gets **3.6× closer to
perfect play** than plain Monte Carlo. At a small budget it does not help: at 1.5M hands plain Monte Carlo was
slightly ahead (gap 27¢ vs 40¢, within noise), because comparing moves needs a reasonable base strategy to play out
the rest of the hand. It is a fine-tuning method, and that is where the shipped experts use it
(`results/race_1500000_hands.json` has the small-budget run).

Monte Carlo beats the temporal-difference methods because hands are short and the reward comes at the end, so
learning from the true result is unbiased; Q-learning and SARSA bootstrap from their own estimates. The neural
network approximates a table that only needs a few hundred entries while training ~50× slower per hand. Neural
networks earn their keep when the state space is too large to tabulate.

## 🔬 For researchers

| File | What it gives you |
|---|---|
| `solver.py` | Exact values of every move; `policy_ev(policy)` and `regret(policy)` grade **any** strategy exactly; `house_edge(Rules(...))` for rule variations |
| `blackjack_env.py` | A **Gymnasium** environment with double and split (Gymnasium's built-in Blackjack has only hit/stand), action masks, seeding |
| `experiments.py` | Multi-seed algorithm race scored exactly, reports and CSV exports, with Python version and git commit recorded |
| `tests/` | 26 tests: engine rules, solver vs simulation, environment vs solver, reproducibility, chat helpers |

```python
from blackjack_env import BlackjackEnv
import solver

env = BlackjackEnv()                        # Gymnasium API: reset(seed), step(action)
obs, info = env.reset(seed=0)               # obs = (total, dealer card, soft, can double, pair)

result = solver.regret(my_policy)           # my_policy(state, legal_actions) -> 0 hit / 1 stand / 2 double / 3 split
print(result["regret"] * 10000, "cents per $100 from perfect play")
```

**Reproduce everything**

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
python blackjack_rl.py train 30000000      # seeded Monte Carlo self-play
python blackjack_rl.py refine 10000000     # precision practice on the saved experts
python experiments.py race --seeds 0 1 2 --budget 6000000
python experiments.py report               # results/report.md
python experiments.py export               # exact value tables and Q-tables as CSV
```

**Assumptions.** Infinite deck for the base agents and the solver (a 6-deck shoe moves the house edge by less than
0.1% and flips a handful of near-tie decisions); dealer peeks; double on any two cards and after a split; one split;
split aces get one card. Not modelled: re-splitting, insurance. The house-edge calculator also offers late
surrender and 6:5 payouts. If you use this work, see [CITATION.cff](CITATION.cff).

---

## 🚀 Run it locally

```bash
git clone https://github.com/<your-username>/royal-blackjack.git
cd royal-blackjack
pip install -r requirements.txt
streamlit run app.py
```

To try multiplayer locally, open the Multiplayer page in two browser tabs: create a table in one and join it from the other.

The four trained agents are included as `q_*.pkl` files, so the app starts instantly. If you delete them, the app retrains them on first launch (a few minutes).

To retrain the experts from the command line and grade them exactly:
```bash
python blackjack_rl.py train 30000000    # Monte Carlo self-play
python blackjack_rl.py refine 10000000   # precision practice
```

To retrain the counting agents (25 million shoe rounds each, a few minutes):
```bash
python counting.py 25000000
```

To re-run the algorithm race (several seeds in parallel, scored exactly):
```bash
python experiments.py race --seeds 0 1 2 --budget 6000000
```

Run the tests with `pip install -r requirements-dev.txt && pytest`.

---

## 📁 Project structure

```
app.py            Entry point and page navigation
home.py           Landing page: hero, results, features, theme gallery
table.py          Casino table game (single player)
multiplayer.py    Online multiplayer rooms
trainer.py        Strategy Trainer: drills scored against the exact solver, spaced repetition of mistakes
advisor.py        Casino Advisor and printable strategy card
research.py       Research Lab: exact grade, error heatmaps, algorithm race, house-edge calculator, data
lab.py            AI Lab: charts, learning curve, Q-values, tournaments
card_counter.py   Card Counter page: count value, bet ramp, deviations, simulation
learn.py          Reinforcement learning explained
ask.py            The "Ask the Professor" chat page
blackjack_rl.py   The RL agent: engine, Monte Carlo training, precision practice, evaluation, CLI
solver.py         Exact solver: optimal values, exact value and regret of any strategy, house edge
blackjack_env.py  Gymnasium environment with double and split
experiments.py    Reproducible multi-seed experiments, reports and CSV exports
algorithms.py     Q-learning, SARSA and a from-scratch neural DQN, for the algorithm race
counting.py       The card-counting agent: shoe simulation, bet sizing, deviation measurement
llm.py            Optional chat support (Ollama or Gemini): model selection, grounding, streaming
professor_rl.py   Thompson-sampling bandit that learns the chat's best teaching style from 👍/👎
shared.py         Shared agents, dealer rules, theme engine and table renderer
tests/            pytest suite
```

---

## 💬 Optional: chat with the Professor

The chat is optional — nothing else depends on it. Set up either provider, or both and pick in the app:

**A model on your own computer** — free and private, but local only:
```bash
ollama pull llama3.2     # about 2 GB, one time
ollama serve             # then reload the app
```

**Google Gemini** — has a free tier and works on the deployed app. Get a key from
[aistudio.google.com](https://aistudio.google.com), then put it in `.streamlit/secrets.toml` locally and under
**Settings → Secrets** on Streamlit Cloud:
```toml
GEMINI_API_KEY = "your-key-here"
```
The key is read from secrets or the environment and is never written into the code or committed. Model names are
discovered from the API at runtime rather than hard-coded, because Google renames and retires them often.

Every question is sent with the agent's **real numbers** attached (the learned value of each move in that exact
situation, the bust odds from the shoe, the true count), so the model explains the reinforcement learning results
rather than making up its own Blackjack advice.

## ⚠️ Limitations and future work

- The base agents and the solver use an endless deck (the counting agent plays a real shoe). A composition-dependent
  solver for finite shoes would let the counting agent be graded exactly too.
- Only one split per hand; insurance and re-splitting are not implemented in the game.
- Training is pure Python (about 150k hands per second); vectorising it would make large experiments much faster.
- Chat ratings and multiplayer rooms live on the server, so they reset when the app restarts.

---

## 📄 License

MIT. See [LICENSE](LICENSE). Built by Aryan.