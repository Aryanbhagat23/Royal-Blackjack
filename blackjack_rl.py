"""
Blackjack Reinforcement Learning Agent  (v2: precision training + splitting)
--------------------------------------------------------------------------
The agent learns with Monte Carlo control from self-play.

State  : (player_total, dealer_up_value, usable_ace, can_double, pair_value)
           pair_value = card value of a splittable pair (2-11), otherwise 0
Actions: 0 = hit, 1 = stand, 2 = double, 3 = split
Reward : money won per unit bet (+1 win, -1 loss, 0 push; doubled after a double;
         after a split, the split decision earns the total of both hands)

Precision training (Monte Carlo control with exploring decisions):
  * Each hand the agent plays its current best (greedy) strategy, except that at one
    randomly chosen decision it tries a random move.
  * Only decisions from that exploring move onward are learned from, and hands with no
    exploring move are skipped. That way every sample honestly measures "take this action,
    then play my best strategy", with no bias from how the hand happened to end.
  * Values are running averages (step size 1/N), so they settle on the true values
    instead of wobbling the way a fixed learning rate does.

Run directly to train agents for both dealer rules and grade them:
    python blackjack_rl.py
"""

import math
import os
import pickle
import random

ACTIONS = ["hit", "stand", "double", "split"]
Q_VERSION = 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CARD_VALUES = {
    "A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10,
}
DRAW = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]   # one entry per rank (infinite-deck draws)


# =========================================================
# Hand helpers used by the app (cards are dicts with "rank" and "value")
# =========================================================

def hand_info(cards):
    """Return (score, usable_ace). usable_ace = an ace is still counted as 11."""
    s = sum(c["value"] for c in cards)
    aces = sum(1 for c in cards if c["rank"] == "A")
    while s > 21 and aces:
        s -= 10
        aces -= 1
    return s, aces > 0


def score(cards):
    return hand_info(cards)[0]


def is_blackjack(cards):
    return len(cards) == 2 and score(cards) == 21


def is_pair(cards):
    return len(cards) == 2 and cards[0]["value"] == cards[1]["value"]


def dealer_should_hit(cards, hit_soft17=False):
    s, soft = hand_info(cards)
    return s < 17 or (hit_soft17 and s == 17 and soft)


def get_state(cards, dealer_up_card, can_double, can_split=False):
    s, usable = hand_info(cards)
    pair = cards[0]["value"] if (can_split and is_pair(cards)) else 0
    return (s, dealer_up_card["value"], usable, bool(can_double), pair)


def legal_actions(can_double, can_split=False):
    return [0, 1] + ([2] if can_double else []) + ([3] if can_split else [])


def greedy(Q, state, legal, random_unseen=False):
    q = Q.get(state)
    if q is None:
        if random_unseen:
            return random.choice(legal)
        return 1 if state[0] >= 17 else 0          # sensible default for unseen situations
    return max(legal, key=lambda a: q[a])


UNTRIED = -9.0   # marks a move the agent hasn't practiced enough to recommend


def best_action(Q, cards, dealer_up_card, can_double, can_split=False):
    """Used by the app. Returns (action_name, {action_name: expected_reward})."""
    can_split = bool(can_split) and is_pair(cards)
    state = get_state(cards, dealer_up_card, can_double, can_split)
    legal = legal_actions(can_double, can_split)
    a = greedy(Q, state, legal)
    q = Q.get(state, [0.0] * 4)
    return ACTIONS[a], {ACTIONS[i]: q[i] for i in legal if q[i] > UNTRIED}


def robust_q(Q, N, min_visits=200, min_special=30):
    """Make a trained table safe for rarely seen situations.

    With an endless deck, the value of hitting or standing depends only on the total, not on how
    many cards made it. So when a two-card situation was rarely practiced (for example a hard 20
    right after splitting tens), its hit and stand values are borrowed from the well-practiced
    same total with three or more cards. Doubles and splits tried fewer than `min_special` times
    are never recommended."""
    R = {s: list(q) for s, q in Q.items()}
    for s, q in Q.items():
        t, up, soft, can_d, pair = s
        n = N.get(s, [0] * 4)
        twin = (t, up, soft, False, 0)
        if can_d and twin in Q:
            tn = N.get(twin, [0] * 4)
            for a in (0, 1):
                if n[a] < min_visits and tn[a] > n[a]:
                    R[s][a] = Q[twin][a]
        for a in (2, 3):
            if n[a] < min_special:
                R[s][a] = UNTRIED
        # Guaranteed rules (true in every Blackjack game) as a final guardrail for rare situations:
        # hitting a hard 11 or less, or a soft 17 or less, can never bust, so never stand there;
        # and never hit a hard 20 or 21.
        if (not soft and t <= 11) or (soft and t <= 17):
            R[s][1] = min(R[s][1], R[s][0] - 0.001)
        if not soft and t >= 20:
            R[s][0] = min(R[s][0], R[s][1] - 0.001)
    return R


# =========================================================
# Fast simulation core (cards are plain values 2-11, infinite deck)
# =========================================================

def _add(total, soft, v):
    total += v
    if v == 11:
        soft += 1
    while total > 21 and soft:
        total -= 10
        soft -= 1
    return total, soft


def _play_hand(policy, up, c1, c2, from_split, record, idx, draw):
    """Play one hand. Returns "split", or (total, bet, busted)."""
    t, s = _add(*_add(0, 0, c1), c2)
    if from_split and c1 == 11:              # split aces get exactly one card
        return t, 1, False
    n = 2
    while True:
        can_d = n == 2
        can_s = n == 2 and not from_split and c1 == c2
        state = (t, up, s > 0, can_d, c1 if can_s else 0)
        legal = [0, 1] + ([2] if can_d else []) + ([3] if can_s else [])
        a = policy(state, legal)
        if a == 3:
            if record is not None:
                record.append((state, 3, -1))
            return "split"
        if record is not None:
            record.append((state, a, idx))
        if a == 1:
            return t, 1, False
        t, s = _add(t, s, draw(DRAW))
        n += 1
        if a == 2:
            return t, 2, t > 21
        if t > 21:
            return t, 1, True
        if t == 21:
            return t, 1, False


def _play_round(policy, up, hole, p1, p2, hs17, record=None, draw=random.choice):
    """Play a full round (dealer already checked for blackjack). Returns per-hand payouts."""
    r = _play_hand(policy, up, p1, p2, False, record, 0, draw)
    if r == "split":
        hands = [_play_hand(policy, up, p1, draw(DRAW), True, record, 0, draw),
                 _play_hand(policy, up, p2, draw(DRAW), True, record, 1, draw)]
    else:
        hands = [r]
    if all(b for _, _, b in hands):
        return [-bet for _, bet, _ in hands]
    d, ds = _add(*_add(0, 0, up), hole)
    while d < 17 or (hs17 and d == 17 and ds):
        d, ds = _add(d, ds, draw(DRAW))
    out = []
    for t, bet, busted in hands:
        if busted:
            out.append(-bet)
        elif d > 21 or t > d:
            out.append(bet)
        elif t < d:
            out.append(-bet)
        else:
            out.append(0)
    return out


def _deal(draw=random.choice):
    return draw(DRAW), draw(DRAW), draw(DRAW), draw(DRAW)   # player1, player2, up, hole


def _natural(a, b):
    return a + b == 21


# =========================================================
# Training
# =========================================================

def train(episodes=2_000_000, hit_soft17=False, Q=None, N=None, explore=0.5, progress=None, seed=None):
    """Precision Monte Carlo control. Pass Q and N to continue training. Returns Q (N is updated in place).
    Pass `seed` for a reproducible run."""
    if seed is not None:
        random.seed(seed)
    Q = {} if Q is None else Q
    N = {} if N is None else N
    rnd = random.random
    choice = random.choice
    for i in range(episodes):
        p1, p2, up, hole = _deal()
        if _natural(p1, p2) or _natural(up, hole):
            continue                                  # no decisions to learn from
        rec, start = [], [None]

        def policy(state, legal):
            if start[0] is None and rnd() < explore:
                start[0] = len(rec)
                return choice(legal)
            q = Q.get(state)
            if q is None:
                return 1 if state[0] >= 17 else 0
            return max(legal, key=q.__getitem__)

        pays = _play_round(policy, up, hole, p1, p2, hit_soft17, rec)
        if start[0] is None:
            # No exploring move this hand. Skipping it keeps the data unbiased: whether a decision is
            # learned from must never depend on how the rest of the hand turned out.
            continue
        total = sum(pays)
        for state, a, idx in rec[start[0]:]:
            g = total if idx == -1 else pays[idx]
            q = Q.get(state)
            if q is None:
                q = Q[state] = [0.0] * 4
                N[state] = [0] * 4
            n = N[state]
            n[a] += 1
            q[a] += (g - q[a]) / n[a]
        if progress and i % 20_000 == 0:
            progress(i / episodes)
    return Q


def evaluate(Q, hands=50_000, hit_soft17=False, random_unseen=False, policy=None, seed=None):
    """Play greedily (or with `policy(state, legal)`). Returns win/loss/tie rates, avg reward per hand,
    and its standard error (`stderr`), so every simulated number can carry a margin of error."""
    if seed is not None:
        random.seed(seed)
    pol = policy or (lambda s, legal: greedy(Q, s, legal, random_unseen))
    wins = losses = ties = 0
    total = total_sq = 0.0
    for _ in range(hands):
        p1, p2, up, hole = _deal()
        pn, dn = _natural(p1, p2), _natural(up, hole)
        if pn or dn:
            r = 0 if pn and dn else (1.5 if pn else -1)
        else:
            r = sum(_play_round(pol, up, hole, p1, p2, hit_soft17))
        total += r
        total_sq += r * r
        wins += r > 0
        losses += r < 0
        ties += r == 0
    mean = total / hands
    var = max(total_sq / hands - mean * mean, 0.0)
    return {"win_rate": wins / hands, "loss_rate": losses / hands, "tie_rate": ties / hands,
            "avg_reward": mean, "stderr": math.sqrt(var / hands)}


def learning_curve(total=200_000, checkpoints=10, eval_hands=20_000, hit_soft17=False, progress=None):
    """Train a brand-new agent in chunks, testing after each. Returns (hands_trained, avg_reward, win_rate) points."""
    Q, N, chunk = {}, {}, total // checkpoints
    e = evaluate(Q, eval_hands, hit_soft17, random_unseen=True)
    points = [(0, e["avg_reward"], e["win_rate"])]
    for k in range(checkpoints):
        train(chunk, hit_soft17, Q, N)
        e = evaluate(Q, eval_hands, hit_soft17, random_unseen=True)
        points.append(((k + 1) * chunk, e["avg_reward"], e["win_rate"]))
        if progress:
            progress((k + 1) / checkpoints)
    return points


# =========================================================
# Precision refinement: targeted practice until every first decision is statistically settled
# =========================================================
# Random self-play gives rare hands (a pair of 4s against a 3, soft 16 against a 5) about 50x less
# practice than common ones, and that is exactly where a Monte Carlo agent's mistakes live. Refinement
# keeps learning from experience alone, but spends it where the agent is still unsure:
#
#   * exploring starts: practice hands are dealt directly into an unsettled first decision
#   * paired comparisons: every legal move is tried against the *same* upcoming cards, so luck
#     cancels out of the comparison and small differences show up with far fewer hands
#   * a stopping rule: a decision is settled once the best move beats every other by more than
#     `z` standard errors, or the remaining moves are shown to be within `tol` of it (a true tie)
#
# This is best-arm identification (a bandit problem) run inside Monte Carlo control. The exact solver
# in solver.py is never consulted; it is only used afterwards to grade the result.

def first_decisions():
    """Every two-card first decision: (state, cards) with a representative pair of cards."""
    out = []
    for up in DEALER_COLS:
        for t in range(5, 20):
            hi = min(10, t - 2)
            if hi * 2 == t:
                hi -= 1
            out.append(((t, up, False, True, 0), (hi, t - hi)))
        for t in range(13, 21):
            out.append(((t, up, True, True, 0), (11, t - 11)))
        for v in range(2, 12):
            out.append(((12 if v == 11 else 2 * v, up, v == 11, True, v), (v, v)))
    return out


def later_decisions():
    """Every hit-or-stand decision after taking a card: (state, None). Hard 4-11 can't bust and hard 21
    is automatic, so the real choices are hard 12-20 and soft 13-20."""
    out = []
    for up in DEALER_COLS:
        for t in range(12, 21):
            out.append(((t, up, False, False, 0), None))
        for t in range(13, 21):
            out.append(((t, up, True, False, 0), None))
    return out


def _play_from(policy, up, hole, total, soft, hs17, draw):
    """Finish a hand that already has three or more cards (hit or stand only), then settle it."""
    while True:
        a = policy((total, up, soft > 0, False, 0), [0, 1])
        if a == 1:
            break
        total, soft = _add(total, soft, draw(DRAW))
        if total > 21:
            return -1
        if total == 21:
            break
    d, ds = _add(*_add(0, 0, up), hole)
    while d < 17 or (hs17 and d == 17 and ds):
        d, ds = _add(d, ds, draw(DRAW))
    return 1 if d > 21 or total > d else -1 if total < d else 0


def _paired_returns(Q, state, cards, hs17, legal, n_future=24):
    """Play the same deal once per legal move, all against the same future cards. `cards` is the
    starting two cards for a first decision, or None for a later hit-or-stand decision."""
    up = state[1]
    while True:
        hole = random.choice(DRAW)
        if up + hole != 21:
            break
    future = [random.choice(DRAW) for _ in range(n_future)]
    out = {}
    for a in legal:
        it = iter(future)
        draw = lambda _deck, it=it: next(it, None) or random.choice(DRAW)
        forced = [a]

        def pol(s, lg, forced=forced):
            if forced:
                return forced.pop()
            return greedy(Q, s, lg)
        if cards is None:
            out[a] = _play_from(pol, up, hole, state[0], 1 if state[2] else 0, hs17, draw)
        else:
            out[a] = sum(_play_round(pol, up, hole, cards[0], cards[1], hs17, None, draw))
    return out


def _settled(stats, legal, tol, z):
    means = {a: stats[a][1] for a in legal}
    best = max(legal, key=means.get)
    nb, mb, vb = stats[best][0], stats[best][1], stats[best][2] / max(stats[best][0] - 1, 1)
    for a in legal:
        if a == best:
            continue
        diff_n, diff_mean, diff_m2 = stats[("d", best, a)] if ("d", best, a) in stats else stats[("d", a, best)]
        sign = 1 if ("d", best, a) in stats else -1
        gap = sign * diff_mean
        se = math.sqrt(diff_m2 / max(diff_n - 1, 1) / max(diff_n, 1))
        if gap - z * se > 0 or gap + z * se < tol:
            continue
        return False
    return nb >= 200


def refine(Q, N, hit_soft17=False, budget=2_000_000, tol=0.001, z=3.0, batch=400, stats=None,
           progress=None, seed=None):
    """Targeted, paired Monte Carlo practice on unsettled decisions (see the notes above): every
    two-card first decision and every later hit-or-stand decision.

    `budget` is the number of practice deals. Q and N are updated in place (the refined estimate of a
    first decision is the paired-sample mean). Returns (stats, report) where stats can be passed back in
    to continue, and report = {"deals", "settled", "total"}."""
    if seed is not None:
        random.seed(seed)
    stats = {} if stats is None else stats
    decisions = first_decisions() + later_decisions()
    deals = 0
    unsettled = decisions
    while deals < budget:
        unsettled = []
        for state, cards in decisions:
            legal = legal_actions(state[3], bool(state[4]))
            st = stats.get(state)
            if st is None or not _settled(st, legal, tol, z):
                unsettled.append((state, cards, legal))
        if not unsettled:
            break
        for state, cards, legal in unsettled:
            if state not in stats:
                q0, n0 = Q.get(state, [0.0] * 4), N.get(state, [0] * 4)
                stats[state] = {"base": (list(q0), list(n0))}
            st = stats[state]
            for _ in range(batch):
                r = _paired_returns(Q, state, cards, hit_soft17, legal)
                for a in legal:
                    _welford(st.setdefault(a, [0, 0.0, 0.0]), r[a])
                for i, a in enumerate(legal):
                    for b in legal[i + 1:]:
                        _welford(st.setdefault(("d", a, b), [0, 0.0, 0.0]), r[a] - r[b])
            deals += batch
            _write_back(Q, N, state, st, legal, tol, z)
            if deals >= budget:
                break
        if progress:
            progress(min(deals / budget, 1.0))
    settled = sum(1 for state, _ in decisions
                  if state in stats and _settled(stats[state], legal_actions(state[3], bool(state[4])), tol, z))
    return stats, {"deals": deals, "settled": settled, "total": len(decisions)}


def _write_back(Q, N, state, st, legal, tol, z):
    """Pool the refined samples with what self-play already learned. Once the paired comparison has
    settled a clear winner, the move it proved best is the one kept (it is far more precise)."""
    q0, n0 = st["base"]
    q = Q.setdefault(state, [0.0] * 4)
    n = N.setdefault(state, [0] * 4)
    for a in legal:
        nr, mr = st[a][0], st[a][1]
        q[a] = (n0[a] * q0[a] + nr * mr) / (n0[a] + nr) if n0[a] + nr else 0.0
        n[a] = n0[a] + nr
    paired_best = max(legal, key=lambda a: st[a][1])
    if max(legal, key=q.__getitem__) != paired_best and _settled(st, legal, tol, z):
        for a in legal:
            q[a] = st[a][1]


def _welford(acc, x):
    acc[0] += 1
    d = x - acc[1]
    acc[1] += d / acc[0]
    acc[2] += d * (x - acc[1])


# =========================================================
# Basic strategy (6 decks, double after split, dealer peeks) for grading
# H = hit, S = stand, D = double (else hit), E = double (else stand), P = split
# Columns: dealer 2 3 4 5 6 7 8 9 10 A
# =========================================================

_HARD = {**{t: "HHHHHHHHHH" for t in range(5, 9)}, 9: "HDDDDHHHHH", 10: "DDDDDDDDHH", 11: "DDDDDDDDDH",
         12: "HHSSSHHHHH", **{t: "SSSSSHHHHH" for t in range(13, 17)}, **{t: "SSSSSSSSSS" for t in range(17, 21)}}
_SOFT = {13: "HHHDDHHHHH", 14: "HHHDDHHHHH", 15: "HHDDDHHHHH", 16: "HHDDDHHHHH", 17: "HDDDDHHHHH",
         18: "SEEEESSHHH", 19: "SSSSSSSSSS", 20: "SSSSSSSSSS"}
_PAIR = {2: "PPPPPPHHHH", 3: "PPPPPPHHHH", 4: "HHHPPHHHHH", 5: "DDDDDDDDHH", 6: "PPPPPHHHHH",
         7: "PPPPPPHHHH", 8: "PPPPPPPPPP", 9: "PPPPPSPPSS", 10: "SSSSSSSSSS", 11: "PPPPPPPPPP"}
_H17_CHANGES = {("hard", 11, 11): "D", ("soft", 18, 2): "E", ("soft", 19, 6): "E"}
_LETTER = {"H": 0, "S": 1, "D": 2, "E": 2, "P": 3}
DEALER_COLS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def basic_strategy(kind, total, up, hit_soft17=False):
    """Chart letter for a two-card hand. kind: hard | soft | pair (total = pair card value for pairs)."""
    letter = {"hard": _HARD, "soft": _SOFT, "pair": _PAIR}[kind][total][DEALER_COLS.index(up)]
    if hit_soft17:
        letter = _H17_CHANGES.get((kind, total, up), letter)
    return letter


def simple_rule_policy(stand_on=17):
    """The simplest possible Blackjack model: copy the dealer. Hit until you reach `stand_on`.
    No doubling, no splitting. Used as a baseline and as the 'Rulebook' player at the table."""
    def pol(state, legal):
        return 1 if state[0] >= stand_on else 0
    return pol


def basic_strategy_policy(hit_soft17=False):
    """A policy function that plays the textbook chart (for comparisons)."""
    def pol(state, legal):
        t, up, soft, can_d, pair = state
        if pair:
            letter = basic_strategy("pair", pair, up, hit_soft17)
        elif soft and 13 <= t <= 20:
            letter = basic_strategy("soft", t, up, hit_soft17)
        elif t >= 21:
            letter = "S"
        elif soft:
            letter = "S" if t >= 19 else "H"
        else:
            letter = basic_strategy("hard", max(5, min(t, 20)), up, hit_soft17)
        a = _LETTER[letter]
        if a == 2 and not can_d:
            a = 1 if letter == "E" else 0
        if a == 3 and 3 not in legal:
            a = 0
        return a
    return pol


_SD = [1.0, 1.0, 2.0, 2.2]   # rough spread of returns per action, for "statistical tie" checks


def grade(Q, N=None, hit_soft17=False):
    """Compare the agent with basic strategy on every two-card decision.
    Returns dict with matches, total, and a list of disagreements."""
    rows = []
    for up in DEALER_COLS:
        for t in range(5, 20):                       # hard 20 as two cards is always a pair of tens
            rows.append(("hard", t, (t, up, False, True, 0), up))
        for t in range(13, 21):
            rows.append(("soft", t, (t, up, True, True, 0), up))
        for v in range(2, 12):
            tot = 12 if v == 11 else 2 * v
            rows.append(("pair", v, (tot, up, v == 11, True, v), up))
    matches, diffs = 0, []
    for kind, t, state, up in rows:
        chart = basic_strategy(kind, t, up, hit_soft17)
        want = _LETTER[chart]
        legal = [0, 1, 2] + ([3] if kind == "pair" else [])
        got = greedy(Q, state, legal)
        if got == want:
            matches += 1
            continue
        q = Q.get(state, [0.0] * 4)
        gap = q[got] - q[want]
        n = N.get(state, [1] * 4) if N else [10_000] * 4
        se = math.sqrt((_SD[got] ** 2) / max(n[got], 1) + (_SD[want] ** 2) / max(n[want], 1))
        diffs.append({"kind": kind, "total": t, "up": up, "agent": ACTIONS[got], "chart": ACTIONS[want],
                      "gap": gap, "tie": abs(gap) < 2 * se})
    return {"matches": matches, "total": len(rows), "disagreements": diffs}


# =========================================================
# Charts for display
# =========================================================

def strategy_table(Q, usable_ace=False, can_double=True):
    """Learned action for player totals x dealer up-card, as letters H/S/D."""
    letters = {0: "H", 1: "S", 2: "D", 3: "P"}
    low = 13 if usable_ace else 5
    rows = {total: [letters[greedy(Q, (total, d, usable_ace, can_double, 0), legal_actions(can_double))]
                    for d in DEALER_COLS]
            for total in range(20, low - 1, -1)}
    return rows, ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]


def pair_table(Q):
    """Learned action for each pair (rows) x dealer up-card, letters H/S/D/P."""
    letters = {0: "H", 1: "S", 2: "D", 3: "P"}
    rows = {}
    for v in range(11, 1, -1):
        tot = 12 if v == 11 else 2 * v
        label = "A,A" if v == 11 else ("10,10" if v == 10 else f"{v},{v}")
        rows[label] = [letters[greedy(Q, (tot, d, v == 11, True, v), [0, 1, 2, 3])] for d in DEALER_COLS]
    return rows, ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]


def chart_letter(kind, total, up, hit_soft17=False):
    letter = basic_strategy(kind, total, up, hit_soft17)
    return {"E": "D"}.get(letter, letter)


# =========================================================
# Save / load
# =========================================================

def save_q(Q, path, N=None, hands=0, refined=None):
    """`refined` optionally records the precision-refinement run: {"deals", "settled", "total", ...}."""
    data = {"version": Q_VERSION, "Q": Q, "N": N or {}, "hands": hands}
    if refined:
        data["refined"] = refined
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_bundle(path):
    """Returns {"Q", "N", "hands"}; raises ValueError for files from an older version."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or data.get("version") != Q_VERSION:
        raise ValueError("old Q-table format")
    return data


def load_q(path):
    return load_bundle(path)["Q"]


def _main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Train or refine the Blackjack agents.")
    ap.add_argument("mode", choices=["train", "refine"], nargs="?", default="train",
                    help="train: fresh Monte Carlo self-play. refine: targeted practice on the saved experts.")
    ap.add_argument("budget", type=int, nargs="?", default=None,
                    help="hands of self-play (train, default 2,000,000) or practice deals (refine, default 10,000,000)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tol", type=float, default=0.001, help="refine: moves this close in value count as a tie")
    ap.add_argument("--z", type=float, default=3.0, help="refine: standard errors required to settle a decision")
    args = ap.parse_args()

    for rule, hs17 in (("S17", False), ("H17", True)):
        path = os.path.join(BASE_DIR, f"q_expert_{rule}.pkl")
        t0 = time.time()
        if args.mode == "train":
            hands = args.budget or 2_000_000
            Q, N = {}, {}
            train(hands, hs17, Q, N, seed=args.seed)
            save_q(Q, path, N, hands)
            info = f"{hands:,} hands"
        else:
            bundle = load_bundle(path)
            Q, N, hands = bundle["Q"], bundle["N"], bundle["hands"]
            budget = args.budget or 10_000_000
            _, report = refine(Q, N, hs17, budget, tol=args.tol, z=args.z, seed=args.seed,
                               progress=lambda p: print(f"\r  {rule} refining... {p:.0%}", end="", flush=True))
            report.update(tol=args.tol, z=args.z, seed=args.seed)
            save_q(Q, path, N, hands, refined=report)
            print()
            info = f"{report['deals']:,} practice deals, {report['settled']}/{report['total']} decisions settled"
        try:
            import solver
            r = solver.regret(solver.greedy_policy(robust_q(Q, N)), solver.AGENT_RULES[rule])
            exact = (f"exact EV {r['policy_ev'] * 100:+.4f} vs perfect {r['optimal_ev'] * 100:+.4f} per $100 "
                     f"(gap {r['regret'] * 10000:.2f}¢), perfect on {r['optimal_share']:.1%} of first decisions")
        except ImportError:
            exact = ""
        print(f"{rule}: {info} in {time.time() - t0:.0f}s | {exact}")


if __name__ == "__main__":
    _main()
