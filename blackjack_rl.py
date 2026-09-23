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

def train(episodes=2_000_000, hit_soft17=False, Q=None, N=None, explore=0.5, progress=None):
    """Precision Monte Carlo control. Pass Q and N to continue training. Returns Q (N is updated in place)."""
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


def evaluate(Q, hands=50_000, hit_soft17=False, random_unseen=False, policy=None):
    """Play greedily (or with `policy(state, legal)`). Returns win/loss/tie rates and avg reward per hand."""
    pol = policy or (lambda s, legal: greedy(Q, s, legal, random_unseen))
    wins = losses = ties = 0
    total = 0.0
    for _ in range(hands):
        p1, p2, up, hole = _deal()
        pn, dn = _natural(p1, p2), _natural(up, hole)
        if pn or dn:
            r = 0 if pn and dn else (1.5 if pn else -1)
        else:
            r = sum(_play_round(pol, up, hole, p1, p2, hit_soft17))
        total += r
        wins += r > 0
        losses += r < 0
        ties += r == 0
    return {"win_rate": wins / hands, "loss_rate": losses / hands, "tie_rate": ties / hands,
            "avg_reward": total / hands}


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

def save_q(Q, path, N=None, hands=0):
    with open(path, "wb") as f:
        pickle.dump({"version": Q_VERSION, "Q": Q, "N": N or {}, "hands": hands}, f)


def load_bundle(path):
    """Returns {"Q", "N", "hands"}; raises ValueError for files from an older version."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or data.get("version") != Q_VERSION:
        raise ValueError("old Q-table format")
    return data


def load_q(path):
    return load_bundle(path)["Q"]


if __name__ == "__main__":
    import sys
    import time

    hands = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    for rule, hs17 in (("S17", False), ("H17", True)):
        t0 = time.time()
        Q, N = {}, {}
        train(hands, hs17, Q, N)
        save_q(Q, os.path.join(BASE_DIR, f"q_expert_{rule}.pkl"), N, hands)
        g = grade(Q, N, hs17)
        e = evaluate(Q, 200_000, hs17)
        b = evaluate(None, 200_000, hs17, policy=basic_strategy_policy(hs17))
        ties = sum(d["tie"] for d in g["disagreements"])
        print(f"{rule}: {hands:,} hands in {time.time() - t0:.0f}s | matches basic strategy on "
              f"{g['matches']}/{g['total']} ({g['matches'] / g['total']:.1%}), {ties} of the "
              f"{len(g['disagreements'])} differences are statistical ties")
        print(f"     agent {e['avg_reward'] * 100:+.2f} vs basic strategy {b['avg_reward'] * 100:+.2f} per $100")
