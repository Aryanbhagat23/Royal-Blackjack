"""
Exact Blackjack solver: the answer key for the reinforcement learning agents
===========================================================================
Reinforcement learning *estimates* the value of each move from random hands, so its answers always
carry some noise. For this game the true values can be computed exactly instead, by working through
every possible card that can come next (dynamic programming / expectimax over an infinite deck).

That gives the project a ground truth:

* ``optimal_values(rules)``  — the exact expected value of hit / stand / double / split / surrender in
  every decision the agent can face, and therefore the mathematically perfect strategy.
* ``policy_ev(policy, rules)`` — the exact expected value of *any* strategy, including a trained
  agent. No simulation, no margin of error.
* ``regret(policy, rules)`` — how much money a strategy gives up per hand compared with perfect play,
  plus the exact cost of every individual decision it gets wrong.

The game model matches ``blackjack_rl`` exactly, so the numbers are directly comparable with the
agents' Q-values:

* infinite deck (each rank equally likely, ten-valued cards 4/13), dealer peeks for blackjack
* the agent's values are conditional on neither side having blackjack (that is when decisions happen)
* hitting to 21 stands automatically; double = one card for twice the bet
* one split per round; split aces receive one card each; a split hand totalling 21 is not a blackjack
* optional rule variations for the house-edge calculator: dealer hits soft 17, double after split,
  late surrender, and the blackjack payout (3:2 or 6:5)

Run directly for a summary:
    python solver.py
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import blackjack_rl as rl

CARDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
P = {c: (4 / 13 if c == 10 else 1 / 13) for c in CARDS}
HIT, STAND, DOUBLE, SPLIT, SURRENDER = 0, 1, 2, 3, 4
NAMES = {HIT: "hit", STAND: "stand", DOUBLE: "double", SPLIT: "split", SURRENDER: "surrender"}
UPCARDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class Rules:
    """Table rules. The defaults are exactly the game the agents are trained on."""
    hit_soft17: bool = False
    double_after_split: bool = True
    surrender: bool = False            # late surrender (after the dealer checks for blackjack)
    blackjack_pays: float = 1.5        # 1.5 = 3:2, 1.2 = 6:5

    def label(self):
        return (f"{'H17' if self.hit_soft17 else 'S17'}, {'DAS' if self.double_after_split else 'no DAS'}, "
                f"{'late surrender' if self.surrender else 'no surrender'}, "
                f"blackjack pays {'3:2' if self.blackjack_pays == 1.5 else '6:5' if self.blackjack_pays == 1.2 else self.blackjack_pays}")


AGENT_RULES = {"S17": Rules(hit_soft17=False), "H17": Rules(hit_soft17=True)}


def _add(total, soft, card):
    return rl._add(total, soft, card)


# =========================================================
# Dealer
# =========================================================

@lru_cache(maxsize=None)
def _dealer_from(total, soft, hit_soft17):
    """Final-total distribution {17..21, 22 = bust} for a dealer currently holding (total, soft)."""
    if total > 21:
        return ((22, 1.0),)
    if total > 17 or (total == 17 and not (hit_soft17 and soft)):
        return ((total, 1.0),)
    out = {}
    for c in CARDS:
        for final, p in _dealer_from(*_add(total, soft, c), hit_soft17):
            out[final] = out.get(final, 0.0) + P[c] * p
    return tuple(sorted(out.items()))


@lru_cache(maxsize=None)
def dealer_distribution(up, hit_soft17=False):
    """Dealer's final total given the up card, *given the dealer does not have blackjack*."""
    out, mass = {}, 0.0
    for hole in CARDS:
        if up + hole == 21:
            continue                                   # peeked blackjack: the hand never gets played
        mass += P[hole]
        t, s = _add(*_add(0, 0, up), hole)
        for final, p in _dealer_from(t, s, hit_soft17):
            out[final] = out.get(final, 0.0) + P[hole] * p
    return {k: v / mass for k, v in sorted(out.items())}


def dealer_blackjack_chance(up):
    return P[10] if up == 11 else P[11] if up == 10 else 0.0


# =========================================================
# Player values (exact, conditional on no dealer blackjack)
# =========================================================

@lru_cache(maxsize=None)
def stand_ev(total, up, hit_soft17=False):
    if total > 21:
        return -1.0
    ev = 0.0
    for final, p in dealer_distribution(up, hit_soft17).items():
        ev += p * (1 if final == 22 or total > final else -1 if total < final else 0)
    return ev


@lru_cache(maxsize=None)
def _continue(total, soft, up, hit_soft17):
    """Best value of a hand that may only hit or stand (three or more cards)."""
    if total > 21:
        return -1.0
    if total == 21:
        return stand_ev(21, up, hit_soft17)
    return max(stand_ev(total, up, hit_soft17), _hit(total, soft, up, hit_soft17))


@lru_cache(maxsize=None)
def _hit(total, soft, up, hit_soft17):
    ev = 0.0
    for c in CARDS:
        ev += P[c] * _continue(*_add(total, soft, c), up, hit_soft17)
    return ev


@lru_cache(maxsize=None)
def _double(total, soft, up, hit_soft17):
    ev = 0.0
    for c in CARDS:
        t, _ = _add(total, soft, c)
        ev += P[c] * stand_ev(t, up, hit_soft17)
    return 2 * ev


def _two_card_best(total, soft, up, rules):
    """Best value of a two-card hand made by splitting (no further split, no surrender)."""
    options = [stand_ev(total, up, rules.hit_soft17), _hit(total, soft, up, rules.hit_soft17)]
    if rules.double_after_split:
        options.append(_double(total, soft, up, rules.hit_soft17))
    return max(options)


@lru_cache(maxsize=None)
def _split(card, up, rules):
    ev = 0.0
    for c in CARDS:
        t, s = _add(*_add(0, 0, card), c)
        if card == 11:
            ev += P[c] * stand_ev(t, up, rules.hit_soft17)      # split aces: one card each
        else:
            ev += P[c] * _two_card_best(t, s, up, rules)
    return 2 * ev


def action_values(state, rules=Rules(), first=True):
    """Exact value of every legal move in an agent state (total, up, soft, can_double, pair),
    assuming perfect play afterwards. Surrender is only offered on the first decision of a round.
    Returns {action: value}."""
    total, up, soft, can_d, pair = state
    hs = rules.hit_soft17
    s = 1 if soft else 0
    vals = {STAND: stand_ev(total, up, hs), HIT: _hit(total, s, up, hs) if total < 21 else -1.0}
    if can_d:
        vals[DOUBLE] = _double(total, s, up, hs)
        if rules.surrender and first:
            vals[SURRENDER] = -0.5
    if pair:
        vals[SPLIT] = _split(pair, up, rules)
    return vals


def best_action(state, rules=Rules(), legal=None):
    vals = action_values(state, rules)
    if legal is not None:
        vals = {a: v for a, v in vals.items() if a in legal}
    return max(vals, key=vals.get), vals


def optimal_policy(rules=Rules()):
    """A policy(state, legal) function that plays perfectly (for simulations and the table)."""
    def pol(state, legal):
        return best_action(state, rules, legal)[0]
    return pol


# =========================================================
# Whole-round expectations
# =========================================================

def _round_ev(first_decision_value, rules):
    """EV per round, given a function value(state) for the first decision of a non-natural hand."""
    ev = 0.0
    for up in UPCARDS:
        p_up = P[up]
        p_dbj = dealer_blackjack_chance(up)
        for c1 in CARDS:
            for c2 in CARDS:
                p = p_up * P[c1] * P[c2]
                natural = c1 + c2 == 21
                if natural:
                    ev += p * (1 - p_dbj) * rules.blackjack_pays
                    continue
                ev += p * p_dbj * -1.0
                t, s = _add(*_add(0, 0, c1), c2)
                state = (t, up, s > 0, True, c1 if c1 == c2 else 0)
                ev += p * (1 - p_dbj) * first_decision_value(state)
    return ev


def house_edge(rules=Rules()):
    """Expected result per $1 initial bet with perfect play (negative = the house wins)."""
    return _round_ev(lambda st: max(action_values(st, rules).values()), rules)


def decision_weights(rules=Rules()):
    """How often each first decision comes up per round (so costs can be weighted by frequency)."""
    w = {}
    for up in UPCARDS:
        live = P[up] * (1 - dealer_blackjack_chance(up))
        for c1 in CARDS:
            for c2 in CARDS:
                if c1 + c2 == 21:
                    continue
                t, s = _add(*_add(0, 0, c1), c2)
                st = (t, up, s > 0, True, c1 if c1 == c2 else 0)
                w[st] = w.get(st, 0.0) + live * P[c1] * P[c2]
    return w


# =========================================================
# Exact value of an arbitrary policy (for grading agents)
# =========================================================

def policy_ev(policy, rules=Rules()):
    """Exact expected value per round of `policy(state, legal) -> action` (0-3), no simulation.
    Only rules the agents are trained on are supported (no surrender)."""
    hs = rules.hit_soft17
    memo = {}

    def play(total, soft, up, n_cards, from_split):
        """Value of a hand played by the policy from (total, soft) holding n_cards cards
        (split hands start at two cards; other hands reach here after a hit)."""
        key = (total, soft, up, n_cards == 2, from_split)
        if key in memo:
            return memo[key]
        if total > 21:
            return -1.0
        can_d = n_cards == 2 and (rules.double_after_split or not from_split)
        state = (total, up, soft > 0, can_d, 0)
        legal = rl.legal_actions(can_d, False)
        a = policy(state, legal)
        if a == STAND:
            v = stand_ev(total, up, hs)
        elif a == DOUBLE:
            v = _double(total, soft, up, hs)
        else:
            v = 0.0
            for c in CARDS:
                t, s = _add(total, soft, c)
                if t > 21:
                    v += P[c] * -1.0
                elif t == 21:
                    v += P[c] * stand_ev(21, up, hs)
                else:
                    v += P[c] * play(t, s, up, n_cards + 1, from_split)
        memo[key] = v
        return v

    def split_hand(card, up):
        v = 0.0
        for c in CARDS:
            t, s = _add(*_add(0, 0, card), c)
            if card == 11:
                v += P[c] * stand_ev(t, up, hs)
            else:
                v += P[c] * play(t, s, up, 2, True)
        return v

    def first(state):
        total, up, soft, _, pair = state
        legal = rl.legal_actions(True, bool(pair))
        a = policy(state, legal)
        s = 1 if soft else 0
        if a == SPLIT and pair:
            return 2 * split_hand(pair, up)
        if a == STAND:
            return stand_ev(total, up, hs)
        if a == DOUBLE:
            return _double(total, s, up, hs)
        return _hit_with_policy(total, s, up)

    def _hit_with_policy(total, s, up):
        v = 0.0
        for c in CARDS:
            t, s2 = _add(total, s, c)
            if t > 21:
                v -= P[c]
            elif t == 21:
                v += P[c] * stand_ev(21, up, hs)
            else:
                v += P[c] * play(t, s2, up, 3, False)
        return v

    return _round_ev(first, rules)


def greedy_policy(Q):
    """The policy an agent's Q-table plays (same tie-breaking and unseen-state default as the app)."""
    return lambda state, legal: rl.greedy(Q, state, legal)


def two_card_states():
    """Every first decision (hard 5-19, soft 13-20, pairs), as agent states, per dealer card."""
    out = []
    for up in UPCARDS:
        for t in range(5, 20):
            out.append(("hard", t, (t, up, False, True, 0)))
        for t in range(13, 21):
            out.append(("soft", t, (t, up, True, True, 0)))
        for v in range(2, 12):
            out.append(("pair", v, (12 if v == 11 else 2 * v, up, v == 11, True, v)))
    return out


def regret(policy, rules=Rules()):
    """How far a policy is from perfect play.

    Returns a dict with
      optimal_ev, policy_ev   exact EV per round (per $1 initial bet)
      regret                  optimal_ev - policy_ev (>= 0): money given up per round
      decisions               per first decision: optimal move, policy move, exact cost of the policy's
                              move (assuming perfect play afterwards) and how often it happens per round
      optimal_share           share of first decisions where the policy picks a perfect move
    """
    opt = house_edge(rules)
    pol = policy_ev(policy, rules)
    weights = decision_weights(rules)
    rows, hits = [], 0
    for kind, t, state in two_card_states():
        legal = rl.legal_actions(True, bool(state[4]))
        best, vals = best_action(state, rules, legal)
        got = policy(state, legal)
        cost = vals[best] - vals.get(got, vals[best])
        if cost <= 1e-12:
            hits += 1
        rows.append({"kind": kind, "total": t, "up": state[1], "state": state,
                     "optimal": NAMES[best], "agent": NAMES[got], "cost": cost,
                     "weight": weights.get(state, 0.0), "values": {NAMES[a]: v for a, v in vals.items()}})
    return {"optimal_ev": opt, "policy_ev": pol, "regret": opt - pol, "decisions": rows,
            "optimal_share": hits / len(rows)}


def strategy_letters(rules=Rules()):
    """The perfect strategy as chart letters: {kind: {row_label: [letters for dealer 2..A]}}."""
    letter = {HIT: "H", STAND: "S", DOUBLE: "D", SPLIT: "P", SURRENDER: "R"}
    out = {"hard": {}, "soft": {}, "pair": {}}
    for t in range(19, 4, -1):
        out["hard"][str(t)] = [letter[best_action((t, u, False, True, 0), rules)[0]] for u in UPCARDS]
    for t in range(20, 12, -1):
        out["soft"][f"A+{t - 11}"] = [letter[best_action((t, u, True, True, 0), rules)[0]] for u in UPCARDS]
    for v in range(11, 1, -1):
        lab = "A,A" if v == 11 else ("10,10" if v == 10 else f"{v},{v}")
        st = lambda u: (12 if v == 11 else 2 * v, u, v == 11, True, v)
        out["pair"][lab] = [letter[best_action(st(u), rules)[0]] for u in UPCARDS]
    return out


if __name__ == "__main__":
    import os
    for key, rules in AGENT_RULES.items():
        print(f"{key}: perfect play = {house_edge(rules) * 100:+.3f} per $100  ({rules.label()})")
        path = os.path.join(rl.BASE_DIR, f"q_expert_{key}.pkl")
        if os.path.exists(path):
            b = rl.load_bundle(path)
            r = regret(greedy_policy(rl.robust_q(b["Q"], b["N"])), rules)
            wrong = [d for d in r["decisions"] if d["cost"] > 1e-12]
            print(f"     expert agent  = {r['policy_ev'] * 100:+.3f} per $100 → regret "
                  f"{r['regret'] * 100:.4f} per $100, perfect on {r['optimal_share']:.1%} of first decisions "
                  f"({len(wrong)} imperfect)")
