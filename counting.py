"""
Card-counting agent
-------------------
The base agent in `blackjack_rl.py` trains on an endless deck, so it can't know which cards are
gone. This module trains an agent that *does*, using a real 6-deck shoe.

It learns two things at once:

1. **Playing deviations.** The state gains the Hi-Lo true count:
   (total, dealer card, soft, can_double, pair, true_count_bucket)
   so the agent can learn, for example, to stand on 16 vs 10 when the shoe is rich in tens.
   Situations it hasn't practiced fall back on the base agent, so it always plays sensibly.

2. **Bet sizing** (a contextual bandit). Before each round it picks a bet from a menu of sizes
   given the current true count, and the reward is the money that bet actually won or lost.
   Betting more when the count is high is what turns Blackjack from a losing game into a winning one.

Run directly to train and report:
    python counting.py 25000000
"""

import math
import os
import pickle
import random

import blackjack_rl as rl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DECKS = 6
PENETRATION = 0.75                 # fraction of the shoe dealt before reshuffling
BETS = [1, 2, 4, 8, 12]            # bet menu, in units
TC_MIN, TC_MAX = -3, 5             # true-count buckets
HILO = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1}
COUNT_VERSION = 1


def counting_path(rule):
    return os.path.join(BASE_DIR, f"q_counter_{rule}.pkl")


def bucket(tc):
    return max(TC_MIN, min(TC_MAX, int(math.floor(tc))))


# ---------------------------------------------------------
# Shoe
# ---------------------------------------------------------

class Shoe:
    __slots__ = ("cards", "i", "cut", "count")

    def __init__(self, decks=DECKS):
        self.cards = [v for v in (11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10) for _ in range(4 * decks)]
        random.shuffle(self.cards)
        self.i = 0
        self.cut = int(len(self.cards) * PENETRATION)
        self.count = 0

    def spent(self):
        return self.i >= self.cut

    def draw(self):
        v = self.cards[self.i]
        self.i += 1
        self.count += HILO[v]
        return v

    def true_count(self):
        decks_left = max((len(self.cards) - self.i) / 52, 0.5)
        return self.count / decks_left


# ---------------------------------------------------------
# One round
# ---------------------------------------------------------

def _add(total, soft, v):
    total += v
    if v == 11:
        soft += 1
    while total > 21 and soft:
        total -= 10
        soft -= 1
    return total, soft


def _play_hand(policy, shoe, up, c1, c2, from_split, tcb, record, idx):
    t, s = _add(*_add(0, 0, c1), c2)
    if from_split and c1 == 11:
        return t, 1, False
    n = 2
    while True:
        can_d = n == 2
        can_s = n == 2 and not from_split and c1 == c2
        state = (t, up, s > 0, can_d, c1 if can_s else 0, tcb)
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
        t, s = _add(t, s, shoe.draw())
        n += 1
        if a == 2:
            return t, 2, t > 21
        if t >= 21:
            return t, 1, t > 21


def play_round(policy, shoe, hit_soft17, record=None):
    """Deal and play one round with a one-unit base bet. Returns payout per unit bet."""
    tcb = bucket(shoe.true_count())          # count before the cards come out
    p1, up, p2 = shoe.draw(), shoe.draw(), shoe.draw()
    hole = shoe.cards[shoe.i]                # dealt face down: taken out of the shoe now...
    shoe.i += 1                              # ...but only counted when it is turned over
    player_bj, dealer_bj = p1 + p2 == 21, up + hole == 21
    if dealer_bj or player_bj:
        shoe.count += HILO[hole]
        if player_bj and dealer_bj:
            return 0.0
        return 1.5 if player_bj else -1.0

    r = _play_hand(policy, shoe, up, p1, p2, False, tcb, record, 0)
    if r == "split":
        hands = [_play_hand(policy, shoe, up, p1, shoe.draw(), True, tcb, record, 0),
                 _play_hand(policy, shoe, up, p2, shoe.draw(), True, tcb, record, 1)]
    else:
        hands = [r]

    shoe.count += HILO[hole]                 # dealer turns over the hole card
    if all(b for _, _, b in hands):
        return -sum(bet for _, bet, _ in hands)
    d, ds = _add(*_add(0, 0, up), hole)
    while d < 17 or (hit_soft17 and d == 17 and ds):
        d, ds = _add(d, ds, shoe.draw())
    total = 0.0
    for t, bet, busted in hands:
        if busted:
            total -= bet
        elif d > 21 or t > d:
            total += bet
        elif t < d:
            total -= bet
    return total


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

MIN_TRUST = 500     # times a count-specific move must be practiced before it overrides the base agent


def make_policy(base_Q, Q, explore_state=None, record=None, explore=0.35, N=None):
    """Count-aware policy that falls back to the base agent for unpracticed situations.
    If `explore_state` is given, one random decision per hand is taken and its position in
    `record` is stored, so training can learn from that decision onward."""
    def pol(state, legal):
        if explore_state is not None and explore_state[0] is None and random.random() < explore:
            explore_state[0] = len(record)
            return random.choice(legal)
        base = base_Q.get(state[:5])
        q = Q.get(state)
        if q is None and base is None:
            return 1 if state[0] >= 17 else 0
        if q is None:
            return max(legal, key=base.__getitem__)
        if base is None or N is None:
            return max(legal, key=q.__getitem__)
        n = N.get(state, [0] * 4)
        # a count-specific value is only trusted once it has been practiced enough
        return max(legal, key=lambda a: q[a] if n[a] >= MIN_TRUST else base[a])
    return pol


def train_counting(rounds=25_000_000, hit_soft17=False, base_Q=None, Q=None, N=None,
                   ev=None, ev_n=None, play_share=0.5, progress=None):
    """Learn count-aware play and bet sizing.

    Each round is used for one job or the other, never both:
      * a *practice* round takes one random exploring move and updates the playing values;
      * a *measuring* round plays the agent's best strategy and records what the true count was
        worth, which is what the bet sizing is learned from.
    Mixing the two would poison the bet sizing with the losses from deliberate random moves."""
    base_Q = base_Q or {}
    Q = {} if Q is None else Q
    N = {} if N is None else N
    ev = {b: 0.0 for b in range(TC_MIN, TC_MAX + 1)} if ev is None else ev
    ev_n = {b: 0 for b in range(TC_MIN, TC_MAX + 1)} if ev_n is None else ev_n

    shoe = Shoe()
    greedy_pol = make_policy(base_Q, Q, N=N)
    for i in range(rounds):
        if shoe.spent():
            shoe = Shoe()
        tcb = bucket(shoe.true_count())
        if random.random() < play_share:                    # practice round
            rec, flag = [], [None]
            pol = make_policy(base_Q, Q, flag, rec, N=N)
            per_unit = play_round(pol, shoe, hit_soft17, rec)
            if flag[0] is not None:
                for state, a, idx in rec[flag[0]:]:
                    q = Q.get(state)
                    if q is None:
                        q = Q[state] = [0.0] * 4
                        N[state] = [0] * 4
                    n = N[state]
                    n[a] += 1
                    q[a] += (per_unit - q[a]) / n[a]
        else:                                               # measuring round
            per_unit = play_round(greedy_pol, shoe, hit_soft17)
            ev_n[tcb] += 1
            ev[tcb] += (per_unit - ev[tcb]) / ev_n[tcb]
        if progress and i % 100_000 == 0:
            progress(i / rounds)
    return {"Q": Q, "N": N, "ev": ev, "ev_n": ev_n}


# ---------------------------------------------------------
# Reading the result
# ---------------------------------------------------------

def bet_ramp(bundle):
    """{true_count_bucket: bet in units}. Each bet size b earns b x (value of this count), so the
    agent bets the most when the count is profitable and the minimum when it isn't."""
    ramp = {}
    for b in range(TC_MIN, TC_MAX + 1):
        value = bundle["ev"][b]
        seen = bundle["ev_n"][b]
        if seen < 2_000:                       # not enough evidence: bet the minimum
            ramp[b] = BETS[0]
            continue
        ramp[b] = BETS[max(range(len(BETS)), key=lambda i: BETS[i] * value)]
    return ramp


def ev_by_count(bundle):
    """{bucket: (expected profit per $1 bet, rounds measured)}"""
    return {b: (bundle["ev"][b], bundle["ev_n"][b]) for b in range(TC_MIN, TC_MAX + 1)}


def deviations(bundle, base_Q, min_visits=4_000, min_gap=0.01, hit_soft17=False):
    """Where the counting agent plays differently from the base agent, with enough evidence."""
    out = []
    for state, q in bundle["Q"].items():
        n = bundle["N"][state]
        t, up, soft, can_d, pair, tcb = state
        if not can_d or t < 5:
            continue
        legal = rl.legal_actions(can_d, pair > 0)
        base = base_Q.get(state[:5])
        if base is None:
            continue
        base_a = max(legal, key=base.__getitem__)
        a = max(legal, key=q.__getitem__)
        if a == base_a or n[a] < min_visits or n[base_a] < min_visits or q[a] - q[base_a] < min_gap:
            continue
        kind = "pair" if pair else ("soft" if soft else "hard")
        out.append({"kind": kind, "total": pair if pair else t, "up": up, "tc": tcb,
                    "base": rl.ACTIONS[base_a], "counting": rl.ACTIONS[a], "gain": q[a] - q[base_a],
                    "visits": n[a] + n[base_a]})
    out.sort(key=lambda d: -d["gain"])
    return out


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def simulate(bundle, base_Q, rounds=200_000, hit_soft17=False, counting=True, flat_bet=1, track=0):
    """Play rounds with either counting (bet ramp + deviations) or flat betting with the base agent.
    Returns dict with profit per 100 units wagered, per round, and optionally a bankroll curve."""
    ramp = bet_ramp(bundle) if counting else None
    Q = bundle["Q"] if counting else {}
    pol = make_policy(base_Q, Q, N=bundle["N"] if counting else None)
    shoe = Shoe()
    bankroll, wagered, curve = 0.0, 0.0, []
    step = max(1, rounds // track) if track else 0
    for i in range(rounds):
        if shoe.spent():
            shoe = Shoe()
        bet = ramp[bucket(shoe.true_count())] if counting else flat_bet
        bankroll += play_round(pol, shoe, hit_soft17) * bet
        wagered += bet
        if step and i % step == 0:
            curve.append(round(bankroll, 2))
    return {"profit": bankroll, "per_100_wagered": bankroll / wagered * 100, "per_round": bankroll / rounds,
            "wagered": wagered, "curve": curve}


def save(bundle, path, rounds=0):
    with open(path, "wb") as f:
        pickle.dump({"version": COUNT_VERSION, "rounds": rounds, **bundle}, f)


def load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data.get("version") != COUNT_VERSION:
        raise ValueError("old counting file")
    return data


if __name__ == "__main__":
    import sys
    import time

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 25_000_000
    for rule, hs17 in (("S17", False), ("H17", True)):
        base = rl.load_bundle(os.path.join(BASE_DIR, f"q_expert_{rule}.pkl"))
        base_Q = rl.robust_q(base["Q"], base["N"])
        t0 = time.time()
        bundle = train_counting(rounds, hs17, base_Q)
        save(bundle, counting_path(rule), rounds)
        print(f"{rule}: {rounds:,} rounds in {time.time() - t0:.0f}s")
        print("   bet ramp:", bet_ramp(bundle))
        c = simulate(bundle, base_Q, 500_000, hs17, counting=True)
        f = simulate(bundle, base_Q, 500_000, hs17, counting=False)
        print(f"   counting {c['per_100_wagered']:+.2f} per $100 wagered, {c['per_round']:+.3f} units/round | "
              f"flat {f['per_100_wagered']:+.2f} per $100")
        print(f"   deviations found: {len(deviations(bundle, base_Q))}")


# ---------------------------------------------------------
# Targeted measurement of deviations
# ---------------------------------------------------------
# Random exploration spreads too thinly across counts, so close-call decisions are measured
# directly: build a shoe at a chosen true count, deal an exact hand, force each action, and play
# the rest normally. The difference between the two measurements is the value of deviating.

LOW, HIGH, NEUTRAL = [2, 3, 4, 5, 6], [10, 11], [7, 8, 9]


def shoe_at_count(tc, decks_left=4.5, tries=400):
    """A 6-deck shoe dealt down to `decks_left` whose Hi-Lo running count matches `tc`.

    The cards removed are chosen at random, then swapped one at a time until the count matches.
    Only the imbalance the count implies is imposed; everything else stays as a normal shoe would
    be, which matters because reshaping the deck any further would change the odds by itself."""
    shoe = Shoe()
    pool = shoe.cards
    removed = len(pool) - int(decks_left * 52)
    target = round(tc * decks_left)
    idx = list(range(len(pool)))
    random.shuffle(idx)
    take, keep = idx[:removed], idx[removed:]
    running = sum(HILO[pool[i]] for i in take)
    for _ in range(tries):
        if running == target:
            break
        need_up = running < target
        # to raise the count, move a low card into the removed pile and take a card back out.
        # each low-for-high swap shifts the count by 2, so when it is off by exactly 1 the card
        # coming back out must be a neutral 7-9, which shifts it by 1.
        want_in = LOW if need_up else HIGH
        want_out = NEUTRAL if abs(running - target) == 1 else (HIGH if need_up else LOW)
        spot_in = next((j for j, i in enumerate(keep) if pool[i] in want_in), None)
        spot_out = next((j for j, i in enumerate(take) if pool[i] in want_out), None)
        if spot_in is None or spot_out is None:
            return None
        take[spot_out], keep[spot_in] = keep[spot_in], take[spot_out]
        running = sum(HILO[pool[i]] for i in take)
    if running != target:
        return None
    taken = set(take)
    shoe.cards = [v for i, v in enumerate(pool) if i not in taken]
    shoe.count = running
    shoe.i = 0
    shoe.cut = len(shoe.cards)
    return shoe


def _take(shoe, value):
    """Deal one card of this value out of the shoe, taken from a random position.

    Taking the first match instead would leave every card it skipped sitting on top of the shoe,
    and those cards are never the value being searched for, so the next draw would be biased."""
    spots = [j for j in range(shoe.i, len(shoe.cards)) if shoe.cards[j] == value]
    if not spots:
        return None
    del shoe.cards[random.choice(spots)]
    shoe.count += HILO[value]
    return value


def measure_action(policy, c1, c2, up, action, tc, hit_soft17, trials=6000, per_shoe=10):
    """Average payout from taking `action` on (c1, c2) vs `up` at true count `tc`, then playing on."""
    total, done, shoe = 0.0, 0, None
    while done < trials:
        if shoe is None or done % per_shoe == 0:
            shoe = shoe_at_count(tc)
            if shoe is None:
                return None
        if None in (_take(shoe, c1), _take(shoe, c2), _take(shoe, up)):
            shoe = None
            continue
        hole = shoe.cards[shoe.i]
        shoe.i += 1
        if up + hole == 21:
            continue                                   # dealer blackjack: no decision happens
        first = [True]

        def pol(state, legal):
            if first[0]:
                first[0] = False
                return action if action in legal else legal[0]
            return policy(state, legal)

        r = _play_hand(pol, shoe, up, c1, c2, False, bucket(tc), None, 0)
        if r == "split":
            hands = [_play_hand(policy, shoe, up, c1, shoe.draw(), True, bucket(tc), None, 0),
                     _play_hand(policy, shoe, up, c2, shoe.draw(), True, bucket(tc), None, 1)]
        else:
            hands = [r]
        shoe.count += HILO[hole]
        if all(b for _, _, b in hands):
            total -= sum(bet for _, bet, _ in hands)
        else:
            d, ds = _add(*_add(0, 0, up), hole)
            while d < 17 or (hit_soft17 and d == 17 and ds):
                d, ds = _add(d, ds, shoe.draw())
            for t, bet, busted in hands:
                if busted:
                    total -= bet
                elif d > 21 or t > d:
                    total += bet
                elif t < d:
                    total -= bet
        done += 1
    return total / trials


def candidates(base_Q, margin=0.14):
    """Two-card decisions where the best and second-best moves are close enough for the count to flip."""
    out = []
    for up in range(2, 12):
        hands = ([(("hard", t), (min(10, t - 2), max(2, t - min(10, t - 2)))) for t in range(8, 17)]
                 + [(("soft", t), (11, t - 11)) for t in range(13, 20)]
                 + [(("pair", v), (v, v)) for v in (2, 3, 4, 6, 7, 8, 9, 10)])
        for (kind, total), (c1, c2) in hands:
            t, s = _add(*_add(0, 0, c1), c2)
            pair = c1 if kind == "pair" else 0
            state = (t, up, s > 0, True, pair)
            q = base_Q.get(state)
            if not q:
                continue
            legal = rl.legal_actions(True, pair > 0)
            ranked = sorted(legal, key=lambda a: -q[a])
            if q[ranked[0]] - q[ranked[1]] <= margin:
                out.append({"kind": kind, "total": total, "cards": (c1, c2), "up": up,
                            "base": ranked[0], "alt": ranked[1]})
    return out


def find_deviations(base_Q, hit_soft17=False, tcs=(-3, -1, 1, 3, 5), trials=5000, progress=None):
    """Measure every close call at several true counts and keep the significant flips."""
    policy = make_policy(base_Q, {})
    found = []
    cands = candidates(base_Q)
    for i, c in enumerate(cands):
        c1, c2 = c["cards"]
        for tc in tcs:
            a = measure_action(policy, c1, c2, c["up"], c["base"], tc, hit_soft17, trials)
            b = measure_action(policy, c1, c2, c["up"], c["alt"], tc, hit_soft17, trials)
            if a is None or b is None:
                continue
            se = math.sqrt(2 * 1.25 / trials) * 2          # ~2 standard errors
            if b - a > se:
                found.append({"kind": c["kind"], "total": c["total"], "up": c["up"], "tc": tc,
                              "base": rl.ACTIONS[c["base"]], "counting": rl.ACTIONS[c["alt"]],
                              "gain": b - a})
        if progress:
            progress((i + 1) / len(cands))
    found.sort(key=lambda d: -d["gain"])
    return found
