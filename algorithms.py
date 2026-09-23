"""
Algorithm comparison
--------------------
The main agent (`blackjack_rl.py`) learns with **Monte Carlo control**. This module implements three
alternatives on exactly the same game, so they can be raced against each other:

* **Q-learning** — off-policy temporal-difference learning. Updates after every move using its own
  best estimate of the next state, instead of waiting for the hand to finish.
* **SARSA** — on-policy temporal-difference learning. Same idea, but it updates toward the action it
  actually takes next, including exploratory ones, so it learns a slightly more cautious strategy.
* **Deep Q-Network (DQN)** — the same Q-learning idea, but instead of a lookup table it uses a small
  neural network (written here with numpy alone, no PyTorch), trained from a replay buffer with a
  target network. Blackjack is small enough that a table wins easily, which is exactly the point:
  neural networks are for games too big to tabulate.

Every learner is scored the same two ways: how often it matches the official basic strategy chart,
and how much money it wins or loses per $100 bet.

Run directly to produce the comparison used by the app:
    python algorithms.py
"""

import json
import math
import os
import random

import numpy as np

import blackjack_rl as rl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "algorithm_race.json")
ALGORITHMS = ["Monte Carlo", "Q-learning", "SARSA", "Deep Q-Network"]


# =========================================================
# Shared environment helpers (one decision step at a time)
# =========================================================

def deal():
    p1, p2, up, hole = rl._deal()
    return p1, p2, up, hole


def start_hand(hit_soft17):
    """Deal until there is a decision to make. Returns (state, context)."""
    while True:
        p1, p2, up, hole = deal()
        if rl._natural(p1, p2) or rl._natural(up, hole):
            continue
        t, s = rl._add(*rl._add(0, 0, p1), p2)
        state = (t, up, s > 0, True, p1 if p1 == p2 else 0)
        return state, {"up": up, "hole": hole, "soft": s, "n": 2, "bet": 1, "hs17": hit_soft17,
                       "cards": (p1, p2)}


def step(state, action, ctx):
    """Take one action. Returns (reward, next_state or None if the hand is over)."""
    t, up, soft_flag, can_d, pair = state
    s, hole, hs17 = ctx["soft"], ctx["hole"], ctx["hs17"]

    if action == 3:                                    # split: play it out with the table policy
        payout = rl._play_round(lambda st, legal: rl.greedy(ctx["ref_Q"], st, legal),
                                up, hole, *ctx["cards"], hs17)
        return sum(payout), None
    if action == 1 or action == 2:                     # stand or double: the hand ends
        bet = 1
        if action == 2:
            t, s = rl._add(t, s, random.choice(rl.DRAW))
            bet = 2
            if t > 21:
                return -bet, None
        d, ds = rl._add(*rl._add(0, 0, up), hole)
        while d < 17 or (hs17 and d == 17 and ds):
            d, ds = rl._add(d, ds, random.choice(rl.DRAW))
        if d > 21 or t > d:
            return bet, None
        if t < d:
            return -bet, None
        return 0, None
    # hit
    t, s = rl._add(t, s, random.choice(rl.DRAW))
    ctx["soft"], ctx["n"] = s, ctx["n"] + 1
    if t > 21:
        return -1, None
    if t == 21:
        return step((t, up, s > 0, False, 0), 1, ctx)
    return 0, (t, up, s > 0, False, 0)


def legal_for(state):
    return rl.legal_actions(state[3], state[4] > 0)


# =========================================================
# Tabular temporal-difference learners
# =========================================================

def train_td(steps=3_000_000, hit_soft17=False, sarsa=False, alpha=0.05, eps_end=0.12, ref_Q=None,
             progress=None, checkpoints=None, on_checkpoint=None):
    """Q-learning (off-policy) or SARSA (on-policy) with a decaying learning rate and epsilon."""
    Q, N = {}, {}
    marks = set(checkpoints or [])

    def q(s):
        if s not in Q:
            Q[s] = [0.0] * 4
            N[s] = [0] * 4
        return Q[s]

    def pick(s, eps):
        legal = legal_for(s)
        if random.random() < eps:
            return random.choice(legal)
        vals = q(s)
        return max(legal, key=vals.__getitem__)

    for i in range(steps):
        eps = max(eps_end, 1.0 - i / (steps * 0.6))
        state, ctx = start_hand(hit_soft17)
        ctx["ref_Q"] = ref_Q or {}
        action = pick(state, eps)
        while True:
            reward, nxt = step(state, action, ctx)
            cur = q(state)
            n = N[state]
            n[action] += 1
            # Robbins-Monro decay: big steps early, small ones later. A fixed step size never
            # settles — the estimate keeps bouncing by several tenths forever.
            step_size = max(0.002, 1.0 / (n[action] ** 0.8))
            if nxt is None:
                target = reward
                cur[action] += step_size * (target - cur[action])
                break
            nxt_action = pick(nxt, eps)
            future = q(nxt)[nxt_action] if sarsa else max(q(nxt)[a] for a in legal_for(nxt))
            cur[action] += step_size * (reward + future - cur[action])
            state, action = nxt, nxt_action
        if i + 1 in marks and on_checkpoint:
            on_checkpoint(i + 1, dict(Q))
        if progress and i % 20_000 == 0:
            progress(i / steps)
    return Q


# =========================================================
# Deep Q-Network (numpy, no deep-learning library)
# =========================================================

def features(state):
    """Turn a state into numbers the network can read: 24 inputs."""
    t, up, soft, can_d, pair = state
    x = np.zeros(24, dtype=np.float32)
    x[0] = (t - 4) / 17.0
    x[1] = (up - 2) / 9.0
    x[2] = 1.0 if soft else 0.0
    x[3] = 1.0 if can_d else 0.0
    x[4] = 1.0 if pair else 0.0
    x[5 + min(max(t - 4, 0), 17)] = 1.0          # one-hot bucket for the total
    x[23] = (pair - 2) / 9.0 if pair else 0.0
    return x


class MLP:
    """Two hidden layers, ReLU, trained with plain gradient descent (Adam)."""

    def __init__(self, n_in=24, hidden=64, n_out=4, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, math.sqrt(2 / n_in), (n_in, hidden)).astype(np.float32),
                  rng.normal(0, math.sqrt(2 / hidden), (hidden, hidden)).astype(np.float32),
                  rng.normal(0, math.sqrt(2 / hidden), (hidden, n_out)).astype(np.float32)]
        self.b = [np.zeros(hidden, np.float32), np.zeros(hidden, np.float32), np.zeros(n_out, np.float32)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        h1 = np.maximum(x @ self.W[0] + self.b[0], 0)
        h2 = np.maximum(h1 @ self.W[1] + self.b[1], 0)
        return h2 @ self.W[2] + self.b[2], (x, h1, h2)

    def predict(self, x):
        return self.forward(x)[0]

    def train_step(self, x, actions, targets, lr=1e-3):
        out, (x0, h1, h2) = self.forward(x)
        grad_out = np.zeros_like(out)
        rows = np.arange(len(x))
        grad_out[rows, actions] = 2 * (out[rows, actions] - targets) / len(x)
        gW2 = h2.T @ grad_out
        gb2 = grad_out.sum(0)
        d2 = (grad_out @ self.W[2].T) * (h2 > 0)
        gW1 = h1.T @ d2
        gb1 = d2.sum(0)
        d1 = (d2 @ self.W[1].T) * (h1 > 0)
        gW0 = x0.T @ d1
        gb0 = d1.sum(0)
        self.t += 1
        for i, (gw, gb) in enumerate(((gW0, gb0), (gW1, gb1), (gW2, gb2))):
            for arr, m, v, g in ((self.W, self.mW, self.vW, gw), (self.b, self.mb, self.vb, gb)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mhat = m[i] / (1 - 0.9 ** self.t)
                vhat = v[i] / (1 - 0.999 ** self.t)
                arr[i] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        return float(((out[rows, actions] - targets) ** 2).mean())


class DQNTable:
    """Wraps a trained network so it can be read like a Q-table by the rest of the project."""

    def __init__(self, net):
        self.net = net
        self.cache = {}

    def get(self, state, default=None):
        if state not in self.cache:
            self.cache[state] = [float(v) for v in self.net.predict(features(state)[None])[0]]
        return self.cache[state]

    def __contains__(self, state):
        return True

    def __getitem__(self, state):
        return self.get(state)


def train_dqn(steps=300_000, hit_soft17=False, hidden=64, batch=64, buffer_size=50_000,
              target_sync=1_000, lr=1e-3, ref_Q=None, progress=None, checkpoints=None,
              on_checkpoint=None, seed=0):
    """Deep Q-learning with a replay buffer and a target network."""
    net, target = MLP(hidden=hidden, seed=seed), MLP(hidden=hidden, seed=seed)
    target.W = [w.copy() for w in net.W]
    target.b = [b.copy() for b in net.b]
    buf_x = np.zeros((buffer_size, 24), np.float32)
    buf_a = np.zeros(buffer_size, np.int64)
    buf_r = np.zeros(buffer_size, np.float32)
    buf_nx = np.zeros((buffer_size, 24), np.float32)
    buf_done = np.zeros(buffer_size, np.float32)
    buf_mask = np.zeros((buffer_size, 4), np.float32)
    filled = idx = 0
    marks = set(checkpoints or [])

    for i in range(steps):
        eps = max(0.05, 1.0 - i / (steps * 0.6))
        state, ctx = start_hand(hit_soft17)
        ctx["ref_Q"] = ref_Q or {}
        while True:
            legal = legal_for(state)
            if random.random() < eps:
                action = random.choice(legal)
            else:
                qs = net.predict(features(state)[None])[0]
                action = max(legal, key=lambda a: qs[a])
            reward, nxt = step(state, action, ctx)
            buf_x[idx] = features(state)
            buf_a[idx] = action
            buf_r[idx] = reward
            buf_done[idx] = 1.0 if nxt is None else 0.0
            buf_nx[idx] = 0 if nxt is None else features(nxt)
            mask = np.full(4, -1e9, np.float32)
            if nxt is not None:
                for a in legal_for(nxt):
                    mask[a] = 0.0
            buf_mask[idx] = mask
            idx = (idx + 1) % buffer_size
            filled = min(filled + 1, buffer_size)
            if nxt is None:
                break
            state = nxt

        if filled >= batch * 8:
            pick = np.random.randint(0, filled, batch)
            nq = target.predict(buf_nx[pick]) + buf_mask[pick]
            targets = buf_r[pick] + (1 - buf_done[pick]) * nq.max(1)
            net.train_step(buf_x[pick], buf_a[pick], targets, lr)
        if i % target_sync == 0:
            target.W = [w.copy() for w in net.W]
            target.b = [b.copy() for b in net.b]
        if i + 1 in marks and on_checkpoint:
            on_checkpoint(i + 1, DQNTable(MLP_copy(net)))
        if progress and i % 2_000 == 0:
            progress(i / steps)
    return DQNTable(net)


def MLP_copy(net):
    clone = MLP(hidden=net.W[0].shape[1])
    clone.W = [w.copy() for w in net.W]
    clone.b = [b.copy() for b in net.b]
    return clone


# =========================================================
# Scoring and the race
# =========================================================

def score(Q, hit_soft17=False, hands=60_000):
    g = rl.grade(Q, None, hit_soft17)
    e = rl.evaluate(Q, hands, hit_soft17)
    return {"match": g["matches"] / g["total"], "per_100": e["avg_reward"] * 100}


def race(budget=1_500_000, dqn_budget=200_000, hit_soft17=False, points=6, eval_hands=60_000,
         progress=None):
    """Train all four learners with checkpoints and score each one along the way."""
    ref = rl.load_bundle(os.path.join(BASE_DIR, "q_expert_S17.pkl" if not hit_soft17 else "q_expert_H17.pkl"))
    ref_Q = rl.robust_q(ref["Q"], ref["N"])
    marks = [int(budget * (k + 1) / points) for k in range(points)]
    dqn_marks = [int(dqn_budget * (k + 1) / points) for k in range(points)]
    out = {name: [] for name in ALGORITHMS}
    done = [0]
    total_jobs = 4

    def note(job):
        done[0] += 1
        if progress:
            progress(done[0] / total_jobs)

    # Monte Carlo (the project's main algorithm)
    Q, N = {}, {}
    for k, m in enumerate(marks):
        chunk = m - (marks[k - 1] if k else 0)
        rl.train(chunk, hit_soft17, Q, N)
        out["Monte Carlo"].append({"hands": m, **score(Q, hit_soft17, eval_hands)})
    note("mc")

    for name, sarsa in (("Q-learning", False), ("SARSA", True)):
        results = []
        train_td(budget, hit_soft17, sarsa=sarsa, ref_Q=ref_Q, checkpoints=marks,
                 on_checkpoint=lambda m, q, r=results: r.append({"hands": m, **score(q, hit_soft17, eval_hands)}))
        out[name] = results
        note(name)

    results = []
    train_dqn(dqn_budget, hit_soft17, ref_Q=ref_Q, checkpoints=dqn_marks,
              on_checkpoint=lambda m, q, r=results: r.append({"hands": m, **score(q, hit_soft17, eval_hands)}))
    out["Deep Q-Network"] = results
    note("dqn")
    return out


def save_race(data, path=RESULTS_PATH):
    with open(path, "w") as f:
        json.dump(data, f)


def load_race(path=RESULTS_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    import time

    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
    t0 = time.time()
    data = race(budget)
    save_race(data)
    print(f"done in {time.time() - t0:.0f}s")
    for name, rows in data.items():
        last = rows[-1]
        print(f"{name:>16}: {last['hands']:>9,} hands → chart match {last['match']:.1%}, "
              f"{last['per_100']:+.2f} per $100")
