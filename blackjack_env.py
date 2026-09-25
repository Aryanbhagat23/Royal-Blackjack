"""
A full-rules Blackjack environment with the Gymnasium API
========================================================
Gymnasium's built-in ``Blackjack-v1`` only allows hit and stand. This environment plays the same game
as the agents in this project, including **double** and **split**, so any RL library (Stable-Baselines3,
RLlib, CleanRL, your own code) can be trained here and graded exactly with ``solver.py``.

    from blackjack_env import BlackjackEnv
    env = BlackjackEnv(hit_soft17=False)
    obs, info = env.reset(seed=0)
    while True:
        action = env.action_space.sample(mask=info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

Observation: (player_total, dealer_up_card, usable_ace, can_double, pair_value) — the same tuple the
project's agents use, so a Q-table learned here plugs straight into ``solver.regret``.
Actions: 0 = hit, 1 = stand, 2 = double, 3 = split. ``info["action_mask"]`` marks the legal ones; an
illegal double or split is played as a hit (and flagged in ``info["illegal_action"]``).
Reward: money won per unit bet, paid when the round ends (a split round pays both hands together).
Rules: infinite deck, dealer peeks for blackjack, double on any two cards (also after a split), one
split per round, split aces get one card, blackjack pays 3:2.

By default rounds where someone has a natural blackjack are skipped (there is no decision to make),
which matches how the agents are trained and how ``solver`` values are defined. Pass
``skip_naturals=False`` to play every round; a natural then ends the round on the first step.

If Gymnasium is installed the class is a real ``gymnasium.Env`` with ``spaces``; otherwise it offers
the same methods and a tiny stand-in for ``action_space`` so the project has no extra dependency.
"""

import random

import blackjack_rl as rl

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:                                   # optional dependency
    gym = spaces = None
    _Base = object


class _Discrete:
    """Minimal stand-in for gymnasium.spaces.Discrete when Gymnasium isn't installed."""

    def __init__(self, n, rng):
        self.n, self._rng = n, rng

    def sample(self, mask=None):
        options = [i for i in range(self.n) if mask is None or mask[i]]
        return self._rng.choice(options)

    def contains(self, x):
        return isinstance(x, int) and 0 <= x < self.n


class BlackjackEnv(_Base):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, hit_soft17=False, skip_naturals=True, render_mode=None):
        self.hit_soft17 = hit_soft17
        self.skip_naturals = skip_naturals
        self.render_mode = render_mode
        self._rng = random.Random()
        if spaces is not None:
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(12), spaces.Discrete(2),
                                                   spaces.Discrete(2), spaces.Discrete(12)))
        else:
            self.action_space = _Discrete(4, self._rng)
            self.observation_space = None

    # ---------------- helpers ----------------
    def _draw(self):
        return self._rng.choice(rl.DRAW)

    def _hand_state(self):
        h = self._hands[self._i]
        can_d = h["n"] == 2
        can_s = h["n"] == 2 and not self._split and h["cards"][0] == h["cards"][1]
        return (h["total"], self._up, h["soft"] > 0, can_d, h["cards"][0] if can_s else 0)

    def _info(self, **extra):
        if self._done:
            mask = [0, 0, 0, 0]
        else:
            _, _, _, can_d, pair = self._hand_state()
            mask = [1, 1, int(can_d), int(bool(pair))]
        return {"action_mask": mask, "hand": self._i, "hands": len(self._hands), **extra}

    def _new_hand(self, c1, c2):
        t, s = rl._add(*rl._add(0, 0, c1), c2)
        return {"cards": [c1, c2], "total": t, "soft": s, "n": 2, "bet": 1, "done": False}

    # ---------------- Gymnasium API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        if spaces is not None:
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
        while True:
            p1, p2, up, hole = (self._draw() for _ in range(4))
            natural = rl._natural(p1, p2) or rl._natural(up, hole)
            if natural and self.skip_naturals:
                continue
            break
        self._up, self._hole = up, hole
        self._hands = [self._new_hand(p1, p2)]
        self._i, self._split, self._done = 0, False, False
        self._natural = None
        if natural:
            pn, dn = rl._natural(p1, p2), rl._natural(up, hole)
            self._natural = 0.0 if pn and dn else (1.5 if pn else -1.0)
        return self._hand_state(), self._info(natural=natural)

    def step(self, action):
        if self._done:
            raise RuntimeError("The round is over. Call reset().")
        if self._natural is not None:
            self._done = True
            return self._hand_state(), self._natural, True, False, self._info()
        action = int(action)
        _, _, _, can_d, pair = self._hand_state()
        illegal = (action == 2 and not can_d) or (action == 3 and not pair) or action not in (0, 1, 2, 3)
        if illegal:
            action = 0
        h = self._hands[self._i]
        if action == 3:
            c = h["cards"][0]
            self._split = True
            self._hands = [self._new_hand(c, self._draw()), self._new_hand(c, self._draw())]
            if c == 11:                                  # split aces: one card each, then stand
                for hand in self._hands:
                    hand["done"] = True
        elif action == 1:
            h["done"] = True
        else:
            card = self._draw()
            h["cards"].append(card)
            h["total"], h["soft"] = rl._add(h["total"], h["soft"], card)
            h["n"] += 1
            if action == 2:
                h["bet"] = 2
                h["done"] = True
            elif h["total"] >= 21:
                h["done"] = True
        while self._i < len(self._hands) and self._hands[self._i]["done"]:
            self._i += 1
        if self._i < len(self._hands):
            return self._hand_state(), 0.0, False, False, self._info(illegal_action=illegal)
        self._i = len(self._hands) - 1
        reward = float(sum(self._settle()))
        self._done = True
        return self._hand_state(), reward, True, False, self._info(illegal_action=illegal)

    def _settle(self):
        if all(h["total"] > 21 for h in self._hands):
            return [-h["bet"] for h in self._hands]
        d, ds = rl._add(*rl._add(0, 0, self._up), self._hole)
        while d < 17 or (self.hit_soft17 and d == 17 and ds):
            d, ds = rl._add(d, ds, self._draw())
        self._dealer_total = d
        out = []
        for h in self._hands:
            t, bet = h["total"], h["bet"]
            out.append(-bet if t > 21 else bet if d > 21 or t > d else -bet if t < d else 0)
        return out

    def render(self):
        names = {11: "A"}
        hands = " | ".join("+".join(names.get(c, str(c)) for c in h["cards"]) + f" ({h['total']})"
                           for h in self._hands)
        return f"Dealer shows {names.get(self._up, self._up)} · You: {hands}"


def register():
    """Register as 'RoyalBlackjack-v0' with Gymnasium (if installed)."""
    if gym is not None:
        gym.register(id="RoyalBlackjack-v0", entry_point="blackjack_env:BlackjackEnv")
