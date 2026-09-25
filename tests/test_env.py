"""The Gymnasium environment plays the same game as the engine and the solver."""

import math

import blackjack_env as E
import solver as S


def run(env, policy, rounds, seed):
    env.reset(seed=seed)
    total = sq = 0.0
    for _ in range(rounds):
        obs, info = env.reset()
        while True:
            legal = [a for a in range(4) if info["action_mask"][a]]
            obs, r, term, trunc, info = env.step(policy(obs, legal))
            if term:
                break
        total += r
        sq += r * r
    m = total / rounds
    return m, math.sqrt(max(sq / rounds - m * m, 0) / rounds)


def test_env_matches_exact_values():
    w = S.decision_weights()
    exact = sum(p * max(S.action_values(s).values()) for s, p in w.items()) / sum(w.values())
    m, se = run(E.BlackjackEnv(), S.optimal_policy(), 150_000, seed=1)
    assert abs(m - exact) < 4 * se


def test_seeded_resets_are_reproducible():
    a, b = E.BlackjackEnv(), E.BlackjackEnv()
    assert [a.reset(seed=4)[0] for _ in range(1)] == [b.reset(seed=4)[0] for _ in range(1)]


def test_action_mask_and_illegal_actions():
    env = E.BlackjackEnv()
    obs, info = env.reset(seed=0)
    assert info["action_mask"][:2] == [1, 1]
    obs, r, term, trunc, info = env.step(0)                      # hit
    if not term:
        assert info["action_mask"][2] == 0                      # no double after hitting
        obs, r, term, trunc, info = env.step(2)                  # illegal double becomes a hit
        assert info["illegal_action"]


def test_split_plays_two_hands():
    env = E.BlackjackEnv()
    for seed in range(500):
        obs, info = env.reset(seed=seed)
        if info["action_mask"][3]:
            obs, r, term, trunc, info = env.step(3)
            assert info["hands"] == 2
            return
    raise AssertionError("no pair dealt in 500 rounds")
