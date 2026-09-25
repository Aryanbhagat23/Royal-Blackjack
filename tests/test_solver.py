"""The exact solver is the answer key, so it is checked against the engine and known facts."""

import math

import blackjack_rl as rl
import solver as S


def test_dealer_distribution_sums_to_one():
    for up in S.UPCARDS:
        for hs17 in (False, True):
            assert abs(sum(S.dealer_distribution(up, hs17).values()) - 1) < 1e-12


def test_well_known_decisions():
    best = lambda st: S.NAMES[S.best_action(st)[0]]
    assert best((16, 10, False, True, 0)) == "hit"
    assert best((11, 6, False, True, 0)) == "double"
    assert best((16, 10, False, True, 8)) == "split"            # always split 8s
    assert best((12, 10, True, True, 11)) == "split"            # always split aces
    assert best((20, 6, False, True, 10)) == "stand"            # never split tens
    assert best((12, 4, False, False, 0)) == "stand"


def test_house_edge_is_in_the_known_range():
    s17, h17 = S.house_edge(S.Rules()), S.house_edge(S.Rules(hit_soft17=True))
    assert -0.0070 < s17 < -0.0045
    assert h17 < s17                                            # hitting soft 17 is worse for the player
    assert S.house_edge(S.Rules(blackjack_pays=1.2)) < s17 - 0.012   # 6:5 costs about 1.4%
    assert S.house_edge(S.Rules(surrender=True)) > s17          # surrender helps the player
    assert S.house_edge(S.Rules(double_after_split=False)) < s17


def test_optimal_policy_value_equals_house_edge():
    rules = S.Rules()
    assert abs(S.policy_ev(S.optimal_policy(rules), rules) - S.house_edge(rules)) < 1e-12


def test_regret_of_perfect_play_is_zero_and_never_negative():
    r = S.regret(S.optimal_policy())
    assert abs(r["regret"]) < 1e-12 and r["optimal_share"] == 1.0
    chart = S.regret(rl.basic_strategy_policy())
    assert chart["regret"] >= 0


def test_exact_value_matches_simulation():
    """Exact EV of basic strategy vs 400k simulated hands of the real engine (within 4 standard errors)."""
    policy = rl.basic_strategy_policy()
    e = rl.evaluate(None, 400_000, policy=policy, seed=5)
    assert abs(e["avg_reward"] - S.policy_ev(policy)) < 4 * e["stderr"]


def test_state_value_matches_engine_simulation():
    """Standing on hard 16 vs a 10, played through the engine."""
    import random
    random.seed(9)
    total = n = 0
    while n < 200_000:
        hole = random.choice(rl.DRAW)
        if hole == 11:
            continue                                            # dealer would have blackjack
        total += rl._play_round(lambda s, l: 1, 10, hole, 10, 6, False)[0]
        n += 1
    se = 1 / math.sqrt(n)
    assert abs(total / n - S.stand_ev(16, 10)) < 4 * se
