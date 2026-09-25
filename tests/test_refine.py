"""Precision refinement settles decisions from experience alone."""

import blackjack_rl as rl
import solver as S


def test_refine_settles_a_clear_decision_correctly():
    Q, N = {}, {}
    rl.train(30_000, Q=Q, N=N, seed=1)
    stats, report = rl.refine(Q, N, budget=30_000, seed=2)
    assert report["deals"] <= 30_000 + 400
    # 11 vs 6 is a clear double; after targeted practice the agent must know it
    st = (11, 6, False, True, 0)
    assert rl.greedy(Q, st, [0, 1, 2]) == S.best_action(st)[0] == 2


def test_first_and_later_decisions_cover_every_choice():
    first = rl.first_decisions()
    assert len(first) == 330 and len({s for s, _ in first}) == 330
    for state, (c1, c2) in first:
        t, s = rl._add(*rl._add(0, 0, c1), c2)
        assert (t, s > 0) == (state[0], state[2])
    assert len(rl.later_decisions()) == 170
