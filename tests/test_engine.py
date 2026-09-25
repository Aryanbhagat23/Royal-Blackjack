"""The game engine: card arithmetic, dealer rules, payouts and splits."""

import itertools

import blackjack_rl as rl


def card(rank):
    return {"rank": rank, "value": rl.CARD_VALUES[rank]}


def scripted(cards):
    """A draw function that deals the given card values in order."""
    it = iter(cards)
    return lambda _deck: next(it)


def stand(state, legal):
    return 1


def test_hand_values_and_soft_aces():
    assert rl.hand_info([card("A"), card("6")]) == (17, True)
    assert rl.hand_info([card("A"), card("6"), card("10")]) == (17, False)
    assert rl.hand_info([card("A"), card("A"), card("9")]) == (21, True)
    assert rl.is_blackjack([card("A"), card("K")])
    assert not rl.is_blackjack([card("A"), card("5"), card("5")])


def test_dealer_soft_17_rule():
    soft17 = [card("A"), card("6")]
    assert not rl.dealer_should_hit(soft17, hit_soft17=False)
    assert rl.dealer_should_hit(soft17, hit_soft17=True)
    assert not rl.dealer_should_hit([card("10"), card("7")], hit_soft17=True)


def test_add_never_leaves_a_busted_soft_hand():
    for cards in itertools.product(rl.DRAW[:10], repeat=3):
        t, s = 0, 0
        for c in cards:
            t, s = rl._add(t, s, c)
        assert t <= 21 or s == 0


def test_round_payouts():
    # player 10+8 stands on 18; dealer 10+7 stands on 17 -> win
    assert rl._play_round(stand, 10, 7, 10, 8, False) == [1]
    # push
    assert rl._play_round(stand, 10, 8, 10, 8, False) == [0]
    # dealer 6+10 draws a 10 and busts
    assert rl._play_round(stand, 6, 10, 10, 2, False, draw=scripted([10])) == [1]


def test_double_pays_twice():
    double_then_stand = lambda s, legal: 2 if 2 in legal else 1
    # 6+5=11, doubles and draws a 10 -> 21; dealer 10+7=17 -> +2
    assert rl._play_round(double_then_stand, 10, 7, 6, 5, False, draw=scripted([10])) == [2]


def test_split_aces_get_one_card_each():
    split_first = lambda s, legal: 3 if 3 in legal else 0      # would keep hitting if allowed
    pays = rl._play_round(split_first, 10, 7, 11, 11, False, draw=scripted([10, 9]))
    assert pays == [1, 1]                                       # A+10=21 and A+9=20 both stand vs 17


def test_greedy_uses_q_and_default():
    Q = {(16, 10, False, True, 0): [-0.54, -0.55, -1.07, 0.0]}
    assert rl.greedy(Q, (16, 10, False, True, 0), [0, 1, 2]) == 0
    assert rl.greedy({}, (18, 10, False, False, 0), [0, 1]) == 1


def test_training_is_reproducible_with_a_seed():
    a, b = {}, {}
    rl.train(20_000, Q=a, N={}, seed=11)
    rl.train(20_000, Q=b, N={}, seed=11)
    assert a == b


def test_evaluate_reports_standard_error():
    e = rl.evaluate(None, 20_000, policy=rl.basic_strategy_policy(), seed=3)
    assert 0.005 < e["stderr"] < 0.012
