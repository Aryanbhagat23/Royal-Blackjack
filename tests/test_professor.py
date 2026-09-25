"""Chat helpers: hand parsing, history cleanup, and the feedback bandit."""

import random

import llm
import professor_rl as prl


def test_parse_hands():
    kinds = lambda q: [([c["rank"] for c in cards], up["rank"]) for cards, up in llm.parse_hands(q)]
    assert kinds("should I hit 16 vs 10?") == [(["10", "6"], "10")]
    assert kinds("pair of 8s against a 6") == [(["8", "8"], "6")]
    assert kinds("A,7 vs 3") == [(["A", "7"], "3")]
    assert kinds("soft 18 against the dealer's 9") == [(["A", "7"], "9")]
    assert kinds("explain monte carlo") == []


def test_clean_history_alternates_and_starts_with_user():
    h = llm.clean_history([{"role": "assistant", "content": "hi"}, {"role": "user", "content": "a"},
                           {"role": "user", "content": "b"}, {"role": "assistant", "content": ""}])
    assert h == [{"role": "user", "content": "a\n\nb"}]


def test_gemini_ranking_prefers_newest_stable_flash():
    ranked = llm.rank_gemini(["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash",
                              "gemini-3.5-pro-preview"])
    assert ranked[:2] == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


def test_bandit_learns_the_liked_style(tmp_path, monkeypatch):
    monkeypatch.setattr(prl, "FEEDBACK_PATH", str(tmp_path / "fb.json"))
    rng = random.Random(0)
    for _ in range(200):
        s = prl.choose_style(rng)
        prl.record(s, s == "numbers", "q", "a")
    top = prl.stats()[0]
    assert top[0] == prl.STYLES["numbers"][0]
