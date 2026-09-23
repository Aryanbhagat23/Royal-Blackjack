"""Casino Advisor — enter your cards and the dealer's up card, get the best move."""

import streamlit as st

import blackjack_rl as rl
from shared import (DEALERS, LEVELS, bust_chance, dealer_bust_chance, explain, get_agents, make_card,
                    page_header, render_html, strategy_card_html, theme, value_bars)

AGENTS = get_agents()
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]  # 10 also covers J, Q, K
ss = st.session_state
ss.setdefault("adv_hand", [])

t = theme()
st.markdown(f"<style>:root {{ --felt1: {t['felt'][0]}; --felt3: {t['felt'][2]}; --rail: {t['rail'][0]}; --face: {t['face']}; --faceink: {t['face_black']}; }}</style>",
            unsafe_allow_html=True)
st.markdown("""
<style>
.felt { background: radial-gradient(ellipse at center, var(--felt1), var(--felt3)); border: 8px solid var(--rail); border-radius: 18px;
  padding: 14px; display: flex; gap: 40px; justify-content: center; flex-wrap: wrap; margin: 8px 0 16px; }
.spot { text-align: center; } .spot b { display: block; color: var(--acc) !important; font-family: var(--head); margin-bottom: 6px; }
.row { display: flex; justify-content: center; min-height: 96px; padding-left: 28px; }
.c { color: var(--faceink) !important; width: 66px; height: 94px; margin-left: -28px; background: var(--face); border-radius: 8px; border: 1px solid #ccc;
  box-shadow: 2px 3px 8px rgba(0,0,0,.5); display: flex; align-items: center; justify-content: center;
  font: bold 26px Arial; }
.c.empty { background: transparent; border: 2px dashed rgba(255,255,255,.35); box-shadow: none; }
.verdict { text-align: center; border-radius: 16px; padding: 16px; margin: 6px 0 12px; box-shadow: 0 6px 20px rgba(0,0,0,.5); }
.verdict .big { font: 900 46px Georgia; letter-spacing: 4px; color: #111 !important; }
.verdict .small { color: #222 !important; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

page_header("🧭 Casino Advisor", "At a real table? Tap the dealer's card and yours, and the Professor tells you the best move.")


# -----------------------------
# Inputs
# -----------------------------

rule = st.segmented_control(
    "Table rule (printed on the felt)", list(DEALERS), default="S17", key="adv_rule",
    format_func=lambda k: "Dealer stands on 17" if k == "S17" else "Dealer hits soft 17",
) or "S17"

up_rank = st.pills("① Dealer's up card", RANKS, key="adv_up")


def add_card():
    if ss.adv_pick:
        ss.adv_hand.append(ss.adv_pick)
    ss.adv_pick = None


st.pills("② Your cards: tap each card in order (10 = any 10, J, Q or K)", RANKS,
         key="adv_pick", on_change=add_card)

b1, b2, b3 = st.columns(3)
if b1.button("↩️ Undo last card", width="stretch") and ss.adv_hand:
    ss.adv_hand.pop()
    st.rerun()
if b2.button("🗑️ Clear hand", width="stretch"):
    ss.adv_hand = []
    st.rerun()
doubling_allowed = b3.toggle("Doubling allowed", value=True,
                             help="Some casinos only allow doubling on certain totals")

hand = [make_card(r) for r in ss.adv_hand]


def cards_row(ranks, slots=2):
    cards = "".join(f'<div class="c">{r}</div>' for r in ranks)
    cards += '<div class="c empty"></div>' * max(0, slots - len(ranks))
    return f'<div class="row">{cards}</div>'


total_txt = ""
if hand:
    t, soft = rl.hand_info(hand)
    total_txt = f" · {'soft ' if soft and t < 21 else ''}{t}"
st.markdown(
    f'<div class="felt"><div class="spot"><b>DEALER</b>{cards_row([up_rank] if up_rank else [], 1)}</div>'
    f'<div class="spot"><b>YOU{total_txt}</b>{cards_row(ss.adv_hand)}</div></div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Advice
# -----------------------------

if not up_rank or len(hand) < 2:
    st.info("Pick the dealer's up card and at least your first two cards.")
else:
    up = make_card(up_rank)
    total, soft = rl.hand_info(hand)
    if total > 21:
        st.error(f"💥 That's {total}, a bust. Tap **Clear hand** for the next round.")
    elif rl.is_blackjack(hand):
        st.success("🃏 **Blackjack!** Nothing to decide; it pays 3 to 2 unless the dealer also has one.")
    elif total == 21:
        st.success("✋ You have 21. **Stand**, always.")
    else:
        can_double = len(hand) == 2 and doubling_allowed
        splittable = len(hand) == 2 and rl.is_pair(hand)
        best, values = rl.best_action(AGENTS[("expert", rule)], hand, up, can_double, splittable)
        bust = bust_chance(hand)
        dbust = dealer_bust_chance(up_rank, DEALERS[rule]["hit_soft17"])

        colors = {"hit": "#f4b6b6", "stand": "#b9e4bf", "double": "#b9ccf2", "split": "#f5dc8a"}
        icons = {"hit": "🃏 HIT", "stand": "✋ STAND", "double": "⏬ DOUBLE", "split": "✂️ SPLIT"}
        st.markdown(
            f'<div class="verdict" style="background:{colors[best]}"><div class="big">{icons[best]}</div>'
            f'<div class="small">Recommended by the RL agent ({LEVELS["expert"]:,} hands of training)</div></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎓 Why")
            st.markdown(explain(best, hand, up, bust, dbust))
            if best == "split":
                st.caption("👉 After splitting, clear the hand and enter each new hand separately "
                           "(your split card plus the new card you're dealt).")
            if best == "double" and len(hand) == 2 and not doubling_allowed:
                st.caption("Doubling is off, so the Professor picked the best non-double move.")
        with c2:
            st.markdown("#### 📊 The numbers")
            st.markdown(f"🎯 Bust chance if you hit: **{bust:.0%}**  \n"
                        f"💥 Dealer bust chance: **{dbust:.0%}**")
            st.markdown(value_bars(values, best), unsafe_allow_html=True)

        if best == "hit":
            st.caption("👉 After you get your new card, tap it above and the Professor will update the advice.")


# -----------------------------
# Printable strategy card
# -----------------------------

st.divider()
st.markdown("### 🖨️ Printable strategy card")
st.write("Many casinos don't allow phones at the table, but a printed strategy card is usually fine. "
         "This one is generated from what the AI actually learned.")
card_html = strategy_card_html(AGENTS[("expert", rule)], DEALERS[rule])
st.download_button("⬇️ Download strategy card (open it and print)", card_html,
                   file_name=f"blackjack_strategy_{rule}.html", mime="text/html", width="stretch")
with st.expander("Preview the card"):
    render_html(card_html, 820)
