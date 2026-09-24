"""Casino Table — full game with AI players, the Professor, themes and card counting."""

import os
import random
import re

import streamlit as st

import blackjack_rl as rl
import counting as C
import llm
from shared import (DEALERS, LEVELS, action_button_css, cards_html, get_bundles, chip_html, explain, get_agents, md_bold, page_header,
                    q_path, render_html, scene_css, scene_effects, theme, value_bars)

NUM_DECKS = 6
START_BANK = 1000

AI_PLAYERS = {
    "veteran": {"name": "Veteran Vic", "emoji": "🎩", "level": "expert", "bet": 50,
                "desc": "30M hands of experience", "short": "30M hands"},
    "rookie": {"name": "Rookie Riley", "emoji": "🧢", "level": "rookie", "bet": 25,
               "desc": "Only 5k hands of experience", "short": "5k hands"},
    "rulebook": {"name": "Rulebook Rita", "emoji": "📏", "level": "rule", "bet": 25,
                 "desc": "No learning: copies the dealer", "short": "no learning"},
}
HILO = {"2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 0, "8": 0, "9": 0,
        "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1}
SUITS = [("♠", False), ("♥", True), ("♣", False), ("♦", True)]

AGENTS = get_agents()


# =========================================================
# State
# =========================================================

def new_shoe():
    shoe = [{"rank": r, "value": v, "symbol": s, "red": red}
            for _ in range(NUM_DECKS) for s, red in SUITS for r, v in rl.CARD_VALUES.items()]
    random.shuffle(shoe)
    return shoe


def init_state():
    ss = st.session_state
    ss.shoe, ss.running_count = new_shoe(), 0
    ss.bankroll, ss.bet = START_BANK, 25
    ss.phase = "betting"                     # betting -> playing -> done
    ss.order, ss.hands, ss.dealer = [], {}, []
    ss.hole_revealed = False
    ss.ai_bank = {k: START_BANK for k in AI_PLAYERS}
    ss.log = []
    ss.prof = {"decisions": 0, "correct": 0, "last": None}
    ss.stats = {"hands": 0, "wins": 0, "losses": 0, "pushes": 0, "blackjacks": 0}
    ss.history = [START_BANK]
    ss.review = []
    ss.anim = {}                             # cards already animated per hand
    ss.toast = None


ss = st.session_state
if "shoe" not in ss:
    init_state()
ss.setdefault("dealer_key", "S17")
ss.setdefault("seats", ["veteran", "rookie", "rulebook"])
ss.setdefault("show_prof", True)
ss.setdefault("show_count", True)


def rule():
    return DEALERS[ss.dealer_key]


@st.cache_data(ttl=60, show_spinner=False)
def llm_status():
    """Which chat providers are usable? Cached so it never slows the table down."""
    return llm.providers()


@st.cache_resource(show_spinner=False)
def counting_bundle(rule_key):
    """The counting agent for this dealer, if one has been trained."""
    try:
        return C.load(C.counting_path(rule_key))
    except (OSError, ValueError, EOFError):
        return None


def add_log(msg):
    ss.log.insert(0, msg)
    del ss.log[15:]


def draw(visible=True):
    card = ss.shoe.pop()
    if visible:
        ss.running_count += HILO[card["rank"]]
    return card


def true_count():
    return ss.running_count / max(len(ss.shoe) / 52, 0.5)


def reveal_hole():
    if not ss.hole_revealed and ss.dealer:
        ss.running_count += HILO[ss.dealer[0]["rank"]]
        ss.hole_revealed = True


# =========================================================
# Game flow
# =========================================================

def new_hand(cards, bet, from_split=False):
    return {"cards": cards, "bet": bet, "status": "playing", "result": "", "payout": 0, "from_split": from_split}


def seat_hands(k):
    return ss.hands[k]["hands"]


def active_hand(k):
    """The hand this seat is currently playing (or its last hand)."""
    seat = ss.hands[k]
    return seat["hands"][min(seat["active"], len(seat["hands"]) - 1)]


def seat_playing(k):
    return any(h["status"] == "playing" for h in seat_hands(k))


def committed(k="you"):
    return sum(h["bet"] for h in seat_hands(k))


def can_split(k, h):
    return (len(h["cards"]) == 2 and not h["from_split"] and len(seat_hands(k)) == 1
            and rl.is_pair(h["cards"]))


def advance_seat(k):
    seat = ss.hands[k]
    while seat["active"] < len(seat["hands"]) and seat["hands"][seat["active"]]["status"] != "playing":
        seat["active"] += 1


def deal_round():
    if len(ss.shoe) < 52 * NUM_DECKS * 0.25:
        ss.shoe, ss.running_count = new_shoe(), 0
        add_log("🔀 Cut card reached. A fresh 6-deck shoe is shuffled.")
        ss.toast = ("🔀 New shoe shuffled, count reset", None)

    ss.order = [k for k in AI_PLAYERS if k in ss.seats] + ["you"]
    ss.hands = {k: {"hands": [new_hand([], ss.bet if k == "you" else AI_PLAYERS[k]["bet"])], "active": 0}
                for k in ss.order}
    ss.dealer, ss.hole_revealed, ss.anim = [], False, {}
    ss.prof["last"] = None
    ss.review = []

    for _ in range(2):
        for k in ss.order:
            seat_hands(k)[0]["cards"].append(draw())
        ss.dealer.append(draw(visible=len(ss.dealer) == 1))  # first dealer card is the hole card

    ss.phase = "playing"
    for k in ss.order:
        h = seat_hands(k)[0]
        if rl.is_blackjack(h["cards"]):
            h["status"] = "blackjack"

    if rl.is_blackjack(ss.dealer):
        reveal_hole()
        add_log(f"{rule()['emoji']} Dealer peeks... Blackjack!")
        settle_all()
        return

    for k in ss.order:                       # AI seats act first (they sit to your right)
        if k != "you" and seat_playing(k):
            ai_play_seat(k)
    if not seat_playing("you"):
        finish_round()


def apply_action(k, action):
    h = active_hand(k)
    if action == "hit":
        h["cards"].append(draw())
        s = rl.score(h["cards"])
        h["status"] = "bust" if s > 21 else "stood" if s == 21 else "playing"
    elif action == "stand":
        h["status"] = "stood"
    elif action == "double":
        h["bet"] *= 2
        h["cards"].append(draw())
        h["status"] = "bust" if rl.score(h["cards"]) > 21 else "stood"
    else:  # split: two hands, one card each, then a new second card for each
        second = h["cards"].pop()
        h["from_split"] = True
        other = new_hand([second], h["bet"], from_split=True)
        seat_hands(k).append(other)
        for hand in (h, other):
            hand["cards"].append(draw())
            if second["rank"] == "A" or rl.score(hand["cards"]) == 21:
                hand["status"] = "stood"       # split Aces get one card; 21 stands automatically
    advance_seat(k)


def seat_best(k, level="expert"):
    h = active_hand(k)
    bank_ok = k != "you" or ss.bankroll >= committed() + h["bet"]
    if level == "rule":
        # The simplest possible model: no learning at all, just "hit until 17", like the dealer.
        return ("stand" if rl.score(h["cards"]) >= 17 else "hit"), {}
    Q = AGENTS[(level, ss.dealer_key)]
    return rl.best_action(Q, h["cards"], ss.dealer[1], len(h["cards"]) == 2 and bank_ok,
                          can_split(k, h) and bank_ok)


def ai_play_seat(k):
    moves = []
    while seat_playing(k):
        action, _ = seat_best(k, AI_PLAYERS[k]["level"])
        moves.append(action)
        apply_action(k, action)
    totals = " & ".join(f"{rl.score(h['cards'])}{' bust' if h['status'] == 'bust' else ''}" for h in seat_hands(k))
    add_log(f"{AI_PLAYERS[k]['emoji']} **{AI_PLAYERS[k]['name']}**: {' → '.join(moves)} ({totals})")


def professor_best():
    return seat_best("you")


def player_action(action):
    best, values = professor_best()
    h = active_hand("you")
    ss.review.append({"total": rl.score(h["cards"]), "soft": rl.hand_info(h["cards"])[1],
                      "cards": [c["rank"] for c in h["cards"]], "action": action, "best": best,
                      "values": dict(values), "bet": h["bet"]})
    ss.prof["decisions"] += 1
    if action == best:
        ss.prof["correct"] += 1
        ss.prof["last"] = ("good", f"✅ Great play! **{action.upper()}** was the best move.")
    else:
        ss.prof["last"] = ("bad", f"⚠️ The Professor would **{best.upper()}** here "
                                  f"({values[best]:+.2f} vs {values.get(action, 0):+.2f} per $1).")
    apply_action("you", action)
    if not seat_playing("you"):
        finish_round()


def finish_round():
    reveal_hole()
    if any(h["status"] == "stood" for k in ss.order for h in seat_hands(k)):
        while rl.dealer_should_hit(ss.dealer, rule()["hit_soft17"]):
            ss.dealer.append(draw())
    settle_all()


def settle_all():
    d, dealer_bj = rl.score(ss.dealer), rl.is_blackjack(ss.dealer)
    for k in ss.order:
        for h in seat_hands(k):
            p, bet, bj = rl.score(h["cards"]), h["bet"], h["status"] == "blackjack"
            if dealer_bj:
                pay, res = (0, "PUSH") if bj else (-bet, "LOSE")
            elif bj:
                pay, res = 1.5 * bet, "BLACKJACK"
            elif h["status"] == "bust":
                pay, res = -bet, "BUST"
            elif d > 21 or p > d:
                pay, res = bet, "WIN"
            elif p < d:
                pay, res = -bet, "LOSE"
            else:
                pay, res = 0, "PUSH"
            h.update(payout=pay, result=res, status="done")
            if k == "you":
                ss.bankroll += pay
                ss.stats["hands"] += 1
                ss.stats["wins" if pay > 0 else "losses" if pay < 0 else "pushes"] += 1
                ss.stats["blackjacks"] += res == "BLACKJACK"
            else:
                ss.ai_bank[k] += pay
    res, pay = my_result()
    ss.history.append(ss.bankroll)
    icon = "🃏" if res == "BLACKJACK" else "🎉" if pay > 0 else "🤝" if pay == 0 else "💸"
    ss.toast = (f"{icon} {res}  {pay:+,.0f}", "balloons" if res == "BLACKJACK" else None)
    add_log(f"{rule()['emoji']} Dealer finishes on **{d}**{' (bust)' if d > 21 else ''}. You: **{res}** {pay:+,.0f}")
    ss.phase = "done"


def what_if(entry):
    """What the alternative move would have produced, using the dealer's real final hand."""
    d = rl.score(ss.dealer)
    dealer_bust = d > 21
    if entry["action"] in ("hit", "double"):
        t = entry["total"]
        if t > 21:
            return None
        if dealer_bust:
            outcome, why = "won", f"the dealer busted with {d}"
        elif t > d:
            outcome, why = "won", f"{t} beats the dealer's {d}"
        elif t < d:
            outcome, why = "lost", f"{t} loses to the dealer's {d}"
        else:
            outcome, why = "pushed", f"{t} ties the dealer's {d}"
        return f"Standing on {t} would have **{outcome}**: {why}."
    return None


def my_result():
    """Overall (label, payout) for your seat, combining split hands."""
    hands = seat_hands("you")
    pay = sum(h["payout"] for h in hands)
    if len(hands) == 1:
        return hands[0]["result"], pay
    return ("WIN" if pay > 0 else "LOSE" if pay < 0 else "PUSH"), pay


# =========================================================
# Table scene (HTML)
# =========================================================

def score_label(cards):
    total, soft = rl.hand_info(cards)
    if rl.is_blackjack(cards):
        return "BLACKJACK"
    return f"{total - 10} / {total}" if soft and total < 21 else str(total)


def seat_html(k, pos, n, seen, delay):
    me = k == "you"
    info = None if me else AI_PLAYERS[k]
    name = "YOU" if me else f'{info["emoji"]} {info["name"]}'
    bank = ss.bankroll if me else ss.ai_bank[k]
    blurb = info["short"] if (info and n >= 4) else (info["desc"] if info else "")
    sub = f"Bankroll ${bank:,.0f}" if me else f'{blurb} · ${bank:,.0f}'
    seat = ss.hands.get(k) if ss.phase != "betting" else None
    mid = (n - 1) / 2
    lift = 0 if n == 1 else 14 - 14 * abs(pos - mid) / mid
    cls = "seat me" if me else "seat"
    if seat and me and seat_playing(k):
        cls += " active"
    if seat:
        groups = ""
        multi = len(seat["hands"]) > 1
        for i, h in enumerate(seat["hands"]):
            key = f"{k}:{i}"
            body, delay = cards_html(h["cards"], prev=ss.anim.get(key, 0), delay_start=delay)
            seen[key] = len(h["cards"])
            badge = ""
            if h["result"]:
                kind = "win" if h["payout"] > 0 else "lose" if h["payout"] < 0 else "push"
                badge = f'<div class="badge {kind}">{h["result"]}<small>{h["payout"]:+,.0f}</small></div>'
            focus = " focus" if multi and me and h["status"] == "playing" and i == seat["active"] else ""
            groups += (f'<div class="hgroup{focus}">{badge}<div class="hand">{body}</div>'
                       f'<div class="pill">{score_label(h["cards"])}</div>{chip_html(h["bet"]) if multi else ""}</div>')
        bet = sum(h["bet"] for h in seat["hands"])
        spot = "" if multi else f'<div class="betspot">{chip_html(bet)}</div>'
        body_html = f'<div class="hands">{groups}</div>{spot}'
    else:
        bet = ss.bet if me else info["bet"]
        body_html = (f'<div class="hands"><div class="hgroup"><div class="hand"><div class="ghost"></div><div class="ghost"></div></div>'
                     f'<div class="pill dim">—</div></div></div><div class="betspot">{chip_html(bet)}</div>')
    html = (f'<div class="{cls}" style="transform:translateY({lift:.0f}px)">{body_html}'
            f'<div class="name">{name}</div><div class="sub">{sub}</div></div>')
    return html, delay


def render_table():
    t = theme()
    d = rule()
    hide = ss.phase == "playing"
    seen, delay = {}, 0.0

    if ss.dealer:
        flip = not hide and ss.anim.get("hole_hidden", False)
        dcards, delay = cards_html(ss.dealer, prev=ss.anim.get("dealer", 0), hide_first=hide, flip_first=flip)
        seen["dealer"] = len(ss.dealer)
        dscore = f"SHOWING {rl.score(ss.dealer[1:])}" if hide else score_label(ss.dealer)
        dpill = f'<div class="pill">{dscore}</div>'
    else:
        dcards, dpill = '<div class="ghost"></div><div class="ghost"></div>', '<div class="pill dim">—</div>'

    ai = [k for k in AI_PLAYERS if k in ss.seats]
    display = ai[:1] + ["you"] + ai[1:]          # you sit in the middle of the arc
    SEAT_SIZES = {1: (200, 62, 88), 2: (186, 60, 86), 3: (162, 56, 80), 4: (132, 48, 68)}
    seat_w, card_w, card_h = SEAT_SIZES.get(len(display), (112, 42, 60))
    seats = ""
    for i, k in enumerate(display):
        html, delay = seat_html(k, i, len(display), seen, delay)
        seats += html
    ss.anim = {**seen, "hole_hidden": hide}

    center, arc_sub = "", f"DEALER {d['rule'].upper()} · {NUM_DECKS} DECKS"
    if ss.phase == "done":
        res, pay = my_result()
        kind = "win" if pay > 0 else "lose" if pay < 0 else "push"
        center = f'<div class="result {kind}">{res}<span>{pay:+,.0f}</span></div>'
    elif ss.phase == "betting":
        arc_sub = '<span class="live">PLACE YOUR BET</span>'
    elif seat_playing("you"):
        arc_sub = '<span class="live pulse">YOUR TURN</span>'

    pct = len(ss.shoe) / (52 * NUM_DECKS)
    count = (f'<div class="count">RC <b>{ss.running_count:+d}</b> · TC <b>{true_count():+.1f}</b></div>'
             if ss.show_count else "")
    html = f"""<style>{scene_css(t, seat_w=seat_w, card_w=card_w, card_h=card_h,
                                  gap=18 if len(display) <= 3 else 10,
                                  side_pad=46 if len(display) <= 3 else 22)}</style>
<div class="rail"><div class="felt">{scene_effects(t)}
  <div class="shoe"><div class="shoebox"><div class="shoefill" style="width:{pct * 100:.0f}%"></div></div>{len(ss.shoe)} cards{count}</div>
  <div class="dealer"><div class="dname">{d['emoji']} {d['name'].upper()}</div><div class="hand">{dcards}</div>{dpill}</div>
  <div class="center">{center}</div>
  <div class="arc"><b>BLACKJACK PAYS 3 TO 2</b>{arc_sub}</div>
  <div class="seats">{seats}</div>
</div></div>"""
    render_html(html, 492)


# =========================================================
# Page-specific styling
# =========================================================

action_button_css("act_")
st.markdown("""
<style>
.st-key-deal button { height: 56px; font-size: 18px; letter-spacing: 2px; }
/* keep the chips and action buttons reachable without scrolling */
.st-key-controls { position: sticky; bottom: 0; z-index: 90; padding: 10px 8px 6px; border-radius: 16px;
  background: linear-gradient(180deg, transparent, var(--bg) 22%); backdrop-filter: blur(3px); }
.hud { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.hud div { flex: 1; min-width: 110px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 5px 12px; }
.hud small { color: var(--muted); font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
.hud b { display: block; color: var(--acc); font-size: 18px; font-weight: 800; }
.logline { font-size: 13px; color: var(--text); opacity: .85; border-bottom: 1px solid var(--border); padding: 5px 0; }
.prof-move { font: 900 32px var(--head); letter-spacing: 3px; color: var(--acc); margin: 2px 0 8px; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# Page
# =========================================================

if ss.toast:
    msg, effect = ss.toast
    st.toast(msg)
    if effect == "balloons":
        st.balloons()
    ss.toast = None

head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown(f'<div class="page-title" style="font-size:26px !important;margin-bottom:2px">🎰 Casino Table'
                f'<span style="font-size:13px;color:var(--muted);font-family:var(--body);margin-left:12px">'
                f"{rule()['emoji']} {rule()['name']} · {rule()['rule']} · "
                f"{len(ss.seats)} AI player{'s' if len(ss.seats) != 1 else ''}</span></div>",
                unsafe_allow_html=True)
with head_r:
    locked = ss.phase == "playing"
    with st.popover("⚙️ Table settings", width="stretch"):
        st.caption("🎨 Change the look with the theme picker at the top of the page.")
        st.radio("Dealer", list(DEALERS), key="dealer_key", disabled=locked,
                 format_func=lambda k: f"{DEALERS[k]['emoji']} {DEALERS[k]['name']} ({DEALERS[k]['rule']})")
        st.multiselect("AI players", list(AI_PLAYERS), key="seats", disabled=locked,
                       format_func=lambda k: f"{AI_PLAYERS[k]['emoji']} {AI_PLAYERS[k]['name']}")
        if locked:
            st.caption("Dealer and players can be changed between hands.")
        st.toggle("🎓 Professor coaching", key="show_prof")
        st.toggle("🧮 Show Hi-Lo card count", key="show_count")
        st.divider()
        if st.button("🔄 Reset bankroll & stats", width="stretch"):
            init_state()
            st.rerun()
        if st.button("🧠 Retrain all agents", width="stretch"):
            for lvl in LEVELS:
                for r in DEALERS:
                    if os.path.exists(q_path(lvl, r)):
                        os.remove(q_path(lvl, r))
            get_bundles.clear()
            st.rerun()

s = ss.stats
acc = f"{ss.prof['correct'] / ss.prof['decisions']:.0%}" if ss.prof["decisions"] else "—"
winrate = f"{s['wins'] / s['hands']:.0%}" if s["hands"] else "—"
st.markdown(
    f'<div class="hud"><div><small>Bankroll</small><b>${ss.bankroll:,.0f}</b></div>'
    f'<div><small>Session</small><b>{ss.bankroll - START_BANK:+,.0f}</b></div>'
    f'<div><small>Hands</small><b>{s["hands"]}</b></div>'
    f'<div><small>Win rate</small><b>{winrate}</b></div>'
    f'<div><small>Professor accuracy</small><b>{acc}</b></div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([2.5, 1], gap="large")

with left:
    render_table()

    controls = st.container(key="controls")
    if ss.phase in ("betting", "done"):
      with controls:
          if ss.bankroll < 10:
              st.error("You're out of chips!")
              if st.button("💵 Buy back in for $1,000", width="stretch"):
                  ss.bankroll = START_BANK
                  ss.history.append(ss.bankroll)
                  st.rerun()
          else:
              chips = st.columns([1, 1, 1, 1, 1.4, 2.6], vertical_alignment="center")
              for col, amt in zip(chips[:4], (10, 25, 50, 100)):
                  with col:
                      if st.button(f"${amt}", key=f"act_chip{amt}", help=f"Add ${amt} to your bet"):
                          ss.bet = min(ss.bet + amt if ss.phase == "betting" else amt, int(ss.bankroll))
                          ss.phase = "betting"
                          st.rerun()
              with chips[4]:
                  if st.button("✖ Clear", width="stretch"):
                      ss.bet, ss.phase = 10, "betting"
                      st.rerun()
              with chips[5]:
                  label = f"🃏 DEAL  ·  ${ss.bet}" if ss.phase == "betting" else f"🔁 SAME BET  ·  ${ss.bet}"
                  if st.button(label, key="deal", width="stretch", type="primary"):
                      ss.bet = max(10, min(ss.bet, int(ss.bankroll)))
                      deal_round()
                      st.rerun()
              if ss.phase == "done":
                  st.caption("Tap a chip to start a new bet, or deal again with the same bet.")
    else:
      with controls:
          me = active_hand("you")
          extra_ok = ss.bankroll >= committed() + me["bet"]
          can_double = len(me["cards"]) == 2 and extra_ok
          splittable = can_split("you", me) and extra_ok
          if len(seat_hands("you")) > 1:
              st.caption(f"✂️ Playing split hand {ss.hands['you']['active'] + 1} of {len(seat_hands('you'))}")
          b = st.columns(5 if splittable else 4)
          if b[0].button("HIT", key="act_hit", width="stretch"):
              player_action("hit")
              st.rerun()
          if b[1].button("STAND", key="act_stand", width="stretch"):
              player_action("stand")
              st.rerun()
          if b[2].button("DOUBLE", key="act_double", width="stretch", disabled=not can_double,
                         help="Double your bet and take exactly one card"):
              player_action("double")
              st.rerun()
          if splittable and b[3].button("SPLIT", key="act_split", width="stretch",
                                        help="Split your pair into two hands, each with its own bet"):
              player_action("split")
              st.rerun()
          if b[-1].button("🤖 AUTO", key="act_auto", width="stretch", help="Let the expert agent finish your hand"):
              while ss.phase == "playing":
                  player_action(professor_best()[0])
              st.rerun()

with right:
    tab_prof, tab_stats, tab_log = st.tabs(["🎓 Professor", "📈 Stats", "📜 Log"])

    with tab_prof:
        if not ss.show_prof:
            st.caption("Professor coaching is off. Turn it on in ⚙️ Table settings.")
        elif ss.phase == "playing" and seat_playing("you"):
            me = active_hand("you")
            best, values = professor_best()
            up = ss.dealer[1]
            bust = sum(rl.score(me["cards"] + [c]) > 21 for c in ss.shoe) / len(ss.shoe)
            sims, busts = 0, 0
            while sims < 1500:
                draws = random.sample(ss.shoe, 10)
                hand = [up, draws[0]]
                if rl.is_blackjack(hand):
                    continue
                i = 1
                while rl.dealer_should_hit(hand, rule()["hit_soft17"]):
                    hand.append(draws[i])
                    i += 1
                busts += rl.score(hand) > 21
                sims += 1
            dbust = busts / sims
            icons = {"hit": "🃏 HIT", "stand": "✋ STAND", "double": "⏬ DOUBLE", "split": "✂️ SPLIT"}
            st.markdown(f'<div class="panel"><small>THE PROFESSOR SAYS</small>'
                        f'<div class="prof-move">{icons[best]}</div>{value_bars(values, best)}</div>',
                        unsafe_allow_html=True)
            why = explain(best, me["cards"], up, bust, dbust)
            if theme()["effect"] == "cyber":
                st.markdown(f'<div class="typewriter" style="font-family:var(--body)">&gt; {md_bold(why)}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(why)
            m1, m2 = st.columns(2)
            m1.metric("Your bust if hit", f"{bust:.0%}")
            m2.metric("Dealer bust", f"{dbust:.0%}")

            provs = llm_status()
            if provs:
                with st.expander("💬 Ask about this hand"):
                    q = st.text_input("Your question", key="hand_q",
                                      placeholder="e.g. why not double here?", label_visibility="collapsed")
                    if q:
                        ctx = llm.hand_context(me["cards"], up, values, best, bust, dbust,
                                               true_count() if ss.show_count else None, ss.bankroll)
                        pid, plabel, pmodels = provs[0]
                        st.write_stream(llm.stream(pid, llm.default_model(pid, pmodels),
                                                   [{"role": "user", "content": q}], ctx, fallbacks=pmodels))
                        st.caption(f"Answered by {plabel}, using the agent's real numbers for this exact hand.")
            tc = true_count()
            if ss.show_count and abs(tc) >= 2:
                st.info(f"{'📈' if tc > 0 else '📉'} True count **{tc:+.1f}**: the shoe favors "
                        f"{'you (rich in tens and aces)' if tc > 0 else 'the dealer (lots of small cards)'}.")
        else:
            st.markdown('<div class="panel">Place a bet and deal. I\'ll explain the best move for every hand '
                        'using the expert agent and live odds from the shoe.</div>', unsafe_allow_html=True)
            cb = counting_bundle(ss.dealer_key)
            if ss.show_count and cb and ss.phase in ("betting", "done"):
                tc = true_count()
                b = C.bucket(tc)
                units = C.bet_ramp(cb)[b]
                value = cb["ev"][b]
                suggested = max(10, min(int(ss.bankroll), units * 10))
                st.markdown("#### 🧮 Counting coach")
                st.markdown(f"True count **{tc:+.1f}** · the counting agent measured this count at "
                            f"**{value * 100:+.2f}%** per dollar.")
                if units > C.BETS[0]:
                    st.success(f"📈 Favourable shoe: bet big, around **${suggested}** ({units}× the minimum).")
                else:
                    st.info(f"📉 Not favourable: bet the minimum, around **${suggested}**.")
                st.caption("Betting with the count is what turns the house edge around. See the Card Counter page.")
        if ss.prof["last"]:
            kind, msg = ss.prof["last"]
            (st.success if kind == "good" else st.warning)(msg)

        if ss.phase == "done" and ss.review:
            lost = sum(max(0.0, e["values"].get(e["best"], 0) - e["values"].get(e["action"], 0)) * e["bet"]
                       for e in ss.review)
            title = "🔍 What if? — replay your decisions"
            with st.expander(title, expanded=len(ss.review) <= 2):
                for i, e in enumerate(ss.review, 1):
                    hand = " ".join(e["cards"]) + f" = {'soft ' if e['soft'] and e['total'] < 21 else ''}{e['total']}"
                    mark = "✅" if e["action"] == e["best"] else "⚠️"
                    st.markdown(f"**{mark} Decision {i}: {hand} vs dealer {ss.dealer[1]['rank']}** — you "
                                f"**{e['action'].upper()}**" +
                                ("" if e["action"] == e["best"] else f", best was **{e['best'].upper()}**"))
                    bits = " · ".join(f"{a} {v:+.2f}" for a, v in e["values"].items())
                    st.caption(f"Expected per $1 → {bits}")
                    alt = what_if(e)
                    if alt:
                        st.caption(alt)
                if lost > 0.005:
                    st.warning(f"Those choices gave up about **${lost * 1:.2f} of expected value** this hand "
                               f"(on a ${ss.review[0]['bet']} bet).")
                else:
                    st.success("You played this hand exactly as the expert agent would have. 👏")

    with tab_stats:
        if len(ss.history) > 1:
            st.markdown("**Bankroll over time**")
            st.line_chart({"Bankroll": ss.history}, height=180, color="#d4af37")
        else:
            st.caption("Play a hand to start your bankroll chart.")
        c1, c2 = st.columns(2)
        c1.metric("Wins", s["wins"])
        c2.metric("Losses", s["losses"])
        c1.metric("Pushes", s["pushes"])
        c2.metric("Blackjacks", s["blackjacks"])
        if ss.seats:
            st.markdown("**AI players**")
            for k in ss.seats:
                st.markdown(f"{AI_PLAYERS[k]['emoji']} {AI_PLAYERS[k]['name']}: "
                            f"**${ss.ai_bank[k]:,.0f}** ({ss.ai_bank[k] - START_BANK:+,.0f})")

    with tab_log:
        if ss.log:
            for m in ss.log:
                m = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", m)
                st.markdown(f'<div class="logline">{m}</div>', unsafe_allow_html=True)
        else:
            st.caption("Nothing yet. Deal the first hand!")
