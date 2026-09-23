"""Multiplayer — online rooms. Friends join the same table with a room code and play against one dealer.

Rooms live in the server's memory (shared by every browser session), protected by a lock.
Each player's page refreshes itself every 1.5 seconds to show what the others did.
"""

import random
import re
import threading
import time
import uuid

import streamlit as st
from streamlit.errors import StreamlitAPIException

import blackjack_rl as rl
from shared import (DEALERS, action_button_css, bust_chance, cards_html, chip_html, dealer_bust_chance,
                    explain, get_agents, page_header, render_html, scene_css, scene_effects, theme, value_bars)

AGENTS = get_agents()

NUM_DECKS = 6
START_BANK = 1000
MAX_PLAYERS = 4
TURN_SECONDS = 30        # a player who doesn't act in time automatically stands
INACTIVE_SECONDS = 25    # a player whose page stops refreshing is removed
AUTO_NEXT_SECONDS = 12   # the table resets for betting after a round
AI_BET = 50
CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
SUITS = [("♠", False), ("♥", True), ("♣", False), ("♦", True)]
REACTIONS = ["👏", "😂", "😱", "🔥", "🍀", "😭"]


# =========================================================
# Shared server state
# =========================================================

@st.cache_resource
def server():
    return {"lock": threading.Lock(), "rooms": {}}


SERVER = server()
LOCK, ROOMS = SERVER["lock"], SERVER["rooms"]


def new_shoe():
    shoe = [{"rank": r, "value": v, "symbol": s, "red": red}
            for _ in range(NUM_DECKS) for s, red in SUITS for r, v in rl.CARD_VALUES.items()]
    random.shuffle(shoe)
    return shoe


def log(room, msg):
    room["log"].insert(0, msg)
    del room["log"][20:]


def create_room(pid, name):
    code = "".join(random.choice(CODE_CHARS) for _ in range(5))
    while code in ROOMS:
        code = "".join(random.choice(CODE_CHARS) for _ in range(5))
    ROOMS[code] = {
        "code": code, "host": pid, "created": time.time(), "dealer_key": "S17", "ai": True,
        "players": {}, "seat_order": [], "phase": "betting", "shoe": new_shoe(),
        "dealer": [], "hands": {}, "order": [], "turn": 0, "turn_started": 0.0, "done_at": 0.0,
        "round": 0, "ai_bank": START_BANK, "log": [], "chat": [],
    }
    join_room(ROOMS[code], pid, name)
    return code


def join_room(room, pid, name):
    if pid in room["players"]:
        room["players"][pid]["name"] = name
        return None
    if len(room["players"]) >= MAX_PLAYERS:
        return "This table is full (4 players max)."
    room["players"][pid] = {"name": name, "bank": START_BANK, "bet": 25, "ready": False,
                            "last_seen": time.time(), "correct": 0, "decisions": 0}
    room["seat_order"].append(pid)
    log(room, f"👋 **{name}** sat down.")
    return None


def new_hand(cards, bet, from_split=False):
    return {"cards": cards, "bet": bet, "status": "playing", "result": "", "payout": 0, "from_split": from_split}


def seat_hands(room, k):
    return room["hands"][k]["hands"]


def active_hand(room, k):
    seat = room["hands"][k]
    return seat["hands"][min(seat["active"], len(seat["hands"]) - 1)]


def seat_playing(room, k):
    return any(h["status"] == "playing" for h in seat_hands(room, k))


def can_split(room, k, h):
    return len(h["cards"]) == 2 and not h["from_split"] and len(seat_hands(room, k)) == 1 and rl.is_pair(h["cards"])


def seat_result(room, k):
    """(label, total payout) for a seat, combining split hands."""
    hands = seat_hands(room, k)
    pay = sum(h["payout"] for h in hands)
    if len(hands) == 1:
        return hands[0]["result"], pay
    return ("WIN" if pay > 0 else "LOSE" if pay < 0 else "PUSH"), pay


def remove_player(room, pid, reason="left"):
    p = room["players"].pop(pid, None)
    if not p:
        return
    room["seat_order"].remove(pid)
    log(room, f"🚪 **{p['name']}** {reason}.")
    if room["phase"] == "playing" and pid in room["hands"] and seat_playing(room, pid):
        was_turn = current_turn(room) == pid
        for h in seat_hands(room, pid):
            if h["status"] == "playing":
                h["status"] = "stood"
        if was_turn:
            room["turn"] += 1
            advance(room)
    if room["players"] and room["host"] not in room["players"]:
        room["host"] = room["seat_order"][0]
        log(room, f"👑 **{room['players'][room['host']]['name']}** is the new host.")


def current_turn(room):
    if room["phase"] == "playing" and room["turn"] < len(room["order"]):
        return room["order"][room["turn"]]
    return None


def draw(room):
    return room["shoe"].pop()


def dealer_rule(room):
    return DEALERS[room["dealer_key"]]


def start_round(room):
    eligible = [pid for pid in room["seat_order"]
                if room["players"][pid]["ready"] and 10 <= room["players"][pid]["bet"] <= room["players"][pid]["bank"]]
    if not eligible:
        return
    if len(room["shoe"]) < 52 * NUM_DECKS * 0.25:
        room["shoe"] = new_shoe()
        log(room, "🔀 Fresh 6-deck shoe shuffled.")

    room["order"] = eligible + (["ai"] if room["ai"] else [])
    room["hands"] = {k: {"hands": [new_hand([], AI_BET if k == "ai" else room["players"][k]["bet"])], "active": 0}
                     for k in room["order"]}
    room["dealer"] = []
    for _ in range(2):
        for k in room["order"]:
            seat_hands(room, k)[0]["cards"].append(draw(room))
        room["dealer"].append(draw(room))
    for p in room["players"].values():
        p["ready"] = False
    room["round"] += 1
    room["phase"], room["turn"] = "playing", 0
    for k in room["order"]:
        h = seat_hands(room, k)[0]
        if rl.is_blackjack(h["cards"]):
            h["status"] = "blackjack"
    if rl.is_blackjack(room["dealer"]):
        log(room, "🎩 Dealer peeks... Blackjack!")
        settle(room)
        return
    advance(room)


def apply_action(room, k, action):
    seat = room["hands"][k]
    h = active_hand(room, k)
    if action == "hit":
        h["cards"].append(draw(room))
        s = rl.score(h["cards"])
        h["status"] = "bust" if s > 21 else "stood" if s == 21 else "playing"
    elif action == "stand":
        h["status"] = "stood"
    elif action == "double":
        h["bet"] *= 2
        h["cards"].append(draw(room))
        h["status"] = "bust" if rl.score(h["cards"]) > 21 else "stood"
    else:  # split
        second = h["cards"].pop()
        h["from_split"] = True
        other = new_hand([second], h["bet"], from_split=True)
        seat["hands"].append(other)
        for hand in (h, other):
            hand["cards"].append(draw(room))
            if second["rank"] == "A" or rl.score(hand["cards"]) == 21:
                hand["status"] = "stood"
    while seat["active"] < len(seat["hands"]) and seat["hands"][seat["active"]]["status"] != "playing":
        seat["active"] += 1


def committed(room, k):
    return sum(h["bet"] for h in seat_hands(room, k))


def seat_best(room, k, level="expert"):
    h = active_hand(room, k)
    bank_ok = k == "ai" or room["players"][k]["bank"] >= committed(room, k) + h["bet"]
    return rl.best_action(AGENTS[(level, room["dealer_key"])], h["cards"], room["dealer"][1],
                          len(h["cards"]) == 2 and bank_ok, can_split(room, k, h) and bank_ok)


def advance(room):
    """Move to the next seat that still needs to act; AI seats act instantly."""
    while room["turn"] < len(room["order"]):
        k = room["order"][room["turn"]]
        if not seat_playing(room, k):
            room["turn"] += 1
            continue
        if k == "ai":
            moves = []
            while seat_playing(room, "ai"):
                a, _ = seat_best(room, "ai")
                moves.append(a)
                apply_action(room, "ai", a)
            totals = " & ".join(str(rl.score(h["cards"])) for h in seat_hands(room, "ai"))
            log(room, f"🎩 **Veteran Vic**: {' → '.join(moves)} ({totals})")
            room["turn"] += 1
            continue
        room["turn_started"] = time.time()
        return
    if any(h["status"] == "stood" for k in room["order"] for h in seat_hands(room, k)):
        while rl.dealer_should_hit(room["dealer"], dealer_rule(room)["hit_soft17"]):
            room["dealer"].append(draw(room))
    settle(room)


def act(room, pid, action):
    if current_turn(room) != pid:
        return
    best, _ = seat_best(room, pid)
    p = room["players"][pid]
    p["decisions"] += 1
    p["correct"] += action == best
    apply_action(room, pid, action)
    room["turn_started"] = time.time()          # each split hand gets its own timer
    if not seat_playing(room, pid):
        room["turn"] += 1
        advance(room)


def settle(room):
    d, dealer_bj = rl.score(room["dealer"]), rl.is_blackjack(room["dealer"])
    winners = []
    for k in room["order"]:
        for h in seat_hands(room, k):
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
        _, total = seat_result(room, k)
        if k == "ai":
            room["ai_bank"] += total
        elif k in room["players"]:
            room["players"][k]["bank"] += total
            if total > 0:
                winners.append(room["players"][k]["name"])
    log(room, f"🎩 Dealer finishes on **{d}**{' (bust)' if d > 21 else ''}."
              + (f" Winners: {', '.join(winners)} 🎉" if winners else ""))
    room["phase"], room["done_at"] = "done", time.time()


def to_betting(room):
    room.update(phase="betting", dealer=[], hands={}, order=[], turn=0)


def tick(room, pid):
    """Housekeeping on every refresh: heartbeats, timeouts, auto-start."""
    now = time.time()
    if pid in room["players"]:
        room["players"][pid]["last_seen"] = now
    for other, p in list(room["players"].items()):
        if now - p["last_seen"] > INACTIVE_SECONDS:
            remove_player(room, other, "disconnected")
    if not room["players"]:
        ROOMS.pop(room["code"], None)
        return
    cur = current_turn(room)
    if cur and cur in room["players"] and now - room["turn_started"] > TURN_SECONDS:
        log(room, f"⏱️ **{room['players'][cur]['name']}** ran out of time and stands.")
        for h in seat_hands(room, cur):
            if h["status"] == "playing":
                h["status"] = "stood"
        room["turn"] += 1
        advance(room)
    if room["phase"] == "done" and now - room["done_at"] > AUTO_NEXT_SECONDS:
        to_betting(room)
    if room["phase"] == "betting" and all(p["ready"] for p in room["players"].values()):
        start_round(room)


# =========================================================
# Table rendering
# =========================================================

def label(cards):
    total, soft = rl.hand_info(cards)
    if rl.is_blackjack(cards):
        return "BLACKJACK"
    return f"{total - 10} / {total}" if soft and total < 21 else str(total)


def table_html(room, me):
    ss = st.session_state
    if ss.get("mp_anim_round") != room["round"]:
        ss.mp_anim_round, ss.mp_anim = room["round"], {}
    prev, seen, delay = ss.mp_anim, {}, 0.0
    t = theme()
    playing = room["phase"] == "playing"
    cur = current_turn(room)
    d = dealer_rule(room)

    if room["dealer"]:
        dcards, delay = cards_html(room["dealer"], prev=prev.get("dealer", 0), hide_first=playing)
        seen["dealer"] = len(room["dealer"])
        dlabel = f"SHOWING {rl.score(room['dealer'][1:])}" if playing else label(room["dealer"])
    else:
        dcards, dlabel = '<div class="ghost"></div><div class="ghost"></div>', "—"

    seats = ""
    for k in list(room["seat_order"]) + (["ai"] if room["ai"] else []):
        is_ai = k == "ai"
        p = None if is_ai else room["players"][k]
        name = "🎩 VETERAN VIC" if is_ai else ("YOU" if k == me else p["name"].upper())
        crown = "👑 " if k == room["host"] else ""
        bank = room["ai_bank"] if is_ai else p["bank"]
        seat = room["hands"].get(k)
        cls = "seat" + (" me" if k == me else "") + (" active" if k == cur else "")
        if seat:
            multi = len(seat["hands"]) > 1
            groups = ""
            for i, h in enumerate(seat["hands"]):
                key = f"{k}:{i}"
                body, delay = cards_html(h["cards"], prev=prev.get(key, 0), delay_start=delay)
                seen[key] = len(h["cards"])
                badge = ""
                if h["result"]:
                    kind = "win" if h["payout"] > 0 else "lose" if h["payout"] < 0 else "push"
                    badge = f'<div class="badge {kind}">{h["result"]}<small>{h["payout"]:+,.0f}</small></div>'
                focus = " focus" if multi and k == cur and h["status"] == "playing" and i == seat["active"] else ""
                groups += (f'<div class="hgroup{focus}">{badge}<div class="hand">{body}</div>'
                           f'<div class="pill">{label(h["cards"])}</div>{chip_html(h["bet"]) if multi else ""}</div>')
            spot = "" if multi else f'<div class="betspot">{chip_html(seat["hands"][0]["bet"])}</div>'
            inner = f'<div class="hands">{groups}</div>{spot}'
        else:
            bet = AI_BET if is_ai else p["bet"]
            ready = ('<div class="ready">READY</div>'
                     if not is_ai and room["phase"] == "betting" and p["ready"] else "")
            inner = (f'<div class="hands"><div class="hgroup">{ready}<div class="hand"><div class="ghost"></div>'
                     f'<div class="ghost"></div></div><div class="pill">—</div></div></div>'
                     f'<div class="betspot">{chip_html(bet)}</div>')
        seats += (f'<div class="{cls}">{inner}<div class="name">{crown}{name}</div>'
                  f'<div class="sub">${bank:,.0f}</div></div>')

    center, arc_sub = "", f"DEALER {d['rule'].upper()}"
    if room["phase"] == "betting":
        ready = sum(p["ready"] for p in room["players"].values())
        arc_sub = f'<span class="live">PLACE YOUR BETS · {ready}/{len(room["players"])} READY</span>'
    elif playing:
        who = ("YOUR TURN" if cur == me else f"{room['players'][cur]['name'].upper()}'S TURN"
               if cur in room["players"] else "")
        arc_sub = f'<span class="live pulse">{who}</span>'
    else:
        if me in room["hands"]:
            res, pay = seat_result(room, me)
            kind = "win" if pay > 0 else "lose" if pay < 0 else "push"
            center = f'<div class="result {kind}">{res}<span>{pay:+,.0f}</span></div>'
        else:
            arc_sub = '<span class="live">ROUND OVER</span>'
    ss.mp_anim = seen

    css = scene_css(t, felt_h=438, center_top=146, arc_top=176, seats_top=206, seat_w=150, card_w=54, card_h=78)
    return f"""<style>{css}</style>
<div class="rail"><div class="felt">{scene_effects(t)}
<div class="code">ROOM {room['code']}</div>
<div class="dealer"><div class="dname">{d['emoji']} {d['name'].upper()}</div><div class="hand">{dcards}</div><div class="pill">{dlabel}</div></div>
<div class="center">{center}</div>
<div class="arc"><b>BLACKJACK PAYS 3 TO 2</b>{arc_sub}</div>
<div class="seats">{seats}</div>
</div></div>"""


# =========================================================
# Page
# =========================================================

ss = st.session_state
ss.setdefault("mp_pid", uuid.uuid4().hex)
ss.setdefault("mp_name", f"Player{random.randint(100, 999)}")
ss.setdefault("mp_code", None)
ME = ss.mp_pid

action_button_css("mp_")
st.markdown("""
<style>
.st-key-mpcontrols { position: sticky; bottom: 0; z-index: 90; padding: 10px 8px 6px; border-radius: 16px;
  background: linear-gradient(180deg, transparent, var(--bg) 22%); backdrop-filter: blur(3px); }
.lb { display: flex; justify-content: space-between; padding: 7px 10px; border-bottom: 1px solid var(--border); font-size: 14px; }
.lb b { color: var(--acc); }
.chatline { font-size: 13px; padding: 3px 0; color: var(--text); }
.chatline b { color: var(--acc); }
</style>
""", unsafe_allow_html=True)


def refresh():
    """Redraw just the table if possible, otherwise the whole page."""
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def bold(m):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", m)


# ---------------- Lobby ----------------

def lobby():
    page_header("👥 Multiplayer", "Play at the same table as your friends. Create a room and share the code.")
    ss.mp_name = st.text_input("Your name", ss.mp_name, max_chars=14).strip() or ss.mp_name

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="feature"><div class="ico">🎲</div><h3>Create a table</h3>'
                    '<p>You become the host. Share the room code and up to 3 friends can join.</p></div>',
                    unsafe_allow_html=True)
        if st.button("Create new table", width="stretch", type="primary"):
            with LOCK:
                ss.mp_code = create_room(ME, ss.mp_name)
            st.query_params["room"] = ss.mp_code
            st.rerun()
    with c2:
        st.markdown('<div class="feature"><div class="ico">🔑</div><h3>Join with a code</h3>'
                    '<p>Enter the 5-letter code a friend gave you.</p></div>', unsafe_allow_html=True)
        code = st.text_input("Room code", st.query_params.get("room", ""), max_chars=5,
                             label_visibility="collapsed", placeholder="e.g. K7QXM").strip().upper()
        if st.button("Join table", width="stretch"):
            with LOCK:
                room = ROOMS.get(code)
                err = "No table with that code. Check it and try again." if not room else join_room(room, ME, ss.mp_name)
            if err:
                st.error(err)
            else:
                ss.mp_code = code
                st.query_params["room"] = code
                st.rerun()

    st.markdown("### 🟢 Open tables")
    with LOCK:
        open_rooms = [(c, r["players"][r["host"]]["name"], len(r["players"]), r["phase"])
                      for c, r in ROOMS.items() if r["players"] and r["host"] in r["players"]]
    if not open_rooms:
        st.caption("No tables are open right now. Create one and invite your friends!")
    for c, host, n, phase in open_rooms:
        a, b, cc, d = st.columns([1, 2, 1, 1], vertical_alignment="center")
        a.markdown(f"**`{c}`**")
        b.markdown(f"👑 {host}'s table · {'in a hand' if phase == 'playing' else 'taking bets'}")
        cc.markdown(f"{n}/{MAX_PLAYERS} players")
        if d.button("Join", key=f"join_{c}", width="stretch", disabled=n >= MAX_PLAYERS):
            with LOCK:
                err = join_room(ROOMS[c], ME, ss.mp_name) if c in ROOMS else "That table just closed."
            if err:
                st.error(err)
            else:
                ss.mp_code = c
                st.query_params["room"] = c
                st.rerun()

    with st.expander("ℹ️ How multiplayer works"):
        st.markdown(
            f"- Everyone plays against the same dealer, from the same shoe, each with their own ${START_BANK:,} bankroll.\n"
            "- Place a bet and press **Ready**. The hand deals when everyone is ready, or when the host deals.\n"
            f"- Players act in seat order. You have **{TURN_SECONDS} seconds** per turn, or you automatically stand.\n"
            "- The host can change the dealer rules and invite Veteran Vic, the expert AI, to play along.\n"
            "- Testing alone? Open this page in a second browser tab and join your own table."
        )


# ---------------- Room ----------------

@st.fragment(run_every=1.5)
def room_view():
    with LOCK:
        room = ROOMS.get(ss.mp_code)
        if room:
            tick(room, ME)
        if not room or ss.mp_code not in ROOMS or ME not in room["players"]:
            ss.mp_code = None
            gone = True
        else:
            gone = False
            html = table_html(room, ME)
            me = room["players"][ME]
            snap = {
                "phase": room["phase"], "turn": current_turn(room), "host": room["host"] == ME,
                "bank": me["bank"], "bet": me["bet"], "ready": me["ready"],
                "hand": active_hand(room, ME) if ME in room["hands"] else None,
                "n_hands": len(seat_hands(room, ME)) if ME in room["hands"] else 0,
                "hand_no": room["hands"][ME]["active"] + 1 if ME in room["hands"] else 0,
                "my_playing": ME in room["hands"] and seat_playing(room, ME),
                "result": seat_result(room, ME) if ME in room["hands"] and room["phase"] == "done" else None,
                "extra_ok": ME in room["hands"] and me["bank"] >= committed(room, ME) + active_hand(room, ME)["bet"],
                "can_split": ME in room["hands"] and can_split(room, ME, active_hand(room, ME)),
                "dealer_up": room["dealer"][1] if len(room["dealer"]) > 1 else None,
                "dealer_key": room["dealer_key"], "ai": room["ai"],
                "time_left": max(0, TURN_SECONDS - (time.time() - room["turn_started"])),
                "next_in": max(0, AUTO_NEXT_SECONDS - (time.time() - room["done_at"])),
                "turn_name": room["players"][current_turn(room)]["name"] if current_turn(room) in room["players"] else "",
                "players": [(pid, p["name"], p["bank"], p["ready"], p["correct"], p["decisions"])
                            for pid, p in room["players"].items()],
                "host_pid": room["host"], "log": list(room["log"]), "chat": list(room["chat"]),
                "ready_any": any(p["ready"] for p in room["players"].values()),
            }
    if gone:
        st.warning("You left the table or it closed.")
        st.query_params.pop("room", None)
        st.rerun(scope="app")

    left, right = st.columns([2.5, 1], gap="large")

    with left:
        render_html(html, 470)
        phase = snap["phase"]
        controls = st.container(key="mpcontrols")

        with controls:
            if phase == "betting":
                if snap["ready"]:
                    st.success(f"✅ You're in for **${snap['bet']}**. Waiting for the other players...")
                    if st.button("Cancel ready", width="stretch"):
                        with LOCK:
                            ROOMS[ss.mp_code]["players"][ME]["ready"] = False
                        refresh()
                elif snap["bank"] < 10:
                    st.error("You're out of chips!")
                    if st.button("💵 Buy back in for $1,000", width="stretch"):
                        with LOCK:
                            ROOMS[ss.mp_code]["players"][ME]["bank"] = START_BANK
                        refresh()
                else:
                    cols = st.columns([1, 1, 1, 1, 1.3, 2.7], vertical_alignment="center")
                    for col, amt in zip(cols[:4], (10, 25, 50, 100)):
                        if col.button(f"${amt}", key=f"mp_chip{amt}"):
                            with LOCK:
                                p = ROOMS[ss.mp_code]["players"][ME]
                                p["bet"] = min(p["bet"] + amt, int(p["bank"]))
                            refresh()
                    if cols[4].button("✖ Clear", width="stretch"):
                        with LOCK:
                            ROOMS[ss.mp_code]["players"][ME]["bet"] = 10
                        refresh()
                    if cols[5].button(f"✅ READY · ${snap['bet']}", width="stretch", type="primary"):
                        with LOCK:
                            p = ROOMS[ss.mp_code]["players"][ME]
                            p["bet"] = max(10, min(p["bet"], int(p["bank"])))
                            p["ready"] = True
                            tick(ROOMS[ss.mp_code], ME)
                        refresh()
                if snap["host"] and snap["ready_any"]:
                    if st.button("🃏 Deal now (players who are ready)", width="stretch"):
                        with LOCK:
                            start_round(ROOMS[ss.mp_code])
                        refresh()

            elif phase == "playing":
                h = snap["hand"]
                if snap["turn"] == ME:
                    st.markdown(f"### 🎯 Your turn · ⏱️ {snap['time_left']:.0f}s")
                    can_double = len(h["cards"]) == 2 and snap["extra_ok"]
                    splittable = snap["can_split"] and snap["extra_ok"]
                    if snap["n_hands"] > 1:
                        st.caption(f"✂️ Playing split hand {snap['hand_no']} of {snap['n_hands']}")
                    opts = [("mp_hit", "HIT", "hit", True), ("mp_stand", "STAND", "stand", True),
                            ("mp_double", "DOUBLE", "double", can_double)]
                    if splittable:
                        opts.append(("mp_split", "SPLIT", "split", True))
                    for col, (key, label_, action, ok) in zip(st.columns(len(opts)), opts):
                        if col.button(label_, key=key, width="stretch", disabled=not ok):
                            with LOCK:
                                act(ROOMS[ss.mp_code], ME, action)
                            refresh()
                    if ss.get("mp_hints", True):
                        best, values = rl.best_action(AGENTS[("expert", snap["dealer_key"])], h["cards"],
                                                      snap["dealer_up"], can_double, splittable)
                        bust = bust_chance(h["cards"])
                        dbust = dealer_bust_chance(snap["dealer_up"]["rank"], DEALERS[snap["dealer_key"]]["hit_soft17"])
                        with st.container(border=True):
                            st.markdown(f"🎓 **The Professor whispers: {best.upper()}**")
                            st.markdown(explain(best, h["cards"], snap["dealer_up"], bust, dbust))
                            st.markdown(value_bars(values, best), unsafe_allow_html=True)
                elif h and snap["my_playing"]:
                    st.info(f"⏳ Waiting for **{snap['turn_name']}** ({snap['time_left']:.0f}s). You're up soon.")
                elif h:
                    st.info(f"⏳ Your hand is finished. Waiting for **{snap['turn_name']}** ({snap['time_left']:.0f}s)...")
                else:
                    st.info("🍿 You're sitting out this hand. Place a bet when the round ends.")

            else:  # done
                if snap["result"]:
                    res, pay = snap["result"]
                    (st.success if pay > 0 else st.error if pay < 0 else st.info)(
                        f"**{res}**  {pay:+,.0f}  ·  Bankroll ${snap['bank']:,.0f}")
                if st.button(f"🔁 Next hand now (auto in {snap['next_in']:.0f}s)", width="stretch", type="primary"):
                    with LOCK:
                        r = ROOMS[ss.mp_code]
                        if r["phase"] == "done":
                            to_betting(r)
                    refresh()

    with right:
        tab_players, tab_chat, tab_log = st.tabs(["🏆 Players", "💬 Chat", "📜 Log"])
        with tab_players:
            ranked = sorted(snap["players"], key=lambda p: -p[2])
            rows = ""
            for i, (pid, name, bank, ready, correct, decisions) in enumerate(ranked):
                medal = ["🥇", "🥈", "🥉", "4️⃣"][i]
                tags = ("👑" if pid == snap["host_pid"] else "") + (" ▶️" if pid == snap["turn"] else "") + \
                       (" ✅" if ready and snap["phase"] == "betting" else "")
                you = " (you)" if pid == ME else ""
                acc = f" · 🎓 {correct / decisions:.0%}" if decisions else ""
                rows += f'<div class="lb"><span>{medal} {name}{you} {tags}</span><span><b>${bank:,.0f}</b>{acc}</span></div>'
            st.markdown(rows, unsafe_allow_html=True)
            st.caption("🎓 = how often each player's moves matched the expert AI.")
        with tab_chat:
            rc = st.columns(len(REACTIONS))
            for col, emo in zip(rc, REACTIONS):
                if col.button(emo, key=f"react_{emo}"):
                    with LOCK:
                        chat = ROOMS[ss.mp_code]["chat"]
                        chat.insert(0, (ss.mp_name, emo))
                        del chat[30:]
                    refresh()
            with st.form("chat_form", clear_on_submit=True, border=False):
                msg = st.text_input("Message", max_chars=120, label_visibility="collapsed", placeholder="Say something...")
                if st.form_submit_button("Send", width="stretch") and msg.strip():
                    with LOCK:
                        chat = ROOMS[ss.mp_code]["chat"]
                        chat.insert(0, (ss.mp_name, msg.strip()))
                        del chat[30:]
                    refresh()
            for name, m in snap["chat"][:15]:
                safe = m.replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f'<div class="chatline"><b>{name.replace("<", "&lt;")}:</b> {safe}</div>',
                            unsafe_allow_html=True)
        with tab_log:
            for m in snap["log"][:15]:
                st.markdown(f'<div class="chatline">{bold(m)}</div>', unsafe_allow_html=True)


def room_header():
    code = ss.mp_code
    c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="center")
    with c1:
        page_header(f"👥 Table {code}", "Share this code with friends so they can join from their own device.")
    with c2:
        with st.popover("📨 Invite friends", width="stretch"):
            st.markdown(f"Room code: **`{code}`**")
            url = getattr(st.context, "url", None)
            if url:
                st.code(f"{url.split('?')[0]}?room={code}", language=None)
                st.caption("Send this link. Your friends pick a name and press **Join table**.")
            else:
                st.caption("Friends open the Multiplayer page and enter the code.")
    with c3:
        with st.popover("⚙️ Room", width="stretch"):
            st.toggle("🎓 Professor hints on my turn", key="mp_hints", value=True)
            with LOCK:
                room = ROOMS.get(code)
                is_host = bool(room and room["host"] == ME)
                cur_rule, cur_ai, phase = (room["dealer_key"], room["ai"], room["phase"]) if room else ("S17", True, "")
            if is_host:
                new_rule = st.radio("Dealer (host only)", list(DEALERS), index=list(DEALERS).index(cur_rule),
                                    disabled=phase == "playing",
                                    format_func=lambda k: f"{DEALERS[k]['emoji']} {DEALERS[k]['name']} ({DEALERS[k]['rule']})")
                new_ai = st.toggle("🎩 Veteran Vic (expert AI) plays", value=cur_ai, disabled=phase == "playing")
                if (new_rule, new_ai) != (cur_rule, cur_ai):
                    with LOCK:
                        if code in ROOMS and ROOMS[code]["phase"] != "playing":
                            ROOMS[code]["dealer_key"], ROOMS[code]["ai"] = new_rule, new_ai
                    st.rerun()
            else:
                st.caption("Only the host 👑 can change the dealer and AI settings.")
            if st.button("🚪 Leave table", width="stretch"):
                with LOCK:
                    if code in ROOMS:
                        remove_player(ROOMS[code], ME)
                        if not ROOMS[code]["players"]:
                            ROOMS.pop(code, None)
                ss.mp_code = None
                st.query_params.pop("room", None)
                st.rerun()


if ss.mp_code and ss.mp_code in ROOMS and ME in ROOMS[ss.mp_code]["players"]:
    room_header()
    room_view()
else:
    ss.mp_code = None
    lobby()