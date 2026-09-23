"""Shared pieces used by every page: dealer rules, trained agents, advice logic."""

import os
import pickle
import random

import streamlit as st

import blackjack_rl as rl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEALERS = {
    "S17": {"name": "Dealer Sophia", "emoji": "👩‍💼", "rule": "Stands on all 17s", "hit_soft17": False},
    "H17": {"name": "Dealer Rex", "emoji": "🧔", "rule": "Hits soft 17", "hit_soft17": True},
}

# Training size per experience level (hands of self-play)
LEVELS = {"rookie": 5_000, "expert": 30_000_000}


def q_path(level, rule):
    return os.path.join(BASE_DIR, f"q_{level}_{rule}.pkl")


def expert_label():
    """Human-friendly training size, e.g. '30M'."""
    n = LEVELS["expert"]
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


@st.cache_resource(show_spinner=False)
def get_bundles():
    """Load the four trained agents (2 levels x 2 dealer rules): {key: {"Q", "N", "hands"}}.
    Any that are missing or from an older version are trained here (a few minutes for the experts)."""
    bundles, bar = {}, None
    jobs = [(lvl, rule) for lvl in LEVELS for rule in DEALERS]
    for i, (lvl, rule) in enumerate(jobs):
        path = q_path(lvl, rule)
        try:
            bundles[(lvl, rule)] = rl.load_bundle(path)
            continue
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            pass
        if bar is None:
            bar = st.progress(0.0, text="Training AI agents (first run only, this can take a few minutes)...")
        label = f"🤖 Training {lvl} agent against {DEALERS[rule]['name']} ({LEVELS[lvl]:,} hands)"
        Q, N = {}, {}
        rl.train(LEVELS[lvl], DEALERS[rule]["hit_soft17"], Q, N,
                 progress=lambda p, i=i, label=label: bar.progress((i + p) / len(jobs), text=label))
        rl.save_q(Q, path, N, LEVELS[lvl])
        bundles[(lvl, rule)] = {"Q": Q, "N": N, "hands": LEVELS[lvl]}
    if bar:
        bar.empty()
    return bundles


@st.cache_resource(show_spinner=False)
def _robust_agents(_bundles_id):
    return {k: (rl.robust_q(b["Q"], b["N"]) if k[0] == "expert" else b["Q"]) for k, b in get_bundles().items()}


def get_agents():
    """{(level, rule): Q-table}. The Expert's table is made safe for rarely seen situations;
    the Rookie keeps its raw, inexperienced table on purpose."""
    return _robust_agents(id(get_bundles()))


# -----------------------------
# Advice helpers (infinite-deck odds, for use without a known shoe)
# -----------------------------

def make_card(rank):
    return {"rank": rank, "value": rl.CARD_VALUES[rank]}


INFINITE_DECK = [make_card(r) for r in rl.CARD_VALUES]  # 13 equally likely ranks


def bust_chance(cards):
    """Chance the next card busts this hand."""
    return sum(rl.score(cards + [c]) > 21 for c in INFINITE_DECK) / len(INFINITE_DECK)


@st.cache_data(show_spinner=False)
def dealer_bust_chance(up_rank, hit_soft17, trials=20_000):
    """Simulated dealer bust chance from an up card, given no dealer blackjack."""
    up = make_card(up_rank)
    busts = done = 0
    while done < trials:
        hand = [up, random.choice(INFINITE_DECK)]
        if rl.is_blackjack(hand):
            continue
        while rl.dealer_should_hit(hand, hit_soft17):
            hand.append(random.choice(INFINITE_DECK))
        busts += rl.score(hand) > 21
        done += 1
    return busts / trials


def explain(best, cards, up, bust, dbust):
    """Plain-English reasoning for the agent's move (rule-based templates)."""
    total, soft = rl.hand_info(cards)
    up_name = "Ace" if up["rank"] == "A" else up["rank"]
    weak = 2 <= up["value"] <= 6
    hand_name = f"soft {total}" if soft else f"hard {total}"

    if best == "stand":
        if total >= 17:
            return f"With {hand_name}, a hit busts you {bust:.0%} of the time. Stand and let the dealer work."
        if weak:
            return (f"The dealer shows a weak **{up_name}** and busts about **{dbust:.0%}** of the time. "
                    f"Don't risk a {bust:.0%} bust yourself; make the dealer draw.")
        return f"Standing on {hand_name} edges out hitting here (bust risk if you hit: {bust:.0%})."
    if best == "hit":
        if soft or bust == 0:
            return f"You have {hand_name}; one more card **cannot bust you**. Free chance to improve."
        if not weak:
            return (f"The dealer shows a strong **{up_name}** and only busts ~{dbust:.0%}. Standing on "
                    f"{total} usually loses, so the {bust:.0%} bust risk from hitting is worth taking.")
        return f"{hand_name.capitalize()} is too weak to stand on. Hit (bust risk {bust:.0%})."
    if best == "split":
        v = cards[0]["rank"] if cards[0]["value"] != 10 else "10"
        why = {
            "A": "Two Aces as one hand is just a soft 12. Split them and each Ace starts fresh toward 21.",
            "8": "A pair of 8s is 16, the worst total in Blackjack. Split them into two hands starting at 8.",
        }.get(v)
        if why:
            return why
        if weak:
            return (f"The dealer shows a weak **{up_name}** (busts ~{dbust:.0%}). Splitting your {v}s puts twice "
                    f"the money on the table while the dealer is most likely to bust.")
        return f"Your pair of {v}s plays better as two separate hands than as a total of {total}."
    return (f"{hand_name.capitalize()} vs a {up_name} is a **doubling spot**. You're the favorite to finish "
            f"strong, so put more money on the table.")


# -----------------------------
# Printable strategy card
# -----------------------------

def strategy_card_html(Q, dealer):
    colors = {"H": "#f4b6b6", "S": "#b9e4bf", "D": "#b9ccf2", "P": "#f5dc8a"}

    def pairs():
        rows, cols = rl.pair_table(Q)
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join(f"<tr><th>{label}</th>" + "".join(f'<td style="background:{colors[a]}">{a}</td>' for a in acts) + "</tr>"
                       for label, acts in rows.items())
        return f"<h3>Pairs</h3><table><tr><th>You \\ Dealer</th>{head}</tr>{body}</table>"

    def table(usable_ace, title):
        rows, cols = rl.strategy_table(Q, usable_ace=usable_ace)
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = ""
        for total, acts in rows.items():
            label = f"A+{total - 11}" if usable_ace else str(total)
            cells = "".join(f'<td style="background:{colors[a]}">{a}</td>' for a in acts)
            body += f"<tr><th>{label}</th>{cells}</tr>"
        return f"<h3>{title}</h3><table><tr><th>You \\ Dealer</th>{head}</tr>{body}</table>"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Blackjack Strategy Card</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h1 {{ margin: 0; font-size: 22px; }} h3 {{ margin: 16px 0 6px; }}
p {{ margin: 4px 0; font-size: 12px; color: #444; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #999; padding: 4px 8px; text-align: center; }}
th {{ background: #eee; }}
.legend span {{ display: inline-block; padding: 2px 8px; margin-right: 6px; border: 1px solid #999; }}
.wrap {{ display: flex; gap: 32px; flex-wrap: wrap; }}
button {{ margin-top: 16px; padding: 8px 16px; }}
@media print {{ button {{ display: none; }} }}
</style></head><body>
<h1>♠ Blackjack Strategy Card</h1>
<p>Learned by a reinforcement learning agent after {LEVELS['expert']:,} simulated hands · Dealer {dealer['rule'].lower()}</p>
<p class="legend"><span style="background:{colors['H']}">H = Hit</span><span style="background:{colors['S']}">S = Stand</span><span style="background:{colors['D']}">D = Double (if not allowed, hit)</span><span style="background:{colors['P']}">P = Split</span></p>
<div class="wrap"><div>{table(False, "Hard totals")}</div><div>{table(True, "Soft totals (Ace counted as 11)")}{pairs()}</div></div>
<button onclick="window.print()">🖨️ Print</button>
</body></html>"""


# -----------------------------
# Themes — every page and both tables read their colors from here
# -----------------------------

THEMES = {
    "Classic Gold": dict(
        icon="👑", tagline="Timeless Monte Carlo elegance",
        bg="#0b0b0b", bg2="#262015", surface="#141414", surface2="#1d1a12", border="#3a2f12",
        accent="#f1d27a", accent_dark="#b8912d", accent2="#5fd07a", text="#f3efe2", muted="#9a9486", on_accent="#2a1a05",
        head_font="'Playfair Display', Georgia, serif", body_font="'Inter', sans-serif", hero_font="'Playfair Display', Georgia, serif",
        fonts="Playfair+Display:wght@700;900&family=Inter:wght@400;600;800",
        felt=("#2a9657", "#17693a", "#0c4424"), rail=("#6b3e1d", "#3b200c"), back=("#7a0f1a", "#a3172a"),
        face="#fffdf7", face_black="#111111", face_red="#c8102e",
        hit=("#ff6b6b", "#b3001b"), stand=("#5fd07a", "#1d7a3a"), double=("#7aa7ff", "#1f4fa3"), btn_text="#ffffff",
        effect="classic",
    ),
    "Neon Vegas": dict(
        icon="🌃", tagline="Bright lights, big wins",
        bg="#0d0018", bg2="#3a0060", surface="#16002a", surface2="#1f0038", border="#5c1a7a",
        accent="#ff2e88", accent_dark="#b3005c", accent2="#00e5ff", text="#fdf0ff", muted="#b596c9", on_accent="#ffffff",
        head_font="'Bungee', sans-serif", body_font="'Inter', sans-serif", hero_font="'Monoton', 'Bungee', sans-serif",
        fonts="Bungee&family=Monoton&family=Inter:wght@400;600;800",
        felt=("#4a0f78", "#27004a", "#0d0018"), rail=("#ff2e88", "#6a0040"), back=("#00b8d4", "#ff2e88"),
        face="#ffffff", face_black="#1a0033", face_red="#ff0055",
        hit=("#ff2e88", "#b3005c"), stand=("#00e5ff", "#008ba3"), double=("#ffe600", "#b3a100"), btn_text="#1a0033",
        effect="neon",
    ),
    "Tokyo Night": dict(
        icon="🌸", tagline="Quiet focus under the cherry blossoms",
        bg="#0e1024", bg2="#2b2452", surface="#171a36", surface2="#1f2345", border="#3d3766",
        accent="#ffb7c5", accent_dark="#e07a93", accent2="#e63946", text="#fdf6f0", muted="#a9a3c4", on_accent="#1b1f3b",
        head_font="'Shippori Mincho', Georgia, serif", body_font="'Zen Kaku Gothic New', 'Inter', sans-serif",
        hero_font="'Shippori Mincho', Georgia, serif",
        fonts="Shippori+Mincho:wght@700;800&family=Zen+Kaku+Gothic+New:wght@400;700",
        felt=("#323b6e", "#1b1f3b", "#0b0d1f"), rail=("#8c2a33", "#44111a"), back=("#e63946", "#b02230"),
        face="#fdf6f0", face_black="#1b1f3b", face_red="#d62839",
        hit=("#e63946", "#9e1f2a"), stand=("#ffb7c5", "#c97a8d"), double=("#8fb3ff", "#4a6fc2"), btn_text="#ffffff",
        effect="petals",
    ),
    "Pirate Tavern": dict(
        icon="🏴‍☠️", tagline="Doubloons, rum, and a risky hand",
        bg="#120b06", bg2="#3d2817", surface="#1f140c", surface2="#2b1c10", border="#6b4a2b",
        accent="#ffc300", accent_dark="#b38600", accent2="#2ec4b6", text="#f3e3c3", muted="#b39b7a", on_accent="#3a2618",
        head_font="'Pirata One', Georgia, serif", body_font="'Inter', sans-serif", hero_font="'Pirata One', Georgia, serif",
        fonts="Pirata+One&family=Inter:wght@400;600;800",
        felt=("#8a6040", "#5c4033", "#3a2618"), rail=("#2b1a0e", "#120a04"), back=("#0f4c5c", "#1b6b80"),
        face="#f3e3c3", face_black="#2b1a0e", face_red="#8b1a1a",
        hit=("#ffc300", "#b38600"), stand=("#2ec4b6", "#0f4c5c"), double=("#e76f51", "#9c3b22"), btn_text="#1f140c",
        effect="pirate",
    ),
    "Cyber Casino": dict(
        icon="🤖", tagline="Hack the house with machine learning",
        bg="#050508", bg2="#0c2410", surface="#0f0f16", surface2="#13131d", border="#1f4a1a",
        accent="#39ff14", accent_dark="#1f9e0a", accent2="#ff6b00", text="#d8ffd0", muted="#6f8f6a", on_accent="#050508",
        head_font="'Orbitron', sans-serif", body_font="'JetBrains Mono', monospace", hero_font="'Orbitron', sans-serif",
        fonts="Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;700",
        felt=("#14301a", "#0b1a0f", "#040805"), rail=("#1f9e0a", "#0a2a05"), back=("#0d0d12", "#16361a"),
        face="#0d0d12", face_black="#39ff14", face_red="#ff6b00",
        hit=("#39ff14", "#1f9e0a"), stand=("#ff6b00", "#b34a00"), double=("#00e5ff", "#008ba3"), btn_text="#050508",
        effect="cyber",
    ),
}
DEFAULT_THEME = "Classic Gold"


def theme():
    name = st.session_state.get("ui_theme", DEFAULT_THEME)
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def md_bold(text):
    """Turn **bold** markdown into <b> for use inside HTML blocks."""
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


# ---- Page-wide styling -------------------------------------------------

PETALS = [(5, 0, 11), (15, 3, 14), (27, 6, 12), (38, 1, 16), (49, 8, 13), (61, 4, 15),
          (72, 9, 12), (83, 2, 14), (92, 5, 17), (33, 11, 13), (66, 12, 15), (10, 14, 12)]


def page_effects(t):
    """Decorative, click-through overlays for the whole page."""
    e = t["effect"]
    if e == "petals":
        spans = "".join(f'<span class="petal" style="left:{x}%;animation-delay:{d}s;animation-duration:{dur}s">🌸</span>'
                        for x, d, dur in PETALS)
        return f'<div class="fx">{spans}</div>'
    if e == "cyber":
        return '<div class="fx scan"></div>'
    return ""


def apply_base_css():
    t = theme()
    a, ad, a2 = t["accent"], t["accent_dark"], t["accent2"]
    glow = (f"text-shadow: 0 0 6px {a}, 0 0 18px {a}, 0 0 34px {a};" if t["effect"] == "neon" else
            f"text-shadow: 0 0 8px {a}99;" if t["effect"] == "cyber" else "")
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={t['fonts']}&display=swap');
:root {{ --acc: {a}; --acc-dark: {ad}; --acc2: {a2}; --bg: {t['bg']}; --surface: {t['surface']}; --surface2: {t['surface2']};
  --border: {t['border']}; --text: {t['text']}; --muted: {t['muted']}; --on-acc: {t['on_accent']};
  --head: {t['head_font']}; --body: {t['body_font']}; --hero: {t['hero_font']}; }}
.stApp {{ background: radial-gradient(circle at 50% -10%, {t['bg2']} 0%, {t['bg']} 60%) fixed; color: var(--text); }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button {{ font-family: var(--body); }}
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, [class*="material-symbols"] {{
  font-family: "Material Symbols Rounded" !important; }}
.stApp p, .stApp li, .stApp label, .stMarkdown {{ color: var(--text); }}
h1, h2, h3, h4, .stApp h1 span, .stApp h2 span, .stApp h3 span {{ font-family: var(--head) !important; color: var(--acc) !important; letter-spacing: .5px; }}
.block-container {{ padding-top: 3.6rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: {t['bg']}ee; border-bottom: 1px solid var(--border); }}
[data-testid="stSidebar"] {{ background: var(--surface); }}

div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {{
  background: linear-gradient(180deg, {a}, {ad}); color: var(--on-acc); border: 0; border-radius: 12px; font-weight: 800;
  box-shadow: 0 4px 14px rgba(0,0,0,.45){', 0 0 14px ' + a + '88' if t['effect'] in ('neon', 'cyber') else ''};
  transition: transform .08s, filter .15s; }}
div.stButton > button p, div.stDownloadButton > button p, div.stFormSubmitButton > button p {{ color: var(--on-acc); font-weight: 800; }}
div.stButton > button:hover, div.stDownloadButton > button:hover {{ filter: brightness(1.1); transform: translateY(-2px); color: var(--on-acc); }}
div.stButton > button:disabled {{ filter: grayscale(1) brightness(.55); }}
[data-testid="stPageLink"] a {{ border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }}
[data-testid="stPageLink"] a:hover {{ border-color: var(--acc); }}
[data-testid="stPageLink"] a p {{ color: var(--acc) !important; font-weight: 700; }}
[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; }}
[data-testid="stMetricValue"] {{ color: var(--acc); }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 700; }}
.stTabs [aria-selected="true"] p {{ color: var(--acc) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--acc); }}

.hero-title {{ font-family: var(--hero) !important; font-weight: 900; font-size: clamp(38px, 5.5vw, 72px) !important; line-height: 1.2; margin: 0; padding: 8px 0 4px;
  color: var(--acc); {glow} }}
.hero-sub {{ color: var(--muted); letter-spacing: 3px; font-size: 13px; text-transform: uppercase; }}
.page-title {{ font-family: var(--head) !important; font-weight: 900; font-size: 30px !important; color: var(--acc); margin: 0; line-height: 1.3; {glow} }}
.page-sub {{ color: var(--muted); margin: 0 0 8px; font-size: 14px; }}
.panel {{ background: linear-gradient(180deg, var(--surface2), var(--surface)); border: 1px solid var(--border); border-radius: 16px; padding: 16px 18px; }}
.panel small {{ color: var(--muted); letter-spacing: 2px; }}
.feature {{ background: linear-gradient(160deg, var(--surface2), var(--surface)); border: 1px solid var(--border); border-radius: 18px;
  padding: 20px; min-height: 200px; margin-bottom: 8px; transition: border-color .2s, transform .2s; }}
.feature:hover {{ border-color: var(--acc); transform: translateY(-3px); }}
.feature .ico {{ font-size: 34px; }}
.feature h3 {{ margin: 6px 0 4px; font-size: 21px; }}
.feature p {{ color: var(--muted) !important; font-size: 14px; margin: 0; }}
.pill-stat {{ display: inline-block; background: var(--surface); border: 1px solid var(--border); border-radius: 30px;
  padding: 6px 14px; margin: 4px; font-size: 13px; color: var(--text); }}
.bar {{ height: 10px; background: rgba(255,255,255,.08); border-radius: 5px; overflow: hidden; margin: 3px 0 10px; }}
.bar div {{ height: 100%; border-radius: 5px; }}
.sticky-controls {{ position: sticky; bottom: 0; z-index: 90; background: linear-gradient(180deg, transparent, {t['bg']} 18%);
  padding: 10px 6px 8px; margin-top: -4px; border-radius: 16px; }}
.typewriter {{ animation: reveal 1.6s steps(40) both; }}
.typewriter::after {{ content: "▋"; color: var(--acc); animation: blink 1s steps(1) infinite; }}
@keyframes reveal {{ from {{ clip-path: inset(0 100% 0 0); }} to {{ clip-path: inset(0 0 0 0); }} }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}

.fx {{ position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }}
.petal {{ position: absolute; top: -40px; font-size: 18px; opacity: .55; animation: fall linear infinite; }}
@keyframes fall {{ 0% {{ transform: translate(0, -40px) rotate(0); }} 100% {{ transform: translate(80px, 110vh) rotate(540deg); }} }}
.scan {{ background: repeating-linear-gradient(0deg, rgba(57,255,20,.035) 0 1px, transparent 1px 3px); animation: flick 4s infinite; }}
@keyframes flick {{ 97% {{ opacity: 1; }} 98% {{ opacity: .6; }} 99% {{ opacity: 1; }} }}
</style>
{page_effects(t)}
"""
    st.markdown(css, unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def value_bars(values, best):
    """Horizontal bars for expected return per action."""
    t = theme()
    lo, hi = min(values.values()), max(values.values())
    out = ""
    for a, v in values.items():
        w = 15 + 85 * ((v - lo) / (hi - lo) if hi > lo else 1)
        col = f"linear-gradient(90deg,{t['accent_dark']},{t['accent']})" if a == best else "rgba(255,255,255,.18)"
        out += (f'<div style="font-size:13px">{a.capitalize()}<span style="float:right">{v:+.2f} per $1</span></div>'
                f'<div class="bar"><div style="width:{w:.0f}%;background:{col}"></div></div>')
    return out


def action_button_css(prefix):
    """Colored HIT / STAND / DOUBLE buttons and casino chips for a page. Keys: {prefix}hit, {prefix}chip10 ..."""
    t = theme()
    rules = ""
    for act in ("hit", "stand", "double", "split"):
        c1, c2 = t.get(act, (t["accent"], t["accent_dark"]))
        rules += (f".st-key-{prefix}{act} button {{ background: linear-gradient(180deg, {c1}, {c2}) !important; height: 56px; letter-spacing: 2px; }}"
                  f".st-key-{prefix}{act} button p {{ color: {t['btn_text']} !important; font-size: 17px; }}")
    chips = CHIP_COLORS[t["effect"]]
    for amt, (bg, fg, edge) in chips.items():
        rules += (f".st-key-{prefix}chip{amt} button {{ background: {bg} !important; border-color: {edge} !important; }}"
                  f".st-key-{prefix}chip{amt} button p {{ color: {fg} !important; }}")
    rules += (f'[class*="st-key-{prefix}chip"] button {{ border-radius: 50% !important; width: 62px !important; height: 62px !important;'
              f' padding: 0 !important; border: 5px dashed; box-shadow: 0 4px 0 rgba(0,0,0,.4), 0 6px 14px rgba(0,0,0,.5) !important; }}'
              f'[class*="st-key-{prefix}chip"] button p {{ font-size: 13px !important; font-weight: 800; }}')
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


# ---- Casino table scene (inside an iframe, so colors are baked in) -----

CHIP_COLORS = {  # amount: (background, text, dashed edge)
    "classic": {10: ("#1f5fbf", "#fff", "#fff"), 25: ("#1d8a3e", "#fff", "#fff"), 50: ("#d9480f", "#fff", "#fff"), 100: ("#151515", "#f1d27a", "#f1d27a")},
    "neon":    {10: ("#00b8d4", "#1a0033", "#fff"), 25: ("#ff2e88", "#fff", "#fff"), 50: ("#ffe600", "#1a0033", "#1a0033"), 100: ("#1a0033", "#ff2e88", "#ff2e88")},
    "petals":  {10: ("#8fb3ff", "#1b1f3b", "#fff"), 25: ("#ffb7c5", "#1b1f3b", "#fff"), 50: ("#e63946", "#fff", "#fff"), 100: ("#1b1f3b", "#ffb7c5", "#ffb7c5")},
    "pirate":  {10: ("#c9a227", "#3a2618", "#7a5a10"), 25: ("#e0b83a", "#3a2618", "#7a5a10"), 50: ("#ffc300", "#3a2618", "#7a5a10"), 100: ("#ffd84d", "#3a2618", "#7a5a10")},
    "cyber":   {10: ("#0d0d12", "#00e5ff", "#00e5ff"), 25: ("#0d0d12", "#39ff14", "#39ff14"), 50: ("#0d0d12", "#ff6b00", "#ff6b00"), 100: ("#0d0d12", "#ff2e88", "#ff2e88")},
}


def chip_html(amount):
    t = theme()
    amt = 100 if amount >= 100 else 50 if amount >= 50 else 25 if amount >= 25 else 10
    bg, fg, edge = CHIP_COLORS[t["effect"]][amt]
    return f'<div class="chip" style="background:{bg};color:{fg};border-color:{edge}">${amount:,.0f}</div>'


def scene_css(t, felt_h=452, center_top=150, arc_top=182, seats_top=214, seat_w=162, card_w=56, card_h=80, gap=18, side_pad=46):
    f1, f2, f3 = t["felt"]
    r1, r2 = t["rail"]
    b1, b2 = t["back"]
    a, e = t["accent"], t["effect"]
    ov = int(card_w * .46)
    felt_bg = (f"repeating-linear-gradient(90deg, rgba(0,0,0,.12) 0 3px, transparent 3px 26px), radial-gradient(ellipse at 50% 30%, {f1}, {f2} 55%, {f3})"
               if e == "pirate" else f"radial-gradient(ellipse at 50% 30%, {f1} 0%, {f2} 55%, {f3} 100%)")
    glow = f"0 0 8px {a}, 0 0 22px {a}" if e in ("neon", "cyber") else "0 1px 2px #000"
    rail_extra = f", 0 0 30px {a}aa" if e in ("neon", "cyber") else ""
    card_border = f"1px solid {a}" if e == "cyber" else "1px solid rgba(0,0,0,.18)"
    card_font = t["body_font"] if e == "cyber" else "Arial, sans-serif"
    return f"""
@import url('https://fonts.googleapis.com/css2?family={t['fonts']}&display=swap');
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: transparent; font-family: {t['body_font']}; overflow: hidden; }}
.rail {{ background: linear-gradient(180deg, {r1}, {r2}); padding: 15px; border-radius: 40px 40px 320px 320px / 40px 40px 170px 170px;
        box-shadow: 0 18px 50px rgba(0,0,0,.75), inset 0 2px 0 rgba(255,255,255,.15){rail_extra}; }}
.felt {{ position: relative; height: {felt_h}px; border-radius: 28px 28px 300px 300px / 28px 28px 155px 155px; background: {felt_bg};
        box-shadow: inset 0 0 90px rgba(0,0,0,.6); padding: 16px 20px; color: #fff; overflow: hidden; }}
.dealer {{ display: flex; flex-direction: column; align-items: center; gap: 8px; position: relative; z-index: 2; }}
.dname {{ font: 700 13px {t['head_font']}; letter-spacing: 2px; background: rgba(0,0,0,.45); padding: 5px 16px; border-radius: 20px;
        border: 1px solid {a}; color: {a}; text-shadow: {glow}; }}
.code {{ position: absolute; top: 16px; left: 22px; font: 800 13px {t['body_font']}; color: {a}; background: rgba(0,0,0,.45); padding: 6px 12px; border-radius: 10px; letter-spacing: 3px; z-index: 3; }}
.arc {{ position: absolute; left: 0; right: 0; top: {arc_top}px; text-align: center; color: {a}; opacity: .85; letter-spacing: 4px;
       font: 700 12px {t['head_font']}; text-shadow: {glow}; }}
.arc b {{ display: block; font-size: 20px; letter-spacing: 6px; }}
.center {{ position: absolute; left: 0; right: 0; top: {center_top}px; display: flex; justify-content: center; z-index: 5; }}
.prompt {{ font: 800 14px {t['body_font']}; letter-spacing: 5px; color: {a}; text-shadow: {glow}; }}
.arc .live {{ display: inline-block; margin-top: 4px; padding: 2px 12px; border-radius: 12px; background: rgba(0,0,0,.45);
            border: 1px solid {a}; color: {a}; letter-spacing: 3px; font: 800 12px {t['body_font']}; }}
.pulse {{ animation: pulse 1.4s ease-in-out infinite; }}
@keyframes pulse {{ 50% {{ opacity: .35; }} }}
.result {{ font: 900 38px {t['hero_font']}; letter-spacing: 5px; padding: 8px 32px; border-radius: 14px; display: flex; gap: 16px; align-items: baseline;
          box-shadow: 0 10px 30px rgba(0,0,0,.6); animation: pop .5s cubic-bezier(.2,1.6,.4,1) both .5s; }}
.result span {{ font: 800 22px {t['body_font']}; }}
.result.win {{ background: linear-gradient(180deg, {a}, {t['accent_dark']}); color: {t['on_accent']}; {'box-shadow: 0 0 30px ' + a + ';' if e in ('neon', 'cyber') else ''} }}
.result.lose {{ background: linear-gradient(180deg, #e0344b, #8a0f1f); color: #fff; }}
.result.push {{ background: linear-gradient(180deg, #f0f0f0, #a9a9a9); color: #222; }}
.seats {{ position: absolute; left: 0; right: 0; top: {seats_top}px; display: flex; justify-content: center;
         gap: {gap}px; padding: 0 {side_pad}px; z-index: 2; }}
.hands {{ display: flex; gap: 10px; justify-content: center; }}
.hgroup {{ position: relative; display: flex; flex-direction: column; align-items: center; gap: 5px; padding: 4px; border-radius: 12px; }}
.hgroup.focus {{ background: rgba(255,255,255,.08); outline: 2px solid {a}; }}
.hgroup .chip {{ width: 40px; height: 40px; font-size: 10px; border-width: 4px; }}
.seat {{ position: relative; min-width: {seat_w}px; display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 6px 6px 8px; border-radius: 18px; }}
.seat.me {{ background: rgba(255,255,255,.06); border: 1px dashed rgba(255,255,255,.3); }}
.seat.active {{ border: 2px solid {a}; box-shadow: 0 0 28px {a}99; }}
.hand {{ display: flex; min-height: {card_h + 2}px; justify-content: center; padding-left: {ov}px; }}
.card {{ width: {card_w}px; height: {card_h}px; margin-left: -{ov}px; border-radius: {'4px' if e == 'cyber' else '10px'}; background: {t['face']};
        color: {t['face_black']}; position: relative; box-shadow: 2px 4px 10px rgba(0,0,0,.5); border: {card_border}; font-family: {card_font}; }}
.card.red {{ color: {t['face_red']}; {'border-color: ' + t['face_red'] + ';' if e == 'cyber' else ''} }}
.card .tl, .card .br {{ position: absolute; font-weight: 800; font-size: {int(card_w * .2)}px; line-height: 1; text-align: center; }}
.card .tl {{ top: 6px; left: 7px; }}
.card .br {{ bottom: 6px; right: 7px; transform: rotate(180deg); }}
.card .mid {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: {int(card_w * .5)}px;
             {'text-shadow: 0 0 10px currentColor;' if e == 'cyber' else ''} }}
.card.back {{ background: repeating-linear-gradient(45deg, {b1}, {b1} 6px, {b2} 6px, {b2} 12px); border: {'1px solid ' + a if e == 'cyber' else '4px solid ' + t['face']}; }}
.deal {{ animation: deal .45s cubic-bezier(.2,.8,.3,1) both; }}
@keyframes deal {{ from {{ opacity: 0; transform: translate(420px,-260px) rotate(35deg) scale(.7); }} to {{ opacity: 1; transform: none; }} }}
.flip {{ animation: flip .6s ease-out both; }}
@keyframes flip {{ 0% {{ transform: rotateY(90deg); }} 100% {{ transform: rotateY(0); }} }}
.ghost {{ width: {card_w}px; height: {card_h}px; margin-left: -{ov}px; border: 2px dashed rgba(255,255,255,.2); border-radius: 10px; }}
.pill {{ background: rgba(0,0,0,.75); border: 1px solid {a}; color: {a}; font: 800 12px {t['body_font']}; padding: 3px 14px; border-radius: 14px; letter-spacing: 1px; }}
.pill.dim {{ opacity: .4; }}
.badge, .ready {{ position: absolute; top: 42px; z-index: 4; font: 900 15px {t['body_font']}; padding: 6px 14px; border-radius: 10px; text-align: center;
         transform: rotate(-8deg); box-shadow: 0 6px 16px rgba(0,0,0,.55); }}
.badge {{ animation: stamp .45s ease-out both .6s; }}
.badge small {{ display: block; font-size: 11px; }}
.badge.win {{ background: {a}; color: {t['on_accent']}; }} .badge.lose {{ background: #b3001b; color: #fff; }} .badge.push {{ background: #ddd; color: #222; }}
.ready {{ background: {t['accent2']}; color: #fff; transform: none; letter-spacing: 2px; }}
@keyframes stamp {{ from {{ transform: scale(2.4) rotate(-8deg); opacity: 0; }} to {{ transform: scale(1) rotate(-8deg); opacity: 1; }} }}
@keyframes pop {{ from {{ transform: scale(.2); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}
.betspot {{ width: 52px; height: 52px; border-radius: 50%; border: 2px solid rgba(255,255,255,.4); display: flex; align-items: center; justify-content: center; }}
.chip {{ width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font: 800 11px {t['body_font']};
        border: 5px dashed; box-shadow: 0 3px 0 rgba(0,0,0,.35), 0 5px 10px rgba(0,0,0,.5); }}
.name {{ font: 800 13px {t['body_font']}; letter-spacing: 1px; text-shadow: 0 1px 2px #000; text-align: center; }}
.sub {{ font-size: 11px; color: rgba(255,255,255,.75); text-align: center; }}
.shoe {{ position: absolute; top: 18px; right: 24px; text-align: center; font-size: 11px; color: rgba(255,255,255,.8); z-index: 3; }}
.shoebox {{ width: 84px; height: 52px; background: linear-gradient(135deg, #2b2b2b, #0e0e0e); border-radius: 6px 22px 6px 6px; border: 1px solid #555;
           position: relative; overflow: hidden; margin-bottom: 4px; }}
.shoefill {{ position: absolute; bottom: 0; left: 0; height: 100%; background: repeating-linear-gradient(90deg, {b1}, {b1} 2px, {t['face']} 2px, {t['face']} 3px); }}
.count {{ margin-top: 3px; color: {a}; }}
.fx {{ position: absolute; inset: 0; pointer-events: none; z-index: 1; overflow: hidden; }}
.petal {{ position: absolute; top: -30px; font-size: 16px; opacity: .7; animation: fall linear infinite; }}
@keyframes fall {{ 0% {{ transform: translate(0,-30px) rotate(0); }} 100% {{ transform: translate(70px, 700px) rotate(540deg); }} }}
.scan {{ background: repeating-linear-gradient(0deg, rgba(57,255,20,.05) 0 1px, transparent 1px 3px); }}
.lights {{ position: absolute; left: 30px; right: 30px; top: 6px; display: flex; justify-content: space-between; }}
.lights i {{ width: 7px; height: 7px; border-radius: 50%; background: {a}; box-shadow: 0 0 10px {a}; animation: bulb 1.2s infinite; }}
.lights i:nth-child(odd) {{ background: {t['accent2']}; box-shadow: 0 0 10px {t['accent2']}; animation-delay: .6s; }}
@keyframes bulb {{ 50% {{ opacity: .25; }} }}
.dot {{ position: absolute; inset: 0; opacity: .07; background-image: radial-gradient(#000 1px, transparent 1px); background-size: 4px 4px; pointer-events: none; }}
"""


def scene_effects(t):
    """Theme decorations drawn on the felt."""
    e = t["effect"]
    if e == "petals":
        return '<div class="fx">' + "".join(
            f'<span class="petal" style="left:{x}%;animation-delay:{d}s;animation-duration:{dur}s">🌸</span>'
            for x, d, dur in PETALS[:9]) + "</div>"
    if e == "cyber":
        return '<div class="fx scan"></div>'
    if e == "neon":
        return '<div class="lights">' + "<i></i>" * 24 + "</div>"
    if e == "classic":
        return '<div class="dot"></div>'
    return ""


def cards_html(cards, prev=0, hide_first=False, flip_first=False, delay_start=0.0):
    """HTML for a hand. Cards at index >= prev get the deal animation. Returns (html, next_delay)."""
    out, delay = "", delay_start
    for i, c in enumerate(cards):
        cls, style = "card", ""
        if i >= prev:
            cls += " deal"
            style = f' style="animation-delay:{delay:.2f}s"'
            delay += 0.18
        if hide_first and i == 0:
            out += f'<div class="{cls} back"{style}></div>'
            continue
        if flip_first and i == 0:
            cls += " flip"
        if c.get("red"):
            cls += " red"
        out += (f'<div class="{cls}"{style}><span class="tl">{c["rank"]}<br>{c["symbol"]}</span>'
                f'<span class="mid">{c["symbol"]}</span><span class="br">{c["rank"]}<br>{c["symbol"]}</span></div>')
    return out, delay


def render_html(html, height):
    """Render a full HTML scene in an iframe (st.iframe on new Streamlit, components.html on old)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)