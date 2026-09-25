"""Home page — the landing screen: hero, quick play, results, how it learns, and theme gallery."""

import streamlit as st

import blackjack_rl as rl
from shared import LEVELS, THEMES, exact_grade, expert_label, get_bundles, render_html, scene_css, theme

_b = get_bundles()[("expert", "S17")]  # loads (or trains) the agents on first launch
_grade = exact_grade("expert", "S17")
t = theme()

from shared import THEMES as _T
_fonts = "&".join("family=" + th["fonts"].replace("&family=", "&family=") for th in _T.values())
st.markdown(f"<style>@import url('https://fonts.googleapis.com/css2?{_fonts}&display=swap');</style>", unsafe_allow_html=True)
st.markdown("""
<style>
.eyebrow { display: inline-block; font-size: 12px; letter-spacing: 3px; color: var(--acc); border: 1px solid var(--acc);
  border-radius: 30px; padding: 5px 14px; margin-bottom: 10px; text-transform: uppercase; }
.lead { font-size: 18px !important; color: var(--text); opacity: .9; max-width: 560px; line-height: 1.6; margin: 10px 0 22px; }
.st-key-cta_play button, .st-key-cta_friends button { height: 58px; font-size: 18px; }
.st-key-cta_friends button { background: transparent !important; border: 2px solid var(--acc) !important; box-shadow: none !important; }
.st-key-cta_friends button p { color: var(--acc) !important; }
.stat { background: linear-gradient(160deg, var(--surface2), var(--surface)); border: 1px solid var(--border); border-radius: 18px;
  padding: 18px 20px; text-align: center; }
.stat b { display: block; font-family: var(--head); font-size: 32px; color: var(--acc); line-height: 1.2; }
.stat span { color: var(--muted); font-size: 13px; }
.section { font-family: var(--head) !important; color: var(--acc); font-size: 30px !important; margin: 38px 0 2px; line-height: 1.3; }
.section-sub { color: var(--muted); margin-bottom: 16px; }
.step { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 20px; height: 100%; position: relative; }
.step .n { width: 38px; height: 38px; border-radius: 50%; background: var(--acc); color: var(--on-acc); display: flex;
  align-items: center; justify-content: center; font-weight: 900; font-family: var(--head); margin-bottom: 10px; }
.step h4 { margin: 0 0 6px; }
.step p { color: var(--muted) !important; font-size: 14px; margin: 0; }
.swatch { border-radius: 16px; padding: 14px; border: 2px solid transparent; }
.swatch.on { border-color: var(--acc); }
.swatch .dots span { display: inline-block; width: 22px; height: 22px; border-radius: 50%; margin-right: 4px; border: 2px solid rgba(255,255,255,.3); }
.swatch h4 { margin: 8px 0 2px; font-size: 17px; }
.swatch p { margin: 0 0 10px; font-size: 12px; opacity: .8; }
.foot { text-align: center; color: var(--muted); font-size: 13px; margin: 40px 0 10px; border-top: 1px solid var(--border); padding-top: 18px; }
</style>
""", unsafe_allow_html=True)


# ---------------- Hero ----------------

left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")
with left:
    st.markdown('<span class="eyebrow">🧠 Reinforcement learning project</span>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Royal Blackjack</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{t["icon"]} {t["tagline"]}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lead">An AI that started knowing nothing, played '
        f'<b>{LEVELS["expert"]:,}</b> hands against itself, and taught itself the strategy '
        'professional players memorize. Now it coaches you, plays beside you, and challenges your friends.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    if c1.button("🎰  Play now", key="cta_play", width="stretch"):
        st.switch_page("table.py")
    if c2.button("👥  Play with friends", key="cta_friends", width="stretch"):
        st.switch_page("multiplayer.py")

with right:
    hero = f"""<style>{scene_css(t, card_w=92, card_h=132)}
.stage {{ position: relative; height: 400px; }}
.oval {{ position: absolute; inset: 20px 10px 30px; border-radius: 50%; background: radial-gradient(ellipse at 50% 35%, {t['felt'][0]}, {t['felt'][1]} 60%, {t['felt'][2]});
        border: 14px solid {t['rail'][0]}; box-shadow: 0 20px 50px rgba(0,0,0,.7), inset 0 0 60px rgba(0,0,0,.6)
        {', 0 0 40px ' + t['accent'] if t['effect'] in ('neon', 'cyber') else ''}; overflow: hidden; }}
.fan {{ position: absolute; left: 50%; top: 110px; width: 0; }}
.fan .card {{ position: absolute; left: -46px; transform-origin: 50% 110%; margin: 0; }}
.fan .c1 {{ animation: fan1 4s ease-in-out infinite; }} .fan .c2 {{ animation: fan2 4s ease-in-out infinite; }} .fan .c3 {{ animation: fan3 4s ease-in-out infinite; }}
@keyframes fan1 {{ 0%,100% {{ transform: rotate(-22deg) translateX(-30px); }} 50% {{ transform: rotate(-26deg) translate(-36px,-6px); }} }}
@keyframes fan2 {{ 0%,100% {{ transform: rotate(0deg) translateY(-8px); }} 50% {{ transform: rotate(0deg) translateY(-18px); }} }}
@keyframes fan3 {{ 0%,100% {{ transform: rotate(22deg) translateX(30px); }} 50% {{ transform: rotate(26deg) translate(36px,-6px); }} }}
.bubble {{ position: absolute; right: 0; top: 8px; max-width: 210px; background: rgba(0,0,0,.78); border: 1px solid {t['accent']};
          color: #fff; border-radius: 14px 14px 4px 14px; padding: 10px 14px; font-size: 13px; line-height: 1.4; animation: float 3s ease-in-out infinite; }}
.bubble b {{ color: {t['accent']}; display: block; font-size: 15px; letter-spacing: 1px; }}
@keyframes float {{ 50% {{ transform: translateY(-8px); }} }}
.stack {{ position: absolute; left: 60px; bottom: 70px; }}
.stack .chip {{ position: absolute; left: 0; width: 58px; height: 58px; }}
.label {{ position: absolute; left: 0; right: 0; bottom: 56px; text-align: center; color: {t['accent']}; font: 700 13px {t['head_font']}; letter-spacing: 5px; opacity: .85; }}
</style>
<div class="stage"><div class="oval"></div>
<div class="fan">
  <div class="card c1 red"><span class="tl">A<br>♥</span><span class="mid">♥</span><span class="br">A<br>♥</span></div>
  <div class="card c2"><span class="tl">K<br>♠</span><span class="mid">♠</span><span class="br">K<br>♠</span></div>
  <div class="card c3 red"><span class="tl">Q<br>♦</span><span class="mid">♦</span><span class="br">Q<br>♦</span></div>
</div>
<div class="bubble"><b>🎓 STAND</b>Dealer shows a 6 and busts about 42% of the time. Let them take the risk.</div>
<div class="stack">{''.join(f'<div class="chip" style="bottom:{i * 7}px;background:{c};color:#fff;border-color:#fff"></div>'
                           for i, c in enumerate([t['hit'][1], t['stand'][1], t['double'][1], t['accent_dark'], t['hit'][0]]))}</div>
<div class="label">BLACKJACK PAYS 3 TO 2</div>
</div>"""
    render_html(hero, 410)


# ---------------- Results strip ----------------

stats = [(expert_label(), "hands of self-play"),
         (f"{_grade['optimal_share']:.1%}", "of decisions provably perfect"),
         (f"{_grade['regret'] * 10000:.2f}¢", "per $100 from perfect play, exactly"),
         ("−$43 → " + f"−${-_grade['policy_ev'] * 100:.2f}", "loss per $100: random vs. trained")]
for col, (big, small) in zip(st.columns(4), stats):
    col.markdown(f'<div class="stat"><b>{big}</b><span>{small}</span></div>', unsafe_allow_html=True)


# ---------------- Choose your game ----------------

st.markdown('<div class="section">Choose your game</div><div class="section-sub">Everything in one place.</div>',
            unsafe_allow_html=True)
FEATURES = [
    ("table.py", "🎰", "Casino Table", "Chips, a 6-deck shoe, AI players, and the Professor grading every move."),
    ("trainer.py", "🎯", "Strategy Trainer",
     "Drills scored against perfect play. See what each mistake costs, and missed hands come back for review."),
    ("multiplayer.py", "👥", "Multiplayer", "Create a room, share the code, and play with friends on their own devices."),
    ("advisor.py", "🧭", "Casino Advisor", "At a real table? Tap your cards and get the best move instantly."),
    ("research.py", "🔬", "Research Lab",
     "Exact grading against a solver, a five-algorithm race over seeds, a house-edge calculator, and open data."),
    ("lab.py", "🧪", "AI Lab", "Watch the AI learn, compare agents, explore Q-values, run tournaments."),
    ("card_counter.py", "🧮", "Card Counter",
     "The agent that learns what each count is worth, bets accordingly, and actually beats the house edge."),
    ("learn.py", "🎓", "How the AI Works", "States, actions, rewards, and Monte Carlo control, explained simply."),
]
row1, row2 = st.columns(4, gap="medium"), st.columns(4, gap="medium")
for col, (page, icon, title, text) in zip(list(row1) + list(row2), FEATURES):
    with col:
        st.markdown(f'<div class="feature"><div class="ico">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
                    unsafe_allow_html=True)
        st.page_link(page, label=f"Open {title}", icon="➡️", width="stretch")


# ---------------- How it learned ----------------

st.markdown('<div class="section">How the AI taught itself</div>'
            '<div class="section-sub">No strategy was programmed in. It learned from wins and losses alone.</div>',
            unsafe_allow_html=True)
steps = [
    ("1", "🎲 Play", "The agent plays a hand. Early on it explores by choosing moves at random."),
    ("2", "🏆 Get a reward", "At the end of the hand it gets +1 for a win, −1 for a loss, or 0 for a push."),
    ("3", "📈 Improve", "It updates its estimate of every move it made, then plays again, 30 million times."),
    ("4", "🎯 Practise", "It drills the rare hands it's still unsure about, comparing every move on the same cards."),
]
for col, (n, title, text) in zip(st.columns(4, gap="medium"), steps):
    col.markdown(f'<div class="step"><div class="n">{n}</div><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
st.page_link("lab.py", label="Watch it learn live in the AI Lab", icon="🧪")


# ---------------- Theme gallery ----------------

st.markdown('<div class="section">Pick your casino</div>'
            '<div class="section-sub">Every page, table, card, and chip changes with the theme.</div>',
            unsafe_allow_html=True)


def use_theme(name):
    st.session_state.ui_theme = name


current = st.session_state.get("ui_theme")
for col, (name, th) in zip(st.columns(len(THEMES), gap="small"), THEMES.items()):
    with col:
        dots = "".join(f'<span style="background:{c}"></span>' for c in (th["accent"], th["accent2"], th["felt"][0]))
        st.markdown(
            f'<div class="swatch {"on" if name == current else ""}" style="background:linear-gradient(160deg,{th["bg2"]},{th["bg"]})">'
            f'<div class="dots">{dots}</div><h4 style="color:{th["accent"]} !important;font-family:{th["head_font"]} !important">'
            f'{th["icon"]} {name}</h4><p style="color:{th["text"]} !important">{th["tagline"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.button("✓ Active" if name == current else "Use theme", key=f"theme_{name}", width="stretch",
                  on_click=use_theme, args=(name,), disabled=name == current)

st.markdown('<div class="foot">♠ Royal Blackjack · Built by Aryan with Python, Streamlit, and reinforcement learning ♥</div>',
            unsafe_allow_html=True)
