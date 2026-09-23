"""Royal Blackjack — entry point.  Run with:  streamlit run app.py"""

import streamlit as st

from shared import DEFAULT_THEME, THEMES, apply_base_css

st.set_page_config(page_title="Royal Blackjack", page_icon="🃏", layout="wide")
if "ui_theme" not in st.session_state:  # remember the theme across page reloads via the URL
    saved = st.query_params.get("theme")
    st.session_state.ui_theme = saved if saved in THEMES else DEFAULT_THEME
apply_base_css()

pages = {
    "": [st.Page("home.py", title="Home", icon="🏠", default=True)],
    "Play": [
        st.Page("table.py", title="Casino Table", icon="🎰"),
        st.Page("multiplayer.py", title="Multiplayer", icon="👥"),
        st.Page("advisor.py", title="Casino Advisor", icon="🧭"),
    ],
    "Learn": [
        st.Page("lab.py", title="AI Lab", icon="🧪"),
        st.Page("card_counter.py", title="Card Counter", icon="🧮"),
        st.Page("learn.py", title="How the AI Works", icon="🎓"),
    ],
}
nav = st.navigation(pages, position="top")

# Theme switcher, top-right of every page
_, pick = st.columns([5, 1])
with pick:
    st.selectbox("🎨 Theme", list(THEMES), key="ui_theme",
                 format_func=lambda n: f"{THEMES[n]['icon']}  {n}", label_visibility="collapsed")

if st.query_params.get("theme") != st.session_state.ui_theme:
    st.query_params["theme"] = st.session_state.ui_theme

nav.run()
