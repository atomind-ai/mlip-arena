import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="MLIP Arena",
    page_icon=":shark:",
    initial_sidebar_state="expanded",
    menu_items={
        "About": 'https://github.com/atomind-ai/mlip-arena',
        "Report a bug": "https://github.com/atomind-ai/mlip-arena/issues/new",
    }
)

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# def login():
#     if st.button("Log in"):
#         st.session_state.logged_in = True
#         st.rerun()

# def logout():
#     if st.button("Log out"):
#         st.session_state.logged_in = False
#         st.rerun()

# login_page = st.Page(login, title="Log in", icon=":material/login:")
# logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

leaderboard = st.Page(
    "models/leaderboard.py", title="Leaderboard", icon=":material/trophy:"
)
bugs = st.Page("models/bugs.py", title="Bug reports", icon=":material/bug_report:")
alerts = st.Page(
    "models/alerts.py", title="System alerts", icon=":material/notification_important:"
)

search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")
ptable = st.Page("tools/ptable.py", title="Periodic table", icon=":material/gradient:")

diatomics = st.Page("tasks/homonuclear-diatomics.py", title="Homonuclear diatomics", icon=":material/target:", default=True)

# if st.session_state.logged_in:
pg = st.navigation(
    {
        # "Account": [logout_page],
        # "Reports": [dashboard, bugs, alerts],
        # "Tools": [search, history, ptable],
        "Models": [leaderboard],
        "Tasks": [diatomics],
        "Tools": [ptable],
    }
)
# else:
#     pg = st.navigation([login_page])

pg.run()