import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="MLIP Arena",
    page_icon=":shark:",
    # initial_sidebar_state="expanded",
    menu_items=None
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

dashboard = st.Page(
    "serve/reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
)
bugs = st.Page("serve/reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
alerts = st.Page(
    "serve/reports/alerts.py", title="System alerts", icon=":material/notification_important:"
)

search = st.Page("serve/tools/search.py", title="Search", icon=":material/search:")
history = st.Page("serve/tools/history.py", title="History", icon=":material/history:")

diatomics = st.Page("serve/tasks/homonuclear-diatomics.py", title="Homonuclear diatomics", icon="")

# if st.session_state.logged_in:
pg = st.navigation(
    {
        # "Account": [logout_page],
        "Reports": [dashboard, bugs, alerts],
        "Tools": [search, history],
        "Tasks": [diatomics],
    }
)
# else:
#     pg = st.navigation([login_page])

pg.run()