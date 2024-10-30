from collections import defaultdict

import streamlit as st

from mlip_arena.tasks import REGISTRY as TASKS

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
    "leaderboard.py", title="Leaderboard", icon=":material/trophy:", default=True
)

search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")
ptable = st.Page("tools/ptable.py", title="Periodic table", icon=":material/gradient:")


nav = defaultdict(list)
nav[""].append(leaderboard)

wide_pages, centered_pages = [], []

for task in TASKS:
    page = st.Page(
        f"tasks/{TASKS[task]['task-page']}.py", title=task, icon=":material/target:"
    )
    nav[TASKS[task]["category"]].append(page)
    if TASKS[task]["task-layout"] == "wide":
        wide_pages.append(page)
    else:
        centered_pages.append(page)

# mbd = st.page_link(page="https://matbench-discovery.materialsproject.org/", label="Matbench Discovery", icon=":material/extension:")
# nav["Other benchmarks"].append(mbd)

pg = st.navigation(nav)

if pg in centered_pages:
    st.set_page_config(
        layout="centered",
        page_title="MLIP Arena",
        page_icon=":shark:",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "https://github.com/atomind-ai/mlip-arena",
            "Report a bug": "https://github.com/atomind-ai/mlip-arena/issues/new",
        },
    )
else:
    st.set_page_config(
        layout="wide",
        page_title="MLIP Arena",
        page_icon=":shark:",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "https://github.com/atomind-ai/mlip-arena",
            "Report a bug": "https://github.com/atomind-ai/mlip-arena/issues/new",
        },
    )

st.toast(
    "MLIP Arena is currently in **pre-alpha**. The results are not stable. Please interpret them with care. Contributions are welcome. For more information, visit https://github.com/atomind-ai/mlip-arena.",
    icon="🍞",
)

# st.sidebar.markdown(
# """
# Other benchmarks
# """
# )
# st.sidebar.page_link(
#     "https://matbench-discovery.materialsproject.org/", label="Matbench Discovery", icon=":material/extension:"
# )

pg.run()
