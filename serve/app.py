from collections import defaultdict

import streamlit as st

from mlip_arena.tasks import REGISTRY as TASKS


leaderboard = st.Page(
    "leaderboard.py", title="Leaderboard", icon=":material/trophy:", default=True
)

nav = defaultdict(list)
nav[""].append(leaderboard)

wide_pages, centered_pages = [], []

for task in TASKS:
    if TASKS[task]['task-page'] is None:
        continue
    page = st.Page(
        f"tasks/{TASKS[task]['task-page']}.py", title=task, icon=":material/target:"
    )
    nav[TASKS[task]["category"]].append(page)
    if TASKS[task]["task-layout"] == "wide":
        wide_pages.append(page)
    else:
        centered_pages.append(page)

pg = st.navigation(nav, expanded=True)

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


# st.toast(
#     "MLIP Arena is currently in **pre-alpha**. The results are not stable. Please interpret them with care. Contributions are welcome. For more information, visit https://github.com/atomind-ai/mlip-arena.",
#     icon="🍞",
# )

st.sidebar.page_link(
    "https://github.com/atomind-ai/mlip-arena", label="GitHub Repository", icon=":material/code:"
)

st.sidebar.markdown(
"""
Complementary Benchmarks
"""
)
st.sidebar.page_link(
    "https://matbench-discovery.materialsproject.org/", label="Matbench Discovery", icon=":material/link:"
)
st.sidebar.page_link(
    "https://openkim.org/", label="OpenKIM", icon=":material/link:"
)

pg.run()
