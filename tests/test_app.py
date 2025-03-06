import streamlit as st
from streamlit.testing.v1 import AppTest
import pytest
from pathlib import Path

path = Path(__file__).parents[1] / "serve" 

@pytest.fixture
def home():
    at = AppTest.from_file(str(path / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at

def test_leaderboard(home):
    # Test the leaderboard page by simulating navigation.
    at = home.switch_page(str(path / "leaderboard.py"))
    assert not at.exception
    
def test_task_pages(home):
    # Test each task page using the TASKS registry.
    from mlip_arena.tasks import REGISTRY as TASKS

    for task, details in TASKS.items():
        page_path = str(path / f"tasks/{details['task-page']}.py")
        at = home.switch_page(page_path)
        assert not at.exception
