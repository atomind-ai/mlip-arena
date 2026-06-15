import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_prefect_home(tmp_path_factory, worker_id):
    """Isolate PREFECT_HOME for each test runner worker to prevent SQLite database lock conflicts / race conditions."""
    if worker_id == "master":
        # Single process run
        temp_dir = tmp_path_factory.mktemp("prefect_home")
    else:
        # Multi-process run under pytest-xdist
        temp_dir = tmp_path_factory.mktemp(f"prefect_home_{worker_id}")

    os.environ["PREFECT_HOME"] = str(temp_dir)
    # Also isolate the Prefect API URL and ensure it uses an in-memory database or isolated database
    os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = f"sqlite+aiosqlite:///{temp_dir}/prefect.db"
