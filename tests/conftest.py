import os
import tempfile

# Set PREFECT_HOME and SQLite DB URL at top-level of conftest.py
# pytest loads conftest.py before importing any test modules.
# This ensures Prefect reads the isolated env vars upon initial import.
worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
temp_dir = os.path.join(tempfile.gettempdir(), f"prefect_home_{worker_id}")
os.makedirs(temp_dir, exist_ok=True)

os.environ["PREFECT_HOME"] = temp_dir
os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = f"sqlite+aiosqlite:///{temp_dir}/prefect.db"
