import os
import tempfile
import threading
import time
import pandas as pd
from mlip_arena.data.local import SafeHDFStore


def test_safe_hdf_store_basic():
    """Test basic initialization, usage, lock creation, and cleanup of SafeHDFStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf_path = os.path.join(tmpdir, "test.h5")
        lock_path = f"{hdf_path}.lock"

        # Initially, the lock file should not exist
        assert not os.path.exists(lock_path)

        with SafeHDFStore(hdf_path, mode="w") as store:
            # Check lock file is created
            assert os.path.exists(lock_path)

            # Perform some simple pandas hdf store operations
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            store.put("data", df)

            # Read it back
            res = store.get("data")
            pd.testing.assert_frame_equal(df, res)

        # After exiting, the lock file should be cleaned up
        assert not os.path.exists(lock_path)


def test_safe_hdf_store_concurrency():
    """Test that SafeHDFStore blocks when lock exists, and proceeds when lock is released."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf_path = os.path.join(tmpdir, "test_concurrent.h5")
        lock_path = f"{hdf_path}.lock"

        # Create lock file manually to simulate another process holding the lock
        flock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        # Start a thread that will release the lock after a short delay
        def release_lock():
            time.sleep(0.5)
            os.close(flock)
            os.remove(lock_path)

        thread = threading.Thread(target=release_lock)
        start_time = time.time()
        thread.start()

        # Try to open the store. It should wait (probe_interval=0.1) until the thread releases the lock.
        with SafeHDFStore(hdf_path, mode="w", probe_interval=0.1) as store:
            duration = time.time() - start_time
            # Ensure it blocked for at least some time
            assert duration >= 0.4

            df = pd.DataFrame({"x": [10]})
            store.put("df", df)

        thread.join()
        assert not os.path.exists(lock_path)
