from pathlib import Path

import pytest

from app.core import persistent_logs


@pytest.fixture(scope="session", autouse=True)
def isolate_stream_session_artifacts(tmp_path_factory):
    """Never let backend QA overwrite artifacts belonging to real streams."""
    original = persistent_logs.SESSION_LOG_DIR
    persistent_logs.SESSION_LOG_DIR = Path(tmp_path_factory.mktemp("stream-session-artifacts"))
    try:
        yield
    finally:
        persistent_logs.SESSION_LOG_DIR = original
