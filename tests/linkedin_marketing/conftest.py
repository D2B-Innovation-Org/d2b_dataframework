import json

import pytest

from d2b_data.linkedin_marketing import LinkedinMarketing


@pytest.fixture
def token_file(tmp_path):
    """A real token file on disk containing a valid access_token."""
    path = tmp_path / "token.json"
    path.write_text(json.dumps({"access_token": "fake_token_123"}))
    return str(path)


@pytest.fixture
def marketing(token_file):
    """LinkedinMarketing with a valid token loaded from a real file."""
    return LinkedinMarketing(token_path=token_file)


@pytest.fixture
def marketing_no_file():
    """LinkedinMarketing with no token file specified."""
    return LinkedinMarketing()


@pytest.fixture
def marketing_bad_file(tmp_path):
    """LinkedinMarketing pointing at a token file that does not exist."""
    return LinkedinMarketing(token_path=str(tmp_path / "missing.json"))


class RecordingLogger:
    """Minimal logger that records messages instead of emitting them."""

    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.critical_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def critical(self, message: str) -> None:
        self.critical_messages.append(message)

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)


@pytest.fixture
def logger():
    """A recording logger that can be injected into the client."""
    return RecordingLogger()
