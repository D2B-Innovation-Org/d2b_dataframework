import csv
import json

import pytest


class RecordingLogger:
    """Logger double that records what the module logged."""

    def __init__(self):
        self.logs = []
        self.criticals = []

    def log(self, message):
        self.logs.append(message)

    def critical(self, message, current_workflow_name=None):
        self.criticals.append((message, current_workflow_name))

    def has_log(self, fragment):
        return any(fragment in message for message in self.logs)

    def has_critical(self, fragment):
        return any(fragment in message for message, _ in self.criticals)


@pytest.fixture
def logger():
    return RecordingLogger()


@pytest.fixture
def credentials_csv(tmp_path):
    """Builds a credentials CSV with the exact headers the module expects."""

    def _build(rows, headers=("Nombre", "project_id", "JSON")):
        path = tmp_path / "credentials.csv"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
        return str(path)

    return _build


@pytest.fixture
def valid_credentials():
    return json.dumps({
        "type": "service_account",
        "project_id": "d2b-cliente-uno",
        "private_key": "-----BEGIN PRIVATE KEY-----fake-----END PRIVATE KEY-----",
    })
