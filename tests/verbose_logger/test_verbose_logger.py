"""Tests for the legacy Verbose logger.

This module is slated for removal once the callers migrate to WorkflowLogger,
but it is still used across the framework, so it stays covered meanwhile.
"""

import pytest

from d2b_data.verbose_logger import Verbose, verbose_logger


BOT_URL = "https://us-central1-d2b-data-management.cloudfunctions.net/innovation-messenger-hangout"


@pytest.fixture
def post(mocker):
    """Verbose imports requests lazily inside critical(), so patch the module."""
    return mocker.patch("requests.post")


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_instance_is_created_with_defaults():
    """Defaults enable logging and alerts."""
    v = Verbose()
    assert v.active is True
    assert v.alerts_enabled is True
    assert v.workflow_name == "UnknownWorkflow"
    assert v.bot_url == BOT_URL


def test_module_level_singleton_is_active():
    """The shared instance other modules import is enabled by default."""
    assert isinstance(verbose_logger, Verbose)
    assert verbose_logger.active is True


# --------------------------------------------------------------------- #
# log
# --------------------------------------------------------------------- #
def test_log_prints_with_timestamp(capsys):
    """Messages are prefixed with a bracketed timestamp."""
    Verbose(workflow_name="wf").log("hola")
    out = capsys.readouterr().out
    assert out.startswith("[")
    assert "]: hola" in out


def test_log_is_silent_when_inactive(capsys):
    """active=False suppresses regular logs."""
    Verbose(active=False).log("hola")
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------- #
# set_workflow_name
# --------------------------------------------------------------------- #
def test_set_workflow_name_updates_the_name():
    """The workflow name can be changed at runtime."""
    v = Verbose(workflow_name="viejo")
    v.set_workflow_name("nuevo")
    assert v.workflow_name == "nuevo"


# --------------------------------------------------------------------- #
# critical
# --------------------------------------------------------------------- #
def test_critical_prints_with_workflow_prefix(post, capsys):
    """The critical message carries the workflow name."""
    Verbose(workflow_name="etl-ga4").critical("todo mal")
    assert "CRITICAL: [etl-ga4] todo mal" in capsys.readouterr().out


def test_critical_uses_the_override_workflow_name(post, capsys):
    """current_workflow_name wins over the instance name."""
    Verbose(workflow_name="etl-ga4").critical("todo mal", current_workflow_name="etl-bq")
    assert "[etl-bq] todo mal" in capsys.readouterr().out


def test_critical_falls_back_to_unknown_workflow(post, capsys):
    """An empty workflow name degrades to 'UnknownWorkflow'."""
    v = Verbose(workflow_name="")
    v.critical("todo mal")
    assert "[UnknownWorkflow]" in capsys.readouterr().out


def test_critical_is_silent_but_still_alerts_when_inactive(post, capsys):
    """active=False hides the print but the alert still goes out."""
    Verbose(active=False, workflow_name="wf").critical("todo mal")
    assert "CRITICAL" not in capsys.readouterr().out
    post.assert_called_once()


def test_critical_posts_the_alert(post):
    """The alert payload carries the prefixed message."""
    post.return_value.status_code = 200
    Verbose(workflow_name="wf").critical("todo mal")

    post.assert_called_once()
    assert post.call_args.args[0] == BOT_URL
    assert post.call_args.kwargs["json"] == {"message": "[wf] todo mal"}
    assert post.call_args.kwargs["timeout"] == 15


def test_critical_confirms_a_successful_alert(post, capsys):
    """A 2xx response is reported as sent."""
    post.return_value.status_code = 200
    Verbose(workflow_name="wf").critical("todo mal")
    assert "Critical alert sent successfully" in capsys.readouterr().out


def test_critical_reports_a_rejected_alert(post, capsys):
    """A non-2xx response is reported with its status and body."""
    post.return_value.status_code = 500
    post.return_value.text = "server error"
    Verbose(workflow_name="wf").critical("todo mal")

    out = capsys.readouterr().out
    assert "Failed to send critical alert" in out
    assert "500" in out and "server error" in out


def test_critical_swallows_network_errors(post, capsys):
    """An exception sending the alert never propagates to the caller."""
    post.side_effect = RuntimeError("sin red")
    Verbose(workflow_name="wf").critical("todo mal")
    assert "Exception sending critical alert" in capsys.readouterr().out


def test_critical_skips_the_alert_when_disabled(post):
    """alerts_enabled=False never hits the network."""
    Verbose(alerts_enabled=False).critical("todo mal")
    post.assert_not_called()


def test_critical_alert_status_is_hidden_when_inactive(post, capsys):
    """With active=False no status line is printed even on failure."""
    post.return_value.status_code = 500
    post.return_value.text = "nope"
    Verbose(active=False).critical("todo mal")
    assert capsys.readouterr().out == ""
