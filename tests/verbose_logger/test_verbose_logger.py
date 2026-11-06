"""Tests for the legacy Verbose logger.

This module is slated for removal once the callers migrate to WorkflowLogger,
but it is still used across the framework, so it stays covered meanwhile.
"""

import pytest

from d2b_data.verbose_logger import Verbose, verbose_logger


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


def test_critical_is_silent_when_inactive(post, capsys):
    """active=False hides the critical print."""
    Verbose(active=False, workflow_name="wf").critical("todo mal")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("alerts_enabled", [True, False])
def test_critical_never_sends_alerts(post, alerts_enabled):
    """Verbose no longer posts to any webhook; alerts live in WorkflowLogger."""
    Verbose(alerts_enabled=alerts_enabled, workflow_name="wf").critical("todo mal")
    post.assert_not_called()


def test_no_webhook_in_source():
    """The public repo must not carry a webhook URL in Verbose."""
    import inspect

    import d2b_data.verbose_logger as module

    source = inspect.getsource(module)
    assert "http" not in source
    assert "requests" not in source
