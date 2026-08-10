import logging
from contextlib import contextmanager

import pytest
import requests

from d2b_data.workflow_logger import WorkflowLogger


WEBHOOK = "https://example.com/hook"


@contextmanager
def capture(wf, caplog):
    """Routes the logger's records into caplog.

    WorkflowLogger sets propagate=False, so records never reach the root
    logger that caplog listens on by default. Attaching caplog's own handler
    to the instance is what makes the assertions see anything.
    """
    caplog.clear()
    caplog.handler.setLevel(logging.NOTSET)
    wf.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        wf.logger.removeHandler(caplog.handler)


@pytest.fixture(autouse=True)
def clean_loggers():
    """WorkflowLogger reuses logging.getLogger(name); reset between tests."""
    yield
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("wf-"):
            logger = logging.getLogger(name)
            logger.handlers.clear()


def build(name="wf-test", **kwargs):
    return WorkflowLogger(workflow_name=name, **kwargs)


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_instance_is_created_with_defaults():
    """Defaults enable both logging and alerts."""
    wf = build()
    assert wf.workflow_name == "wf-test"
    assert wf.active is True
    assert wf.alerts_enabled is True
    assert wf.webhook_url is None


def test_console_handler_is_configured():
    """A stream handler with the expected format is attached."""
    wf = build("wf-handler")
    assert len(wf.logger.handlers) == 1
    handler = wf.logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert "%(levelname)s" in handler.formatter._fmt


def test_handler_is_not_duplicated_on_second_instance():
    """Two loggers with the same name must not stack handlers."""
    build("wf-dup")
    wf = build("wf-dup")
    assert len(wf.logger.handlers) == 1


def test_propagate_is_disabled():
    """Messages must not bubble up to the root logger and print twice."""
    assert build("wf-prop").logger.propagate is False


def test_custom_level_is_applied():
    """The level argument reaches both the logger and its handler."""
    wf = build("wf-level", level=logging.DEBUG)
    assert wf.logger.level == logging.DEBUG
    assert wf.logger.handlers[0].level == logging.DEBUG


# --------------------------------------------------------------------- #
# Niveles de log
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("method,level", [
    ("debug", "DEBUG"),
    ("info", "INFO"),
    ("warning", "WARNING"),
    ("error", "ERROR"),
])
def test_each_level_emits_a_record(method, level, caplog):
    """Every level method emits a record at its own severity."""
    wf = build("wf-levels", level=logging.DEBUG, alerts_enabled=False)
    with capture(wf, caplog):
        getattr(wf, method)("mensaje de prueba")
    assert caplog.records[-1].levelname == level
    assert caplog.records[-1].message == "mensaje de prueba"


def test_critical_emits_a_record(caplog):
    """critical logs at CRITICAL severity."""
    wf = build("wf-crit", alerts_enabled=False)
    with capture(wf, caplog):
        wf.critical("todo mal")
    assert caplog.records[-1].levelname == "CRITICAL"


@pytest.mark.parametrize("method", ["debug", "info", "warning", "error", "critical"])
def test_active_false_silences_every_level(method, caplog):
    """active=False suppresses all log output."""
    wf = build("wf-silent", active=False, level=logging.DEBUG, alerts_enabled=False)
    with capture(wf, caplog):
        getattr(wf, method)("no debería aparecer")
    assert caplog.records == []


def test_error_forwards_exc_info(caplog, mocker):
    """exc_info is passed through to the underlying logger."""
    wf = build("wf-exc")
    spy = mocker.spy(wf.logger, "error")
    wf.error("falló", exc_info=True)
    assert spy.call_args.kwargs["exc_info"] is True


# --------------------------------------------------------------------- #
# set_workflow_name
# --------------------------------------------------------------------- #
def test_set_workflow_name_updates_both_names():
    """Renaming updates the attribute and the underlying logger name."""
    wf = build("wf-old")
    wf.set_workflow_name("wf-new")
    assert wf.workflow_name == "wf-new"
    assert wf.logger.name == "wf-new"


# --------------------------------------------------------------------- #
# Alertas por webhook
# --------------------------------------------------------------------- #
def test_critical_sends_alert_when_webhook_configured(mocker):
    """A critical with a webhook posts the message with the workflow prefix."""
    post = mocker.patch("d2b_data.workflow_logger.requests.post")
    wf = build("wf-alert", webhook_url=WEBHOOK)

    wf.critical("BigQuery caído")

    post.assert_called_once()
    assert post.call_args.args[0] == WEBHOOK
    assert post.call_args.kwargs["json"] == {"message": "[wf-alert] BigQuery caído"}
    assert post.call_args.kwargs["timeout"] == 15


def test_critical_skips_alert_when_alerts_disabled(mocker):
    """alerts_enabled=False never hits the network."""
    post = mocker.patch("d2b_data.workflow_logger.requests.post")
    build("wf-noalert", alerts_enabled=False, webhook_url=WEBHOOK).critical("x")
    post.assert_not_called()


def test_critical_skips_alert_when_send_alert_false(mocker):
    """send_alert=False suppresses the alert for a single call."""
    post = mocker.patch("d2b_data.workflow_logger.requests.post")
    build("wf-nosend", webhook_url=WEBHOOK).critical("x", send_alert=False)
    post.assert_not_called()


def test_alert_without_webhook_logs_a_warning(caplog):
    """Without a webhook URL the alert is skipped with a warning."""
    wf = build("wf-nourl")
    with capture(wf, caplog):
        wf.critical("sin webhook")
    assert any("webhook_url no está configurado" in r.message for r in caplog.records)


def test_alert_logs_error_on_http_failure(caplog, mocker):
    """A failing webhook is logged, never raised."""
    response = mocker.MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("500")
    mocker.patch("d2b_data.workflow_logger.requests.post", return_value=response)

    wf = build("wf-httpfail", webhook_url=WEBHOOK)
    with capture(wf, caplog):
        wf.critical("boom")
    assert any("No se pudo enviar la alerta" in r.message for r in caplog.records)


def test_alert_logs_error_on_connection_failure(caplog, mocker):
    """A network error is caught the same way."""
    mocker.patch(
        "d2b_data.workflow_logger.requests.post",
        side_effect=requests.ConnectionError("sin red"),
    )
    wf = build("wf-connfail", webhook_url=WEBHOOK)
    with capture(wf, caplog):
        wf.critical("boom")
    assert any("No se pudo enviar la alerta" in r.message for r in caplog.records)


def test_alert_logs_success(caplog, mocker):
    """A successful alert is confirmed in the log."""
    mocker.patch("d2b_data.workflow_logger.requests.post", return_value=mocker.MagicMock())
    wf = build("wf-ok", webhook_url=WEBHOOK)
    with capture(wf, caplog):
        wf.critical("boom")
    assert any("Alerta crítica enviada" in r.message for r in caplog.records)


def test_alert_is_sent_even_when_logging_is_inactive(mocker):
    """active=False silences logs but must not silence alerts."""
    post = mocker.patch("d2b_data.workflow_logger.requests.post")
    build("wf-inactive", active=False, webhook_url=WEBHOOK).critical("boom")
    post.assert_called_once()
