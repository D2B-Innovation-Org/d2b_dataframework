import logging
from typing import Optional

import requests


class WorkflowLogger:
    """
    Centralized logger for workflow execution.

    Wraps Python's built-in logging module and provides optional
    webhook-based notifications for critical events.

    Features:
        - Standard logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - Workflow context in log messages
        - Configurable console output
        - Optional webhook alerts for critical failures
        - Runtime enable/disable of logging and alerts

    Intended for ETL pipelines, Cloud Run services, scheduled jobs,
    and other automated workflows.
    """

    def __init__(
        self,
        workflow_name: str = "unknown",
        active: bool = True,
        alerts_enabled: bool = True,
        webhook_url: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        self.workflow_name = workflow_name
        self.active = active
        self.alerts_enabled = alerts_enabled
        self.webhook_url = webhook_url

        self.logger = logging.getLogger(workflow_name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        self._configure_console_handler(level)

    def _configure_console_handler(self, level: int) -> None:
        if self.logger.handlers:
            return

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_workflow_name(self, workflow_name: str) -> None:
        """
        Setter function used to change the name of the workflow during execution
        Args: A string with the workflow name.

        Return:
            None.
        """
        self.workflow_name = workflow_name
        self.logger.name = workflow_name

    def debug(self, message: str) -> None:
        if self.active:
            self.logger.debug(message)

    def info(self, message: str) -> None:
        if self.active:
            self.logger.info(message)

    def warning(self, message: str) -> None:
        if self.active:
            self.logger.warning(message)

    def error(
        self,
        message: str,
        *,
        exc_info: bool = False,
    ) -> None:
        if self.active:
            self.logger.error(message, exc_info=exc_info)

    def critical(
        self,
        message: str,
        *,
        exc_info: bool = False,
        send_alert: bool = True,
    ) -> None:
        if self.active:
            self.logger.critical(message, exc_info=exc_info)

        if self.alerts_enabled and send_alert:
            self._send_alert(message)

    def _send_alert(self, message: str) -> None:
        if not self.webhook_url:
            self.logger.warning(
                "No se envió la alerta: webhook_url no está configurado."
            )
            return

        payload = {
            "message": f"[{self.workflow_name}] {message}",
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()

            if self.active:
                self.logger.info("Alerta crítica enviada correctamente.")

        except requests.RequestException as error:
            self.logger.error(
                "No se pudo enviar la alerta crítica: %s",
                error,
                exc_info=True,
            )
