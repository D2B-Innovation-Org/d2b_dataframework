import logging
import warnings
from typing import Any, Optional

import requests


def resolve_logger(logger: Any, verbose_logger: Any, owner: str) -> Any:
    """Elige el logger entre el parámetro nuevo y el alias obsoleto.

    Las clases de d2b_data recibían el logger como `verbose_logger`; ahora el
    parámetro se llama `logger`. Mientras haya funciones que usen el nombre
    antiguo se acepta, pero se avisa con un DeprecationWarning.

    Args:
        logger: Valor del parámetro `logger`.
        verbose_logger: Valor del parámetro obsoleto `verbose_logger`.
        owner: Nombre de la clase o función, para el mensaje del aviso.

    Returns:
        `logger` si viene; si no, `verbose_logger`; None si no viene ninguno.
    """
    if verbose_logger is None:
        return logger
    warnings.warn(
        f"{owner}: el parámetro 'verbose_logger' está obsoleto, usa 'logger'",
        DeprecationWarning,
        stacklevel=3,
    )
    return logger if logger is not None else verbose_logger


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

    def log(self, message: str) -> None:
        """Registra un mensaje informativo; alias de info().

        Existe para que WorkflowLogger reemplace a Verbose sin cambiar las
        clases de d2b_data que llaman a `verbose.log(...)`.

        Args:
            message: Texto a registrar.
        """
        self.info(message)

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
        current_workflow_name: Optional[str] = None,
    ) -> None:
        """Registra un error crítico y, si corresponde, envía la alerta.

        Args:
            message: Texto del error.
            exc_info: Si es True, adjunta el traceback de la excepción activa.
            send_alert: Si es False, no envía la alerta en esta llamada.
            current_workflow_name: Nombre a usar en la alerta en vez del
                workflow_name de la instancia. Mantiene la firma de
                Verbose.critical para que las clases existentes funcionen.
        """
        if self.active:
            self.logger.critical(message, exc_info=exc_info)

        if self.alerts_enabled and send_alert:
            self._send_alert(message, current_workflow_name)

    def _send_alert(
        self, message: str, workflow_name: Optional[str] = None
    ) -> None:
        if not self.webhook_url:
            self.logger.warning(
                "No se envió la alerta: webhook_url no está configurado."
            )
            return

        payload = {
            "message": f"[{workflow_name or self.workflow_name}] {message}",
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
