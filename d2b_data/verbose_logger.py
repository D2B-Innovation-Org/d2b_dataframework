"""Logger legado de consola.

Verbose ya no envía alertas: solo imprime. Las alertas por webhook viven en
WorkflowLogger (d2b_data.workflow_logger), que recibe la URL desde Secret
Manager. Los flujos que necesiten alertas deben migrar a WorkflowLogger.
"""

from datetime import datetime


class Verbose:
    def __init__(self, active=True, alerts_enabled=True, workflow_name="UnknownWorkflow"):
        """Crea el logger.

        Args:
            active: Si es False no imprime nada.
            alerts_enabled: Se mantiene para no romper a quienes lo pasan; ya no
                tiene efecto porque Verbose no envía alertas.
            workflow_name: Prefijo de los mensajes de critical().
        """
        self.active = active
        self.alerts_enabled = alerts_enabled
        self.workflow_name = workflow_name

    def log(self, msg):
        if self.active:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}]: {msg}")

    def critical(self, msg, current_workflow_name=None):
        """Imprime un error crítico. No envía alertas; para eso usar WorkflowLogger."""
        name_to_use = current_workflow_name or self.workflow_name or "UnknownWorkflow"
        if self.active:
            print(f"CRITICAL: [{name_to_use}] {msg}")

    def set_workflow_name(self, name):
        self.workflow_name = name


# CONFIGURACIÓN GLOBAL
VERBOSE_ACTIVE = True
ALERTS_ENABLED = True
verbose_logger = Verbose(active=VERBOSE_ACTIVE, alerts_enabled=ALERTS_ENABLED)
