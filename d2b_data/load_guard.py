"""Registro de cargas y chequeo de frescura para las funciones el-*.

Cada carga exitosa deja una fila por cuenta en una tabla de control de
BigQuery. Antes de llamar a la API de origen, la función consulta esa tabla:
si todas las cuentas ya se cargaron hoy después de la hora de corte, se salta
la carga. Así varias corridas al día sirven de respaldo sin repetir llamadas
a la API cuando la primera ya funcionó.

Ante cualquier falla de la tabla de control el módulo responde "no cargado"
y la función carga igual: se puede perder el ahorro, nunca datos.
"""

from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from google.cloud import bigquery


DEFAULT_TABLE_ID = "d2b-data-management.ops.load_log"
DEFAULT_TIMEZONE = "America/Santiago"
STATUS_OK = "ok"
STATUS_ERROR = "error"

_ALREADY_LOADED_SQL = """
SELECT DISTINCT account
FROM `{table_id}`
WHERE DATE(loaded_at) >= DATE(@cutoff)
  AND loaded_at >= @cutoff
  AND source = @source
  AND target = @target
  AND account IN UNNEST(@accounts)
  AND status = @status
  AND date_since <= @since
  AND date_until >= @until
"""


class _NullLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass


def normalize_account(account: Any) -> str:
    """Convierte un id de cuenta al formato que se guarda en la tabla.

    Args:
        account: Id de la cuenta, como número o texto, con o sin el
            prefijo 'act_' de Meta.

    Returns:
        El id como texto y sin el prefijo 'act_'.
    """
    return str(account).strip().removeprefix("act_")


class LoadGuard:
    """Consulta y registra cargas en la tabla de control de BigQuery.

    Attributes:
        table_id: Tabla de control, en formato 'proyecto.dataset.tabla'.
        timezone: Zona horaria en la que se interpreta la hora de corte.
    """

    def __init__(
        self,
        client: Optional[bigquery.Client] = None,
        table_id: str = DEFAULT_TABLE_ID,
        timezone_name: str = DEFAULT_TIMEZONE,
        logger: Optional[Any] = None,
    ) -> None:
        """Crea el guard sin conectarse todavía a BigQuery.

        Args:
            client: Cliente de BigQuery. Si es None se crea al primer uso con
                las credenciales del entorno (la cuenta de servicio de la
                función), no con las del cliente.
            table_id: Tabla de control.
            timezone_name: Zona horaria de la hora de corte.
            logger: WorkflowLogger, o cualquier objeto con info() y
                warning(). Si es None no registra nada.
        """
        self._client = client
        self.table_id = table_id
        self.timezone = ZoneInfo(timezone_name)
        self.logger = logger if logger is not None else _NullLogger()

    @property
    def client(self) -> bigquery.Client:
        """Cliente de BigQuery, creado al primer uso."""
        if self._client is None:
            project = self.table_id.split(".")[0]
            self._client = bigquery.Client(project=project)
        return self._client

    def freshness_cutoff(
        self, fresh_after_hour: int, now: Optional[datetime] = None
    ) -> datetime:
        """Calcula desde qué momento de hoy una carga cuenta como fresca.

        Args:
            fresh_after_hour: Hora local (0 a 23) desde la que una carga de
                hoy se considera completa.
            now: Momento de referencia con zona horaria. Por defecto, ahora.

        Returns:
            El corte de hoy a esa hora local, expresado en UTC.

        Raises:
            ValueError: Si fresh_after_hour no está entre 0 y 23.
        """
        if not 0 <= fresh_after_hour <= 23:
            raise ValueError(
                f"fresh_after_hour debe estar entre 0 y 23, llegó {fresh_after_hour}"
            )
        reference = now if now is not None else datetime.now(timezone.utc)
        local_now = reference.astimezone(self.timezone)
        local_cutoff = local_now.replace(
            hour=fresh_after_hour, minute=0, second=0, microsecond=0
        )
        return local_cutoff.astimezone(timezone.utc)

    def already_loaded(
        self,
        source: str,
        target: str,
        accounts: Iterable[Any],
        since_date: date,
        until_date: date,
        fresh_after_hour: int = 6,
        now: Optional[datetime] = None,
    ) -> bool:
        """Indica si todas las cuentas ya se cargaron hoy para el rango pedido.

        Una cuenta cuenta como cargada si tiene una fila con status 'ok',
        loaded_at posterior al corte de hoy y un rango que cubre el pedido
        (date_since <= since_date y date_until >= until_date). Exigir el
        rango completo evita que un backfill de fechas antiguas se salte
        porque ya hubo una carga normal hoy. Si la consulta falla devuelve
        False para que la carga siga.

        Args:
            source: Fuente de datos, por ejemplo 'facebook'.
            target: Destino, en formato 'proyecto.dataset.tabla'.
            accounts: Cuentas que pide la carga.
            since_date: Primer día que debe cubrir la carga.
            until_date: Último día que debe cubrir la carga.
            fresh_after_hour: Hora local de corte, de 0 a 23.
            now: Momento de referencia con zona horaria. Por defecto, ahora.

        Returns:
            True si todas las cuentas ya están cargadas; False en otro caso.
        """
        wanted = {normalize_account(account) for account in accounts}
        if not wanted:
            return False

        cutoff = self.freshness_cutoff(fresh_after_hour, now)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff),
                bigquery.ScalarQueryParameter("source", "STRING", source),
                bigquery.ScalarQueryParameter("target", "STRING", target),
                bigquery.ArrayQueryParameter("accounts", "STRING", sorted(wanted)),
                bigquery.ScalarQueryParameter("status", "STRING", STATUS_OK),
                bigquery.ScalarQueryParameter("since", "DATE", since_date),
                bigquery.ScalarQueryParameter("until", "DATE", until_date),
            ]
        )
        sql = _ALREADY_LOADED_SQL.format(table_id=self.table_id)

        try:
            rows = self.client.query(sql, job_config=job_config).result()
            loaded = {row["account"] for row in rows}
        except Exception as error:
            self.logger.warning(
                f"load_guard | no se pudo leer {self.table_id}, "
                f"se carga igual: {error}"
            )
            return False

        missing = sorted(wanted - loaded)
        if missing:
            self.logger.info(
                f"load_guard | {source} -> {target}: faltan cargas de hoy para "
                f"{missing}"
            )
            return False

        self.logger.info(
            f"load_guard | {source} -> {target}: todas las cuentas ya se "
            f"cargaron hoy después de las {fresh_after_hour:02d}:00"
        )
        return True

    def record(
        self,
        *,
        source: str,
        target: str,
        client_name: str,
        account: Any,
        date_since: date,
        date_until: date,
        rows: int,
        status: str,
        error: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Registra el resultado de la carga de una cuenta.

        Nunca lanza excepciones: si no puede escribir, lo registra en el log
        y devuelve False, porque la carga en sí ya terminó.

        Args:
            source: Fuente de datos, por ejemplo 'facebook'.
            target: Destino, en formato 'proyecto.dataset.tabla'.
            client_name: Nombre del cliente, para consultas y paneles.
            account: Cuenta cargada.
            date_since: Primer día del rango pedido.
            date_until: Último día del rango pedido.
            rows: Filas escritas para esta cuenta; 0 si la API no trajo datos.
            status: STATUS_OK o STATUS_ERROR.
            error: Descripción del error cuando status es STATUS_ERROR.
            now: Momento de la carga con zona horaria. Por defecto, ahora.

        Returns:
            True si la fila se escribió; False si no.

        Raises:
            ValueError: Si status no es STATUS_OK ni STATUS_ERROR.
        """
        if status not in (STATUS_OK, STATUS_ERROR):
            raise ValueError(f"status inválido: {status}")

        loaded_at = now if now is not None else datetime.now(timezone.utc)
        row = {
            "loaded_at": loaded_at.astimezone(timezone.utc).isoformat(),
            "source": source,
            "target": target,
            "client": client_name,
            "account": normalize_account(account),
            "date_since": date_since.isoformat(),
            "date_until": date_until.isoformat(),
            "rows": int(rows),
            "status": status,
            "error": error,
        }

        try:
            insert_errors = self.client.insert_rows_json(self.table_id, [row])
        except Exception as insert_exception:
            insert_errors = [str(insert_exception)]

        if insert_errors:
            self.logger.warning(
                f"load_guard | no se pudo registrar la carga de "
                f"{row['account']} en {self.table_id}: {insert_errors}"
            )
            return False
        return True
