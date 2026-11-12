"""Clasificación de errores y resultado final de las funciones el-*.

Todas las funciones de extracción (el-facebook, el-ga4, el-ga4-adc,
el-linkedin, el-search-console, el-tiktok) siguen el mismo contrato:

- Un error de permiso o credencial (401/403, token inválido) corta la cuenta
  afectada: reintentar no sirve y solo suma llamadas a la API.
- Un error transitorio (429, 5xx, timeout) se reintenta con espera creciente.
- Una cuenta sin filas es una carga correcta, no un error.
- Al terminar, la función responde según lo que pasó con todas las cuentas:

    todo ok o sin datos       -> 200, sin alerta
    algo cargó y algo falló   -> 200 "carga parcial", con alerta
    nada cargó y hubo errores -> 500, con alerta

Las alertas viven aquí y no en los workflows: el workflow solo ve el código
HTTP.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from d2b_data.load_guard import STATUS_ERROR, STATUS_OK


T = TypeVar("T")

# Textos que las APIs usan para errores de permiso o de credencial cuando la
# excepción no trae un código HTTP legible.
_AUTH_MARKERS = (
    "permission_denied",
    "insufficient permission",
    "sufficient permissions",
    "does not have permission",
    "unauthorized",
    "unauthenticated",
    "invalid_grant",
    "invalid_token",
    "access token is invalid",
    "access token has expired",
    "token has been expired or revoked",
)

_TRANSIENT_MARKERS = (
    "rate limit",
    "too many requests",
    "quota exceeded",
    "temporarily unavailable",
    "internal error",
    "backend error",
    "deadline exceeded",
    "timed out",
)

MAX_ALERT_ACCOUNTS = 10


class ErrorKind(str, Enum):
    """Tipo de error, que decide si se reintenta y si se corta la cuenta."""

    AUTH = "auth"
    TRANSIENT = "transient"
    OTHER = "other"


class ExtractionError(RuntimeError):
    """Error de una API de origen con el código HTTP que lo describe.

    Las clases de d2b_data lanzan esta excepción, o una subclase, cuando la
    API responde con error, para que classify_error pueda decidir sin
    depender del texto del mensaje.

    Attributes:
        status_code: Código HTTP equivalente al error, o None si no se sabe.
        api_code: Código propio de la API de origen, si lo trae.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        api_code: Optional[Any] = None,
    ) -> None:
        """Crea el error.

        Args:
            message: Descripción del error, idealmente con el texto de la API.
            status_code: Código HTTP equivalente.
            api_code: Código propio de la API de origen.
        """
        super().__init__(message)
        self.status_code = status_code
        self.api_code = api_code


def _status_code_of(error: BaseException) -> Optional[int]:
    """Busca el código HTTP en los formatos de excepción que usamos.

    Cubre ExtractionError, requests.HTTPError (response.status_code),
    googleapiclient HttpError (resp.status) y google.api_core (code).

    Args:
        error: Excepción a inspeccionar.

    Returns:
        El código HTTP como entero, o None si no se encuentra.
    """
    candidates = [getattr(error, "status_code", None)]

    response = getattr(error, "response", None)
    if response is not None:
        candidates.append(getattr(response, "status_code", None))

    resp = getattr(error, "resp", None)
    if resp is not None:
        candidates.append(getattr(resp, "status", None))

    candidates.append(getattr(error, "code", None))

    for candidate in candidates:
        # google.api_core usa un enum para code; solo sirven los enteros.
        if isinstance(candidate, int) and 100 <= candidate <= 599:
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def classify_error(error: BaseException) -> ErrorKind:
    """Clasifica una excepción de extracción.

    Primero usa el código HTTP; si no hay, el tipo de excepción y luego el
    texto del mensaje. Lo que no calza en ninguna regla es OTHER: no se
    reintenta, pero tampoco corta la cuenta.

    Args:
        error: Excepción lanzada al llamar a la API de origen.

    Returns:
        AUTH, TRANSIENT u OTHER.
    """
    status_code = _status_code_of(error)
    if status_code in (401, 403):
        return ErrorKind.AUTH
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return ErrorKind.TRANSIENT

    if isinstance(error, (TimeoutError, ConnectionError)):
        return ErrorKind.TRANSIENT

    text = str(error).lower()
    if any(marker in text for marker in _AUTH_MARKERS):
        return ErrorKind.AUTH
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return ErrorKind.TRANSIENT
    return ErrorKind.OTHER


def call_with_retries(
    func: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 2.0,
    logger: Optional[Any] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> T:
    """Ejecuta func y la reintenta solo ante errores transitorios.

    Los errores AUTH y OTHER se relanzan de inmediato: reintentar un 403
    no cambia el resultado.

    Args:
        func: Llamada sin argumentos, por ejemplo un lambda sobre la API.
        attempts: Intentos totales, incluido el primero. Mínimo 1.
        base_delay: Segundos de espera antes del segundo intento; se duplica
            en cada intento siguiente.
        logger: Objeto con warning(); si es None no registra nada.
        sleep: Función de espera; por defecto time.sleep, que se busca al
            llamar para que los tests puedan reemplazarla.

    Returns:
        Lo que devuelva func.

    Raises:
        ValueError: Si attempts es menor que 1.
        Exception: La última excepción de func cuando no se reintenta o se
            agotan los intentos.
    """
    if attempts < 1:
        raise ValueError(f"attempts debe ser al menos 1, llegó {attempts}")
    wait_for = sleep if sleep is not None else time.sleep

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:
            kind = classify_error(error)
            if kind is not ErrorKind.TRANSIENT or attempt == attempts:
                raise
            wait = base_delay * (2 ** (attempt - 1)) + random.random()
            if logger is not None:
                logger.warning(
                    f"load_result | error transitorio (intento {attempt} de "
                    f"{attempts}), se reintenta en {wait:.1f}s: {error}"
                )
            wait_for(wait)

    # Inalcanzable: el bucle devuelve o relanza en el último intento.
    raise RuntimeError("call_with_retries terminó sin resultado")


@dataclass
class AccountResult:
    """Lo que pasó con una cuenta durante la carga.

    Attributes:
        account: Id de la cuenta, property o sitio.
        rows: Filas escritas en BigQuery.
        tables_ok: Fragmentos escritos sin error.
        tables_failed: Fragmentos o días que fallaron.
        error: Error de extracción que cortó la cuenta, si hubo.
        error_kind: Tipo de ese error.
    """

    account: str
    rows: int = 0
    tables_ok: int = 0
    tables_failed: list[str] = field(default_factory=list)
    error: Optional[str] = None
    error_kind: Optional[ErrorKind] = None

    def fail(self, error: BaseException) -> ErrorKind:
        """Registra un error de extracción de la cuenta.

        Args:
            error: Excepción de la API de origen.

        Returns:
            El tipo de error, para que la función decida si corta la cuenta.
        """
        kind = classify_error(error)
        self.error = f"{type(error).__name__}: {error}"[:1000]
        self.error_kind = kind
        return kind

    @property
    def has_errors(self) -> bool:
        """True si la cuenta tuvo un error de extracción o una tabla fallida."""
        return self.error is not None or len(self.tables_failed) > 0

    @property
    def guard_status(self) -> str:
        """Estado para LoadGuard.record: error si algo falló, ok si no."""
        return STATUS_ERROR if self.has_errors else STATUS_OK

    @property
    def guard_error(self) -> Optional[str]:
        """Texto de error para LoadGuard.record, o None si no hubo."""
        parts = []
        if self.error is not None:
            parts.append(self.error)
        if self.tables_failed:
            failed_preview = ", ".join(self.tables_failed[:5])
            parts.append(f"{len(self.tables_failed)} tablas fallaron: {failed_preview}")
        if not parts:
            return None
        return " | ".join(parts)[:1000]

    def summary(self) -> str:
        """Una línea legible para la alerta."""
        if self.error is not None:
            kind = self.error_kind.value if self.error_kind else "other"
            return f"{self.account}: {kind} - {self.error[:200]}"
        failed_preview = ", ".join(self.tables_failed[:3])
        return (
            f"{self.account}: {len(self.tables_failed)} tablas fallaron "
            f"({failed_preview}), {self.tables_ok} OK"
        )


class LoadReport:
    """Junta los resultados por cuenta y arma la respuesta HTTP final.

    Attributes:
        source: Fuente de datos, por ejemplo 'tiktok'.
        target: Destino, en formato 'proyecto.dataset.prefijo'.
    """

    def __init__(self, source: str, target: str, logger: Any) -> None:
        """Crea el reporte vacío.

        Args:
            source: Fuente de datos.
            target: Destino en BigQuery.
            logger: WorkflowLogger, o cualquier objeto con info() y
                critical().
        """
        self.source = source
        self.target = target
        self.logger = logger
        self._accounts: dict[str, AccountResult] = {}

    def account(self, account: Any) -> AccountResult:
        """Devuelve el resultado de una cuenta, creándolo si no existe.

        Args:
            account: Id de la cuenta.

        Returns:
            El AccountResult de esa cuenta.
        """
        key = str(account)
        if key not in self._accounts:
            self._accounts[key] = AccountResult(account=key)
        return self._accounts[key]

    @property
    def results(self) -> list[AccountResult]:
        """Resultados por cuenta, en el orden en que se agregaron."""
        return list(self._accounts.values())

    def finish(self) -> tuple[str, int]:
        """Registra el cierre, alerta si corresponde y arma la respuesta.

        Returns:
            Tupla (mensaje, código HTTP) lista para devolver desde la función.
        """
        results = self.results
        total_rows = sum(result.rows for result in results)
        total_tables_ok = sum(result.tables_ok for result in results)
        with_errors = [result for result in results if result.has_errors]
        label = f"{self.source} -> {self.target}"

        if not with_errors:
            if total_rows == 0:
                self.logger.info(f"{label} | END OK, sin datos para el rango")
                return "200 - sin datos", 200
            self.logger.info(
                f"{label} | END OK, {total_rows} filas en {total_tables_ok} tablas"
            )
            return "200", 200

        lines = [result.summary() for result in with_errors[:MAX_ALERT_ACCOUNTS]]
        hidden = len(with_errors) - len(lines)
        if hidden > 0:
            lines.append(f"... y {hidden} cuentas más")
        detail = "\n".join(lines)

        if total_tables_ok > 0:
            self.logger.critical(
                f"Carga parcial en {label}: {len(with_errors)} de {len(results)} "
                f"cuentas con errores, {total_tables_ok} tablas OK\n{detail}"
            )
            return (
                f"200 - carga parcial, {len(with_errors)} cuentas con errores",
                200,
            )

        self.logger.critical(
            f"No se cargó nada en {label}: {len(with_errors)} cuentas con "
            f"errores\n{detail}"
        )
        return f"Carga fallida: {len(with_errors)} cuentas con errores", 500
