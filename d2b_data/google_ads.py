"""Google Ads API reporting through the official ``google-ads`` library.

Runs GAQL queries with ``GoogleAdsService.search_stream`` and returns the rows
exactly as the API sends them: one key per path of the response
``field_mask``, enums by name, repeated fields as lists and nested messages
as dictionaries. No metric is derived and no value is converted
(``metrics.cost_micros`` stays in micros).

``google-ads`` is an optional dependency (``pip install "d2b_data[google_ads]"``)
because its wheel is large and most ``el-*`` functions do not need it. It is
imported only when the first API client is built, so this module can be
imported without it.

Error handling follows the lesson from the Meta and LinkedIn incidents:
retrying against a quota limit only burns more quota.

* Transient errors (gRPC ``UNAVAILABLE``, ``DEADLINE_EXCEEDED``, ``INTERNAL``
  and Google Ads ``InternalError``) are retried with exponential backoff.
* Quota and rate limit errors raise :class:`GoogleAdsQuotaError` at once.
* Anything else (authentication, permissions, invalid query) raises
  :class:`GoogleAdsRequestError` at once, with the ``request_id`` and every
  error message, which is what Google support asks for.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import date
from typing import Any

import pandas as pd

from d2b_data.workflow_logger import WorkflowLogger

ADWORDS_SCOPE = "https://www.googleapis.com/auth/adwords"
INSTALL_HINT = (
    "google-ads is required for GoogleAds: "
    'pip install "d2b_data[google_ads]" (or pip install google-ads)'
)
REQUIRED_CREDENTIALS = (
    "developer_token",
    "client_id",
    "client_secret",
    "refresh_token",
)
CREDENTIAL_KEYS = REQUIRED_CREDENTIALS + ("login_customer_id",)

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 5.0  # Google's error guide suggests 5 s, 10 s, 20 s.

# GAQL resource and field names: snake_case segments joined by dots.
GAQL_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
DATE_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DASHED_CUSTOMER_ID = re.compile(r"^\d{3}-\d{3}-\d{4}$")
PLAIN_CUSTOMER_ID = re.compile(r"^\d{10}$")

# The library's interceptor leaves INTERNAL and RESOURCE_EXHAUSTED as raw
# grpc.RpcError (never wrapped in GoogleAdsException), so both groups are
# matched by status code name as well as by Google Ads error enum.
TRANSIENT_STATUS_CODES = frozenset({"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"})
QUOTA_STATUS_CODES = frozenset({"RESOURCE_EXHAUSTED"})
TRANSIENT_INTERNAL_ERRORS = frozenset(
    {"INTERNAL_ERROR", "TRANSIENT_ERROR", "DEADLINE_EXCEEDED"}
)

_TRANSIENT = "transient"
_QUOTA = "quota"
_FATAL = "fatal"


class GoogleAdsQuotaError(Exception):
    """Raised when Google Ads rejects a request for quota or rate limits.

    The request is not retried: daily operation limits reset once a day and
    retrying only spends operations that the next accounts need.
    """


class GoogleAdsRequestError(Exception):
    """Raised when Google Ads rejects a request for a non-retryable reason.

    Covers authentication, permission and query errors, and transient errors
    that persisted after every retry. The original exception is chained as
    ``__cause__``.

    Attributes:
        request_id: Google Ads request id, or None when the API sent none.
        status: gRPC status code name, such as ``"INVALID_ARGUMENT"``.
    """

    def __init__(
        self,
        message: str,
        request_id: str | None = None,
        status: str | None = None,
    ) -> None:
        """Initializes the error.

        Args:
            message: Human readable description, including every API error.
            request_id: Google Ads request id, if any.
            status: gRPC status code name, if any.
        """
        super().__init__(message)
        self.request_id = request_id
        self.status = status


def normalize_customer_id(customer_id: str | int) -> str:
    """Normalizes a Google Ads customer id to ten digits without dashes.

    Args:
        customer_id: Id as ``"123-456-7890"``, ``"1234567890"`` or an integer.

    Returns:
        The id as a ten digit string.

    Raises:
        ValueError: If the id does not match any accepted format.
    """
    if isinstance(customer_id, bool):
        raise ValueError("customer_id must be a ten digit Google Ads id")
    if isinstance(customer_id, int):
        if not 0 < customer_id < 10**10:
            raise ValueError("customer_id must be a ten digit Google Ads id")
        # Integers lose leading zeros; padding restores the canonical form.
        return str(customer_id).zfill(10)
    if isinstance(customer_id, str):
        value = customer_id.strip()
        if DASHED_CUSTOMER_ID.match(value):
            return value.replace("-", "")
        if PLAIN_CUSTOMER_ID.match(value):
            return value
    raise ValueError(
        "customer_id must look like 123-456-7890, 1234567890 or be an integer"
    )


class GoogleAds:
    """Extract raw reporting data from the Google Ads API.

    Authenticates with user OAuth (developer token, client id, client secret
    and a refresh token with the ``adwords`` scope). The API client is built
    on the first request, so creating the object never calls the API.

    Attributes:
        developer_token: Google Ads developer token.
        client_id: OAuth client id.
        client_secret: OAuth client secret.
        refresh_token: OAuth refresh token with the ``adwords`` scope.
        login_customer_id: Manager (MCC) account id, ten digits, or None.
        api_version: API version such as ``"v26"``, or None for the
            library default.
        logger: Logger used for progress, retries and errors.
    """

    def __init__(
        self,
        developer_token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        login_customer_id: str | None = None,
        token_path: str | None = None,
        logger: WorkflowLogger | None = None,
        api_version: str | None = None,
    ) -> None:
        """Initializes the wrapper without calling the API.

        Explicit arguments take precedence; anything missing is read from
        ``token_path``, a JSON file with the same keys.

        Args:
            developer_token: Google Ads developer token.
            client_id: OAuth client id.
            client_secret: OAuth client secret.
            refresh_token: OAuth refresh token with the ``adwords`` scope.
            login_customer_id: Manager (MCC) account id used as login
                customer. Optional; omit it for accounts with direct access.
            token_path: Path to a JSON file with any of the keys
                ``developer_token``, ``client_id``, ``client_secret``,
                ``refresh_token`` and ``login_customer_id``.
            logger: WorkflowLogger instance. If omitted, an internal logger
                with alerts disabled is created.
            api_version: API version such as ``"v26"``. If omitted, the
                library default version is used.

        Raises:
            ValueError: If a required credential is missing (only the key
                names are reported, never the values) or login_customer_id
                is malformed.
            FileNotFoundError: If token_path does not exist.
        """
        self.logger: WorkflowLogger = logger or WorkflowLogger(
            workflow_name="GoogleAds", alerts_enabled=False
        )

        explicit: dict[str, Any] = {
            "developer_token": developer_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "login_customer_id": login_customer_id,
        }
        from_file: dict[str, Any] = (
            self._read_token_file(token_path) if token_path else {}
        )
        resolved: dict[str, Any] = {
            key: explicit[key] if explicit[key] else from_file.get(key)
            for key in CREDENTIAL_KEYS
        }

        missing = [key for key in REQUIRED_CREDENTIALS if not resolved[key]]
        if missing:
            raise ValueError("Missing Google Ads credentials: " + ", ".join(missing))

        self.developer_token: str = str(resolved["developer_token"])
        self.client_id: str = str(resolved["client_id"])
        self.client_secret: str = str(resolved["client_secret"])
        self.refresh_token: str = str(resolved["refresh_token"])
        self.login_customer_id: str | None = (
            normalize_customer_id(resolved["login_customer_id"])
            if resolved["login_customer_id"]
            else None
        )
        self.api_version: str | None = api_version
        self._client: Any = None

        self.logger.info(
            "GoogleAds initialized "
            f"(login_customer_id={self.login_customer_id or 'none'}, "
            f"api_version={self.api_version or 'library default'})"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_query(
        resource: str,
        fields: list[str],
        start_date: str,
        end_date: str,
        where: list[str] | None = None,
    ) -> str:
        """Builds a GAQL query filtered by ``segments.date``.

        Args:
            resource: GAQL resource, such as ``"ad_group_ad"``.
            fields: Fields to select, such as ``"metrics.cost_micros"``.
            start_date: First day, ``YYYY-MM-DD``.
            end_date: Last day, ``YYYY-MM-DD``, inclusive.
            where: Extra conditions joined with ``AND``. They come from the
                workflow YAML, which is trusted, so they are added verbatim.

        Returns:
            The GAQL query.

        Raises:
            ValueError: If fields is empty, a name is not a GAQL identifier,
                a date is malformed, start_date is after end_date or a
                condition contains ``;``.
        """
        if not fields:
            raise ValueError("fields must contain at least one field")
        for name in [resource, *fields]:
            if not isinstance(name, str) or not GAQL_IDENTIFIER.match(name):
                raise ValueError(f"Invalid GAQL identifier: {name!r}")

        start = _parse_date(start_date, "start_date")
        end = _parse_date(end_date, "end_date")
        if start > end:
            raise ValueError("start_date cannot be greater than end_date")

        conditions = [f"segments.date BETWEEN '{start_date}' AND '{end_date}'"]
        for condition in where or []:
            # GAQL has no statements to chain, but a ';' means the YAML is
            # malformed or someone is trying to sneak something in.
            if ";" in condition:
                raise ValueError(f"WHERE condition cannot contain ';': {condition!r}")
            conditions.append(condition)

        return (
            f"SELECT {', '.join(fields)} FROM {resource} "
            f"WHERE {' AND '.join(conditions)}"
        )

    def search(self, customer_id: str | int, query: str) -> list[dict[str, Any]]:
        """Runs a GAQL query with ``search_stream`` and returns the raw rows.

        Each row is a dict whose keys are exactly the paths of the response
        ``field_mask``, in that order. Values are read path by path instead
        of converting the whole ``GoogleAdsRow``, which has thousands of
        fields and would also drop zeros and empty strings.

        Args:
            customer_id: Account to query, in any format accepted by
                :func:`normalize_customer_id`.
            query: GAQL query.

        Returns:
            One dict per row. Enums are returned by name, repeated fields
            as lists and nested messages as dicts.

        Raises:
            ValueError: If customer_id is malformed.
            ImportError: If google-ads is not installed.
            GoogleAdsQuotaError: On quota or rate limit errors.
            GoogleAdsRequestError: On any other API error, or when a
                transient error persists after every retry.
        """
        account = normalize_customer_id(customer_id)
        service = self._get_client().get_service("GoogleAdsService")
        self.logger.info(f"Querying Google Ads account {account}")

        attempt = 0
        while True:
            try:
                rows = self._read_stream(service, account, query)
                break
            except Exception as error:
                kind = _classify_error(error)
                if kind == _QUOTA:
                    message = _describe_error(error)
                    self.logger.error(
                        f"Quota exceeded for account {account}: {message}"
                    )
                    raise GoogleAdsQuotaError(message) from error
                if kind == _TRANSIENT and attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE_SECONDS * 2**attempt + random.uniform(0, 1)
                    attempt += 1
                    self.logger.warning(
                        f"Transient error for account {account}: "
                        f"{_describe_error(error)}. Retry {attempt}/{MAX_RETRIES} "
                        f"in {delay:.1f} s."
                    )
                    time.sleep(delay)
                    continue
                if kind == _FATAL and not _is_api_error(error):
                    # Not an API response (for example a revoked refresh
                    # token raises google.auth RefreshError): leave it as is.
                    raise
                message = _describe_error(error)
                if kind == _TRANSIENT:
                    message = f"{message} (gave up after {MAX_RETRIES} retries)"
                self.logger.error(
                    f"Google Ads request failed for account {account}: {message}"
                )
                raise GoogleAdsRequestError(
                    message,
                    request_id=_request_id(error),
                    status=_status_name(error),
                ) from error

        self.logger.info(f"Account {account}: {len(rows)} rows")
        return rows

    def get_report_df(
        self,
        customer_id: str | int,
        resource: str,
        fields: list[str],
        start_date: str,
        end_date: str,
        where: list[str] | None = None,
    ) -> pd.DataFrame:
        """Runs a date-filtered report and returns it as a DataFrame.

        Columns are named after the ``field_mask`` paths and values are not
        converted.

        Args:
            customer_id: Account to query.
            resource: GAQL resource, such as ``"ad_group_ad"``.
            fields: Fields to select.
            start_date: First day, ``YYYY-MM-DD``.
            end_date: Last day, ``YYYY-MM-DD``, inclusive.
            where: Extra GAQL conditions joined with ``AND``.

        Returns:
            One row per API row. Without rows, an empty DataFrame whose
            columns are the requested fields.

        Raises:
            ValueError: If the query parameters or customer_id are invalid.
            GoogleAdsQuotaError: On quota or rate limit errors.
            GoogleAdsRequestError: On any other API error.
        """
        query = self.build_query(resource, fields, start_date, end_date, where)
        rows = self.search(customer_id, query)
        if not rows:
            return pd.DataFrame(columns=list(fields))
        return pd.DataFrame(rows)

    def list_accessible_customers(self) -> list[str]:
        """Lists the accounts the refresh token can access directly.

        Useful as a connection test: it needs valid credentials but no
        customer id.

        Returns:
            Ten digit customer ids.

        Raises:
            ImportError: If google-ads is not installed.
            GoogleAdsQuotaError: On quota or rate limit errors.
            GoogleAdsRequestError: On any other API error.
        """
        service = self._get_client().get_service("CustomerService")
        try:
            response = service.list_accessible_customers()
        except Exception as error:
            kind = _classify_error(error)
            if kind == _QUOTA:
                raise GoogleAdsQuotaError(_describe_error(error)) from error
            if not _is_api_error(error):
                raise
            raise GoogleAdsRequestError(
                _describe_error(error),
                request_id=_request_id(error),
                status=_status_name(error),
            ) from error
        # Resource names look like "customers/1234567890".
        return [name.split("/")[-1] for name in response.resource_names]

    @staticmethod
    def generate_refresh_token(
        client_secret_path: str,
        token_path: str | None = None,
        *,
        port: int = 0,
    ) -> str:
        """Runs the local OAuth flow and returns a refresh token.

        Interactive and local only: it opens a browser. Never call it from
        Cloud Run.

        Args:
            client_secret_path: OAuth client JSON downloaded from Google
                Cloud. A "Desktop app" client works with any port.
            token_path: If given, the client id, client secret and refresh
                token are written there as JSON, readable only by the owner.
            port: Local port for the redirect. 0 picks a free port; set it
                when the OAuth client only allows a registered redirect URI.

        Returns:
            The refresh token.

        Raises:
            RuntimeError: If Google did not return a refresh token.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(
            client_secret_path, scopes=[ADWORDS_SCOPE]
        )
        # prompt="consent" forces Google to issue a new refresh token even if
        # the user already authorized this client before.
        credentials = flow.run_local_server(
            port=port, access_type="offline", prompt="consent"
        )
        if not credentials.refresh_token:
            raise RuntimeError("Google did not return a refresh token")

        if token_path:
            payload = {
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "refresh_token": credentials.refresh_token,
            }
            descriptor = os.open(
                token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)

        return credentials.refresh_token

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_token_file(token_path: str) -> dict[str, Any]:
        """Reads credentials from a JSON file.

        Args:
            token_path: Path to the JSON file.

        Returns:
            The credential keys found in the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a JSON object.
        """
        with open(token_path, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("token_path must contain a JSON object")
        return {key: data.get(key) for key in CREDENTIAL_KEYS}

    def _get_client(self) -> Any:
        """Returns the cached GoogleAdsClient, building it on first use.

        Returns:
            A ``google.ads.googleads.client.GoogleAdsClient``.

        Raises:
            ImportError: If google-ads is not installed.
        """
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        """Builds the GoogleAdsClient from the stored credentials.

        Returns:
            A ``google.ads.googleads.client.GoogleAdsClient``.

        Raises:
            ImportError: If google-ads is not installed.
        """
        try:
            from google.ads.googleads.client import GoogleAdsClient
        except ImportError as error:
            raise ImportError(INSTALL_HINT) from error

        config: dict[str, Any] = {
            "developer_token": self.developer_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            # proto-plus exposes enums with .name and plain Python values.
            "use_proto_plus": True,
        }
        if self.login_customer_id:
            config["login_customer_id"] = self.login_customer_id
        return GoogleAdsClient.load_from_dict(config, version=self.api_version)

    @staticmethod
    def _read_stream(
        service: Any, customer_id: str, query: str
    ) -> list[dict[str, Any]]:
        """Consumes a whole ``search_stream`` response.

        Errors surface while iterating, not when the call is made, so a
        retry has to restart the stream from scratch.

        Args:
            service: GoogleAdsService client.
            customer_id: Ten digit customer id.
            query: GAQL query.

        Returns:
            One dict per row, keyed by the field_mask paths.
        """
        from google.ads.googleads.util import get_nested_attr

        rows: list[dict[str, Any]] = []
        stream = service.search_stream(customer_id=customer_id, query=query)
        for batch in stream:
            paths = list(batch.field_mask.paths)
            for row in batch.results:
                rows.append(
                    {path: _to_python(get_nested_attr(row, path)) for path in paths}
                )
        return rows


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #
def _parse_date(value: str, name: str) -> date:
    """Parses a ``YYYY-MM-DD`` date.

    Args:
        value: Date string.
        name: Parameter name, for the error message.

    Returns:
        The parsed date.

    Raises:
        ValueError: If the value is not a valid ``YYYY-MM-DD`` date.
    """
    if not isinstance(value, str) or not DATE_FORMAT.match(value):
        raise ValueError(f"{name} must use YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM-DD format") from error


def _to_python(value: Any) -> Any:
    """Converts a proto-plus value into plain Python without changing it.

    Args:
        value: A value read from a GoogleAdsRow.

    Returns:
        Enum name, dict for messages, list for repeated fields, or the
        scalar unchanged.
    """
    import proto

    # Enums subclass int, so they must be checked before scalars.
    if isinstance(value, proto.Enum):
        return value.name
    if isinstance(value, proto.Message):
        return type(value).to_dict(
            value, preserving_proto_field_name=True, use_integers_for_enums=False
        )
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    return [_to_python(item) for item in value]


def _status_name(error: BaseException) -> str | None:
    """Returns the gRPC status code name of an API error, if any.

    Args:
        error: Exception raised by the library.

    Returns:
        Status name such as ``"UNAVAILABLE"``, or None.
    """
    rpc_error = getattr(error, "error", error)
    code = getattr(rpc_error, "code", None)
    if not callable(code):
        return None
    try:
        return code().name
    except Exception:
        return None


def _ads_error_codes(error: BaseException) -> list[tuple[str, str]]:
    """Lists the Google Ads error codes inside a GoogleAdsException.

    Args:
        error: Exception raised by the library.

    Returns:
        Pairs such as ``("quota_error", "RESOURCE_EXHAUSTED")``. Empty when
        the exception carries no GoogleAdsFailure.
    """
    failure = getattr(error, "failure", None)
    if failure is None:
        return []
    codes = [_error_code(ads_error) for ads_error in failure.errors]
    return [code for code in codes if code]


def _error_code(ads_error: Any) -> tuple[str, str] | None:
    """Reads which error code is set in a GoogleAdsError.

    Args:
        ads_error: A ``GoogleAdsError`` message.

    Returns:
        Pair such as ``("query_error", "UNRECOGNIZED_FIELD")``, or None.
    """
    error_code = ads_error.error_code
    group = type(error_code).pb(error_code).WhichOneof("error_code")
    if not group:
        return None
    return group, getattr(error_code, group).name


def _is_api_error(error: BaseException) -> bool:
    """Tells whether an exception came from the Google Ads API.

    Args:
        error: Exception raised by the library.

    Returns:
        True for GoogleAdsException and grpc.RpcError.
    """
    try:
        import grpc
        from google.ads.googleads.errors import GoogleAdsException
    except ImportError:
        return False
    return isinstance(error, (GoogleAdsException, grpc.RpcError))


def _classify_error(error: BaseException) -> str:
    """Groups an exception as transient, quota or fatal.

    Args:
        error: Exception raised by the library.

    Returns:
        ``"transient"``, ``"quota"`` or ``"fatal"``.
    """
    if not _is_api_error(error):
        return _FATAL
    status = _status_name(error)
    codes = _ads_error_codes(error)
    if status in QUOTA_STATUS_CODES or any(g == "quota_error" for g, _ in codes):
        return _QUOTA
    if codes:
        # Retry only when every error is a server-side hiccup; a single
        # query or permission error makes the request fail again.
        if all(
            group == "internal_error" and name in TRANSIENT_INTERNAL_ERRORS
            for group, name in codes
        ):
            return _TRANSIENT
        return _FATAL
    if status in TRANSIENT_STATUS_CODES:
        return _TRANSIENT
    return _FATAL


def _request_id(error: BaseException) -> str | None:
    """Extracts the Google Ads request id from an API error.

    Args:
        error: Exception raised by the library.

    Returns:
        The request id, or None when the API did not send one.
    """
    request_id = getattr(error, "request_id", None)
    if request_id:
        return str(request_id)
    trailing = getattr(error, "trailing_metadata", None)
    if callable(trailing):
        try:
            for key, value in trailing() or ():
                if key == "request-id":
                    return str(value)
        except Exception:
            return None
    return None


def _describe_error(error: BaseException) -> str:
    """Builds a one-line description with the request id and every error.

    Args:
        error: Exception raised by the library.

    Returns:
        Description such as
        ``"request_id=abc status=INVALID_ARGUMENT errors: [query_error=...]"``.
    """
    parts = [f"request_id={_request_id(error) or 'unknown'}"]
    status = _status_name(error)
    if status:
        parts.append(f"status={status}")

    failure = getattr(error, "failure", None)
    if failure is not None and len(failure.errors):
        details = []
        for ads_error in failure.errors:
            code = _error_code(ads_error)
            label = f"{code[0]}={code[1]}: " if code else ""
            details.append(f"[{label}{ads_error.message}]")
        parts.append("errors: " + " ".join(details))
    else:
        rpc_details = getattr(error, "details", None)
        text = rpc_details() if callable(rpc_details) else str(error)
        if text:
            parts.append(f"details: {text}")
    return " ".join(parts)
