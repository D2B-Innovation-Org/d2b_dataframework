"""Shared Meta Graph API client for the organic extraction classes.

``FacebookOrganic`` and ``InstagramOrganic`` talk to the same Graph API, so
the HTTP plumbing lives here once:

* one pinned API version (``GRAPH_API_VERSION``), overridable per client;
* the access token travels in the ``Authorization: Bearer`` header, never in
  the query string, and is kept out of logs and exception messages;
* Graph API errors (HTTP 400 with a JSON ``error`` body) are parsed before
  raising and surfaced as ``MetaGraphError``, an ``ExtractionError`` that
  carries the HTTP status, the API code and a ``MetaErrorCategory``;
* transient failures (HTTP 5xx, API codes 1 and 2, network errors) are
  retried with exponential backoff; auth, permission and quota errors never
  are, because retrying them only burns more quota.
"""

import logging
import random
import time
from enum import Enum
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

import requests

from d2b_data.load_result import ExtractionError

GRAPH_API_VERSION = "v25.0"
GRAPH_HOST = "https://graph.facebook.com"
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each retry

# Graph API error codes by category.
QUOTA_CODES = frozenset({4, 17, 32, 613}) | frozenset(range(80001, 80015))
AUTH_CODES = frozenset({190, 102})
PERMISSION_CODES = frozenset({10}) | frozenset(range(200, 300))
TRANSIENT_CODES = frozenset({1, 2})

PAGE_FIELDS = "id,name,access_token,instagram_business_account{id,username}"


class MetaErrorCategory(str, Enum):
    """What a Graph API error means for the caller."""

    QUOTA = "quota"
    AUTH = "auth"
    PERMISSION = "permission"
    TRANSIENT = "transient"
    OTHER = "other"


# HTTP status handed to classify_error for each category, so the shared
# el-* contract (401/403 cut the account, 429/5xx are transient) applies.
_EQUIVALENT_STATUS = {
    MetaErrorCategory.QUOTA: 429,
    MetaErrorCategory.AUTH: 401,
    MetaErrorCategory.PERMISSION: 403,
}


def categorize(http_status: Optional[int], api_code: Optional[int]) -> MetaErrorCategory:
    """Map an HTTP status and a Graph API code to an error category.

    The API code wins when it is known, because Meta reports most errors as
    HTTP 400 whatever their cause.

    Args:
        http_status: HTTP status of the response, if any.
        api_code: ``error.code`` from the response body, if any.

    Returns:
        The category of the error.
    """
    if api_code in QUOTA_CODES:
        return MetaErrorCategory.QUOTA
    if api_code in AUTH_CODES:
        return MetaErrorCategory.AUTH
    if api_code in PERMISSION_CODES:
        return MetaErrorCategory.PERMISSION
    if api_code in TRANSIENT_CODES:
        return MetaErrorCategory.TRANSIENT
    if http_status == 429:
        return MetaErrorCategory.QUOTA
    if http_status == 401:
        return MetaErrorCategory.AUTH
    if http_status == 403:
        return MetaErrorCategory.PERMISSION
    if http_status is not None and http_status >= 500:
        return MetaErrorCategory.TRANSIENT
    return MetaErrorCategory.OTHER


class MetaGraphError(ExtractionError):
    """Graph API error with its HTTP status, API code and category.

    ``status_code`` (used by ``classify_error``) is the HTTP status
    equivalent of the category, e.g. 401 for an expired token even though
    Meta answered 400. The real status is kept in ``http_status``.

    Attributes:
        category: Kind of error.
        http_status: HTTP status of the response, or None if there was none.
        api_code: ``error.code`` from the body, or None.
        api_subcode: ``error.error_subcode`` from the body, or None.
    """

    def __init__(
        self,
        message: str,
        category: MetaErrorCategory,
        http_status: Optional[int] = None,
        api_code: Optional[int] = None,
        api_subcode: Optional[int] = None,
    ) -> None:
        if category in _EQUIVALENT_STATUS:
            equivalent = _EQUIVALENT_STATUS[category]
        elif category is MetaErrorCategory.TRANSIENT:
            equivalent = http_status if http_status and http_status >= 500 else 500
        else:
            equivalent = http_status
        super().__init__(message, status_code=equivalent, api_code=api_code)
        self.category = category
        self.http_status = http_status
        self.api_subcode = api_subcode

    @property
    def must_propagate(self) -> bool:
        """True for auth, permission and quota errors, which no item can fix."""
        return self.category in (
            MetaErrorCategory.AUTH,
            MetaErrorCategory.PERMISSION,
            MetaErrorCategory.QUOTA,
        )


class MetaGraphClient:
    """Small Graph API client with auth, error parsing, retries and paging.

    Attributes:
        api_version: Graph API version used in every URL.
        base_url: ``https://graph.facebook.com/<api_version>``.
        session: ``requests.Session`` used for every call.
    """

    def __init__(
        self,
        access_token: str,
        logger: Optional[Any] = None,
        api_version: str = GRAPH_API_VERSION,
    ) -> None:
        """Create the client.

        Args:
            access_token: User, Page or System User access token.
            logger: Object with ``log()`` and ``critical()``; stdlib logging
                is used when None.
            api_version: Graph API version, e.g. 'v25.0'.
        """
        self._access_token = access_token
        self.api_version = api_version
        self.base_url = f"{GRAPH_HOST}/{api_version}"
        self.logger = logger or self._build_default_logger()
        self.session = requests.Session()

    @staticmethod
    def _build_default_logger() -> Any:
        """Build a stdlib-based fallback logger with .log() and .critical()."""
        logger = logging.getLogger("MetaGraphClient")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

        class _Adapter:
            def log(self, message: str) -> None:
                logger.info(message)

            def critical(self, message: str) -> None:
                logger.error(message)

        return _Adapter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redact(self, text: str) -> str:
        """Remove the access token from text that may be logged or raised."""
        if self._access_token:
            return text.replace(self._access_token, "***")
        return text

    def _build_error(
        self, endpoint: str, response: requests.Response
    ) -> MetaGraphError:
        """Turn a failed response into a MetaGraphError, parsing the body first."""
        http_status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        error = error if isinstance(error, dict) else {}

        api_code = error.get("code") if isinstance(error.get("code"), int) else None
        subcode = error.get("error_subcode")
        subcode = subcode if isinstance(subcode, int) else None
        category = categorize(http_status, api_code)
        detail = self._redact(str(error.get("message") or "no error body"))
        message = (
            f"Meta Graph API error on {endpoint}: HTTP {http_status}, "
            f"code {api_code}, subcode {subcode}, type {error.get('type')}: "
            f"{detail}"
        )
        return MetaGraphError(
            message,
            category,
            http_status=http_status,
            api_code=api_code,
            api_subcode=subcode,
        )

    def _request(self, url: str, endpoint: str, params: dict) -> dict:
        """GET a full URL with auth and retry transient failures.

        Args:
            url: Full URL without a query string.
            endpoint: Endpoint label used in logs and messages.
            params: Query parameters (never the token).

        Returns:
            Parsed JSON object of the response.

        Raises:
            MetaGraphError: On any Graph API error or exhausted retries.
        """
        headers = {"Authorization": f"Bearer {self._access_token}"}

        for attempt in range(1, MAX_RETRIES + 1):
            error: MetaGraphError
            try:
                response = self.session.get(
                    url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
                )
            except requests.exceptions.RequestException as exc:
                # Only the exception type: its text can embed the URL.
                error = MetaGraphError(
                    f"Meta Graph request failed on {endpoint}: "
                    f"{type(exc).__name__}",
                    MetaErrorCategory.TRANSIENT,
                )
            else:
                if response.status_code < 400:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise MetaGraphError(
                            f"Meta Graph API returned a non-JSON body on "
                            f"{endpoint}",
                            MetaErrorCategory.OTHER,
                            http_status=response.status_code,
                        ) from None
                    if isinstance(payload, dict) and "error" in payload:
                        # Defensive: an error body with a 2xx status.
                        raise self._build_error(endpoint, response)
                    return payload
                error = self._build_error(endpoint, response)

            if error.category is not MetaErrorCategory.TRANSIENT or attempt == MAX_RETRIES:
                raise error
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.random()
            self.logger.log(
                f"Transient error on {endpoint}, retrying in {wait:.1f}s "
                f"(attempt {attempt}/{MAX_RETRIES}): {error}"
            )
            time.sleep(wait)

        raise RuntimeError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Execute a single GET against the Graph API.

        Args:
            endpoint: Path relative to the versioned base URL, e.g.
                '/12345/posts'.
            params: Query parameters; the token must not be among them.

        Returns:
            Parsed JSON response.

        Raises:
            MetaGraphError: On Graph API errors.
        """
        return self._request(f"{self.base_url}{endpoint}", endpoint, dict(params or {}))

    def paginate(self, endpoint: str, params: Optional[dict] = None) -> list[dict]:
        """Fetch every page of a paginated endpoint.

        Follows ``paging.next``. Meta usually embeds ``access_token`` in
        that URL, so it is stripped and the Bearer header is used instead;
        a ``next`` URL pointing to another host is refused so the token
        never leaves graph.facebook.com.

        Args:
            endpoint: Path relative to the versioned base URL.
            params: Initial query parameters.

        Returns:
            Flat list of the records of all pages.

        Raises:
            MetaGraphError: On Graph API errors or a foreign ``next`` URL.
        """
        results: list[dict] = []
        url = f"{self.base_url}{endpoint}"
        query = dict(params or {})
        page_count = 0

        while True:
            response = self._request(url, endpoint, query)
            data = response.get("data", [])
            results.extend(data)
            page_count += 1
            self.logger.log(
                f"paginate | {endpoint} page {page_count}: {len(data)} records "
                f"(total {len(results)})"
            )

            next_url = (response.get("paging") or {}).get("next")
            if not next_url or not data:
                break

            parts = urlsplit(next_url)
            if f"{parts.scheme}://{parts.netloc}" != GRAPH_HOST:
                raise MetaGraphError(
                    f"Refusing to follow paging.next to another host on {endpoint}",
                    MetaErrorCategory.OTHER,
                )
            url = f"{GRAPH_HOST}{parts.path}"
            query = {
                key: value
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if key != "access_token"
            }

        return results

    def get_pages(self) -> list[dict]:
        """List the Pages the token can manage, with their IG accounts.

        Each dict has ``id``, ``name``, ``access_token`` (the Page token) and,
        when linked, ``instagram_business_account`` with ``id`` and
        ``username``. The Page tokens are secrets: callers must not log them.

        Returns:
            List of Page dicts as returned by ``/me/accounts``.
        """
        pages = self.paginate("/me/accounts", {"fields": PAGE_FIELDS, "limit": 100})
        self.logger.log(f"get_pages | {len(pages)} pages accessible")
        return pages
