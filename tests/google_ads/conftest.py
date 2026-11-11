"""Fixtures for GoogleAds.

No network and no real credentials: the API client is replaced by a fake that
returns stream batches built with the real ``google-ads`` message types, so
the row conversion runs against the same objects the API returns. Tests that
need those types are skipped when ``google-ads`` is not installed.
"""

import importlib.util
import json
from types import SimpleNamespace
from typing import Any

import pytest


def _has_google_ads() -> bool:
    # find_spec raises instead of returning None when "google.ads" is missing.
    try:
        return importlib.util.find_spec("google.ads.googleads") is not None
    except ModuleNotFoundError:
        return False


HAS_GOOGLE_ADS = _has_google_ads()
requires_google_ads = pytest.mark.skipif(
    not HAS_GOOGLE_ADS, reason="google-ads is not installed"
)

CREDENTIALS = {
    "developer_token": "dev-token-secret",
    "client_id": "client-id-secret",
    "client_secret": "client-secret-secret",
    "refresh_token": "refresh-token-secret",
}


class FakeGoogleAdsService:
    """Stand-in for GoogleAdsService.

    Each item of ``outcomes`` is consumed by one ``search_stream`` call: a list
    of batches is streamed, an exception is raised while iterating, which is
    where the real stream raises it.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, str]] = []

    def search_stream(self, customer_id: str, query: str):
        self.calls.append({"customer_id": customer_id, "query": query})
        outcome = self.outcomes.pop(0)

        def _iterate():
            if isinstance(outcome, BaseException):
                raise outcome
            yield from outcome

        return _iterate()


class FakeCustomerService:
    """Stand-in for CustomerService."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    def list_accessible_customers(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return SimpleNamespace(resource_names=self.outcome)


class FakeClient:
    """Stand-in for GoogleAdsClient exposing only get_service."""

    def __init__(self, services: dict[str, Any]) -> None:
        self.services = services

    def get_service(self, name: str) -> Any:
        return self.services[name]


@pytest.fixture
def credentials() -> dict[str, str]:
    """Fake credentials; values end in 'secret' so leaks are easy to spot."""
    return dict(CREDENTIALS)


@pytest.fixture
def token_file(tmp_path):
    """Writes a credentials JSON and returns its path."""

    def _write(data: dict[str, Any]) -> str:
        path = tmp_path / "google_ads_token.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def ads(credentials):
    """GoogleAds instance with fake credentials and no API client yet."""
    from d2b_data.google_ads import GoogleAds

    return GoogleAds(**credentials)


@pytest.fixture
def no_sleep(mocker):
    """Patches time.sleep in the module so retries do not wait."""
    return mocker.patch("d2b_data.google_ads.time.sleep")


@pytest.fixture
def with_stream(ads):
    """Installs a FakeGoogleAdsService on ``ads`` and returns it."""

    def _install(*outcomes: Any) -> FakeGoogleAdsService:
        service = FakeGoogleAdsService(list(outcomes))
        ads._client = FakeClient({"GoogleAdsService": service})
        return service

    return _install


# ---------------------------------------------------------------------- #
# Builders with the real google-ads types
# ---------------------------------------------------------------------- #
def make_batch(paths: list[str], rows: list[Any]):
    """Builds a SearchGoogleAdsStreamResponse with the given mask and rows."""
    from google.ads.googleads.v26.services.types.google_ads_service import (
        SearchGoogleAdsStreamResponse,
    )

    batch = SearchGoogleAdsStreamResponse()
    batch.field_mask.paths.extend(paths)
    batch.results.extend(rows)
    return batch


def make_row(**values: Any):
    """Builds a GoogleAdsRow from dotted paths written with '__'.

    Example: ``make_row(campaign__id=1, metrics__cost_micros=0)``.
    """
    from google.ads.googleads.v26.services.types.google_ads_service import (
        GoogleAdsRow,
    )

    row = GoogleAdsRow()
    for key, value in values.items():
        *parents, leaf = key.split("__")
        target = row
        for parent in parents:
            target = getattr(target, parent)
        if isinstance(value, list):
            getattr(target, leaf).extend(value)
        else:
            setattr(target, leaf, value)
    return row


class FakeRpcError(Exception):
    """grpc.RpcError with a status code, details and trailing metadata."""

    def __init__(self, code_name: str, details: str = "", request_id: str = ""):
        import grpc

        super().__init__(details)
        self._code = getattr(grpc.StatusCode, code_name)
        self._details = details
        self._request_id = request_id

    def code(self):
        return self._code

    def details(self):
        return self._details

    def trailing_metadata(self):
        return [("request-id", self._request_id)] if self._request_id else []


def rpc_error(code_name: str, details: str = "", request_id: str = ""):
    """Builds an instance that really subclasses grpc.RpcError."""
    import grpc

    cls = type("RpcError", (FakeRpcError, grpc.RpcError), {})
    return cls(code_name, details, request_id)


def ads_exception(
    errors: list[tuple[str, str, str]],
    code_name: str = "INVALID_ARGUMENT",
    request_id: str = "req-123",
):
    """Builds a GoogleAdsException.

    Args:
        errors: ``(group, enum name, message)`` such as
            ``("query_error", "UNRECOGNIZED_FIELD", "bad field")``.
        code_name: gRPC status of the underlying RpcError.
        request_id: Google Ads request id.
    """
    from google.ads.googleads.errors import GoogleAdsException
    from google.ads.googleads.v26.errors.types.errors import (
        ErrorCode,
        GoogleAdsError,
        GoogleAdsFailure,
    )

    failure = GoogleAdsFailure()
    for group, name, message in errors:
        error_code = ErrorCode()
        enum_type = type(getattr(error_code, group))
        setattr(error_code, group, enum_type[name])
        failure.errors.append(GoogleAdsError(error_code=error_code, message=message))
    rpc = rpc_error(code_name, "", request_id)
    return GoogleAdsException(rpc, rpc, failure, request_id)
