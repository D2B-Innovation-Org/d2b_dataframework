"""Tests for the shared Meta Graph client. No network: Session.get is mocked."""

from unittest.mock import MagicMock

import pytest
import requests

import d2b_data.meta_graph as mg
from d2b_data.load_result import ErrorKind, ExtractionError, classify_error
from d2b_data.meta_graph import (
    GRAPH_API_VERSION,
    MetaErrorCategory,
    MetaGraphClient,
    MetaGraphError,
    categorize,
)

TOKEN = "EAAB-secret-token-123"


def resp(status=200, payload=None):
    """Fake requests.Response."""
    r = MagicMock()
    r.status_code = status
    if payload is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = payload
    return r


def graph_error(code, status=400, message="boom", subcode=None):
    """Graph API error response: HTTP 400 plus a JSON error body."""
    error = {"code": code, "message": message, "type": "OAuthException"}
    if subcode is not None:
        error["error_subcode"] = subcode
    return resp(status, {"error": error})


class RecordingLogger:
    """Logger that keeps every message, to assert nothing leaks."""

    def __init__(self):
        self.messages = []

    def log(self, message):
        self.messages.append(message)

    def critical(self, message):
        self.messages.append(message)


@pytest.fixture
def logger():
    return RecordingLogger()


@pytest.fixture
def client(logger):
    return MetaGraphClient(TOKEN, logger)


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    return mocker.patch.object(mg.time, "sleep")


class TestVersionAndAuth:
    def test_default_version(self, client):
        assert GRAPH_API_VERSION == "v25.0"
        assert client.api_version == "v25.0"
        assert client.base_url == "https://graph.facebook.com/v25.0"

    def test_version_override(self):
        c = MetaGraphClient(TOKEN, api_version="v26.0")
        assert c.base_url == "https://graph.facebook.com/v26.0"

    def test_uses_requests_session(self, client):
        assert isinstance(client.session, requests.Session)

    def test_token_in_bearer_header_not_params(self, client, mocker):
        get = mocker.patch.object(client.session, "get", return_value=resp(payload={"data": []}))
        client.get("/me", {"fields": "id"})
        args, kwargs = get.call_args
        assert args[0] == "https://graph.facebook.com/v25.0/me"
        assert kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}
        assert "access_token" not in kwargs["params"]
        assert kwargs["params"] == {"fields": "id"}


class TestCategorize:
    @pytest.mark.parametrize("code", [4, 17, 32, 613, 80001, 80008, 80014])
    def test_quota(self, code):
        assert categorize(400, code) is MetaErrorCategory.QUOTA

    @pytest.mark.parametrize("code", [190, 102])
    def test_auth(self, code):
        assert categorize(400, code) is MetaErrorCategory.AUTH

    @pytest.mark.parametrize("code", [10, 200, 230, 299])
    def test_permission(self, code):
        assert categorize(400, code) is MetaErrorCategory.PERMISSION

    @pytest.mark.parametrize("code", [1, 2])
    def test_transient_codes(self, code):
        assert categorize(400, code) is MetaErrorCategory.TRANSIENT

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_transient_http(self, status):
        assert categorize(status, None) is MetaErrorCategory.TRANSIENT

    def test_range_edges_are_other(self, ):
        assert categorize(400, 80000) is MetaErrorCategory.OTHER
        assert categorize(400, 80015) is MetaErrorCategory.OTHER
        assert categorize(400, 300) is MetaErrorCategory.OTHER

    def test_unknown_is_other(self):
        assert categorize(400, 100) is MetaErrorCategory.OTHER
        assert categorize(400, None) is MetaErrorCategory.OTHER

    def test_http_status_fallbacks(self):
        assert categorize(429, None) is MetaErrorCategory.QUOTA
        assert categorize(401, None) is MetaErrorCategory.AUTH
        assert categorize(403, None) is MetaErrorCategory.PERMISSION


class TestErrorHandling:
    @pytest.mark.parametrize(
        "code,category,kind",
        [
            (4, MetaErrorCategory.QUOTA, ErrorKind.TRANSIENT),
            (190, MetaErrorCategory.AUTH, ErrorKind.AUTH),
            (10, MetaErrorCategory.PERMISSION, ErrorKind.AUTH),
            (100, MetaErrorCategory.OTHER, ErrorKind.OTHER),
        ],
    )
    def test_body_parsed_before_raising(self, client, mocker, code, category, kind):
        mocker.patch.object(client.session, "get", return_value=graph_error(code, subcode=463))
        with pytest.raises(MetaGraphError) as info:
            client.get("/x/insights")
        err = info.value
        assert isinstance(err, ExtractionError)
        assert err.category is category
        assert err.http_status == 400
        assert err.api_code == code
        assert err.api_subcode == 463
        assert classify_error(err) is kind

    def test_non_retryable_not_retried(self, client, mocker):
        get = mocker.patch.object(client.session, "get", return_value=graph_error(190))
        with pytest.raises(MetaGraphError):
            client.get("/me")
        assert get.call_count == 1

    def test_quota_not_retried(self, client, mocker):
        get = mocker.patch.object(client.session, "get", return_value=graph_error(4))
        with pytest.raises(MetaGraphError):
            client.get("/me")
        assert get.call_count == 1

    def test_must_propagate(self):
        assert MetaGraphError("x", MetaErrorCategory.AUTH).must_propagate
        assert MetaGraphError("x", MetaErrorCategory.PERMISSION).must_propagate
        assert MetaGraphError("x", MetaErrorCategory.QUOTA).must_propagate
        assert not MetaGraphError("x", MetaErrorCategory.TRANSIENT).must_propagate
        assert not MetaGraphError("x", MetaErrorCategory.OTHER).must_propagate

    def test_non_json_error_body(self, client, mocker):
        mocker.patch.object(client.session, "get", return_value=resp(403))
        with pytest.raises(MetaGraphError) as info:
            client.get("/me")
        assert info.value.category is MetaErrorCategory.PERMISSION
        assert info.value.api_code is None

    def test_error_body_with_200_status(self, client, mocker):
        mocker.patch.object(
            client.session,
            "get",
            return_value=resp(200, {"error": {"code": 190, "message": "bad"}}),
        )
        with pytest.raises(MetaGraphError) as info:
            client.get("/me")
        assert info.value.category is MetaErrorCategory.AUTH


class TestRetry:
    def test_retry_on_5xx_then_ok(self, client, mocker, no_sleep):
        get = mocker.patch.object(
            client.session,
            "get",
            side_effect=[resp(503), resp(500), resp(200, {"data": [{"id": "1"}]})],
        )
        assert client.get("/me") == {"data": [{"id": "1"}]}
        assert get.call_count == 3
        assert no_sleep.call_count == 2

    def test_retry_on_code_1_then_ok(self, client, mocker):
        get = mocker.patch.object(
            client.session, "get", side_effect=[graph_error(1), resp(200, {"ok": 1})]
        )
        assert client.get("/me") == {"ok": 1}
        assert get.call_count == 2

    def test_5xx_exhausted(self, client, mocker):
        get = mocker.patch.object(client.session, "get", return_value=resp(502))
        with pytest.raises(MetaGraphError) as info:
            client.get("/me")
        assert get.call_count == mg.MAX_RETRIES
        assert info.value.category is MetaErrorCategory.TRANSIENT
        assert info.value.http_status == 502
        assert classify_error(info.value) is ErrorKind.TRANSIENT

    def test_network_error_retried_without_leaking_text(self, client, mocker, logger):
        url_error = requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool: Max retries https://graph.facebook.com/me?access_token={TOKEN}"
        )
        mocker.patch.object(client.session, "get", side_effect=url_error)
        with pytest.raises(MetaGraphError) as info:
            client.get("/me")
        assert info.value.category is MetaErrorCategory.TRANSIENT
        assert TOKEN not in str(info.value)
        assert all(TOKEN not in m for m in logger.messages)


class TestPagination:
    def test_follows_next_and_strips_token(self, client, mocker, logger):
        next_url = (
            "https://graph.facebook.com/v25.0/123/posts"
            f"?access_token={TOKEN}&limit=100&after=cur1&fields=id"
        )
        get = mocker.patch.object(
            client.session,
            "get",
            side_effect=[
                resp(200, {"data": [{"id": "1"}], "paging": {"next": next_url}}),
                resp(200, {"data": [{"id": "2"}], "paging": {}}),
            ],
        )
        result = client.paginate("/123/posts", {"fields": "id", "limit": 100})

        assert [r["id"] for r in result] == ["1", "2"]
        assert get.call_count == 2
        second_args, second_kwargs = get.call_args_list[1]
        assert second_args[0] == "https://graph.facebook.com/v25.0/123/posts"
        assert second_kwargs["params"] == {"limit": "100", "after": "cur1", "fields": "id"}
        assert second_kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}
        assert all(TOKEN not in m for m in logger.messages)

    def test_single_page(self, client, mocker):
        mocker.patch.object(
            client.session, "get", return_value=resp(200, {"data": [{"id": "1"}], "paging": {}})
        )
        assert client.paginate("/123/posts") == [{"id": "1"}]

    def test_stops_on_empty_data(self, client, mocker):
        get = mocker.patch.object(
            client.session,
            "get",
            side_effect=[
                resp(200, {"data": [{"id": "1"}], "paging": {"next": "https://graph.facebook.com/v25.0/x?after=a"}}),
                resp(200, {"data": [], "paging": {"next": "https://graph.facebook.com/v25.0/x?after=b"}}),
            ],
        )
        assert len(client.paginate("/x")) == 1
        assert get.call_count == 2

    def test_refuses_foreign_next_host(self, client, mocker):
        get = mocker.patch.object(
            client.session,
            "get",
            return_value=resp(200, {"data": [{"id": "1"}], "paging": {"next": "https://evil.example.com/x?after=a"}}),
        )
        with pytest.raises(MetaGraphError) as info:
            client.paginate("/x")
        assert get.call_count == 1
        assert "evil.example.com" not in str(info.value)

    def test_error_mid_pagination_propagates(self, client, mocker):
        mocker.patch.object(
            client.session,
            "get",
            side_effect=[
                resp(200, {"data": [{"id": "1"}], "paging": {"next": "https://graph.facebook.com/v25.0/x?after=a"}}),
                graph_error(190),
            ],
        )
        with pytest.raises(MetaGraphError) as info:
            client.paginate("/x")
        assert info.value.category is MetaErrorCategory.AUTH


class TestTokenNotLeaked:
    def test_not_in_exception_even_if_api_echoes_it(self, client, mocker):
        mocker.patch.object(
            client.session,
            "get",
            return_value=graph_error(190, message=f"Invalid OAuth access token {TOKEN}"),
        )
        with pytest.raises(MetaGraphError) as info:
            client.get("/me")
        assert TOKEN not in str(info.value)
        assert TOKEN not in repr(info.value)
        assert "***" in str(info.value)

    def test_not_in_exception_for_5xx(self, client, mocker):
        mocker.patch.object(client.session, "get", return_value=resp(500))
        with pytest.raises(MetaGraphError) as info:
            client.get("/me")
        assert TOKEN not in str(info.value)

    def test_not_in_logs_across_calls(self, client, mocker, logger):
        mocker.patch.object(
            client.session,
            "get",
            side_effect=[resp(503), resp(200, {"data": [{"id": "1"}], "paging": {}})],
        )
        client.paginate("/me/accounts")
        assert logger.messages
        assert all(TOKEN not in m for m in logger.messages)


class TestGetPages:
    def test_get_pages(self, client, mocker, logger):
        pages = [
            {
                "id": "p1",
                "name": "Page 1",
                "access_token": "PAGE-TOKEN-1",
                "instagram_business_account": {"id": "ig1", "username": "one"},
            },
            {"id": "p2", "name": "Page 2", "access_token": "PAGE-TOKEN-2"},
        ]
        get = mocker.patch.object(
            client.session, "get", return_value=resp(200, {"data": pages, "paging": {}})
        )
        result = client.get_pages()

        assert result == pages
        args, kwargs = get.call_args
        assert args[0].endswith("/v25.0/me/accounts")
        assert kwargs["params"]["fields"] == (
            "id,name,access_token,instagram_business_account{id,username}"
        )
        assert all("PAGE-TOKEN" not in m for m in logger.messages)

    def test_get_pages_paginates(self, client, mocker):
        mocker.patch.object(
            client.session,
            "get",
            side_effect=[
                resp(200, {"data": [{"id": "p1"}], "paging": {"next": "https://graph.facebook.com/v25.0/me/accounts?after=a"}}),
                resp(200, {"data": [{"id": "p2"}], "paging": {}}),
            ],
        )
        assert [p["id"] for p in client.get_pages()] == ["p1", "p2"]
