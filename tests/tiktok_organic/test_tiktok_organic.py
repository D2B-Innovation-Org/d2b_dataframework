"""Tests for TikTokOrganic. Payload shapes mirror real API responses."""

import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

import d2b_data.tiktok_organic as tt_module
from d2b_data.tiktok_organic import (
    DAILY_FIELDS,
    VIDEO_FIELDS,
    QuotaExhaustedError,
    TikTokApiError,
    TikTokOrganic,
)


def resp(status=200, payload=None):
    """Fake requests.Response."""
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(str(status))
    return r


def ok(data):
    """TikTok success envelope."""
    return resp(payload={"code": 0, "message": "OK", "request_id": "x", "data": data})


def day(d, **kw):
    """One metrics[] item as returned by /business/get/."""
    item = {field: 0 for field in DAILY_FIELDS}
    item.update({"date": d, "audience_activity": [{"hour": "10", "count": 3}]})
    item.update(kw)
    return item


def video(item_id, create_time="1782153270", **kw):
    """One videos[] item as returned by /business/video/list/."""
    item = {
        "item_id": item_id,
        "create_time": create_time,
        "media_type": "VIDEO",
        "video_views": 10,
        "audience_types": [{"type": "NEW_VIEWER", "percentage": 1}],
        "impression_sources": [],
    }
    item.update(kw)
    return item


@pytest.fixture
def today(mocker):
    """Freeze TikTokOrganic's notion of today."""
    mocker.patch.object(tt_module, "_today", return_value=date(2026, 11, 12))


# -- init / token --------------------------------------------------------


class TestInit:
    """Token loading."""

    def test_loads_token_and_open_id(self, tiktok) -> None:
        """A valid file sets token and open_id."""
        assert tiktok.token == "fake_access"
        assert tiktok.open_id == "-000fakeOpenId"

    def test_no_token_path(self) -> None:
        """Without token_path nothing is loaded."""
        assert TikTokOrganic().token is None

    def test_missing_open_id(self, tmp_path) -> None:
        """A token file without open_id is not usable."""
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"access_token": "a"}))
        client = TikTokOrganic(token_path=str(path))
        assert client.token is None

    def test_unreadable_file(self, tmp_path) -> None:
        """A missing file does not raise; token stays None."""
        assert TikTokOrganic(token_path=str(tmp_path / "nope.json")).token is None

    def test_request_without_token_raises(self) -> None:
        """Calling the API without a token is a programming error."""
        with pytest.raises(RuntimeError):
            TikTokOrganic().get_profile_snapshot()


class TestRefreshAccessToken:
    """refresh_access_token."""

    def test_success_rotates_and_keeps_open_id(
        self, tiktok, token_file, mocker
    ) -> None:
        """New tokens are written; fields missing from the response survive."""
        post = mocker.patch.object(
            tiktok.session,
            "post",
            return_value=ok({"access_token": "new_a", "refresh_token": "new_r"}),
        )
        assert tiktok.refresh_access_token("cid", "csecret") is True
        saved = json.loads(token_file.read_text())
        assert saved["access_token"] == "new_a"
        assert saved["refresh_token"] == "new_r"
        assert saved["open_id"] == "-000fakeOpenId"
        assert tiktok.token == "new_a"
        body = post.call_args.kwargs["json"]
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "fake_refresh"
        assert post.call_args.kwargs["timeout"]

    def test_error_code_leaves_file_untouched(self, tiktok, token_file, mocker) -> None:
        """A TikTok error code returns False and does not overwrite the file."""
        before = token_file.read_text()
        mocker.patch.object(
            tiktok.session,
            "post",
            return_value=resp(payload={"code": 40104, "message": "invalid refresh"}),
        )
        assert tiktok.refresh_access_token("cid", "csecret") is False
        assert token_file.read_text() == before

    def test_http_error(self, tiktok, mocker) -> None:
        """Network/HTTP failures return False."""
        mocker.patch.object(tiktok.session, "post", return_value=resp(500))
        assert tiktok.refresh_access_token("cid", "csecret") is False

    def test_no_refresh_token(self, tmp_path) -> None:
        """Without refresh_token in the file there is nothing to exchange."""
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"access_token": "a", "open_id": "o"}))
        assert (
            TikTokOrganic(token_path=str(path)).refresh_access_token("c", "s") is False
        )


# -- HTTP helper ---------------------------------------------------------


class TestRequestGet:
    """_request_get."""

    def test_lists_json_encoded_and_business_id(self, tiktok, mocker) -> None:
        """List params go JSON-encoded and business_id is the open_id."""
        get = mocker.patch.object(tiktok.session, "get", return_value=ok({"a": 1}))
        assert tiktok._request_get("business/get/", {"fields": ["x", "y"]}) == {"a": 1}
        kwargs = get.call_args.kwargs
        assert kwargs["params"]["fields"] == '["x", "y"]'
        assert kwargs["params"]["business_id"] == "-000fakeOpenId"
        assert kwargs["headers"] == {"Access-Token": "fake_access"}
        assert kwargs["timeout"]

    def test_http_429_raises_quota(self, tiktok, mocker) -> None:
        """HTTP 429 stops immediately."""
        mocker.patch.object(tiktok.session, "get", return_value=resp(429))
        with pytest.raises(QuotaExhaustedError):
            tiktok._request_get("business/get/", {})

    def test_rate_limit_code_raises_quota(self, tiktok, mocker) -> None:
        """TikTok's rate-limit code in a 200 body is also a quota stop."""
        mocker.patch.object(
            tiktok.session,
            "get",
            return_value=resp(payload={"code": 40100, "message": "Too many requests"}),
        )
        with pytest.raises(QuotaExhaustedError):
            tiktok._request_get("business/get/", {})

    def test_error_code_raises_api_error(self, tiktok, mocker) -> None:
        """Other non-zero codes raise TikTokApiError carrying the code."""
        mocker.patch.object(
            tiktok.session,
            "get",
            return_value=resp(payload={"code": 40002, "message": "bad field"}),
        )
        with pytest.raises(TikTokApiError) as exc:
            tiktok._request_get("business/get/", {})
        assert exc.value.code == 40002

    def test_transient_then_ok(self, tiktok, mocker) -> None:
        """5xx is retried and the next success is returned."""
        mocker.patch.object(tt_module.time, "sleep")
        mocker.patch.object(
            tiktok.session, "get", side_effect=[resp(503), ok({"done": True})]
        )
        assert tiktok._request_get("business/get/", {}) == {"done": True}

    def test_transient_exhausted(self, tiktok, mocker) -> None:
        """Persistent 5xx ends in an HTTPError."""
        mocker.patch.object(tt_module.time, "sleep")
        mocker.patch.object(tiktok.session, "get", return_value=resp(503))
        with pytest.raises(requests.exceptions.HTTPError):
            tiktok._request_get("business/get/", {}, max_retries=2)


# -- daily ---------------------------------------------------------------


class TestGetDailyStats:
    """get_daily_stats."""

    def test_rows_dates_and_json(self, tiktok, mocker, today) -> None:
        """One row per day, sorted, date as datetime.date, activity as JSON."""
        mocker.patch.object(
            tiktok.session,
            "get",
            return_value=ok(
                {"metrics": [day("2026-11-10"), day("2026-11-09", likes=4)]}
            ),
        )
        df = tiktok.get_daily_stats("2026-11-09", "2026-11-10")
        assert list(df["date"]) == [date(2026, 11, 9), date(2026, 11, 10)]
        assert list(df.columns[:2]) == ["date", "open_id"]
        assert df.loc[0, "likes"] == 4
        assert json.loads(df.loc[0, "audience_activity"]) == [
            {"hour": "10", "count": 3}
        ]

    def test_clamps_to_60_day_window(self, tiktok, mocker, today) -> None:
        """Start dates beyond 60 days are moved to the oldest allowed day."""
        get = mocker.patch.object(
            tiktok.session, "get", return_value=ok({"metrics": []})
        )
        tiktok.get_daily_stats("2026-01-01", "2026-11-11")
        assert get.call_args.kwargs["params"]["start_date"] == "2026-09-14"

    def test_range_fully_outside_window(self, tiktok, mocker, today) -> None:
        """A range entirely older than the window makes no call."""
        get = mocker.patch.object(tiktok.session, "get")
        df = tiktok.get_daily_stats("2026-01-01", "2026-02-01")
        assert df.empty
        get.assert_not_called()

    def test_empty_metrics(self, tiktok, mocker, today) -> None:
        """No days returned gives an empty frame with the expected columns."""
        mocker.patch.object(tiktok.session, "get", return_value=ok({"metrics": []}))
        df = tiktok.get_daily_stats("2026-11-09", "2026-11-10")
        assert df.empty
        assert list(df.columns) == ["date", "open_id", *DAILY_FIELDS]

    def test_start_after_end(self, tiktok) -> None:
        """Inverted ranges are rejected."""
        with pytest.raises(ValueError):
            tiktok.get_daily_stats("2026-11-10", "2026-11-09")

    def test_bad_date_format(self, tiktok) -> None:
        """Malformed dates are rejected."""
        with pytest.raises(ValueError):
            tiktok.get_daily_stats("10-11-2026", "2026-11-11")


# -- profile -------------------------------------------------------------


class TestGetProfileSnapshot:
    """get_profile_snapshot."""

    def test_single_row_with_json_audience(self, tiktok, mocker) -> None:
        """One row; scalars kept, audience lists serialized, metrics dropped."""
        mocker.patch.object(
            tiktok.session,
            "get",
            return_value=ok(
                {
                    "username": "d2bchile",
                    "followers_count": 12,
                    "is_business_account": True,
                    "audience_countries": [{"country": "CL", "percentage": 0.9}],
                    "metrics": [],
                }
            ),
        )
        df = tiktok.get_profile_snapshot()
        assert len(df) == 1
        assert list(df.columns[:3]) == ["snapshot_date", "extracted_at", "open_id"]
        assert "metrics" not in df.columns
        assert df.loc[0, "followers_count"] == 12
        assert json.loads(df.loc[0, "audience_countries"])[0]["country"] == "CL"
        assert isinstance(df.loc[0, "snapshot_date"], date)


# -- videos --------------------------------------------------------------


class TestGetVideos:
    """get_videos."""

    def test_paginates_with_cursor(self, tiktok, mocker) -> None:
        """Follows the cursor while has_more is true."""
        get = mocker.patch.object(
            tiktok.session,
            "get",
            side_effect=[
                ok(
                    {"videos": [video("1"), video("2")], "has_more": True, "cursor": 99}
                ),
                ok({"videos": [video("3")], "has_more": False, "cursor": 0}),
            ],
        )
        df = tiktok.get_videos()
        assert list(df["item_id"]) == ["1", "2", "3"]
        cursors = [c.kwargs["params"]["cursor"] for c in get.call_args_list]
        assert cursors == [0, 99]

    def test_stops_on_empty_page(self, tiktok, mocker) -> None:
        """An empty page with has_more=true does not loop forever."""
        get = mocker.patch.object(
            tiktok.session,
            "get",
            return_value=ok({"videos": [], "has_more": True, "cursor": 5}),
        )
        assert tiktok.get_videos().empty
        assert get.call_count == 1

    def test_types_and_json(self, tiktok, mocker) -> None:
        """create_time becomes a UTC timestamp and nested lists JSON strings."""
        mocker.patch.object(
            tiktok.session,
            "get",
            return_value=ok({"videos": [video("1")], "has_more": False}),
        )
        df = tiktok.get_videos()
        assert df.loc[0, "create_time"] == pd.Timestamp("2026-06-22 18:34:30", tz="UTC")
        assert json.loads(df.loc[0, "audience_types"])[0]["type"] == "NEW_VIEWER"
        assert df.loc[0, "impression_sources"] == "[]"
        assert df.loc[0, "open_id"] == "-000fakeOpenId"

    def test_no_videos(self, tiktok, mocker) -> None:
        """No videos gives an empty frame with the expected columns."""
        mocker.patch.object(
            tiktok.session, "get", return_value=ok({"videos": [], "has_more": False})
        )
        df = tiktok.get_videos()
        assert df.empty
        assert list(df.columns) == [
            "snapshot_date",
            "extracted_at",
            "open_id",
            *VIDEO_FIELDS,
        ]
