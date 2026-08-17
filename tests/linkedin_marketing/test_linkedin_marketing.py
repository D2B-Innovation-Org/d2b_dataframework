from unittest.mock import MagicMock
from urllib.parse import quote

import pandas as pd
import pytest
import requests


# ------------------------------------------------------------------
# Init & Auth
# ------------------------------------------------------------------


def test_instance_with_valid_token(marketing):
    """Token loaded from file sets headers correctly."""
    assert marketing.token == "fake_token_123"
    assert marketing.headers["Authorization"] == "Bearer fake_token_123"
    assert marketing.headers["X-Restli-Protocol-Version"] == "2.0.0"
    assert marketing.headers["Linkedin-Version"] == "202607"
    assert marketing.headers["Content-Type"] == "application/json"


def test_instance_declares_202607_version(marketing_no_file):
    """The client targets the 202607 API version."""
    assert marketing_no_file.linkedin_version == "202607"


def test_instance_no_file_specified(marketing_no_file):
    """Without token_path, token and headers remain None."""
    assert marketing_no_file.token is None
    assert marketing_no_file.headers is None
    assert marketing_no_file.token_path is None


def test_instance_bad_file(marketing_bad_file):
    """When the file read fails, token stays None."""
    assert marketing_bad_file.token is None
    assert marketing_bad_file.headers is None


def test_load_token_missing_access_token_key(mocker):
    """Token file without 'access_token' key leaves the client unauthenticated."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.load", return_value={"refresh_token": "something"})

    from d2b_data.linkedin_marketing import LinkedinMarketing

    li = LinkedinMarketing(token_path="incomplete.json")
    assert li.token is None
    assert li.headers is None


def test_set_headers_without_token(marketing_no_file):
    """_set_headers does nothing when token is None."""
    marketing_no_file.token = None
    marketing_no_file._set_headers()
    assert marketing_no_file.headers is None


def test_set_token_loads_and_sets_headers(marketing_no_file, mocker):
    """set_token reads the file and configures the headers."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.load", return_value={"access_token": "abc"})

    marketing_no_file.set_token("token.json")

    assert marketing_no_file.token_path == "token.json"
    assert marketing_no_file.token == "abc"
    assert marketing_no_file.headers["Authorization"] == "Bearer abc"


def test_set_token_with_bad_file_clears_previous_credentials(marketing, mocker):
    """A failed reload must not leave the old token/headers in place."""
    assert marketing.token == "fake_token_123"

    mocker.patch("builtins.open", side_effect=Exception("gone"))
    marketing.set_token("missing.json")

    assert marketing.token is None
    assert marketing.headers is None


def test_default_logger_is_built_when_none_given(marketing_no_file):
    """A fallback logger exposing .info/.critical is created."""
    assert hasattr(marketing_no_file.logger, "info")
    assert hasattr(marketing_no_file.logger, "critical")


def test_custom_logger_is_used():
    """An injected logger receives the init message."""
    from d2b_data.linkedin_marketing import LinkedinMarketing

    logger = MagicMock()
    LinkedinMarketing(logger=logger)

    assert logger.info.called


# ------------------------------------------------------------------
# _request_get
# ------------------------------------------------------------------


def test_request_get_no_headers_raises(marketing_no_file):
    """Calling _request_get without auth raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Headers not set"):
        marketing_no_file._request_get("https://example.com")


def test_request_get_success(marketing, mocker):
    """Successful GET returns parsed JSON."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"elements": []}

    mocker.patch("requests.Session.send", return_value=fake_response)

    assert marketing._request_get("https://api.linkedin.com/rest/test") == {
        "elements": []
    }


def test_request_get_sends_url_verbatim(marketing, mocker):
    """The prepared URL is not re-encoded by requests."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {}

    send = mocker.patch("requests.Session.send", return_value=fake_response)

    url = "https://api.linkedin.com/rest/adAnalytics?q=statistics&pivots=List(CAMPAIGN)"
    marketing._request_get(url)

    assert send.call_args[0][0].url == url


def test_request_get_429_raises_quota_error(marketing, mocker):
    """429 response raises QuotaExhaustedError immediately, without retrying."""
    from d2b_data.linkedin_marketing import QuotaExhaustedError

    fake_response = MagicMock()
    fake_response.status_code = 429

    send = mocker.patch("requests.Session.send", return_value=fake_response)

    with pytest.raises(QuotaExhaustedError, match="daily quota"):
        marketing._request_get("https://api.linkedin.com/rest/test")

    assert send.call_count == 1


def test_request_get_transient_error_retries(marketing, mocker):
    """Transient 500 errors are retried with exponential backoff."""
    fail = MagicMock()
    fail.status_code = 500

    success = MagicMock()
    success.status_code = 200
    success.json.return_value = {"ok": True}

    mocker.patch("requests.Session.send", side_effect=[fail, success])
    mock_sleep = mocker.patch("d2b_data.linkedin_marketing.time.sleep")

    assert marketing._request_get("https://api.linkedin.com/rest/test") == {"ok": True}
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(1)


@pytest.mark.parametrize("code", [500, 502, 503])
def test_request_get_retries_every_transient_code(marketing, mocker, code):
    """500, 502 and 503 are all treated as transient."""
    fail = MagicMock()
    fail.status_code = code

    success = MagicMock()
    success.status_code = 200
    success.json.return_value = {"ok": True}

    mocker.patch("requests.Session.send", side_effect=[fail, success])
    mocker.patch("d2b_data.linkedin_marketing.time.sleep")

    assert marketing._request_get("https://api.linkedin.com/rest/test") == {"ok": True}


def test_request_get_backoff_is_exponential(marketing, mocker):
    """Waits double on every retry: 1s, 2s, 4s."""
    fail = MagicMock()
    fail.status_code = 503
    fail.raise_for_status.side_effect = requests.exceptions.HTTPError("503")

    mocker.patch("requests.Session.send", return_value=fail)
    mock_sleep = mocker.patch("d2b_data.linkedin_marketing.time.sleep")

    with pytest.raises(requests.exceptions.HTTPError):
        marketing._request_get("https://api.linkedin.com/rest/test", max_retries=3)

    assert [call.args[0] for call in mock_sleep.call_args_list] == [1, 2, 4]


def test_request_get_exhausts_retries(marketing, mocker):
    """After max retries on 500, raise_for_status is called."""
    fail = MagicMock()
    fail.status_code = 500
    fail.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error"
    )

    send = mocker.patch("requests.Session.send", return_value=fail)
    mocker.patch("d2b_data.linkedin_marketing.time.sleep")

    with pytest.raises(requests.exceptions.HTTPError):
        marketing._request_get("https://api.linkedin.com/rest/test", max_retries=2)

    assert send.call_count == 3


def test_request_get_non_retryable_error(marketing, mocker):
    """Non-transient errors (e.g. 403) raise immediately."""
    fail = MagicMock()
    fail.status_code = 403
    fail.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")

    send = mocker.patch("requests.Session.send", return_value=fail)

    with pytest.raises(requests.exceptions.HTTPError):
        marketing._request_get("https://api.linkedin.com/rest/test")

    assert send.call_count == 1


def test_request_get_rejects_negative_retries(marketing):
    """A negative max_retries would skip the loop and silently return None."""
    with pytest.raises(ValueError, match="max_retries"):
        marketing._request_get("https://api.linkedin.com/rest/test", max_retries=-1)


# ------------------------------------------------------------------
# _fetch_paginated_report
# ------------------------------------------------------------------


def test_fetch_paginated_report_single_page(marketing, mocker):
    """A short page ends pagination after one request."""
    mocker.patch.object(
        marketing, "_request_get", return_value={"elements": [{"a": 1}, {"a": 2}]}
    )

    result = marketing._fetch_paginated_report("https://api.linkedin.com/rest/x?q=y")

    assert result == [{"a": 1}, {"a": 2}]
    assert marketing._request_get.call_count == 1


def test_fetch_paginated_report_follows_pages(marketing, mocker):
    """A full page triggers a second request with the offset advanced."""
    full_page = {"elements": [{"i": i} for i in range(50)]}
    last_page = {"elements": [{"i": 50}]}

    mocker.patch.object(marketing, "_request_get", side_effect=[full_page, last_page])

    result = marketing._fetch_paginated_report("https://api.linkedin.com/rest/x?q=y")

    assert len(result) == 51
    called = [call.args[0] for call in marketing._request_get.call_args_list]
    assert called[0].endswith("&count=50&start=0")
    assert called[1].endswith("&count=50&start=50")


def test_fetch_paginated_report_stops_on_empty_elements(marketing, mocker):
    """An empty page ends pagination and returns what was collected."""
    mocker.patch.object(marketing, "_request_get", return_value={"elements": []})

    assert marketing._fetch_paginated_report("https://api.linkedin.com/rest/x?q=y") == []


def test_fetch_paginated_report_stops_on_missing_elements_key(marketing, mocker):
    """A payload without 'elements' does not blow up."""
    mocker.patch.object(marketing, "_request_get", return_value={})

    assert marketing._fetch_paginated_report("https://api.linkedin.com/rest/x?q=y") == []


def test_fetch_paginated_report_propagates_api_error(marketing, mocker):
    """Request errors during pagination are logged and re-raised."""
    mocker.patch.object(
        marketing,
        "_request_get",
        side_effect=requests.exceptions.HTTPError("boom"),
    )

    with pytest.raises(requests.exceptions.HTTPError):
        marketing._fetch_paginated_report("https://api.linkedin.com/rest/x?q=y")


def test_fetch_paginated_report_propagates_quota_error(marketing, mocker):
    """QuotaExhaustedError is not swallowed by the RequestException handler."""
    from d2b_data.linkedin_marketing import QuotaExhaustedError

    mocker.patch.object(
        marketing, "_request_get", side_effect=QuotaExhaustedError("quota")
    )

    with pytest.raises(QuotaExhaustedError):
        marketing._fetch_paginated_report("https://api.linkedin.com/rest/x?q=y")


# ------------------------------------------------------------------
# get_report
# ------------------------------------------------------------------


def _captured_url(marketing):
    """Return the URL passed to _fetch_paginated_report."""
    return marketing._fetch_paginated_report.call_args[0][0]


def test_get_report_requires_pivot(marketing):
    """Statistics reports are meaningless without a pivot."""
    with pytest.raises(ValueError, match="pivot is required"):
        marketing.get_report("123", "2024-01-01", "2024-01-31", "impressions")


def test_get_report_builds_expected_url(marketing, mocker):
    """The URL carries the account URN, granularity, pivot and date range."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report("123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN")

    url = _captured_url(marketing)
    assert url.startswith("https://api.linkedin.com/rest/adAnalytics?q=statistics")
    assert "&pivots=List(CAMPAIGN)" in url
    assert "&timeGranularity=DAILY" in url
    assert (
        "&dateRange=(start:(year:2024,month:1,day:1),end:(year:2024,month:1,day:31))"
        in url
    )
    assert f"&accounts=List({quote('urn:li:sponsoredAccount:123')})" in url


def test_get_report_strips_leading_zeros_from_dates(marketing, mocker):
    """LinkedIn expects integers, so '01' must not reach the URL."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report("123", "2024-01-05", "2024-02-09", "impressions", "CAMPAIGN")

    assert (
        "dateRange=(start:(year:2024,month:1,day:5),end:(year:2024,month:2,day:9))"
        in _captured_url(marketing)
    )


def test_get_report_always_requests_date_range_field(marketing, mocker):
    """Without dateRange in fields, DAILY rows have no date to group by."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report("123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN")

    assert "&fields=impressions,dateRange,pivotValues" in _captured_url(marketing)


def test_get_report_does_not_duplicate_requested_fields(marketing, mocker):
    """A caller that already asks for dateRange/pivotValues gets them once."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report(
        "123", "2024-01-01", "2024-01-31", "impressions,dateRange", "CAMPAIGN"
    )

    url = _captured_url(marketing)
    assert "&fields=impressions,dateRange,pivotValues" in url
    assert url.count("dateRange,") == 1


def test_get_report_strips_whitespace_from_metrics(marketing, mocker):
    """The URL is sent verbatim, so a stray space would corrupt the request."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report(
        "123", "2024-01-01", "2024-01-31", "impressions, clicks", "CAMPAIGN"
    )

    url = _captured_url(marketing)
    assert " " not in url
    assert "&fields=impressions,clicks,dateRange,pivotValues" in url


def test_get_report_strips_whitespace_from_pivots(marketing, mocker):
    """Multiple pivots may be passed with spaces after the comma."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report(
        "123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN, CREATIVE"
    )

    assert "&pivots=List(CAMPAIGN,CREATIVE)" in _captured_url(marketing)


def test_get_report_honours_time_granularity(marketing, mocker):
    """A non-default granularity reaches the URL."""
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    marketing.get_report(
        "123",
        "2024-01-01",
        "2024-01-31",
        "impressions",
        "CAMPAIGN",
        time_granularity="MONTHLY",
    )

    assert "&timeGranularity=MONTHLY" in _captured_url(marketing)


@pytest.mark.parametrize(
    "start, end",
    [
        ("2024-13-01", "2024-01-31"),
        ("01-01-2024", "2024-01-31"),
        ("not-a-date", "2024-01-31"),
        ("2024-01-01", "2024-02-30"),
        ("2024-01", "2024-01-31"),
    ],
)
def test_get_report_rejects_malformed_dates(marketing, mocker, start, end):
    """Bad dates fail fast with a clear message instead of a cryptic unpack error."""
    fetch = mocker.patch.object(marketing, "_fetch_paginated_report", return_value=[])

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        marketing.get_report("123", start, end, "impressions", "CAMPAIGN")

    fetch.assert_not_called()


def test_get_report_rejects_inverted_range(marketing):
    """An end date before the start date is a caller mistake."""
    with pytest.raises(ValueError, match="must not be after"):
        marketing.get_report(
            "123", "2024-02-01", "2024-01-01", "impressions", "CAMPAIGN"
        )


def test_get_report_returns_rows(marketing, mocker):
    """Rows from the paginator are returned untouched."""
    rows = [{"impressions": 10}, {"impressions": 20}]
    mocker.patch.object(marketing, "_fetch_paginated_report", return_value=rows)

    result = marketing.get_report(
        "123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN"
    )
    assert result == rows


def test_get_report_logs_and_reraises_api_error(marketing, mocker):
    """API failures are logged as critical and propagated."""
    marketing.logger = MagicMock()
    mocker.patch.object(
        marketing,
        "_fetch_paginated_report",
        side_effect=requests.exceptions.HTTPError("boom"),
    )

    with pytest.raises(requests.exceptions.HTTPError):
        marketing.get_report("123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN")

    assert marketing.logger.critical.called


# ------------------------------------------------------------------
# get_report_dataframe
# ------------------------------------------------------------------


def test_get_report_dataframe_flattens_nested_fields(marketing, mocker):
    """Nested dicts are flattened with an underscore separator."""
    rows = [
        {
            "impressions": 100,
            "dateRange": {"start": {"year": 2024, "month": 1, "day": 1}},
            "pivotValues": ["urn:li:sponsoredCampaign:1"],
        }
    ]
    mocker.patch.object(marketing, "get_report", return_value=rows)

    df = marketing.get_report_dataframe(
        "123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN"
    )

    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "impressions"] == 100
    assert df.loc[0, "dateRange_start_year"] == 2024


def test_get_report_dataframe_forwards_arguments(marketing, mocker):
    """Every argument is passed through to get_report in order."""
    mock_report = mocker.patch.object(marketing, "get_report", return_value=[])

    marketing.get_report_dataframe(
        "123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN", "MONTHLY"
    )

    mock_report.assert_called_once_with(
        "123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN", "MONTHLY"
    )


def test_get_report_dataframe_empty_result(marketing, mocker):
    """No rows yields an empty DataFrame, not an error."""
    mocker.patch.object(marketing, "get_report", return_value=[])

    df = marketing.get_report_dataframe(
        "123", "2024-01-01", "2024-01-31", "impressions", "CAMPAIGN"
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_report_dataframe_propagates_pivot_validation(marketing):
    """Validation in get_report is not bypassed by the DataFrame wrapper."""
    with pytest.raises(ValueError, match="pivot is required"):
        marketing.get_report_dataframe("123", "2024-01-01", "2024-01-31", "impressions")


# ------------------------------------------------------------------
# Pending methods
# ------------------------------------------------------------------


def test_campaign_name_helpers_are_still_stubs(marketing):
    """Documented as 'Dev Pending' — guards against silent partial rollout."""
    assert marketing.get_campaign_names(["1"]) is None
    assert marketing.get_campaign_group_names(["1"]) is None
