"""Tests for LinkedinMarketing.

These tests exercise only the logic that runs without touching the
LinkedIn API: authentication from disk, header construction, argument
validation and the early-return branches. Nothing is mocked, so every
assertion here reflects real behaviour of the module.
"""

import json

import pytest

from d2b_data.linkedin_marketing import LinkedinMarketing, QuotaExhaustedError

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


def test_load_token_returns_full_payload(tmp_path):
    """_load_token_from_file returns the whole JSON payload, not just the token."""
    path = tmp_path / "token.json"
    payload = {"access_token": "abc", "expires_in": 3600}
    path.write_text(json.dumps(payload))

    li = LinkedinMarketing()
    li.token_path = str(path)

    assert li._load_token_from_file() == payload
    assert li.token == "abc"


def test_load_token_missing_access_token_key(tmp_path):
    """Token file without 'access_token' key leaves the client unauthenticated."""
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"refresh_token": "something"}))

    li = LinkedinMarketing(token_path=str(path))

    assert li.token is None
    assert li.headers is None


def test_load_token_with_invalid_json(tmp_path):
    """A corrupt token file is handled without raising."""
    path = tmp_path / "broken.json"
    path.write_text("{not json")

    li = LinkedinMarketing(token_path=str(path))

    assert li.token is None
    assert li.headers is None


def test_set_headers_without_token(marketing_no_file):
    """_set_headers does nothing when token is None."""
    marketing_no_file.token = None
    marketing_no_file._set_headers()
    assert marketing_no_file.headers is None


def test_set_headers_without_token_logs_critical(logger):
    """The missing-token case is reported as critical, not silently ignored."""
    li = LinkedinMarketing(logger=logger)
    li._set_headers()

    assert any("access token is missing" in msg for msg in logger.critical_messages)


def test_set_token_loads_and_sets_headers(marketing_no_file, tmp_path):
    """set_token reads the file and configures the headers."""
    path = tmp_path / "other_token.json"
    path.write_text(json.dumps({"access_token": "abc"}))

    marketing_no_file.set_token(str(path))

    assert marketing_no_file.token_path == str(path)
    assert marketing_no_file.token == "abc"
    assert marketing_no_file.headers["Authorization"] == "Bearer abc"


def test_set_token_with_bad_file_clears_previous_credentials(marketing, tmp_path):
    """A failed reload must not leave the old token/headers in place."""
    assert marketing.token == "fake_token_123"

    marketing.set_token(str(tmp_path / "gone.json"))

    assert marketing.token is None
    assert marketing.headers is None


def test_default_logger_is_built_when_none_given(marketing_no_file):
    """A fallback logger exposing .info/.critical/.debug is created."""
    assert hasattr(marketing_no_file.logger, "info")
    assert hasattr(marketing_no_file.logger, "critical")
    assert hasattr(marketing_no_file.logger, "debug")


def test_default_logger_methods_run(marketing_no_file):
    """The fallback logger accepts messages without raising."""
    marketing_no_file.logger.info("info message")
    marketing_no_file.logger.critical("critical message")
    marketing_no_file.logger.debug("debug message")


def test_custom_logger_is_used(logger):
    """An injected logger receives the init message."""
    LinkedinMarketing(logger=logger)

    assert logger.info_messages
    assert any("LinkedinMarketing" in msg for msg in logger.info_messages)


# ------------------------------------------------------------------
# _request_get — guard clauses (no network involved)
# ------------------------------------------------------------------


def test_request_get_no_headers_raises(marketing_no_file):
    """Calling _request_get without auth raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Headers not set"):
        marketing_no_file._request_get("https://example.com")


def test_request_get_rejects_negative_retries(marketing):
    """A negative max_retries would skip the loop and silently return None."""
    with pytest.raises(ValueError, match="max_retries"):
        marketing._request_get("https://api.linkedin.com/rest/test", max_retries=-1)


def test_quota_exhausted_error_is_an_exception():
    """QuotaExhaustedError can be raised and caught like any exception."""
    assert issubclass(QuotaExhaustedError, Exception)

    with pytest.raises(QuotaExhaustedError, match="quota"):
        raise QuotaExhaustedError("quota")


# ------------------------------------------------------------------
# Campaign metadata helpers — empty-input branch
# ------------------------------------------------------------------


def test_get_campaign_names_without_ids_returns_empty_dict(marketing):
    """An empty id set short-circuits before any API call."""
    assert marketing._get_campaign_names(set(), "123") == {}


def test_get_campaign_group_names_without_ids_returns_empty_dict(marketing):
    """An empty id set short-circuits before any API call."""
    assert marketing._get_campaign_group_names(set(), "123") == {}


def test_campaign_helpers_log_the_empty_case(logger, token_file):
    """The empty-input branch is reported so it is not mistaken for a failure."""
    li = LinkedinMarketing(token_path=token_file, logger=logger)

    li._get_campaign_names(set(), "123")
    li._get_campaign_group_names(set(), "123")

    empty_logs = [msg for msg in logger.info_messages if "No campaign id" in msg]
    assert len(empty_logs) == 2


# ------------------------------------------------------------------
# get_report — validation (fails before reaching the API)
# ------------------------------------------------------------------


def test_get_report_requires_pivot(marketing):
    """Statistics reports are meaningless without a pivot."""
    with pytest.raises(ValueError, match="pivot is required"):
        marketing.get_report("123", "2024-01-01", "2024-01-31", ["impressions"])


def test_get_report_rejects_empty_pivot_list(marketing):
    """An empty pivot list is treated the same as no pivot at all."""
    with pytest.raises(ValueError, match="pivot is required"):
        marketing.get_report("123", "2024-01-01", "2024-01-31", ["impressions"], [])


def test_get_report_rejects_more_than_three_pivots(marketing):
    """LinkedIn accepts at most 3 pivots per query."""
    pivots = ["CAMPAIGN", "CREATIVE", "ACCOUNT", "COMPANY"]

    with pytest.raises(ValueError, match="3 pivot values"):
        marketing.get_report("123", "2024-01-01", "2024-01-31", ["impressions"], pivots)


def test_get_report_rejects_more_than_twenty_metrics(marketing):
    """LinkedIn accepts at most 20 metrics per query."""
    metrics = [f"metric_{i}" for i in range(21)]

    with pytest.raises(ValueError, match="20 metrics"):
        marketing.get_report("123", "2024-01-01", "2024-01-31", metrics, ["CAMPAIGN"])


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
def test_get_report_rejects_malformed_dates(marketing, start, end):
    """Bad dates fail fast with a clear message instead of a cryptic unpack error."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        marketing.get_report("123", start, end, ["impressions"], ["CAMPAIGN"])


def test_get_report_rejects_inverted_range(marketing):
    """An end date before the start date is a caller mistake."""
    with pytest.raises(ValueError, match="must not be after"):
        marketing.get_report(
            "123", "2024-02-01", "2024-01-01", ["impressions"], ["CAMPAIGN"]
        )


def test_get_report_does_not_mutate_the_metrics_list(marketing):
    """The caller's list must survive the required-fields injection untouched."""
    metrics = ["impressions"]

    with pytest.raises(ValueError, match="must not be after"):
        marketing.get_report("123", "2024-02-01", "2024-01-01", metrics, ["CAMPAIGN"])

    assert metrics == ["impressions"]


# ------------------------------------------------------------------
# get_report_dataframe — validation is not bypassed
# ------------------------------------------------------------------


def test_get_report_dataframe_propagates_pivot_validation(marketing):
    """Validation in get_report is not bypassed by the DataFrame wrapper."""
    with pytest.raises(ValueError, match="pivot is required"):
        marketing.get_report_dataframe(
            "123", "2024-01-01", "2024-01-31", ["impressions"], None
        )


def test_get_report_dataframe_propagates_date_validation(marketing):
    """Malformed dates are rejected before any request is attempted."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        marketing.get_report_dataframe(
            "123", "not-a-date", "2024-01-31", ["impressions"], ["CAMPAIGN"]
        )


@pytest.mark.parametrize("get_campaign_information", [True, False])
def test_get_report_dataframe_validates_in_both_branches(
    marketing, get_campaign_information
):
    """Both the enriched and the plain branch go through get_report's checks."""
    with pytest.raises(ValueError, match="must not be after"):
        marketing.get_report_dataframe(
            "123",
            "2024-02-01",
            "2024-01-01",
            ["impressions"],
            ["CAMPAIGN"],
            get_campaign_information=get_campaign_information,
        )
