from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# get_report_dataframe — happy path
# ---------------------------------------------------------------------------


def test_get_report_dataframe_returns_dataframe(fb, base_params, mocker):
    """Normal report with data returns a populated DataFrame."""
    raw = [
        {
            "impressions": "1000",
            "clicks": "50",
            "spend": "20.5",
            "date_start": "2024-01-01",
            "date_stop": "2024-01-01",
            "account_id": "123",
        },
        {
            "impressions": "2000",
            "clicks": "80",
            "spend": "35.0",
            "date_start": "2024-01-02",
            "date_stop": "2024-01-02",
            "account_id": "123",
        },
    ]
    mocker.patch.object(fb, "get_report", return_value=raw)
    df = fb.get_report_dataframe(base_params)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "impressions" in df.columns


def test_get_report_dataframe_empty_report_returns_empty_df(fb, base_params, mocker):
    """Empty list from API returns empty DataFrame with expected columns."""
    mocker.patch.object(fb, "get_report", return_value=[])
    df = fb.get_report_dataframe(base_params)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    for col in base_params["fields"]:
        assert col in df.columns


# -------------------------


def test_get_report_dataframe_raises_on_invalid_report_type(fb, base_params, mocker):
    """If get_report returns a non-list, ValueError is raised."""
    mocker.patch.object(fb, "get_report", return_value="not a list")
    with pytest.raises(ValueError, match="Bad data"):
        fb.get_report_dataframe(base_params)


def test_get_report_dataframe_raises_on_list_of_non_dicts(fb, base_params, mocker):
    """List containing non-dict elements triggers ValueError."""
    mocker.patch.object(fb, "get_report", return_value=["string", 42])
    with pytest.raises(ValueError, match="Bad data"):
        fb.get_report_dataframe(base_params)


def test_get_report_dataframe_raises_on_none_report(fb, base_params, mocker):
    """None returned by get_report triggers ValueError."""
    mocker.patch.object(fb, "get_report", return_value=None)
    with pytest.raises(ValueError, match="Bad data"):
        fb.get_report_dataframe(base_params)


# ---------------------------------------------------------------------------
# get_report_dataframe — actions expansion
# ---------------------------------------------------------------------------


def test_get_report_dataframe_expands_actions(fb, base_params, mocker):
    """Rows with an 'actions' list get flattened into _action_* columns."""
    raw = [
        {
            "impressions": "500",
            "date_start": "2024-01-01",
            "date_stop": "2024-01-01",
            "account_id": "123",
            "actions": [
                {"action_type": "link_click", "value": "10"},
                {"action_type": "purchase", "value": "2"},
            ],
        }
    ]
    mocker.patch.object(fb, "get_report", return_value=raw)
    df = fb.get_report_dataframe(base_params)

    assert "_action_link_click" in df.columns
    assert "_action_purchase" in df.columns
    assert df["_action_link_click"].iloc[0] == "10"
    assert df["_action_purchase"].iloc[0] == "2"


def test_get_report_dataframe_missing_action_returns_zero(fb, base_params, mocker):
    """Rows missing a given action_type get 0 in the expanded column."""
    raw = [
        {
            "impressions": "100",
            "date_start": "2024-01-01",
            "date_stop": "2024-01-01",
            "account_id": "123",
            "actions": [{"action_type": "link_click", "value": "5"}],
        },
        {
            "impressions": "200",
            "date_start": "2024-01-02",
            "date_stop": "2024-01-02",
            "account_id": "123",
            "actions": [],
        },
    ]
    mocker.patch.object(fb, "get_report", return_value=raw)
    df = fb.get_report_dataframe(base_params)

    assert df["_action_link_click"].iloc[1] == 0


def test_get_report_dataframe_no_duplicate_action_columns(fb, base_params, mocker):
    """Duplicate action_type values across rows don't create duplicate columns."""
    raw = [
        {
            "impressions": "100",
            "date_start": "2024-01-01",
            "date_stop": "2024-01-01",
            "account_id": "123",
            "actions": [{"action_type": "purchase", "value": "1"}],
        },
        {
            "impressions": "200",
            "date_start": "2024-01-02",
            "date_stop": "2024-01-02",
            "account_id": "123",
            "actions": [{"action_type": "purchase", "value": "3"}],
        },
    ]
    mocker.patch.object(fb, "get_report", return_value=raw)
    df = fb.get_report_dataframe(base_params)

    assert df.columns.tolist().count("_action_purchase") == 1


# ---------------------------------------------------------------------------
# get_report_dataframe — multiple accounts
# ---------------------------------------------------------------------------


def test_get_report_dataframe_multiple_accounts_concatenates(fb, base_params, mocker):
    """Passing a list of accounts concatenates results into a single DataFrame."""
    raw = [
        {
            "impressions": "100",
            "date_start": "2024-01-01",
            "date_stop": "2024-01-01",
            "account_id": "111",
        }
    ]
    mocker.patch.object(fb, "get_report", return_value=raw)

    df = fb.get_report_dataframe(base_params, id_account=["111", "222"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# get_report — async job polling
# ---------------------------------------------------------------------------


def _make_async_job(status_sequence, records=None):
    """Build a mock async job that cycles through status_sequence on api_get()."""
    job = MagicMock()
    status_iter = iter(status_sequence)

    def fake_api_get():
        return {"async_status": next(status_iter)}

    job.api_get.side_effect = fake_api_get

    if records is not None:
        mock_records = []
        for r in records:
            rec = MagicMock()
            rec.export_all_data.return_value = r
            mock_records.append(rec)
        job.get_result.return_value = mock_records

    return job


def test_get_report_returns_records_on_job_completed(fb, mocker):
    """get_report polls until Job Completed and returns exported records."""
    records = [{"impressions": "500", "spend": "10"}]
    async_job = _make_async_job(["Job Running", "Job Completed"], records=records)

    mock_account = MagicMock()
    mock_account.get_insights.return_value = async_job
    mocker.patch("d2b_data.Facebook_Marketing.AdAccount", return_value=mock_account)
    mocker.patch("time.sleep")

    result = fb.get_report({}, "act_123")

    assert result == [{"impressions": "500", "spend": "10"}]


def test_get_report_downloads_result_in_large_pages(fb, mocker):
    """Results are paged with page_size rows instead of Meta's default of 25."""
    async_job = _make_async_job(["Job Completed", "Job Completed"], records=[])

    mock_account = MagicMock()
    mock_account.get_insights.return_value = async_job
    mocker.patch("d2b_data.Facebook_Marketing.AdAccount", return_value=mock_account)
    mocker.patch("time.sleep")

    fb.get_report({}, "act_123")
    fb.get_report({}, "act_123", page_size=1000)

    assert async_job.get_result.call_args_list[0].kwargs == {"params": {"limit": 500}}
    assert async_job.get_result.call_args_list[1].kwargs == {"params": {"limit": 1000}}


def test_get_report_raises_on_job_failed(fb, mocker):
    """get_report raises Exception when job status is Job Failed."""
    async_job = _make_async_job(["Job Failed"])

    mock_account = MagicMock()
    mock_account.get_insights.return_value = async_job
    mocker.patch("d2b_data.Facebook_Marketing.AdAccount", return_value=mock_account)
    mocker.patch("time.sleep")

    with pytest.raises(Exception, match="Job falló"):
        fb.get_report({}, "act_123")


def test_get_report_raises_timeout_when_job_never_completes(fb, mocker):
    """get_report raises TimeoutError after 60 polling attempts."""
    async_job = _make_async_job(["Job Running"] * 61)

    mock_account = MagicMock()
    mock_account.get_insights.return_value = async_job
    mocker.patch("d2b_data.Facebook_Marketing.AdAccount", return_value=mock_account)
    mocker.patch("time.sleep")

    with pytest.raises(TimeoutError):
        fb.get_report({}, "act_123")


def test_get_report_returns_empty_list_when_result_is_none(fb, mocker):
    """get_report returns [] when get_result() is None (empty report)."""
    async_job = _make_async_job(["Job Completed"])
    async_job.get_result.return_value = None

    mock_account = MagicMock()
    mock_account.get_insights.return_value = async_job
    mocker.patch("d2b_data.Facebook_Marketing.AdAccount", return_value=mock_account)
    mocker.patch("time.sleep")

    result = fb.get_report({}, "act_123")
    assert result == []


def test_get_report_raises_on_critical_facebook_error(fb, mocker):
    """FacebookRequestError with subcode 99 raises immediately without retrying."""
    from facebook_business.exceptions import FacebookRequestError

    # Build a real FacebookRequestError instance so isinstance() triggers correctly.
    body = '{"error": {"code": 100, "error_subcode": 99, "message": "Auth error", "type": "OAuthException"}}'
    error = FacebookRequestError(
        message="Auth error",
        request_context={},
        http_status=500,
        http_headers={},
        body=body,
    )

    mock_account = MagicMock()
    mock_account.get_insights.side_effect = error
    mocker.patch("d2b_data.Facebook_Marketing.AdAccount", return_value=mock_account)
    mocker.patch("time.sleep")

    with pytest.raises(Exception, match="subcode 99"):
        fb.get_report({}, "act_123")


# ---------------------------------------------------------------------------
# _split_text
# ---------------------------------------------------------------------------


def test_split_text_returns_value_for_matching_action(fb):
    actions = [{"action_type": "purchase", "value": "5"}]
    assert fb._split_text(actions, "purchase") == "5"


def test_split_text_returns_zero_for_missing_action(fb):
    actions = [{"action_type": "link_click", "value": "3"}]
    assert fb._split_text(actions, "purchase") == 0


def test_split_text_returns_zero_for_non_list(fb):
    assert fb._split_text(None, "purchase") == 0
    assert fb._split_text("not_a_list", "purchase") == 0
    assert fb._split_text(0, "purchase") == 0


# ---------------------------------------------------------------------------
# _unique_actions
# ---------------------------------------------------------------------------


def test_unique_actions_extracts_action_types(fb):
    df = pd.DataFrame(
        {
            "spend": ["10", "20"],
            "actions": [
                [{"action_type": "purchase", "value": "1"}],
                [
                    {"action_type": "purchase", "value": "2"},
                    {"action_type": "link_click", "value": "8"},
                ],
            ],
        }
    )
    result = fb._unique_actions(df)
    assert "actions" in result
    assert result["actions"] == {"purchase", "link_click"}


def test_unique_actions_ignores_non_list_columns(fb):
    df = pd.DataFrame({"spend": ["10", "20"], "clicks": ["5", "8"]})
    result = fb._unique_actions(df)
    assert result == {}
