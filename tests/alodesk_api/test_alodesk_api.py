from datetime import date, timedelta

import pandas as pd
import pytest
import requests

from d2b_data.Alodesk_API import Alodesk_API


BASE = "https://automovil.alodesk.io"


@pytest.fixture(autouse=True)
def no_backoff(mocker):
    """tenacity waits up to 30s between retries; skip the real sleeping."""
    return mocker.patch("tenacity.nap.time.sleep")


@pytest.fixture
def logger(mocker):
    return mocker.MagicMock()


@pytest.fixture
def api(logger):
    return Alodesk_API(base_url=BASE + "/", token="fake-token", verbose_logger=logger)


def response(mocker, payload, status_code=200, text=""):
    resp = mocker.MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = payload
    return resp


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_base_url_trailing_slash_is_stripped(api):
    """A trailing slash would produce double slashes in every URL."""
    assert api.base_url == BASE


def test_headers_carry_the_token(api):
    """The bearer token and a descriptive user agent are sent."""
    assert api.headers["Authorization"] == "Bearer fake-token"
    assert api.headers["Accept"] == "application/json"
    assert "alodesk-api" in api.headers["User-Agent"]


def test_init_logs_the_version(api, logger):
    """Construction announces itself through the logger."""
    assert "INIT Alodesk_API" in logger.log.call_args_list[0].args[0]


def test_null_verbose_is_used_when_no_logger_given(capsys):
    """Without a logger, .log() is a no-op and .critical() prints."""
    api = Alodesk_API(base_url=BASE, token="t")
    api.verbose.log("invisible")
    api.verbose.critical("visible")
    out = capsys.readouterr().out
    assert "invisible" not in out
    assert "[CRITICAL] visible" in out


# --------------------------------------------------------------------- #
# dedup_leads
# --------------------------------------------------------------------- #
def test_dedup_leads_empty_dataframe_is_returned_as_is():
    """An empty frame short-circuits without touching columns."""
    df = pd.DataFrame()
    assert Alodesk_API.dedup_leads(df) is df


def test_dedup_leads_keeps_the_most_recent_record():
    """Duplicated lead_ids collapse to the newest updated_at."""
    df = pd.DataFrame({
        "lead_id": [1, 1, 2],
        "updated_at": ["2024-01-01", "2024-03-01", "2024-02-01"],
        "estado": ["nuevo", "cerrado", "nuevo"],
    })
    result = Alodesk_API.dedup_leads(df)
    assert len(result) == 2
    assert result.loc[result["lead_id"] == 1, "estado"].item() == "cerrado"


def test_dedup_leads_does_not_mutate_the_input():
    """The original DataFrame keeps its string timestamps."""
    df = pd.DataFrame({"lead_id": [1], "updated_at": ["2024-01-01"]})
    Alodesk_API.dedup_leads(df)
    assert df["updated_at"].dtype == object


def test_dedup_leads_coerces_invalid_timestamps():
    """Unparseable timestamps become NaT instead of raising."""
    df = pd.DataFrame({"lead_id": [1, 2], "updated_at": ["2024-01-01", "no es fecha"]})
    result = Alodesk_API.dedup_leads(df)
    assert result["updated_at"].isna().sum() == 1


def test_dedup_leads_accepts_custom_column_names():
    """The id and timestamp columns are configurable."""
    df = pd.DataFrame({"id": [1, 1], "fecha": ["2024-01-01", "2024-02-01"], "v": ["a", "b"]})
    result = Alodesk_API.dedup_leads(df, id_col="id", ts_col="fecha")
    assert result["v"].tolist() == ["b"]


# --------------------------------------------------------------------- #
# _fetch
# --------------------------------------------------------------------- #
def test_fetch_builds_the_url_and_returns_json(api, mocker):
    """Endpoint slashes are normalised and the JSON body is returned."""
    get = mocker.patch("requests.get", return_value=response(mocker, {"ok": True}))

    assert api._fetch("/api/leads/") == {"ok": True}
    assert get.call_args.args[0] == f"{BASE}/api/leads/"
    assert get.call_args.kwargs["headers"] == api.headers
    assert get.call_args.kwargs["timeout"] == 30


def test_fetch_forwards_params(api, mocker):
    """Query params are passed through untouched."""
    get = mocker.patch("requests.get", return_value=response(mocker, {}))
    api._fetch("api/leads/", params={"page": 2})
    assert get.call_args.kwargs["params"] == {"page": 2}


def test_fetch_reports_rate_limit(api, logger, mocker):
    """A 429 is escalated as critical before raising."""
    resp = response(mocker, {}, status_code=429, text="slow down")
    resp.raise_for_status.side_effect = requests.HTTPError("429")
    mocker.patch("requests.get", return_value=resp)

    with pytest.raises(requests.HTTPError):
        api._fetch("api/leads/")
    assert "Rate-limit alcanzado" in logger.critical.call_args.args[0]


def test_fetch_retries_and_then_succeeds(api, mocker):
    """A transient connection error is retried, not surfaced."""
    get = mocker.patch("requests.get", side_effect=[
        requests.ConnectionError("sin red"),
        response(mocker, {"ok": True}),
    ])
    assert api._fetch("api/leads/") == {"ok": True}
    assert get.call_count == 2


def test_fetch_gives_up_after_four_attempts(api, mocker):
    """After the retry budget is spent the error is re-raised."""
    get = mocker.patch("requests.get", side_effect=requests.Timeout("lento"))

    with pytest.raises(requests.Timeout):
        api._fetch("api/leads/")
    assert get.call_count == 4


def test_fetch_does_not_retry_unexpected_errors(api, mocker):
    """Only network errors are retried; a ValueError fails immediately."""
    get = mocker.patch("requests.get", side_effect=ValueError("boom"))

    with pytest.raises(ValueError):
        api._fetch("api/leads/")
    assert get.call_count == 1


# --------------------------------------------------------------------- #
# _paginate
# --------------------------------------------------------------------- #
def test_paginate_walks_pages_until_next_is_falsy(api, mocker):
    """Pagination follows 'next' and stops when it disappears."""
    fetch = mocker.patch.object(api, "_fetch", side_effect=[
        {"results": [{"id": 1}], "next": "?page=2"},
        {"results": [{"id": 2}], "next": None},
    ])
    assert list(api._paginate("api/leads/")) == [{"id": 1}, {"id": 2}]
    assert fetch.call_args_list[0].kwargs["params"] == {"page": 1}
    assert fetch.call_args_list[1].kwargs["params"] == {"page": 2}


def test_paginate_merges_base_params(api, mocker):
    """Caller params survive alongside the page number."""
    mocker.patch.object(api, "_fetch", return_value={"results": []})
    list(api._paginate("api/leads/", params={"startDate": "2024-01-01"}))
    assert api._fetch.call_args.kwargs["params"] == {"startDate": "2024-01-01", "page": 1}


def test_paginate_supports_a_custom_page_param(api, mocker):
    """The page parameter name is configurable."""
    mocker.patch.object(api, "_fetch", return_value={"results": []})
    list(api._paginate("api/leads/", page_param="pagina"))
    assert api._fetch.call_args.kwargs["params"] == {"pagina": 1}


def test_paginate_handles_a_plain_list_response(api, mocker):
    """An unpaginated list endpoint yields its items and stops."""
    fetch = mocker.patch.object(api, "_fetch", return_value=[{"id": 1}, {"id": 2}])
    assert list(api._paginate("api/leads/")) == [{"id": 1}, {"id": 2}]
    assert fetch.call_count == 1


def test_paginate_stops_on_empty_list(api, mocker):
    """An empty list ends the iteration."""
    mocker.patch.object(api, "_fetch", return_value=[])
    assert list(api._paginate("api/leads/")) == []


def test_paginate_stops_on_empty_results(api, mocker):
    """A dict with no results ends the iteration."""
    mocker.patch.object(api, "_fetch", return_value={"results": []})
    assert list(api._paginate("api/leads/")) == []


def test_paginate_reports_unexpected_payloads(api, logger, mocker):
    """A response that is neither list nor dict is escalated."""
    mocker.patch.object(api, "_fetch", return_value="texto plano")
    assert list(api._paginate("api/leads/")) == []
    assert "Formato inesperado" in logger.critical.call_args.args[0]


# --------------------------------------------------------------------- #
# download_leads
# --------------------------------------------------------------------- #
def test_download_leads_uses_a_30_day_window_by_default(api, mocker):
    """The default window is today-30 through today."""
    paginate = mocker.patch.object(api, "_paginate", return_value=iter([]))
    api.download_leads()

    params = paginate.call_args.kwargs["params"]
    assert params["endDate"] == date.today().isoformat()
    assert params["startDate"] == (date.today() - timedelta(days=30)).isoformat()


def test_download_leads_honours_days_back(api, mocker):
    """days_back moves the start of the window."""
    paginate = mocker.patch.object(api, "_paginate", return_value=iter([]))
    api.download_leads(days_back=7)
    assert paginate.call_args.kwargs["params"]["startDate"] == (
        date.today() - timedelta(days=7)
    ).isoformat()


def test_download_leads_single_day_collapses_the_window(api, mocker):
    """single_day requests exactly one date on both ends."""
    paginate = mocker.patch.object(api, "_paginate", return_value=iter([]))
    api.download_leads(days_back=1, single_day=True)

    params = paginate.call_args.kwargs["params"]
    expected = (date.today() - timedelta(days=1)).isoformat()
    assert params["startDate"] == params["endDate"] == expected


def test_download_leads_returns_a_dataframe(api, mocker):
    """Rows from the paginator become a DataFrame."""
    mocker.patch.object(api, "_paginate", return_value=iter([
        {"lead_id": 1, "estado": "nuevo"},
        {"lead_id": 2, "estado": "cerrado"},
    ]))
    df = api.download_leads()
    assert len(df) == 2
    assert list(df.columns) == ["lead_id", "estado"]


def test_download_leads_empty_result_is_an_empty_dataframe(api, mocker):
    """No leads yields an empty DataFrame rather than None."""
    mocker.patch.object(api, "_paginate", return_value=iter([]))
    df = api.download_leads()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_download_leads_hits_the_report_endpoint(api, mocker):
    """The leads report endpoint is the one queried."""
    paginate = mocker.patch.object(api, "_paginate", return_value=iter([]))
    api.download_leads()
    assert paginate.call_args.args[0] == "api/leads/report/"
