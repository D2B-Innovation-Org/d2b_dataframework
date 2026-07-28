from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

from d2b_data.search_console import GoogleSearchConsole


def test_instance_is_created_correctly(gsc):
    """Verifies that the object is created with the expected defaults."""
    assert gsc.default_api_name == "searchconsole"
    assert gsc.default_version == "v1"
    assert gsc.auto_paginate is True
    assert gsc.row_limit == 25_000
    assert gsc.use_service_account is False


def test_get_service_returns_service(gsc):
    """get_service returns the authenticated service object."""
    assert gsc.get_service() is gsc.service


def test_get_token_returns_token_path(gsc):
    """get_token returns the configured OAuth token path."""
    assert gsc.get_token() == "fake_token.json"


def test_set_auto_paginate_toggles_value(gsc):
    """set_auto_paginate updates and returns the pagination status."""
    assert gsc.set_auto_paginate(False) is False
    assert gsc.auto_paginate is False
    assert gsc.set_auto_paginate(True) is True
    assert gsc.auto_paginate is True


def test_set_auto_paginate_rejects_non_boolean(gsc):
    """set_auto_paginate raises TypeError for non-boolean input."""
    with pytest.raises(TypeError):
        gsc.set_auto_paginate("yes")


def test_create_query_basic_structure(gsc):
    """_create_query builds the expected request body."""
    query = gsc._create_query(
        start_date="2024-01-01",
        end_date="2024-01-31",
        dimensions=["page", "date"],
    )
    assert query["startDate"] == "2024-01-01"
    assert query["endDate"] == "2024-01-31"
    assert query["dimensions"] == ["page", "date"]
    assert query["rowLimit"] == gsc.row_limit
    assert query["startRow"] == 0
    assert query["type"] == "web"
    assert query["dataState"] == "final"
    assert "dimensionFilterGroups" not in query


def test_create_query_includes_filter_groups(gsc):
    """_create_query includes dimensionFilterGroups when provided."""
    filters = [{"filters": [{"dimension": "country", "expression": "chl"}]}]
    query = gsc._create_query(
        start_date="2024-01-01",
        end_date="2024-01-31",
        dimensions=["page"],
        dimension_filter_groups=filters,
    )
    assert query["dimensionFilterGroups"] == filters


def test_to_df_full_data(gsc, raw_response):
    """_to_df maps keys and metrics into a DataFrame."""
    dimensions = ["page", "date"]
    result = gsc._to_df(raw_response, dimensions)

    assert list(result.columns) == [
        "page",
        "date",
        "clicks",
        "impressions",
        "ctr",
        "position",
    ]
    assert len(result) == 2
    assert result.iloc[0]["page"] == "https://example.com/a"
    assert result.iloc[0]["date"] == "2024-01-01"
    assert result.iloc[0]["clicks"] == 10
    assert result.iloc[0]["impressions"] == 100


def test_to_df_no_rows_returns_empty_with_columns(gsc, empty_response):
    """_to_df returns an empty DataFrame with expected columns when no rows."""
    dimensions = ["page", "date"]
    result = gsc._to_df(empty_response, dimensions)

    assert result.empty
    assert list(result.columns) == [
        "page",
        "date",
        "clicks",
        "impressions",
        "ctr",
        "position",
    ]


def test_to_df_missing_metrics_default_to_zero(gsc):
    """_to_df defaults missing metric values to zero."""
    response = {"rows": [{"keys": ["https://example.com/a", "2024-01-01"]}]}
    result = gsc._to_df(response, ["page", "date"])

    assert result.iloc[0]["clicks"] == 0
    assert result.iloc[0]["impressions"] == 0
    assert result.iloc[0]["ctr"] == 0
    assert result.iloc[0]["position"] == 0


def test_empty_df_columns(gsc):
    """_empty_df includes dimensions plus the metric columns."""
    result = gsc._empty_df(["query", "date"])
    assert list(result.columns) == [
        "query",
        "date",
        "clicks",
        "impressions",
        "ctr",
        "position",
    ]
    assert result.empty


def test_validate_report_parameters_missing_property(gsc):
    """_validate_report_parameters raises when property_uri is missing."""
    with pytest.raises(ValueError, match="property_uri is required"):
        gsc._validate_report_parameters(
            property_uri="",
            start_date="2024-01-01",
            end_date="2024-01-31",
            dimensions=["page"],
        )


def test_validate_report_parameters_missing_dimensions(gsc):
    """_validate_report_parameters raises when dimensions is empty."""
    with pytest.raises(ValueError, match="dimensions is required"):
        gsc._validate_report_parameters(
            property_uri="sc-domain:example.com",
            start_date="2024-01-01",
            end_date="2024-01-31",
            dimensions=[],
        )


def test_validate_report_parameters_bad_date_format(gsc):
    """_validate_report_parameters raises on invalid date format."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        gsc._validate_report_parameters(
            property_uri="sc-domain:example.com",
            start_date="01-01-2024",
            end_date="2024-01-31",
            dimensions=["page"],
        )


def test_validate_report_parameters_start_after_end(gsc):
    """_validate_report_parameters raises when start_date > end_date."""
    with pytest.raises(ValueError, match="start_date cannot be greater"):
        gsc._validate_report_parameters(
            property_uri="sc-domain:example.com",
            start_date="2024-02-01",
            end_date="2024-01-01",
            dimensions=["page"],
        )


def test_get_report_raw_no_errors(gsc, raw_response):
    """_get_report_raw returns the first successful response with no retries."""
    mock_execute = MagicMock(return_value=raw_response)
    mock_query = MagicMock(execute=mock_execute)
    mock_searchanalytics = MagicMock(
        return_value=MagicMock(query=MagicMock(return_value=mock_query))
    )
    gsc.service.searchanalytics = mock_searchanalytics

    result = gsc._get_report_raw("sc-domain:example.com", {})
    assert result == raw_response


def test_get_report_raw_non_recoverable_error_raises(gsc, mocker):
    """A non-recoverable status code raises immediately without retrying."""
    fake_resp = MagicMock()
    fake_resp.status = 403
    http_error = HttpError(resp=fake_resp, content=b"Forbidden")

    mock_execute = MagicMock(side_effect=http_error)
    mock_query = MagicMock(execute=mock_execute)
    gsc.service.searchanalytics = MagicMock(
        return_value=MagicMock(query=MagicMock(return_value=mock_query))
    )
    mock_sleep = mocker.patch("time.sleep")

    with pytest.raises(HttpError):
        gsc._get_report_raw("sc-domain:example.com", {})

    assert mock_sleep.call_count == 0


def test_get_report_raw_429_backoff_exhausts_retries(gsc, mocker):
    """A recoverable 429 error retries with backoff and eventually raises."""
    fake_resp = MagicMock()
    fake_resp.status = 429
    http_error = HttpError(resp=fake_resp, content=b"Too Many Requests")

    mock_execute = MagicMock(side_effect=http_error)
    mock_query = MagicMock(execute=mock_execute)
    gsc.service.searchanalytics = MagicMock(
        return_value=MagicMock(query=MagicMock(return_value=mock_query))
    )
    mock_sleep = mocker.patch("time.sleep")

    with pytest.raises(HttpError):
        gsc._get_report_raw("sc-domain:example.com", {})

    assert mock_sleep.call_count == 5


def test_get_report_df_non_paginated(gsc, raw_response):
    """get_report_df returns a single-page DataFrame when auto_paginate is off."""
    gsc.auto_paginate = False
    gsc._get_report_raw = MagicMock(return_value=raw_response)

    result = gsc.get_report_df(
        property_uri="sc-domain:example.com",
        start_date="2024-01-01",
        end_date="2024-01-31",
        dimensions=["page"],
    )

    assert len(result) == 2
    # "date" is auto-appended to the requested dimensions.
    assert "date" in result.columns
    assert "page" in result.columns


def test_get_report_df_appends_date_dimension(gsc, raw_response):
    """get_report_df appends 'date' to dimensions and passes it downstream."""
    gsc.auto_paginate = True
    captured = {}

    def fake_paginated(property_uri, query, dimensions):
        captured["dimensions"] = dimensions
        return gsc._to_df(raw_response, dimensions)

    gsc._get_paginated_report = MagicMock(side_effect=fake_paginated)

    gsc.get_report_df(
        property_uri="sc-domain:example.com",
        start_date="2024-01-01",
        end_date="2024-01-31",
        dimensions=["page"],
    )

    assert captured["dimensions"] == ["page", "date"]


def test_get_report_df_invalid_params_raise(gsc):
    """get_report_df validates parameters before querying."""
    with pytest.raises(ValueError):
        gsc.get_report_df(
            property_uri="",
            start_date="2024-01-01",
            end_date="2024-01-31",
            dimensions=["page"],
        )


def test_get_paginated_report_single_page(gsc, raw_response):
    """_get_paginated_report stops when a page returns fewer than row_limit rows."""
    gsc.row_limit = 25_000
    gsc._get_report_raw = MagicMock(return_value=raw_response)

    result = gsc._get_paginated_report(
        property_uri="sc-domain:example.com",
        query=gsc._create_query("2024-01-01", "2024-01-31", ["page", "date"]),
        dimensions=["page", "date"],
    )

    assert len(result) == 2
    assert gsc._get_report_raw.call_count == 1


def test_get_paginated_report_multiple_pages(gsc):
    """_get_paginated_report concatenates pages until a short page arrives."""
    gsc.row_limit = 2

    full_page = {
        "rows": [
            {"keys": ["a", "2024-01-01"], "clicks": 1, "impressions": 1,
             "ctr": 1, "position": 1},
            {"keys": ["b", "2024-01-01"], "clicks": 1, "impressions": 1,
             "ctr": 1, "position": 1},
        ]
    }
    short_page = {
        "rows": [
            {"keys": ["c", "2024-01-01"], "clicks": 1, "impressions": 1,
             "ctr": 1, "position": 1},
        ]
    }

    gsc._get_report_raw = MagicMock(side_effect=[full_page, short_page])

    result = gsc._get_paginated_report(
        property_uri="sc-domain:example.com",
        query=gsc._create_query("2024-01-01", "2024-01-31", ["page", "date"]),
        dimensions=["page", "date"],
    )

    assert len(result) == 3
    assert gsc._get_report_raw.call_count == 2


def test_get_paginated_report_no_data(gsc, empty_response):
    """_get_paginated_report returns an empty DataFrame when no rows exist."""
    gsc._get_report_raw = MagicMock(return_value=empty_response)

    result = gsc._get_paginated_report(
        property_uri="sc-domain:example.com",
        query=gsc._create_query("2024-01-01", "2024-01-31", ["page", "date"]),
        dimensions=["page", "date"],
    )

    assert result.empty
    assert list(result.columns) == [
        "page",
        "date",
        "clicks",
        "impressions",
        "ctr",
        "position",
    ]
    assert gsc._get_report_raw.call_count == 1
