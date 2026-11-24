from d2b_data.Google_GA4 import Google_GA4
from unittest.mock import MagicMock
from googleapiclient.errors import HttpError
import pytest

def test_instance_is_created_correctly(ga4):
    """Verifies that the object is created correctly"""
    assert ga4.auto_paginate == True
    assert ga4.debug_status == False
    assert ga4.intraday_limit == 30 * 100000

def test_extract_sampling_info_no_metadata(ga4):
    """Verifies the correct reading of an object that doesn't have metadata"""
    report_data = {}

    # 2. Llama al método
    result = ga4._extract_sampling_info(report_data)

    # 3. Verifica lo que esperas
    assert result['sampled'] == False
    assert result['sampling_percentage'] == 100.0
    assert result['dataLossFromOtherRow'] == False

def test_extract_sampling_info_with_metadata(ga4):
    """Verifies that the object comes with metadata"""

    report_data = {
    "metadata": {
        "samplingMetadatas": [
            {"samplesReadCount": "500", "samplingSpaceSize": "1000"}
        ]
     }
    }
    
    result = ga4._extract_sampling_info(report_data) 

    assert result['sampled'] == True
    assert result['samplesReadCounts'] == 500
    assert result['samplingSpaceSizes'] == 1000
    assert result['sampling_percentage'] == 50.0

def test_to_dataframe_full_data(ga4):
    """Verifies correct implementation when there's full data returned"""
    raw_response = {
    "reports": [{
        "dimensionHeaders": [{"name": "date"}, {"name": "city"}],
        "metricHeaders": [{"name": "sessions"}],
        "rows": [
            {
                "dimensionValues": [{"value": "2024-01-01"}, {"value": "Santiago"}],
                "metricValues": [{"value": "150"}]
            } 
          ]
        }]
    }

    result = ga4._to_df(raw_response)
    assert list(result.columns) == ['date', 'city', 'sessions']
    assert result.iloc[0]['date'] == "2024-01-01"
    assert result.iloc[0]['city'] == "Santiago"
    assert result.iloc[0]['sessions'] == "150"
    
def test_to_dataframe_no_data(ga4):
    """Verifies correct implementation when there's no data returned"""
    raw_response = {
    "reports": []
    }

    result = ga4._to_df(raw_response)

    assert result.empty == True

def test_to_dataframe_no_rows(ga4):
    """Verifies implementation when the DataFrame returns with no rows"""

    raw_response = {
    "reports": [{
        "dimensionHeaders": [{"name": "date"}, {"name": "city"}],
        "metricHeaders": [{"name": "sessions"}],
        "rows": []
        }]
    }

    result = ga4._to_df(raw_response)
    assert len(result) == 0

def test_get_report_raw_no_errors(ga4):
    """Verifies the first answer is correct with no retries"""
    raw_response = {"reports": [{
        "dimensionHeaders": [{"name": "date"}, {"name": "city"}],
        "metricHeaders": [{"name": "sessions"}],
        "rows": [
            {
                "dimensionValues": [{"value": "2024-01-01"}, {"value": "Santiago"}],
                "metricValues": [{"value": "150"}]
            } 
          ]
        }]
    }

    mock_execute = MagicMock(return_value=raw_response)
    mock_batch = MagicMock(execute=mock_execute)
    mock_properties = MagicMock(
        return_value=MagicMock(
            batchRunReports=MagicMock(return_value=mock_batch)
        )
    )
    
    ga4.service.properties = mock_properties

    result = ga4._get_report_raw("properties/123", {"requests": [{}]})
    assert result == raw_response

def test_get_report_429_backoff(ga4, mocker):
    """Testing for the 429 error handling and exponential backoff"""
    
    fake_resp = MagicMock()
    fake_resp.status = 429
    http_error = HttpError(resp=fake_resp, content=b"Too Many Requests")
    mock_execute = MagicMock(side_effect=http_error)
    mock_batch = MagicMock(execute=mock_execute)
    mock_properties = MagicMock(
        return_value=MagicMock(
            batchRunReports=MagicMock(return_value=mock_batch)
        )
    )
    ga4.service.properties = mock_properties

    mock_sleep = mocker.patch("time.sleep")

    with pytest.raises(HttpError):
        ga4._get_report_raw("properties/123", {"requests": [{}]})

    assert mock_sleep.call_count == 5

@pytest.mark.parametrize("status", [502, 504])
def test_get_report_gateway_errors_backoff(ga4, mocker, status):
    """Testing that 502/504 gateway errors are retried like 429"""

    fake_resp = MagicMock()
    fake_resp.status = status
    http_error = HttpError(resp=fake_resp, content=b"Bad Gateway")
    mock_execute = MagicMock(side_effect=http_error)
    mock_batch = MagicMock(execute=mock_execute)
    ga4.service.properties = MagicMock(
        return_value=MagicMock(
            batchRunReports=MagicMock(return_value=mock_batch)
        )
    )

    mock_sleep = mocker.patch("time.sleep")

    with pytest.raises(HttpError):
        ga4._get_report_raw("properties/123", {"requests": [{}]})

    assert mock_sleep.call_count == 5

def test_get_single_report_no_sampling_returns_no_sampling_cols(ga4, df_fake):
    """Tesing that the class handles sampling data, no sampling data and an empty response with sampling data"""
    
    ga4._get_report_raw = MagicMock(return_value={"anything": True}) 
    ga4._to_df = MagicMock(return_value=df_fake)

    result = ga4._get_single_report("properties/123", {})

    sampling_cols = ['samplesReadCounts', 'samplingSpaceSizes', 'sampling_percentage',
                   'sampled', 'dataLossFromOtherRow']
    
    assert not any(col in result.columns for col in sampling_cols)

def test_get_single_report_with_sampling_returns_data_and_sampling_cols(ga4, raw_response_with_sampling, df_fake_with_sampling):
    """Testing thath the report comes with sampling data"""
    ga4._get_report_raw = MagicMock(return_value=raw_response_with_sampling)  
    ga4._to_df = MagicMock(return_value=df_fake_with_sampling)

    result = ga4._get_single_report("properties/123", {})

    sampling_cols = ['samplesReadCounts', 'samplingSpaceSizes', 'sampling_percentage',
                   'sampled', 'dataLossFromOtherRow']
    
    assert all(col in result.columns for col in sampling_cols)

def test_get_paginated_report_returns_concatenated_dataframe(ga4, df_fake):
    """Tests that get paginated report returns all pages and the correct DataFrame"""
    
    ga4.set_batch_size(1)
    query = {
    "requests": [{
        "dateRanges": [{"startDate": "2024-01-01", "endDate": "2024-01-02"}]
    }]
    }

    ga4._get_all_rows_for_day = MagicMock(return_value=[df_fake])
    result = ga4._get_paginated_report("properties/123", query)

    assert ga4._get_all_rows_for_day.call_count == 2 
    assert not result.empty
    assert list(result.columns) == ['date', 'city', 'sessions'] 

def test_get_paginated_report_no_rows(ga4):
    """Tests that get paginated report returns an empty DataFrame when there are no rows"""
    
    ga4.set_batch_size(1)
    query = {
    "requests": [{
        "dateRanges": [{"startDate": "2024-01-01", "endDate": "2024-01-02"}]
    }]
    }

    ga4._get_all_rows_for_day = MagicMock(return_value=[]) 
    result = ga4._get_paginated_report("properties/123", query)

    assert ga4._get_all_rows_for_day.call_count == 2 
    assert result.empty

def test_create_daily__create_daily_query(ga4):
    """Tests that the daily query is created correctly"""
    
    query = {
        "requests": [{
            "dimensions": [{"name": "date"}, {"name": "city"}],
            "metrics": [{"name": "sessions"}],
            "dateRanges": [{"startDate": "2024-01-01", "endDate": "2024-01-31"}] 
        }]
    }

    start_date = "2025-01-01"
    end_date = "2025-01-01"

    result = ga4._create_daily_query(query, start_date, end_date)

    assert result['requests'][0]['dateRanges'][0]['startDate'] == start_date
    assert result['requests'][0]['dateRanges'][0]['endDate'] == end_date


def _make_ga4_report(rows_by_day: list[tuple[str, list[tuple[str, str]]]], sampling=None):
    reports = []
    for idx, (day_str, pairs) in enumerate(rows_by_day):
        report = {
            "dimensionHeaders": [{"name": "date"}, {"name": "city"}],
            "metricHeaders": [{"name": "sessions"}],
            "rows": [
                {
                    "dimensionValues": [{"value": day_str}, {"value": city}],
                    "metricValues": [{"value": sessions}],
                }
                for city, sessions in pairs
            ],
        }
        if sampling and idx in sampling:
            report["metadata"] = sampling[idx]
        reports.append(report)
    return {"reports": reports}


def test_batch_size_and_request_delay_defaults_and_validation(ga4, mocker):
    """Verifies default batch_size=5, request_delay=0.0, and validation errors."""
    assert ga4.batch_size == 5
    assert ga4.request_delay == 0.0

    mock_token_mng = MagicMock()
    mock_token_mng.get_service.return_value = MagicMock()
    mocker.patch(
        "d2b_data.Google_GA4.d2b_data.Google_Token_MNG.Google_Token_MNG",
        return_value=mock_token_mng,
    )

    for invalid_batch in [0, 6, True]:
        with pytest.raises(ValueError):
            Google_GA4("s", "t", batch_size=invalid_batch)
        with pytest.raises(ValueError):
            ga4.set_batch_size(invalid_batch)

    with pytest.raises(ValueError):
        Google_GA4("s", "t", request_delay=-0.1)


def test_paginated_report_batches_five_days_per_call(ga4):
    """7 days with batch_size=5 makes 2 HTTP calls (5 requests + 2 requests)."""
    query = {
        "requests": [{
            "dimensions": [{"name": "date"}, {"name": "city"}],
            "metrics": [{"name": "sessions"}],
            "dateRanges": [{"startDate": "2026-11-01", "endDate": "2026-11-07"}],
        }]
    }

    calls = []

    def fake_raw(property_id, q):
        calls.append(q)
        days = [req["dateRanges"][0]["startDate"] for req in q["requests"]]
        return _make_ga4_report([(d, [("Santiago", "10")]) for d in days])

    ga4._get_report_raw = MagicMock(side_effect=fake_raw)

    result = ga4.get_report_df("properties/123", query)

    assert len(calls) == 2
    assert len(calls[0]["requests"]) == 5
    assert len(calls[1]["requests"]) == 2
    assert [r["dateRanges"][0]["startDate"] for r in calls[0]["requests"]] == [
        "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"
    ]
    assert [r["dateRanges"][0]["startDate"] for r in calls[1]["requests"]] == [
        "2026-11-06", "2026-11-07"
    ]
    assert len(result) == 7


def test_batch_size_5_vs_1_exact_dataframe_parity(ga4):
    """batch_size=5 produces the exact same DataFrame as batch_size=1, including empty days."""
    import pandas as pd

    data_by_day = {
        "2026-11-01": [("Santiago", "10"), ("Valparaiso", "5")],
        "2026-11-02": [],
        "2026-11-03": [("Concepcion", "7")],
        "2026-11-04": [],
        "2026-11-05": [("Temuco", "3")],
        "2026-11-06": [("Antofagasta", "12")],
    }
    query = {
        "requests": [{
            "dimensions": [{"name": "date"}, {"name": "city"}],
            "metrics": [{"name": "sessions"}],
            "dateRanges": [{"startDate": "2026-11-01", "endDate": "2026-11-06"}],
        }]
    }

    def fake_raw(property_id, q):
        days = [req["dateRanges"][0]["startDate"] for req in q["requests"]]
        return _make_ga4_report([(d, data_by_day[d]) for d in days])

    ga4._get_report_raw = MagicMock(side_effect=fake_raw)

    ga4.set_batch_size(1)
    df_single = ga4.get_report_df("properties/123", query)

    ga4.set_batch_size(5)
    df_batched = ga4.get_report_df("properties/123", query)

    pd.testing.assert_frame_equal(df_batched, df_single)


def test_batch_overflow_pagination_only_for_large_day(ga4):
    """When one day in a batch hits LIMIT_PER_REQUEST, only that day paginates by offset."""
    ga4.LIMIT_PER_REQUEST = 2
    query = {
        "requests": [{
            "dimensions": [{"name": "date"}, {"name": "city"}],
            "metrics": [{"name": "sessions"}],
            "dateRanges": [{"startDate": "2026-11-01", "endDate": "2026-11-03"}],
        }]
    }

    calls = []

    def fake_raw(property_id, q):
        calls.append([(r["dateRanges"][0]["startDate"], r.get("offset", 0)) for r in q["requests"]])
        reports = []
        for r in q["requests"]:
            d = r["dateRanges"][0]["startDate"]
            off = r.get("offset", 0)
            if d == "2026-11-02":
                if off == 0:
                    reports.append((d, [("A", "1"), ("B", "2")]))
                elif off == 2:
                    reports.append((d, [("C", "3")]))
                else:
                    reports.append((d, []))
            else:
                reports.append((d, [("X", "9")]))
        return _make_ga4_report(reports)

    ga4._get_report_raw = MagicMock(side_effect=fake_raw)

    result = ga4.get_report_df("properties/123", query)

    assert calls == [
        [("2026-11-01", 0), ("2026-11-02", 0), ("2026-11-03", 0)],
        [("2026-11-02", 2)],
    ]
    assert list(zip(result["date"], result["city"], result["sessions"])) == [
        ("2026-11-01", "X", "9"),
        ("2026-11-02", "A", "1"),
        ("2026-11-02", "B", "2"),
        ("2026-11-02", "C", "3"),
        ("2026-11-03", "X", "9"),
    ]


def test_batch_mode_with_extract_sampling(ga4):
    """Each day in a batch receives its own sampling metadata columns."""
    query = {
        "requests": [{
            "dimensions": [{"name": "date"}, {"name": "city"}],
            "metrics": [{"name": "sessions"}],
            "dateRanges": [{"startDate": "2026-11-01", "endDate": "2026-11-02"}],
        }]
    }
    sampling = {
        0: {},
        1: {"samplingMetadatas": [{"samplesReadCount": "250", "samplingSpaceSize": "1000"}]},
    }
    ga4._get_report_raw = MagicMock(
        return_value=_make_ga4_report(
            [("2026-11-01", [("Santiago", "10")]), ("2026-11-02", [("Lima", "20")])],
            sampling=sampling,
        )
    )

    result = ga4.get_report_df("properties/123", query, extract_sampling=True)

    assert list(result["sampled"]) == [False, True]
    assert list(result["sampling_percentage"]) == [100.0, 25.0]


def test_request_delay_sleeps_between_batches_only(ga4, mocker):
    """request_delay sleeps n_batches - 1 times and not after the last batch."""
    ga4.request_delay = 0.5
    query = {
        "requests": [{
            "dimensions": [{"name": "date"}, {"name": "city"}],
            "metrics": [{"name": "sessions"}],
            "dateRanges": [{"startDate": "2026-11-01", "endDate": "2026-11-11"}],
        }]
    }
    ga4._get_report_raw = MagicMock(
        side_effect=lambda _, q: _make_ga4_report(
            [(r["dateRanges"][0]["startDate"], [("Santiago", "1")]) for r in q["requests"]]
        )
    )
    mock_sleep = mocker.patch("time.sleep")

    result = ga4.get_report_df("properties/123", query)

    assert len(result) == 11
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(0.5)