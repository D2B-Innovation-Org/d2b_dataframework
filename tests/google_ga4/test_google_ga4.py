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