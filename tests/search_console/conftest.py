import pytest
from unittest.mock import MagicMock


@pytest.fixture
def gsc(mocker):
    """GoogleSearchConsole instance with a mocked Google_Token_MNG service."""
    mock_token_mng = MagicMock()
    mock_token_mng.get_service.return_value = MagicMock()

    mocker.patch(
        "d2b_data.search_console.d2b_data.Google_Token_MNG.Google_Token_MNG",
        return_value=mock_token_mng,
    )

    from d2b_data.search_console import GoogleSearchConsole

    return GoogleSearchConsole(
        client_secret="fake_secret.json",
        token_json="fake_token.json",
    )


@pytest.fixture
def raw_response():
    """A raw Search Console API response with two rows."""
    return {
        "rows": [
            {
                "keys": ["https://example.com/a", "2024-01-01"],
                "clicks": 10,
                "impressions": 100,
                "ctr": 0.1,
                "position": 1.5,
            },
            {
                "keys": ["https://example.com/b", "2024-01-01"],
                "clicks": 5,
                "impressions": 50,
                "ctr": 0.1,
                "position": 2.0,
            },
        ]
    }


@pytest.fixture
def empty_response():
    """A raw Search Console API response with no rows."""
    return {}
