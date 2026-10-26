import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def fb(mocker):
    mocker.patch("facebook_business.api.FacebookAdsApi.init")
    from d2b_data.Facebook_Marketing import Facebook_Marketing
    return Facebook_Marketing(
        app_id="fake_app_id",
        app_secret="fake_secret",
        access_token="fake_token",
        id_account="123456789",
    )


@pytest.fixture
def base_params():
    return {
        "level": "campaign",
        "fields": ["impressions", "clicks", "spend"],
        "breakdowns": [],
        "time_range": {"since": "2024-01-01", "until": "2024-01-31"},
    }
