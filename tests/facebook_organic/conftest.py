import pytest

from d2b_data.facebook_organic import FacebookOrganic


@pytest.fixture
def fb():
    return FacebookOrganic(access_token="fake_token")


@pytest.fixture
def metrics():
    return ["post_media_view", "post_clicks"]
