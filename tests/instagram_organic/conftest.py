import pytest

from d2b_data.instagram_organic import InstagramOrganic


@pytest.fixture
def ig():
    return InstagramOrganic(access_token="fake_token")


@pytest.fixture
def feed_metrics():
    return ["views", "reach", "likes"]


@pytest.fixture
def story_metrics():
    return ["views", "reach"]
