import pytest

TOKEN = {
    "access_token": "fake_access",
    "refresh_token": "fake_refresh",
    "open_id": "-000fakeOpenId",
    "expires_in": 86400,
    "refresh_token_expires_in": 31536000,
    "scope": "user.info.basic,video.list",
}


@pytest.fixture
def token_file(tmp_path):
    """Path to a valid TikTok account holder token file."""
    import json

    path = tmp_path / "token.json"
    path.write_text(json.dumps(TOKEN))
    return path


@pytest.fixture
def tiktok(token_file):
    """TikTokOrganic with a valid token loaded from file."""
    from d2b_data.tiktok_organic import TikTokOrganic

    return TikTokOrganic(token_path=str(token_file))
