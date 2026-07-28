import pytest
from unittest.mock import MagicMock


@pytest.fixture
def make_token_mng(mocker):
    """
    Factory that builds a Google_Token_MNG instance while patching the
    heavy authentication path (`create_api`) so no real Google call happens.

    The returned instance exposes `create_api` as a MagicMock, so tests that
    need the real method can restore it with `mocker.patch.object`.
    """

    def _make(**overrides):
        mocker.patch(
            "d2b_data.Google_Token_MNG.Google_Token_MNG.create_api",
            return_value=MagicMock(),
        )

        from d2b_data.Google_Token_MNG import Google_Token_MNG

        params = {
            "client_secret": "fake_secret.json",
            "token": "fake_token.json",
            "api_name": "searchconsole",
            "api_version": "v1",
            "scopes": ["https://www.googleapis.com/auth/webmasters.readonly"],
            "use_service_account": False,
        }
        params.update(overrides)

        return Google_Token_MNG(**params)

    return _make
