from unittest.mock import MagicMock

import pytest


@pytest.fixture
def token_mng(mocker):
    """Patches the token manager so no real Google auth happens."""
    manager = MagicMock()
    manager.get_service.return_value = MagicMock()
    return mocker.patch(
        "d2b_data.Google_Spreadsheet.Google_Token_MNG",
        return_value=manager,
    )


@pytest.fixture
def gs(token_mng):
    from d2b_data.Google_Spreadsheet import Google_Spreadsheet

    return Google_Spreadsheet(credentials_path="fake_token.json", url_id="abc123")


@pytest.fixture
def values(gs):
    """Shortcut to the mocked spreadsheets().values() chain."""
    return gs.service.spreadsheets.return_value.values.return_value
