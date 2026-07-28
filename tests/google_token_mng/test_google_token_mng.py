import json
from unittest.mock import MagicMock

import pytest

from d2b_data.Google_Token_MNG import Google_Token_MNG


def test_instance_stores_configuration(make_token_mng):
    """The constructor stores the provided configuration and builds a service."""
    tm = make_token_mng()
    assert tm.api_name == "searchconsole"
    assert tm.version == "v1"
    assert tm.client_secret == "fake_secret.json"
    assert tm.token == "fake_token.json"
    assert tm.use_sa is False
    assert tm.scopes == ["https://www.googleapis.com/auth/webmasters.readonly"]


def test_get_service_returns_created_service(make_token_mng):
    """get_service returns the object produced by create_api."""
    tm = make_token_mng()
    assert tm.get_service() is tm.service


def test_save_and_open_json_roundtrip(make_token_mng, tmp_path):
    """save_json writes content that open_json can read back."""
    tm = make_token_mng()
    target = tmp_path / "token.json"

    # A JSON object serialized as a string: open_json should return it verbatim.
    content = json.dumps({"access_token": "abc", "refresh_token": "xyz"})
    tm.save_json(str(target), content)

    assert target.exists()
    assert tm.open_json(str(target)) == content


def test_open_json_returns_inner_string_when_double_encoded(make_token_mng, tmp_path):
    """open_json unwraps a double-encoded JSON string."""
    tm = make_token_mng()
    target = tmp_path / "token.json"

    inner = json.dumps({"access_token": "abc"})
    target.write_text(json.dumps(inner), encoding="utf-8")

    assert tm.open_json(str(target)) == inner


def test_open_json_returns_raw_on_non_json(make_token_mng, tmp_path):
    """open_json returns raw content when the file is not valid JSON."""
    tm = make_token_mng()
    target = tmp_path / "token.txt"
    target.write_text("not-json-content", encoding="utf-8")

    assert tm.open_json(str(target)) == "not-json-content"


def test_create_api_oauth_with_existing_token(mocker, tmp_path):
    """
    OAuth path: when a token file exists, credentials are loaded from it and a
    service is built with the authorized http.
    """
    token_file = tmp_path / "token.json"
    token_file.write_text('{"access_token": "abc"}', encoding="utf-8")

    fake_creds = MagicMock()
    fake_creds.authorize.return_value = "http-auth"
    mocker.patch(
        "d2b_data.Google_Token_MNG.client.Credentials.new_from_json",
        return_value=fake_creds,
    )
    mock_build = mocker.patch(
        "d2b_data.Google_Token_MNG.build", return_value="service"
    )

    tm = Google_Token_MNG(
        client_secret="secret.json",
        token=str(token_file),
        api_name="searchconsole",
        api_version="v1",
        scopes=["scope"],
        use_service_account=False,
    )

    assert tm.get_service() == "service"
    fake_creds.authorize.assert_called_once()
    _, kwargs = mock_build.call_args
    assert kwargs["http"] == "http-auth"
    assert kwargs["cache_discovery"] is False


def test_create_api_oauth_requires_scopes(mocker, tmp_path):
    """OAuth path raises when a token is given without scopes."""
    token_file = tmp_path / "token.json"
    token_file.write_text('{"access_token": "abc"}', encoding="utf-8")

    with pytest.raises(ValueError, match="scopes are required"):
        Google_Token_MNG(
            client_secret="secret.json",
            token=str(token_file),
            api_name="searchconsole",
            api_version="v1",
            scopes=None,
            use_service_account=False,
        )


def test_create_api_service_account_from_file(mocker, tmp_path):
    """Service Account path loads credentials from the key file."""
    sa_file = tmp_path / "sa.json"
    sa_file.write_text("{}", encoding="utf-8")

    fake_creds = MagicMock()
    mock_from_file = mocker.patch(
        "d2b_data.Google_Token_MNG.service_account.Credentials."
        "from_service_account_file",
        return_value=fake_creds,
    )
    mock_build = mocker.patch(
        "d2b_data.Google_Token_MNG.build", return_value="sa-service"
    )

    tm = Google_Token_MNG(
        client_secret=str(sa_file),
        token=None,
        api_name="searchconsole",
        api_version="v1",
        scopes=["scope"],
        use_service_account=True,
    )

    assert tm.get_service() == "sa-service"
    mock_from_file.assert_called_once()
    _, kwargs = mock_build.call_args
    assert kwargs["credentials"] is fake_creds


def test_create_api_service_account_falls_back_to_adc(mocker):
    """Service Account path falls back to ADC when no key file is provided."""
    fake_creds = MagicMock()
    mock_default = mocker.patch(
        "d2b_data.Google_Token_MNG.google.auth.default",
        return_value=(fake_creds, "my-project"),
    )
    mock_build = mocker.patch(
        "d2b_data.Google_Token_MNG.build", return_value="adc-service"
    )

    tm = Google_Token_MNG(
        client_secret=None,
        token=None,
        api_name="searchconsole",
        api_version="v1",
        scopes=["scope"],
        use_service_account=True,
    )

    assert tm.get_service() == "adc-service"
    mock_default.assert_called_once()
    _, kwargs = mock_build.call_args
    assert kwargs["credentials"] is fake_creds


def test_create_api_public_no_auth(mocker):
    """When no token and no service account, a public (unauthenticated) service is built."""
    mock_build = mocker.patch(
        "d2b_data.Google_Token_MNG.build", return_value="public-service"
    )

    tm = Google_Token_MNG(
        client_secret=None,
        token=None,
        api_name="searchconsole",
        api_version="v1",
        scopes=None,
        use_service_account=False,
    )

    assert tm.get_service() == "public-service"
    args, kwargs = mock_build.call_args
    assert args == ("searchconsole", "v1")
    assert kwargs["cache_discovery"] is False
    assert "http" not in kwargs
    assert "credentials" not in kwargs


def test_get_credentials_loads_existing_token(mocker, tmp_path):
    """get_credentials loads and returns credentials from an existing token file."""
    mocker.patch(
        "d2b_data.Google_Token_MNG.Google_Token_MNG.create_api",
        return_value=MagicMock(),
    )
    tm = Google_Token_MNG(
        client_secret="secret.json",
        token="token.json",
        api_name="searchconsole",
        api_version="v1",
        scopes=["scope"],
    )

    token_file = tmp_path / "token.json"
    token_file.write_text('{"access_token": "abc"}', encoding="utf-8")

    fake_creds = MagicMock()
    mock_new = mocker.patch(
        "d2b_data.Google_Token_MNG.client.Credentials.new_from_json",
        return_value=fake_creds,
    )

    result = tm.get_credentials(
        secrets="secret.json",
        credentials=str(token_file),
        scopes=["scope"],
    )

    assert result is fake_creds
    mock_new.assert_called_once()


def test_get_credentials_requires_secret_when_no_token(mocker, tmp_path):
    """get_credentials raises when the token is missing and no secret is provided."""
    mocker.patch(
        "d2b_data.Google_Token_MNG.Google_Token_MNG.create_api",
        return_value=MagicMock(),
    )
    tm = Google_Token_MNG(
        client_secret=None,
        token="missing.json",
        api_name="searchconsole",
        api_version="v1",
        scopes=["scope"],
    )

    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="client secret file is required"):
        tm.get_credentials(
            secrets=None,
            credentials=str(missing),
            scopes=["scope"],
        )
