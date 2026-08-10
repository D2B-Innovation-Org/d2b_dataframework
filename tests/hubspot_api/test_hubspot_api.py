import pandas as pd
import pytest
import requests

from d2b_data.hubspot_api import HubSpot_API


@pytest.fixture
def logger(mocker):
    """Verbose-style logger double with .log() and .error()."""
    return mocker.MagicMock()


@pytest.fixture
def hs(logger):
    return HubSpot_API(token="fake-token", verbose_logger=logger)


def http_error(status_code, text="denied"):
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode()
    return requests.exceptions.HTTPError(response=response)


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_instance_sets_auth_headers(hs):
    """The session carries the bearer token and JSON content type."""
    assert hs.token == "fake-token"
    assert hs.session.headers["Authorization"] == "Bearer fake-token"
    assert hs.session.headers["Content-Type"] == "application/json"


def test_base_url_is_the_hubspot_api():
    """The base URL points at the public HubSpot API."""
    assert HubSpot_API.BASE_URL == "https://api.hubapi.com/"


@pytest.mark.parametrize("bad_token", ["", None])
def test_missing_token_raises(bad_token):
    """An empty token is rejected at construction."""
    with pytest.raises(ValueError, match="Se requiere un token"):
        HubSpot_API(token=bad_token)


# --------------------------------------------------------------------- #
# _log
# --------------------------------------------------------------------- #
def test_log_uses_the_injected_logger(hs, logger):
    """Info messages go through the logger's .log()."""
    hs._log("hola")
    logger.log.assert_called_once_with("hola")


def test_log_error_uses_the_error_channel(hs, logger):
    """Error messages go through the logger's .error()."""
    hs._log("mal", "error")
    logger.error.assert_called_once_with("mal")


def test_log_falls_back_to_print(capsys):
    """Without a logger the message is printed with its level."""
    HubSpot_API(token="t")._log("hola")
    assert "[INFO] hola" in capsys.readouterr().out


# --------------------------------------------------------------------- #
# call_api
# --------------------------------------------------------------------- #
def test_call_api_returns_json(hs, mocker):
    """A successful call returns the decoded body."""
    response = mocker.MagicMock()
    response.json.return_value = {"results": [{"id": "1"}]}
    request = mocker.patch.object(hs.session, "request", return_value=response)

    assert hs.call_api("GET", "crm/v3/objects/contacts") == {"results": [{"id": "1"}]}
    request.assert_called_once_with(
        "GET",
        "https://api.hubapi.com/crm/v3/objects/contacts",
        params=None,
        json=None,
        timeout=30,
    )


def test_call_api_forwards_params_and_body(hs, mocker):
    """Query params and JSON body reach the session."""
    request = mocker.patch.object(hs.session, "request")
    hs.call_api("POST", "crm/v3/objects/deals", params={"limit": 5}, json_data={"a": 1})

    assert request.call_args.kwargs["params"] == {"limit": 5}
    assert request.call_args.kwargs["json"] == {"a": 1}


def test_call_api_returns_empty_dict_on_http_error(hs, logger, mocker):
    """A 4xx/5xx is logged and degraded to an empty dict."""
    response = mocker.MagicMock()
    response.raise_for_status.side_effect = http_error(401, "invalid token")
    mocker.patch.object(hs.session, "request", return_value=response)

    assert hs.call_api("GET", "crm/v3/objects/contacts") == {}
    assert "Error HTTP" in logger.error.call_args.args[0]
    assert "401" in logger.error.call_args.args[0]


def test_call_api_returns_empty_dict_on_connection_error(hs, logger, mocker):
    """A network failure is logged and degraded to an empty dict."""
    mocker.patch.object(hs.session, "request", side_effect=requests.ConnectionError("sin red"))

    assert hs.call_api("GET", "crm/v3/objects/contacts") == {}
    assert "Error de conexión" in logger.error.call_args.args[0]


# --------------------------------------------------------------------- #
# test_connection
# --------------------------------------------------------------------- #
def test_test_connection_true_when_results_present(hs, mocker):
    """A response carrying 'results' means the token works."""
    call = mocker.patch.object(hs, "call_api", return_value={"results": []})
    assert hs.test_connection() is True
    call.assert_called_once_with("GET", "crm/v3/objects/contacts", params={"limit": 1})


def test_test_connection_false_without_results(hs, logger, mocker):
    """An unexpected payload means the connection failed."""
    mocker.patch.object(hs, "call_api", return_value={})
    assert hs.test_connection() is False
    assert "falló" in logger.error.call_args.args[0]


def test_test_connection_false_on_exception(hs, logger, mocker):
    """An unexpected exception is caught and reported as a failure."""
    mocker.patch.object(hs, "call_api", side_effect=RuntimeError("boom"))
    assert hs.test_connection() is False
    assert "Excepción al probar" in logger.error.call_args.args[0]


# --------------------------------------------------------------------- #
# to_dataframe
# --------------------------------------------------------------------- #
def test_to_dataframe_empty_records():
    """No records yields an empty DataFrame, not an error."""
    df = HubSpot_API(token="t").to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_to_dataframe_flattens_properties(hs):
    """The nested 'properties' dict is flattened into columns."""
    records = [
        {"id": "1", "createdAt": "2024-01-01", "properties": {"email": "a@x.cl", "firstname": "Ana"}},
        {"id": "2", "createdAt": "2024-01-02", "properties": {"email": "b@x.cl", "firstname": "Beto"}},
    ]
    df = hs.to_dataframe(records)
    assert list(df.columns) == ["hs_object_id", "createdAt", "email", "firstname"]
    assert df["email"].tolist() == ["a@x.cl", "b@x.cl"]


def test_to_dataframe_renames_id_to_hs_object_id(hs):
    """'id' becomes 'hs_object_id' for consistency with HubSpot exports."""
    df = hs.to_dataframe([{"id": "1", "properties": {"email": "a@x.cl"}}])
    assert "hs_object_id" in df.columns
    assert "id" not in df.columns


def test_to_dataframe_keeps_existing_hs_object_id(hs):
    """When the property already exists, 'id' is left alone."""
    df = hs.to_dataframe([{"id": "1", "properties": {"hs_object_id": "999"}}])
    assert df["id"].tolist() == ["1"]
    assert df["hs_object_id"].tolist() == ["999"]


def test_to_dataframe_drops_nested_top_level_values(hs):
    """Nested dicts/lists outside 'properties' are not flattened in."""
    records = [{"id": "1", "associations": {"deals": []}, "tags": ["a"], "properties": {"email": "a@x.cl"}}]
    df = hs.to_dataframe(records)
    assert "associations" not in df.columns
    assert "tags" not in df.columns


def test_to_dataframe_handles_records_without_properties(hs):
    """A record with no properties still produces its top-level fields."""
    df = hs.to_dataframe([{"id": "1", "archived": False}])
    assert df["archived"].tolist() == [False]
