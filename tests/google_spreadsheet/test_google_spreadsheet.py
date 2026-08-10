import pandas as pd
import pytest

from d2b_data.Google_Spreadsheet import Google_Spreadsheet


SHEET_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_instance_is_created_correctly(gs):
    """The client keeps its credentials path and sheet id."""
    assert gs.credentials_path == "fake_token.json"
    assert gs.url_id == "abc123"
    assert gs.service is gs.token_manager.get_service.return_value


def test_oauth_mode_passes_path_as_secret_and_token(token_mng):
    """In OAuth mode the same file acts as client secret and token."""
    Google_Spreadsheet(credentials_path="creds.json")

    kwargs = token_mng.call_args.kwargs
    assert kwargs["client_secret"] == "creds.json"
    assert kwargs["token"] == "creds.json"
    assert kwargs["use_service_account"] is False


def test_service_account_mode_sends_no_token(token_mng):
    """Service accounts do not use an OAuth token file."""
    Google_Spreadsheet(credentials_path="sa.json", use_service_account=True)

    kwargs = token_mng.call_args.kwargs
    assert kwargs["client_secret"] == "sa.json"
    assert kwargs["token"] is None
    assert kwargs["use_service_account"] is True


def test_sheets_api_scope_and_version(token_mng):
    """The client requests the Sheets v4 API with the spreadsheets scope."""
    Google_Spreadsheet(credentials_path="creds.json")

    kwargs = token_mng.call_args.kwargs
    assert kwargs["scopes"] == ["https://www.googleapis.com/auth/spreadsheets"]
    assert kwargs["api_name"] == "sheets"
    assert kwargs["api_version"] == "v4"


def test_get_spreadsheet_returns_the_resource(gs):
    """get_spreadsheet exposes the raw spreadsheets() resource."""
    assert gs.get_spreadsheet() is gs.service.spreadsheets.return_value


# --------------------------------------------------------------------- #
# read_data_dataframe
# --------------------------------------------------------------------- #
def test_read_data_uses_first_row_as_header(gs, values):
    """The first returned row becomes the DataFrame header."""
    values.get.return_value.execute.return_value = {
        "values": [
            ["fecha", "clicks", "impresiones"],
            ["2024-01-01", "10", "100"],
            ["2024-01-02", "20", "200"],
        ]
    }
    df = gs.read_data_dataframe(SHEET_ID, "Hoja1!A1:C3")

    assert list(df.columns) == ["fecha", "clicks", "impresiones"]
    assert len(df) == 2
    assert df.iloc[0]["clicks"] == "10"


def test_read_data_passes_id_and_range(gs, values):
    """The spreadsheet id and range reach the API call."""
    values.get.return_value.execute.return_value = {"values": [["a"], ["1"]]}
    gs.read_data_dataframe(SHEET_ID, "Hoja1!A:A")

    values.get.assert_called_once_with(spreadsheetId=SHEET_ID, range="Hoja1!A:A")


def test_read_data_returns_empty_dataframe_when_sheet_is_empty(gs, values):
    """A response with no 'values' key yields an empty DataFrame."""
    values.get.return_value.execute.return_value = {}
    df = gs.read_data_dataframe(SHEET_ID, "Hoja1!A1:C3")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_read_data_returns_empty_dataframe_on_api_error(gs, values, capsys):
    """An API failure is reported and degraded to an empty DataFrame."""
    values.get.return_value.execute.side_effect = RuntimeError("403 forbidden")
    df = gs.read_data_dataframe(SHEET_ID, "Hoja1!A1:C3")

    assert df.empty
    assert "Error leyendo data" in capsys.readouterr().out


# --------------------------------------------------------------------- #
# delete_data
# --------------------------------------------------------------------- #
def body_of(gs):
    batch = gs.service.spreadsheets.return_value.batchUpdate
    return batch.call_args.kwargs["body"]["requests"][0]["updateCells"]


def test_delete_data_all_clears_the_whole_sheet(gs):
    """Without a vector the range is the entire sheet."""
    assert gs.delete_data(sheetid=0, spreadsheetId=SHEET_ID) is True
    assert body_of(gs)["range"] == {"sheetId": 0}


def test_delete_data_rows_sets_row_bounds(gs):
    """vector='ROWS' translates the indexes to row bounds."""
    gs.delete_data(sheetid=0, spreadsheetId=SHEET_ID, vector="ROWS", start_index=1, end_index=10)
    assert body_of(gs)["range"] == {"sheetId": 0, "startRowIndex": 1, "endRowIndex": 10}


def test_delete_data_columns_sets_column_bounds(gs):
    """vector='COLUMNS' translates the indexes to column bounds."""
    gs.delete_data(sheetid=2, spreadsheetId=SHEET_ID, vector="COLUMNS", start_index=1, end_index=5)
    assert body_of(gs)["range"] == {"sheetId": 2, "startColumnIndex": 1, "endColumnIndex": 5}


def test_delete_data_vector_is_case_insensitive(gs):
    """Lowercase vector names work the same."""
    gs.delete_data(sheetid=0, spreadsheetId=SHEET_ID, vector="rows", start_index=0, end_index=3)
    assert "startRowIndex" in body_of(gs)["range"]


def test_delete_data_omits_missing_bounds(gs):
    """A vector without indexes leaves the range open-ended."""
    gs.delete_data(sheetid=0, spreadsheetId=SHEET_ID, vector="ROWS")
    assert body_of(gs)["range"] == {"sheetId": 0}


@pytest.mark.parametrize("mode,expected", [
    ("VALUES", "userEnteredValue"),
    ("FORMAT", "userEnteredFormat"),
    ("ALL", "*"),
])
def test_delete_data_mode_selects_target_fields(gs, mode, expected):
    """The mode decides which cell fields get wiped."""
    gs.delete_data(sheetid=0, spreadsheetId=SHEET_ID, mode=mode)
    assert body_of(gs)["fields"] == expected


def test_delete_data_executes_the_batch_update(gs):
    """The request is actually sent, not just built."""
    gs.delete_data(sheetid=0, spreadsheetId=SHEET_ID)
    gs.service.spreadsheets.return_value.batchUpdate.return_value.execute.assert_called_once()


# --------------------------------------------------------------------- #
# update_data / append_data
# --------------------------------------------------------------------- #
def test_update_data_sends_values_with_user_entered_option(gs, values):
    """update_data writes the rows using USER_ENTERED parsing."""
    rows = [["fecha", "clicks"], ["2024-01-01", 10]]
    assert gs.update_data(SHEET_ID, "Hoja1!A1", rows) is True

    kwargs = values.update.call_args.kwargs
    assert kwargs["spreadsheetId"] == SHEET_ID
    assert kwargs["range"] == "Hoja1!A1"
    assert kwargs["valueInputOption"] == "USER_ENTERED"
    assert kwargs["body"] == {"values": rows}
    values.update.return_value.execute.assert_called_once()


def test_append_data_sends_values(gs, values):
    """append_data adds rows at the end of the range."""
    rows = [["2024-01-02", 20]]
    assert gs.append_data(SHEET_ID, "Hoja1!A1", rows) is True

    kwargs = values.append.call_args.kwargs
    assert kwargs["body"] == {"values": rows}
    assert kwargs["valueInputOption"] == "USER_ENTERED"
    values.append.return_value.execute.assert_called_once()


def test_append_data_reports_the_row_count(gs, values, capsys):
    """The number of appended rows is printed for traceability."""
    gs.append_data(SHEET_ID, "Hoja1!A1", [["a"], ["b"], ["c"]])
    assert "Agregando 3 filas" in capsys.readouterr().out
