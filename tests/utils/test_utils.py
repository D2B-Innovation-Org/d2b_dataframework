import json
import os

import pandas as pd
import pytest

from d2b_data.utils import extract_and_write_temp_credentials, load_schema_from_csv


WORKFLOW = "test-workflow"


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """load_schema_from_csv reads './schema.csv', so the cwd matters."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_schema(path, rows):
    pd.DataFrame(rows).to_csv(path / "schema.csv", index=False)


# --------------------------------------------------------------------- #
# load_schema_from_csv
# --------------------------------------------------------------------- #
def test_load_schema_returns_none_when_file_is_absent(in_tmp_cwd, logger):
    """Without schema.csv the caller falls back to BigQuery autodetection."""
    assert load_schema_from_csv(logger, WORKFLOW) is None
    assert logger.has_log("no encontrado")


def test_load_schema_reads_english_columns(in_tmp_cwd, logger):
    """A schema.csv with name/type/description is parsed as-is."""
    write_schema(in_tmp_cwd, [
        {"name": "clicks", "type": "integer", "description": "total clicks"},
        {"name": "date", "type": "date", "description": "event date"},
    ])
    schema = load_schema_from_csv(logger, WORKFLOW)
    assert schema == [
        {"name": "clicks", "type": "INTEGER", "description": "total clicks"},
        {"name": "date", "type": "DATE", "description": "event date"},
    ]


def test_load_schema_renames_spanish_columns(in_tmp_cwd, logger):
    """'nombre', 'tipo' and 'descripcion' are mapped to the canonical names."""
    write_schema(in_tmp_cwd, [
        {"nombre": "clicks", "tipo": "integer", "descripcion": "clics totales"},
    ])
    schema = load_schema_from_csv(logger, WORKFLOW)
    assert schema == [{"name": "clicks", "type": "INTEGER", "description": "clics totales"}]


def test_load_schema_uppercases_types(in_tmp_cwd, logger):
    """BigQuery expects uppercase types."""
    write_schema(in_tmp_cwd, [{"name": "x", "type": "string", "description": ""}])
    assert load_schema_from_csv(logger, WORKFLOW)[0]["type"] == "STRING"


def test_load_schema_strips_accents_from_description(in_tmp_cwd, logger):
    """Descriptions go through unidecode."""
    write_schema(in_tmp_cwd, [
        {"name": "sesion", "type": "string", "description": "Métrica de sesión ñandú"},
    ])
    assert load_schema_from_csv(logger, WORKFLOW)[0]["description"] == "Metrica de sesion nandu"


def test_load_schema_defaults_missing_description(in_tmp_cwd, logger):
    """A schema without 'description' gets an empty one and a log line."""
    write_schema(in_tmp_cwd, [{"name": "clicks", "type": "integer"}])
    schema = load_schema_from_csv(logger, WORKFLOW)
    assert schema[0]["description"] == ""
    assert logger.has_log("no encontrada")


def test_load_schema_fails_without_name_column(in_tmp_cwd, logger):
    """A schema missing 'name' is rejected and reported as critical."""
    write_schema(in_tmp_cwd, [{"campo": "clicks", "type": "integer"}])
    assert load_schema_from_csv(logger, WORKFLOW) is None
    assert logger.has_critical("Error procesando schema.csv")
    assert logger.criticals[0][1] == WORKFLOW


def test_load_schema_fails_without_type_column(in_tmp_cwd, logger):
    """A schema missing 'type' is rejected too."""
    write_schema(in_tmp_cwd, [{"name": "clicks", "unidad": "integer"}])
    assert load_schema_from_csv(logger, WORKFLOW) is None
    assert logger.has_critical("Error procesando")


def test_load_schema_reports_unreadable_file(in_tmp_cwd, logger):
    """A corrupt CSV is reported as critical rather than raising."""
    (in_tmp_cwd / "schema.csv").write_text('name,type\n"unterminated', encoding="utf-8")
    assert load_schema_from_csv(logger, WORKFLOW) is None
    assert logger.has_critical("Error procesando")


# --------------------------------------------------------------------- #
# extract_and_write_temp_credentials
# --------------------------------------------------------------------- #
def test_extract_credentials_writes_temp_file(credentials_csv, valid_credentials, logger):
    """A matching client yields a temp file holding its credentials JSON."""
    path = credentials_csv([["Cliente Uno", "d2b-cliente-uno", valid_credentials]])
    project_map = {"Cliente Uno": "d2b-cliente-uno"}

    temp_path, name = extract_and_write_temp_credentials(
        "Cliente Uno", path, logger, WORKFLOW, project_map,
    )
    try:
        assert name == "Cliente Uno"
        assert temp_path.endswith(".json")
        with open(temp_path) as handle:
            assert json.load(handle)["project_id"] == "d2b-cliente-uno"
    finally:
        os.unlink(temp_path)


def test_extract_credentials_match_is_case_insensitive(credentials_csv, valid_credentials, logger):
    """Client lookup ignores case and surrounding whitespace."""
    path = credentials_csv([["  CLIENTE UNO  ", "d2b-cliente-uno", valid_credentials]])
    temp_path, name = extract_and_write_temp_credentials(
        "cliente uno", path, logger, WORKFLOW, {},
    )
    try:
        assert name == "CLIENTE UNO"
    finally:
        os.unlink(temp_path)


def test_extract_credentials_sanitizes_temp_file_prefix(credentials_csv, valid_credentials, logger):
    """Accents, spaces and dashes are removed from the temp file name."""
    path = credentials_csv([["Ñuñoa Marketing - SpA", "d2b-x", valid_credentials]])
    temp_path, _ = extract_and_write_temp_credentials(
        "Ñuñoa Marketing - SpA", path, logger, WORKFLOW, {},
    )
    try:
        basename = os.path.basename(temp_path)
        assert basename.startswith("temp_creds_Nunoa_Marketing___SpA_")
    finally:
        os.unlink(temp_path)


def test_extract_credentials_warns_when_client_not_in_project_map(credentials_csv, valid_credentials, logger):
    """A client missing from the project map logs a warning but still works."""
    path = credentials_csv([["Cliente Uno", "d2b-cliente-uno", valid_credentials]])
    temp_path, _ = extract_and_write_temp_credentials(
        "Cliente Uno", path, logger, WORKFLOW, {},
    )
    try:
        assert logger.has_log("No se encontró mapeo de project_id")
    finally:
        os.unlink(temp_path)


def test_extract_credentials_rejects_project_id_mismatch(credentials_csv, valid_credentials, logger):
    """A project_id that disagrees with the map is a critical failure."""
    path = credentials_csv([["Cliente Uno", "otro", valid_credentials]])
    project_map = {"Cliente Uno": "d2b-otro-proyecto"}

    result = extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, project_map)
    assert result == (None, None)
    assert logger.has_critical("DISCREPANCIA DE PROJECT_ID")


def test_extract_credentials_returns_none_when_client_absent(credentials_csv, valid_credentials, logger):
    """A client that is not in the CSV returns (None, None)."""
    path = credentials_csv([["Otro Cliente", "d2b-otro", valid_credentials]])
    assert extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, {}) == (None, None)
    assert logger.has_log("No se encontró 'Cliente Uno'")


def test_extract_credentials_rejects_wrong_headers(credentials_csv, valid_credentials, logger):
    """The CSV must have exactly the expected headers."""
    path = credentials_csv(
        [["Cliente Uno", "d2b-cliente-uno", valid_credentials]],
        headers=("cliente", "proyecto", "credenciales"),
    )
    assert extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, {}) == (None, None)
    assert logger.has_critical("Encabezados CSV incorrectos")


def test_extract_credentials_rejects_empty_csv(tmp_path, logger):
    """An empty file has no header row to read."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert extract_and_write_temp_credentials("X", str(path), logger, WORKFLOW, {}) == (None, None)
    assert logger.has_critical("CSV malformado")


def test_extract_credentials_skips_incomplete_rows(credentials_csv, valid_credentials, logger):
    """Short or empty rows are skipped, not fatal."""
    path = credentials_csv([
        ["Solo Nombre"],
        [],
        ["Cliente Uno", "d2b-cliente-uno", valid_credentials],
    ])
    temp_path, name = extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, {})
    try:
        assert name == "Cliente Uno"
        assert logger.has_log("vacía o incompleta")
    finally:
        os.unlink(temp_path)


def test_extract_credentials_reports_invalid_json(credentials_csv, logger):
    """A malformed JSON block is reported as critical."""
    path = credentials_csv([["Cliente Uno", "d2b-cliente-uno", "{not valid json"]])
    assert extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, {}) == (None, None)
    assert logger.has_critical("Error decodificando JSON")


def test_extract_credentials_reports_temp_file_failure(credentials_csv, valid_credentials, logger, mocker):
    """A failure writing the temp file is reported instead of propagating."""
    path = credentials_csv([["Cliente Uno", "d2b-cliente-uno", valid_credentials]])
    mocker.patch("d2b_data.utils.tempfile.NamedTemporaryFile", side_effect=OSError("disk full"))

    assert extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, {}) == (None, None)
    assert logger.has_critical("Error escribiendo archivo temporal")


def test_extract_credentials_missing_file(tmp_path, logger):
    """A missing CSV is reported as critical."""
    missing = str(tmp_path / "nope.csv")
    assert extract_and_write_temp_credentials("X", missing, logger, WORKFLOW, {}) == (None, None)
    assert logger.has_critical("no encontrado")


def test_extract_credentials_reports_unexpected_error(credentials_csv, valid_credentials, logger, mocker):
    """Any other failure reading the CSV is caught and reported."""
    path = credentials_csv([["Cliente Uno", "d2b-cliente-uno", valid_credentials]])
    mocker.patch("d2b_data.utils.csv.reader", side_effect=RuntimeError("boom"))

    assert extract_and_write_temp_credentials("Cliente Uno", path, logger, WORKFLOW, {}) == (None, None)
    assert logger.has_critical("Error general leyendo CSV")
