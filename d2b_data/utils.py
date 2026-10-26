import csv
import json
import os
import tempfile

import pandas as pd
from unidecode import unidecode

from d2b_data.workflow_logger import resolve_logger



def load_schema_from_csv(logger=None, wf_name_func=None, *, verbose_logger_func=None):
    """
    Carga un schema desde un archivo CSV local llamado `schema.csv`.
    El archivo debe tener al menos las columnas: 'name' (o 'nombre') y 'type' (o 'tipo').
    Opcionalmente puede tener una columna 'description'.

    Args:
        logger: Logger con métodos `.log()` y `.critical()` (WorkflowLogger o Verbose).
        wf_name_func: Nombre del workflow actual (para trazabilidad en logs críticos).
        verbose_logger_func: Nombre obsoleto de `logger`; se acepta con aviso.

    Returns:
        schema_table_local: lista de dicts con keys `name`, `type`, `description`, o None si no hay archivo o hay error.
    """
    logger = resolve_logger(logger, verbose_logger_func, "load_schema_from_csv")
    if logger is None:
        raise TypeError("load_schema_from_csv requiere el argumento 'logger'")
    schema_table_local = None
    schema_csv_path = "schema.csv"

    if os.path.isfile(schema_csv_path):
        logger.log(f"load_schema_from_csv | Cargando schema desde {schema_csv_path}")
        try:
            schema_df = pd.read_csv(schema_csv_path)
            rename_map = {
                c: "name" if "nombre" in c.lower()
                else "type" if "tipo" in c.lower()
                else "description" if "descrip" in c.lower()
                else c
                for c in schema_df.columns
            }
            schema_df.rename(columns=rename_map, inplace=True)

            if "name" not in schema_df.columns or "type" not in schema_df.columns:
                raise ValueError("schema.csv debe tener columnas 'name' (o 'nombre') y 'type' (o 'tipo').")

            if "description" not in schema_df.columns:
                logger.log("load_schema_from_csv | Columna 'description' no encontrada. Se usará vacía.")
                schema_df["description"] = ""

            schema_df["type"] = schema_df["type"].str.upper()
            schema_df["description"] = schema_df["description"].fillna("").astype(str).map(unidecode)

            schema_table_local = json.loads(
                schema_df[["name", "type", "description"]].to_json(orient="records", force_ascii=False)
            )

            logger.log("load_schema_from_csv | Schema cargado y procesado exitosamente.")

        except Exception as e_schema:
            msg_schema_err = f"Error procesando {schema_csv_path}: {str(e_schema)}"
            logger.critical(msg_schema_err, current_workflow_name=wf_name_func)
            return None
    else:
        logger.log(f"load_schema_from_csv | {schema_csv_path} no encontrado, se usará autodetección de schema en BQ.")

    return schema_table_local

def extract_and_write_temp_credentials(
    client_name_from_map,
    source_csv_path,
    logger=None,
    current_workflow_name_param=None,
    project_map_dict=None,
    *,
    verbose_logger_param=None,
):
    """
    Extrae las credenciales de un cliente desde un CSV y las escribe en un archivo temporal.
    Args:
        client_name_from_map (str): Nombre del cliente a buscar en el CSV.
        source_csv_path (str): Ruta al archivo CSV que contiene las credenciales.
        logger: Logger con métodos `.log()` y `.critical()` (WorkflowLogger o Verbose).
        current_workflow_name_param (str): Nombre del workflow actual para trazabilidad en logs críticos.
        project_map_dict (dict): Mapeo de nombre de cliente a project_id para validar el JSON.
        verbose_logger_param: Nombre obsoleto de `logger`; se acepta con aviso.
    Returns:
        str: Ruta al archivo temporal con las credenciales JSON, o None si hubo error.
        tuple: (ruta al archivo temporal con las credenciales JSON, nombre del cliente encontrado) o (None, None) si hubo error.
    """
    logger = resolve_logger(
        logger, verbose_logger_param, "extract_and_write_temp_credentials"
    )
    # Siguen siendo obligatorios; los defaults existen solo para aceptar el alias por nombre
    if logger is None or project_map_dict is None:
        raise TypeError(
            "extract_and_write_temp_credentials requiere 'logger' y 'project_map_dict'"
        )
    logger.log(f"extract_and_write_temp_credentials | Buscando credenciales para cliente: {client_name_from_map} en CSV: {source_csv_path}")
    project_id_to_check_internally = project_map_dict.get(client_name_from_map)
    if not project_id_to_check_internally:
        logger.log(f"extract_and_write_temp_credentials | ADVERTENCIA: No se encontró mapeo de project_id para validación interna de '{client_name_from_map}'.")

    try:
        with open(source_csv_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
            try:
                header = next(reader)
                expected_headers = ["Nombre", "project_id", "JSON"]
                if header != expected_headers:
                    logger.critical(f"Encabezados CSV incorrectos en '{source_csv_path}'. Esperados: {expected_headers}, Obtenidos: {header}", current_workflow_name=current_workflow_name_param)
                    return None, None
                name_col_idx, json_col_idx = header.index('Nombre'), header.index('JSON')
            except (StopIteration, ValueError, IndexError) as e_header:
                logger.critical(f"CSV malformado o encabezados no encontrados en '{source_csv_path}': {str(e_header)}", current_workflow_name=current_workflow_name_param)
                return None, None

            for row_number, row in enumerate(reader, 2):
                if not row or len(row) <= max(name_col_idx, json_col_idx):
                    logger.log(f"extract_and_write_temp_credentials | Fila CSV {row_number} vacía o incompleta, omitiendo.")
                    continue
                csv_client_name = row[name_col_idx].strip()
                if csv_client_name.lower() == client_name_from_map.strip().lower():
                    logger.log(f"extract_and_write_temp_credentials | Coincidencia encontrada en CSV (fila {row_number}) para: {client_name_from_map}")
                    json_block_str_from_csv = row[json_col_idx]
                    try:
                        credentials_dict = json.loads(json_block_str_from_csv)
                        if project_id_to_check_internally and credentials_dict.get("project_id") != project_id_to_check_internally:
                            logger.critical(f"DISCREPANCIA DE PROJECT_ID: JSON ('{credentials_dict.get('project_id')}') vs mapeo ('{project_id_to_check_internally}') para {client_name_from_map} (fila {row_number}).", current_workflow_name=current_workflow_name_param)
                            return None, None

                        temp_file_prefix = f"temp_creds_{unidecode(client_name_from_map).replace(' ', '_').replace('-', '_')}_"
                        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", prefix=temp_file_prefix)
                        temp_file_path = temp_file.name
                        json.dump(credentials_dict, temp_file, indent=2)
                        temp_file.close()
                        logger.log(f"extract_and_write_temp_credentials | Archivo temporal de credenciales creado: {temp_file_path}")
                        return temp_file_path, csv_client_name
                    except json.JSONDecodeError as e_json_decode:
                        logger.critical(f"Error decodificando JSON para {client_name_from_map} (fila {row_number}): {str(e_json_decode)}. JSON (inicio): {json_block_str_from_csv[:300]}", current_workflow_name=current_workflow_name_param)
                        return None, None
                    except Exception as e_write_temp:
                        logger.critical(f"Error escribiendo archivo temporal para {client_name_from_map}: {str(e_write_temp)}", current_workflow_name=current_workflow_name_param)
                        return None, None
            logger.log(f"extract_and_write_temp_credentials | No se encontró '{client_name_from_map}' en CSV '{source_csv_path}'.")
            return None, None
    except FileNotFoundError:
        logger.critical(f"extract_and_write_temp_credentials | Archivo CSV '{source_csv_path}' no encontrado.", current_workflow_name=current_workflow_name_param)
        return None, None
    except Exception as e_general_csv:
        logger.critical(f"extract_and_write_temp_credentials | Error general leyendo CSV '{source_csv_path}': {str(e_general_csv)}", current_workflow_name=current_workflow_name_param)
        return None, None