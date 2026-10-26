import time
from urllib.parse import urlsplit

import pandas as pd
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.api import FacebookAdsApi
from facebook_business.exceptions import FacebookRequestError


class Facebook_Marketing:
    def __init__(
        self,
        app_id,
        app_secret,
        access_token,
        id_account=None,
        unsampled=False,
        verbose_logger=None,
        api_version="v25.0",
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.unsampled = unsampled
        self.id_account = id_account
        self.verbose = verbose_logger if verbose_logger else self._null_verbose()
        self.verbose.log(
            "--- EXECUTING Facebook_Marketing Class v3.3 - Deployed on 2025-06-25 15:30 ---"
        )
        self.service = FacebookAdsApi.init(
            self.app_id,
            self.app_secret,
            self.access_token,
            api_version=api_version,
        )
        self.service._session.requests.hooks["response"].append(self._log_meta_usage)

    def _log_meta_usage(self, response, *args, **kwargs):
        """
        Registra el uso de cuota que Meta informa en las cabeceras de cada respuesta.

        Se engancha como hook de `requests` en la sesión del SDK, así cubre todas las
        llamadas (lanzar el job, consultar su estado y cada página del resultado).
        Solo se registra la ruta, nunca el query string, porque ahí viajan el token y
        el appsecret_proof.
        """
        try:
            headers = response.headers
            usage = {
                name: headers[header]
                for name, header in (
                    ("insights_throttle", "x-fb-ads-insights-throttle"),
                    ("business_use_case", "x-business-use-case-usage"),
                    ("app_usage", "x-app-usage"),
                )
                if header in headers
            }
            if not usage:
                return
            path = urlsplit(response.url).path
            caller = getattr(self.verbose, "workflow_name", None) or "unknown"
            details = " | ".join(f"{name}={value}" for name, value in usage.items())
            self.verbose.log(
                f"meta_usage | {caller} | {response.request.method} {path} | "
                f"status={response.status_code} | {details}"
            )
        except Exception:
            pass

    def _null_verbose(self):
        class DummyVerbose:
            def log(self, *args, **kwargs):
                pass

            def critical(self, *args, **kwargs):
                pass

        return DummyVerbose()

    def get_report_dataframe(self, params, id_account=None):
        id_account = id_account or self.id_account
        if isinstance(id_account, list):
            return self.def_report_array_accounts(params, id_account)

        df_facebook = pd.DataFrame()
        act_id = f"act_{str(id_account)}"

        if self.unsampled:
            self.verbose.log("get_report_dataframe | Unsampled")
            unsampled_array = []
            date_range = pd.date_range(
                start=params["time_range"]["since"], end=params["time_range"]["until"]
            )

            for idx, date in enumerate(date_range):
                str_date = date.strftime("%Y-%m-%d")
                unsampled_array.append(df_day)

            df_facebook = pd.concat(unsampled_array, ignore_index=True)

        else:  # caso unsampled=False
            report = self.get_report(params, act_id)
            if not isinstance(report, list) or not all(
                isinstance(r, dict) for r in report
            ):
                self.verbose.critical(
                    f"[{act_id}] Facebook devolvió un objeto inválido: {type(report)} - contenido: {report}"
                )
                raise ValueError("Bad data to set object data")

            if len(report) == 0:
                default_cols = (
                    params.get("fields", [])
                    + params.get("breakdowns", [])
                    + ["date_start", "date_stop", "account_id"]
                )
                df_facebook = pd.DataFrame(columns=default_cols)
            else:
                try:
                    self.verbose.log(
                        "Intentando crear el DataFrame desde el reporte crudo..."
                    )
                    df_facebook = pd.DataFrame(report, index=None)
                    self.verbose.log("DataFrame creado exitosamente desde el reporte.")
                except Exception:
                    self.verbose.critical(
                        "¡FALLO CRÍTICO EN LA CREACIÓN DEL DATAFRAME! Guardando datos crudos en GCS para análisis."
                    )

        if not df_facebook.empty:
            df_facebook.reset_index(drop=True, inplace=True)

        self.verbose.log("Iniciando procesamiento de acciones (versión robusta)...")
        actions_dict = self._unique_actions(df_facebook)

        for column, actions in actions_dict.items():
            for action in actions:
                # Validamos que 'action' sea un string válido y no vacío.
                if not isinstance(action, str) or not action:
                    self.verbose.log(
                        f"ADVERTENCIA: Se encontró una 'action_type' inválida o vacía: '{action}'. Se saltará."
                    )
                    continue

                action_col_name = f"_action_{action}"

                # --- LA SOLUCIÓN FINAL ---
                # Verificamos si la columna ya existe en el DataFrame antes de intentar crearla.
                if action_col_name in df_facebook.columns:
                    self.verbose.log(
                        f"ADVERTENCIA: La columna '{action_col_name}' ya existe. Saltando para evitar duplicados."
                    )
                    continue
                df_facebook[action_col_name] = df_facebook[column].apply(
                    lambda x: self._split_text(x, action)
                )

        self.verbose.log("Procesamiento de acciones completado.")

        return df_facebook

    def get_report(self, params, act_id, max_tries=10, page_size=500):
        """
        Ejecuta una consulta de reportes asincrónica a la API de Facebook Ads para una cuenta específica.

        Esta función lanza un job de insights (asincrónico) para la cuenta especificada y espera
        hasta que el job se complete, falle o exceda el tiempo de espera. Si se completa correctamente,
        extrae los datos del reporte y los transforma en una lista de registros exportados.

        Args:
            params (dict): Parámetros de consulta para `get_insights()`. Debe incluir campos como:
                - 'level': nivel de agregación ('campaign', 'adset', etc.)
                - 'fields': lista de métricas/dimensiones solicitadas
                - 'time_range': diccionario con 'since' y 'until'
                - (opcional) 'breakdowns', 'time_increment', etc.
            act_id (str): ID de cuenta publicitaria en formato 'act_XXXXXXXXXXXX'.
            max_tries (int, optional): Número máximo de intentos para iniciar el job. Por defecto es 10.
            page_size (int, optional): Filas por página al descargar el resultado del job. No cambia
                los datos, solo cuántas llamadas se hacen: sin este valor Meta pagina de a 25 filas y
                cada página cuenta contra el límite de Ads Insights de la app. Por defecto es 500.

        Returns:
            list[dict]: Lista de registros extraídos del reporte, cada uno representando una fila con datos exportados.
            En caso de error, puede lanzar una excepción o retornar una lista vacía si el resultado es None.

        Raises:
            Exception: Si la API lanza un error crítico (subcode 99 o status 500), si el job falla,
                    o si se agota el tiempo de espera sin recibir resultados.
        """
        my_account = AdAccount(act_id, api=self.service)
        for attempt in range(max_tries):
            try:
                async_job = my_account.get_insights(params=params, is_async=True)
                self.verbose.log(
                    f"get_report | Intento {attempt + 1} - Job lanzado correctamente para la cuenta {act_id}"
                )
                break
            except FacebookRequestError as e:
                subcode = e.api_error_subcode()
                status = e.http_status()
                message = e.api_error_message()

                log_msg = (
                    f"get_report | Facebook API error\n"
                    f"  Subcode: {subcode}\n"
                    f"  Status:  {status}\n"
                    f"  Message: {message}\n"
                    f"  Params:  {params}\n"
                    f"  Account: {act_id}"
                )

                if subcode == 99 or status == 500:
                    self.verbose.critical("Error crítico en Meta API:\n" + log_msg)
                    raise Exception(
                        "Meta API devolvió error 500 con subcode 99: problema de autenticación o permisos."
                    )
                else:
                    self.verbose.critical(log_msg)
                    raise e
            except Exception as e:
                self.verbose.critical(
                    f"get_report | Error inesperado al iniciar job para {act_id}: {e}"
                )
                time.sleep(2**attempt)
        else:
            raise Exception(
                "get_report | No se pudo iniciar el job después de múltiples intentos"
            )

        time.sleep(10)
        tries = 0
        while tries < 60:
            status = async_job.api_get().get("async_status", "")

            if status == "Job Completed":
                self.verbose.log("get_report | Job completado")
                result = async_job.get_result(params={"limit": page_size})

                if result is None:
                    self.verbose.critical(
                        f"{act_id} | get_report | Resultado vacío o None recibido."
                    )
                    return []

                self.verbose.log(
                    f"{act_id} | get_report | Resultado recibido con {len(result)} registros"
                )
                self.verbose.log(
                    f"get_report | Resultado crudo recibido, tipo: {type(result)}"
                )

                records = []
                for record in result:
                    try:
                        records.append(record.export_all_data())
                    except Exception as e:
                        self.verbose.log(
                            f"get_report | Error procesando un registro: {str(e)}"
                        )
                        continue

                self.verbose.log(
                    f"get_report | Exportación completada con {len(records)} registros."
                )
                return records

            elif status == "Job Failed":
                raise Exception(f"get_report | Job falló para la cuenta {act_id}")

            else:
                self.verbose.log(f"get_report | Esperando... intento {tries + 1}")
                time.sleep(20)
                tries += 1

        raise TimeoutError(f"get_report | Timeout esperando el job para {act_id}")

    def def_report_array_accounts(self, params, id_accounts):
        self.verbose.log("def_report_array_accounts | Procesando múltiples cuentas")
        array_df = []
        for acc in id_accounts:
            self.verbose.log(f"Cuenta: {acc}")
            acc = str(acc)
            df = self.get_report_dataframe(params, acc)
            array_df.append(df)
        return pd.concat(array_df, ignore_index=True)

    def _unique_actions(self, df):
        self.verbose.log("_unique_actions")
        actions_per_column = {}
        for column in df.columns:
            if df[column].apply(lambda x: isinstance(x, list)).any():
                actions_per_column[column] = set()
                for all_actions in df[column].dropna():
                    for action in all_actions:
                        if "action_type" in action:
                            actions_per_column[column].add(action["action_type"])
        return actions_per_column

    def _split_text(self, text, action):
        if not isinstance(text, list):
            return 0
        for element in text:
            if element.get("action_type") == action:
                return element.get("value", 0)
        return 0
