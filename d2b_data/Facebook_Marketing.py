import time
import warnings
from datetime import datetime, timezone
from urllib.parse import urlsplit

import pandas as pd
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.api import FacebookAdsApi
from facebook_business.exceptions import FacebookRequestError
from d2b_data.workflow_logger import resolve_logger

# Columnas de get_ads_creatives, en orden. Se usan también para el DataFrame vacío
# y así el esquema de la tabla de destino no depende de si la cuenta tiene anuncios.
CREATIVE_COLUMNS = [
    "account_id",
    "ad_id",
    "ad_name",
    "effective_status",
    "updated_time",
    "creative_id",
    "object_type",
    "thumbnail_url",
    "image_url",
    "image_hash",
    "video_id",
]
# Piso al achicar la página cuando Meta responde que la respuesta es muy grande.
MIN_ADS_PAGE_SIZE = 25
CREATIVE_FIELDS = [
    "id",
    "object_type",
    "thumbnail_url",
    "image_url",
    "image_hash",
    "video_id",
]


class Facebook_Marketing:
    def __init__(
        self,
        app_id,
        app_secret,
        access_token,
        id_account=None,
        unsampled=False,
        logger=None,
        api_version="v25.0",
        *,
        verbose_logger=None,
    ):
        logger = resolve_logger(logger, verbose_logger, "Facebook_Marketing")
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.unsampled = unsampled
        self.id_account = id_account
        self.logger = logger if logger else self._null_verbose()
        self.logger.log(
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
            caller = getattr(self.logger, "workflow_name", None) or "unknown"
            details = " | ".join(f"{name}={value}" for name, value in usage.items())
            self.logger.log(
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
            self.logger.log("get_report_dataframe | Unsampled")
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
                self.logger.critical(
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
                    self.logger.log(
                        "Intentando crear el DataFrame desde el reporte crudo..."
                    )
                    df_facebook = pd.DataFrame(report, index=None)
                    self.logger.log("DataFrame creado exitosamente desde el reporte.")
                except Exception:
                    self.logger.critical(
                        "¡FALLO CRÍTICO EN LA CREACIÓN DEL DATAFRAME! Guardando datos crudos en GCS para análisis."
                    )

        if not df_facebook.empty:
            df_facebook.reset_index(drop=True, inplace=True)

        self.logger.log("Iniciando procesamiento de acciones (versión robusta)...")
        actions_dict = self._unique_actions(df_facebook)

        for column, actions in actions_dict.items():
            for action in actions:
                # Validamos que 'action' sea un string válido y no vacío.
                if not isinstance(action, str) or not action:
                    self.logger.log(
                        f"ADVERTENCIA: Se encontró una 'action_type' inválida o vacía: '{action}'. Se saltará."
                    )
                    continue

                action_col_name = f"_action_{action}"

                # --- LA SOLUCIÓN FINAL ---
                # Verificamos si la columna ya existe en el DataFrame antes de intentar crearla.
                if action_col_name in df_facebook.columns:
                    self.logger.log(
                        f"ADVERTENCIA: La columna '{action_col_name}' ya existe. Saltando para evitar duplicados."
                    )
                    continue
                df_facebook[action_col_name] = df_facebook[column].apply(
                    lambda x: self._split_text(x, action)
                )

        self.logger.log("Procesamiento de acciones completado.")

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
                self.logger.log(
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
                    self.logger.critical("Error crítico en Meta API:\n" + log_msg)
                    raise Exception(
                        "Meta API devolvió error 500 con subcode 99: problema de autenticación o permisos."
                    )
                else:
                    self.logger.critical(log_msg)
                    raise e
            except Exception as e:
                self.logger.critical(
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
                self.logger.log("get_report | Job completado")
                result = async_job.get_result(params={"limit": page_size})

                if result is None:
                    self.logger.critical(
                        f"{act_id} | get_report | Resultado vacío o None recibido."
                    )
                    return []

                self.logger.log(
                    f"{act_id} | get_report | Resultado recibido con {len(result)} registros"
                )
                self.logger.log(
                    f"get_report | Resultado crudo recibido, tipo: {type(result)}"
                )

                records = []
                for record in result:
                    try:
                        records.append(record.export_all_data())
                    except Exception as e:
                        self.logger.log(
                            f"get_report | Error procesando un registro: {str(e)}"
                        )
                        continue

                self.logger.log(
                    f"get_report | Exportación completada con {len(records)} registros."
                )
                return records

            elif status == "Job Failed":
                raise Exception(f"get_report | Job falló para la cuenta {act_id}")

            else:
                self.logger.log(f"get_report | Esperando... intento {tries + 1}")
                time.sleep(20)
                tries += 1

        raise TimeoutError(f"get_report | Timeout esperando el job para {act_id}")

    def def_report_array_accounts(self, params, id_accounts):
        self.logger.log("def_report_array_accounts | Procesando múltiples cuentas")
        array_df = []
        for acc in id_accounts:
            self.logger.log(f"Cuenta: {acc}")
            acc = str(acc)
            df = self.get_report_dataframe(params, acc)
            array_df.append(df)
        return pd.concat(array_df, ignore_index=True)

    def get_ads_creatives(
        self,
        id_account: str | int | list | None = None,
        thumbnail_size: int = 400,
        effective_status: list[str] | None = None,
        page_size: int = 500,
        updated_since: datetime | None = None,
    ) -> pd.DataFrame:
        """Lista los anuncios de una cuenta con las URLs de imagen de su creative.

        Insights no devuelve las imágenes de los anuncios, por eso es una consulta
        aparte al edge /act_{id}/ads. No usa el límite de Ads Insights. Las URLs que
        entrega Meta son temporales (vencen en uno a cinco días), así que quien las
        consuma debe descargar la imagen y no guardar la URL.

        Args:
            id_account: ID de la cuenta sin el prefijo 'act_', o una lista de IDs.
                Si es None se usa la cuenta del constructor.
            thumbnail_size: Lado en píxeles del thumbnail cuadrado. Meta entrega
                64x64 si no se indica. Por defecto 400.
            effective_status: Estados de anuncio a incluir, por ejemplo
                ['ACTIVE', 'PAUSED', 'ARCHIVED']. Si es None se usa el filtro por
                defecto de Meta, que deja fuera los anuncios archivados y eliminados.
            page_size: Anuncios por página. Por defecto 500.
            updated_since: Si se indica, solo trae los anuncios modificados después
                de ese momento (filtro de Meta por updated_time). Sirve para cargas
                incrementales. Un datetime sin zona horaria se interpreta como UTC.
                Por defecto None, que trae todos.

        Returns:
            DataFrame con una fila por anuncio y las columnas de CREATIVE_COLUMNS.
            Los campos que Meta no informa para un anuncio quedan en None (por
            ejemplo image_url en anuncios de video).

        Raises:
            ValueError: Si thumbnail_size no es un entero positivo.
            TypeError: Si updated_since no es None ni un datetime.
            FacebookRequestError: Si Meta rechaza la consulta.
        """
        # Solo None cae a la cuenta del constructor: una lista vacía significa
        # "ninguna cuenta", no "la cuenta por defecto".
        if id_account is None:
            id_account = self.id_account
        is_valid_size = (
            isinstance(thumbnail_size, int)
            and not isinstance(thumbnail_size, bool)
            and thumbnail_size > 0
        )
        if not is_valid_size:
            raise ValueError(
                f"thumbnail_size debe ser un entero positivo, llegó {thumbnail_size!r}"
            )
        if updated_since is not None and not isinstance(updated_since, datetime):
            raise TypeError(
                f"updated_since debe ser un datetime o None, llegó {updated_since!r}"
            )

        if isinstance(id_account, list):
            frames = [
                self.get_ads_creatives(
                    acc,
                    thumbnail_size,
                    effective_status,
                    page_size,
                    updated_since=updated_since,
                )
                for acc in id_account
            ]
            if not frames:
                return pd.DataFrame(columns=CREATIVE_COLUMNS)
            return pd.concat(frames, ignore_index=True)

        account_id = str(id_account)
        creative_field = (
            f"creative.thumbnail_width({thumbnail_size})"
            f".thumbnail_height({thumbnail_size})"
            "{" + ",".join(CREATIVE_FIELDS) + "}"
        )
        fields = ["id", "name", "effective_status", "updated_time", creative_field]
        params = {"limit": page_size}
        if effective_status:
            params["effective_status"] = list(effective_status)
        if updated_since is not None:
            if updated_since.tzinfo is None:
                updated_since = updated_since.replace(tzinfo=timezone.utc)
            # Meta espera el filtro de updated_time como timestamp unix en segundos.
            params["filtering"] = [
                {
                    "field": "updated_time",
                    "operator": "GREATER_THAN",
                    "value": int(updated_since.timestamp()),
                }
            ]

        self.logger.log(f"get_ads_creatives | act_{account_id} | inicio")
        ad_account = AdAccount(f"act_{account_id}", api=self.service)

        # get_ads ya descarga la primera página, así que también puede fallar por
        # tamaño de respuesta antes de que exista el cursor.
        while True:
            try:
                ads = self._get_ads_cursor(ad_account, fields, params)
                break
            except FacebookRequestError as error:
                self._shrink_page_or_raise(error, params, account_id)

        rows = []
        # Si una página falla por tamaño, el cursor conserva su 'after': se reintenta
        # esa misma página con menos anuncios y no se pierde lo ya descargado.
        while True:
            try:
                for ad in ads:
                    rows.append(self._creative_row(ad, account_id))
                break
            except FacebookRequestError as error:
                self._shrink_page_or_raise(error, ads.params, account_id)

        self.logger.log(f"get_ads_creatives | act_{account_id} | {len(rows)} anuncios")
        return pd.DataFrame(rows, columns=CREATIVE_COLUMNS)

    def _get_ads_cursor(self, ad_account, fields: list[str], params: dict):
        """Pide el edge de anuncios sin el UserWarning falso del SDK.

        El SDK valida los campos contra una lista fija y no reconoce la expansión
        anidada del creative, aunque la Graph API sí la acepta.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ads does not allow field")
            return ad_account.get_ads(fields=fields, params=dict(params))

    def _shrink_page_or_raise(
        self, error: FacebookRequestError, params: dict, account_id: str
    ) -> None:
        """Divide a la mitad el 'limit' de params si la página es muy pesada.

        Meta tiene dos formas de rechazar una página con el creative expandido que
        pesa demasiado (pasa en cuentas grandes con 500 anuncios por página):
        código 1 "Please reduce the amount of data", o código 1 subcódigo 99
        "An unknown error occurred" (HTTP 500), que es su timeout interno y
        aparece de forma intermitente tras unos 30 segundos.

        Args:
            error: Error devuelto por Meta.
            params: Parámetros de la consulta; se modifica 'limit' en el lugar.
            account_id: Cuenta, solo para el log.

        Raises:
            FacebookRequestError: El mismo error, si no es por tamaño o si el
                'limit' ya está en el mínimo.
        """
        message = error.api_error_message() or ""
        is_too_much_data = "reduce the amount of data" in message
        is_meta_timeout = (
            error.api_error_code() == 1 and error.api_error_subcode() == 99
        )
        limit = params.get("limit", MIN_ADS_PAGE_SIZE)
        if not (is_too_much_data or is_meta_timeout) or limit <= MIN_ADS_PAGE_SIZE:
            raise error
        params["limit"] = max(MIN_ADS_PAGE_SIZE, limit // 2)
        self.logger.log(
            f"get_ads_creatives | act_{account_id} | respuesta muy grande "
            f"(código {error.api_error_code()}/{error.api_error_subcode()}), "
            f"limit {limit} -> {params['limit']}"
        )

    @staticmethod
    def _creative_row(ad, account_id: str) -> dict:
        """Aplana un anuncio del SDK a una fila con las columnas de CREATIVE_COLUMNS."""
        data = ad.export_all_data() if hasattr(ad, "export_all_data") else ad
        creative = data.get("creative") or {}
        return {
            "account_id": account_id,
            "ad_id": data.get("id"),
            "ad_name": data.get("name"),
            "effective_status": data.get("effective_status"),
            "updated_time": data.get("updated_time"),
            "creative_id": creative.get("id"),
            "object_type": creative.get("object_type"),
            "thumbnail_url": creative.get("thumbnail_url"),
            "image_url": creative.get("image_url"),
            "image_hash": creative.get("image_hash"),
            "video_id": creative.get("video_id"),
        }

    def _unique_actions(self, df):
        self.logger.log("_unique_actions")
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
