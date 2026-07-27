import copy
import datetime
import random
import time
from datetime import datetime, timedelta

import pandas as pd
from googleapiclient.errors import HttpError

import d2b_data.Google_Token_MNG


class Google_GA4:
    def __init__(
        self,
        client_secret: str | None = None,
        token_json: str | None = None,
        debug: bool = False,
        auto_paginate: bool = True,
        extract_sampling: bool = False,
        intraday_limit: int = 30,
        use_service_account: bool = False,
    ):
        self.default_api_name = "analyticsdata"
        self.default_version = "v1beta"
        self.client_secret = client_secret
        self.debug_status = debug
        self.auto_paginate = auto_paginate
        self.extract_sampling = extract_sampling
        self.intraday_limit = intraday_limit * 100000
        self.service = self.create_service(
            self.client_secret, token_json, use_service_account
        )

    def _extract_sampling_info(self, report_data: dict) -> dict:
        """
        Extracts sampling information from the GA4 report metadata.
        """
        info = {
            "samplesReadCounts": 0,
            "samplingSpaceSizes": 0,
            "sampled": False,
            "sampling_percentage": 100.0,
            "dataLossFromOtherRow": False,
        }

        metadata = report_data.get("metadata", {})

        # Sampling
        sampling_metadatas = metadata.get("samplingMetadatas", [])
        if sampling_metadatas:
            s_data = sampling_metadatas[0]
            samples_read = int(s_data.get("samplesReadCount", 0))
            space_size = int(s_data.get("samplingSpaceSize", 0))

            if space_size > 0:
                info["samplesReadCounts"] = samples_read
                info["samplingSpaceSizes"] = space_size
                info["sampled"] = True
                info["sampling_percentage"] = (samples_read / space_size) * 100

        # Cardinalidad
        if metadata.get("dataLossFromOtherRow"):
            info["dataLossFromOtherRow"] = True

        return info

    def get_service(self):
        """return the service object"""
        return self.service

    def get_token(self):
        """return the token"""
        return self.token

    def get_credentials(self):
        """return credentials"""
        return self.credentials

    def set_auto_paginate(self, auto_paginate: bool = True):
        """
        Activates/deactivates automatic pagination for large reports
        """
        if type(auto_paginate) == bool:
            self.auto_paginate = auto_paginate
        return self.auto_paginate

    def set_extract_sampling(self, extract_sampling: bool = True):
        """
        Activates/deactivates extraction of sampling info into the DataFrame
        """
        if type(extract_sampling) == bool:
            self.extract_sampling = extract_sampling
        return self.extract_sampling

    def create_service(
        self, secrets: str, credentials: str, use_service_account: bool = False
    ):
        """Creates the Google Analytics Data API service object using the provided credentials."""

        token_mng = d2b_data.Google_Token_MNG.Google_Token_MNG(
            client_secret=secrets,  # Si es SA, esto es la ruta al JSON key
            token=credentials,  # Si es SA, esto puede ser None
            scopes=["https://www.googleapis.com/auth/analytics.readonly"],
            api_version=self.default_version,
            api_name=self.default_api_name,
            use_service_account=use_service_account,  # <--- ¡Aquí está la magia!
        )

        self.service = token_mng.get_service()
        self.debug("Conectado a GA4")
        return self.service

    def _to_df(self, raw_server_response: dict) -> pd.DataFrame:
        """
        Transforms the raw GA4 API response into a Pandas DataFrame.
        """
        if not raw_server_response.get("reports"):
            return pd.DataFrame()

        response = raw_server_response.get("reports")[0]

        cols = []
        for dimensions_cols in response.get("dimensionHeaders", []):
            cols.append(dimensions_cols.get("name"))
        for metrics_cols in response.get("metricHeaders", []):
            cols.append(metrics_cols.get("name"))

        results = []
        for row in response.get("rows", []):
            row_array = []
            for dimension in row.get("dimensionValues", []):
                row_array.append(dimension.get("value"))
            for metrics in row.get("metricValues", []):
                row_array.append(metrics.get("value"))
            results.append(row_array)

        return pd.DataFrame(results, columns=cols)

    def debug(self, message: str):
        """Prints debug messages if debug mode is enabled."""
        if self.debug_status:
            print(message)

    def _get_report_raw(self, property_id: str, query: dict):
        """
        Executes the API call to retrieve the raw report data, with retry logic for handling rate limits and transient errors.
        """
        max_retries = 5
        retry_count = 0

        while True:
            try:
                response = (
                    self.service.properties()
                    .batchRunReports(property=property_id, body=query)
                    .execute()
                )
                return response

            except HttpError as e:
                status_code = e.resp.status
                reason = e._get_reason()
                if status_code in [429, 500, 503]:
                    if retry_count >= max_retries:
                        self.debug(
                            f" Error {status_code} ({reason}): max retries {max_retries} exceeded."
                        )
                        raise e

                    sleep_time = (2**retry_count) + random.uniform(0, 1)

                    self.debug(
                        f" Error {status_code}. Retry {retry_count + 1}/{max_retries}. Waiting {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    retry_count += 1
                else:
                    self.debug(f" Not recoverable {status_code}: {reason}")
                    raise e

            except Exception as e:
                if "429" in str(e):
                    if retry_count >= max_retries:
                        raise

                    sleep_time = (2**retry_count) + random.uniform(0, 1)
                    self.debug(
                        f" Generic error detected (possible 429). Waiting {sleep_time:.2f} s..."
                    )

                    time.sleep(sleep_time)
                    retry_count += 1
                else:
                    raise

    def get_report_df(
        self, property_id: str, query: dict, extract_sampling=None
    ) -> pd.DataFrame:
        """Public method to retrieve a report as a DataFrame, with optional automatic pagination and sampling info extraction."""

        should_extract = (
            extract_sampling if extract_sampling is not None else self.extract_sampling
        )

        if not self.auto_paginate:
            return self._get_single_report(property_id, query, should_extract)

        return self._get_paginated_report(property_id, query, should_extract)

    def _get_single_report(
        self, property_id: str, query: dict, extract_sampling: bool = False
    ):
        """Obtains a single report without pagination, with optional sampling info extraction."""
        res = self._get_report_raw(property_id, query)
        df_report = self._to_df(res)

        # Agregar info de sampling si está activada
        if extract_sampling and not df_report.empty:
            report_data = res.get("reports")[0]
            sampling_info = self._extract_sampling_info(report_data)
            df_report["samplesReadCounts"] = sampling_info["samplesReadCounts"]
            df_report["samplingSpaceSizes"] = sampling_info["samplingSpaceSizes"]
            df_report["sampling_percentage"] = sampling_info["sampling_percentage"]
            df_report["sampled"] = sampling_info["sampled"]
            df_report["dataLossFromOtherRow"] = sampling_info["dataLossFromOtherRow"]

        return df_report

    def _get_paginated_report(
        self, property_id: str, query: dict, extract_sampling: bool = False
    ):
        """
        Obtains a report with automatic pagination, iterating day by day to avoid sampling and API limits.
        """
        # Extraer rango de fechas
        original_date_ranges = query["requests"][0]["dateRanges"]
        start_date = original_date_ranges[0]["startDate"]
        end_date = original_date_ranges[0]["endDate"]

        self.debug(f"Paging from {start_date} to {end_date}")

        # Convertir a datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Lista para almacenar DataFrames
        all_dataframes = []
        current_date = start

        # Iterar día por día
        while current_date <= end:
            day_str = current_date.strftime("%Y-%m-%d")
            self.debug(f"Consulting day: {day_str}")

            # Obtener todas las filas del día
            day_dataframes = self._get_all_rows_for_day(
                property_id, query, day_str, extract_sampling
            )

            if day_dataframes:
                all_dataframes.extend(day_dataframes)

            current_date += timedelta(days=1)

        # Combinar resultados
        if all_dataframes:
            result_df = pd.concat(all_dataframes, ignore_index=True)
            self.debug(f"Total of rows obtained: {len(result_df)}")
            return result_df
        else:
            self.debug("No data found for the specified period.")
            return pd.DataFrame()

    def _get_all_rows_for_day(
        self,
        property_id: str,
        query: dict,
        day_str: str,
        extract_sampling: bool = False,
    ):
        """
        Obtains all rows for a specific day using offset
        """
        day_dataframes = []
        offset = 0
        limit_per_request = 250000

        while True:
            # Crear query para este día con offset
            daily_query = self._create_daily_query(query, day_str, day_str)
            daily_query["requests"][0]["offset"] = offset

            self.debug(f"  → offset {offset}")

            # Ejecutar query
            daily_df = self._get_single_report(
                property_id, daily_query, extract_sampling
            )

            if daily_df.empty:
                break

            rows_received = len(daily_df)
            day_dataframes.append(daily_df)

            # Si recibimos menos de 250k, no hay más datos
            if rows_received < limit_per_request:
                break

            offset += limit_per_request

        total_rows = sum(len(df) for df in day_dataframes)
        if total_rows > 0:
            self.debug(f" Total rows for the day: {total_rows}")

        return day_dataframes

    def _create_daily_query(
        self, original_query: dict, start_date: str, end_date: str
    ) -> dict:
        """
        Creates a copy of the query with specific dates
        """
        daily_query = copy.deepcopy(original_query)
        daily_query["requests"][0]["dateRanges"] = [
            {"startDate": start_date, "endDate": end_date}
        ]
        return daily_query
