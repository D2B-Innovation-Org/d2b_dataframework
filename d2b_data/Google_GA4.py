import copy
import datetime
import random
import time
from datetime import datetime, timedelta

import pandas as pd
from googleapiclient.errors import HttpError

import d2b_data.Google_Token_MNG


class Google_GA4:
    LIMIT_PER_REQUEST = 250000

    def __init__(
        self,
        client_secret: str | None = None,
        token_json: str | None = None,
        debug: bool = False,
        auto_paginate: bool = True,
        extract_sampling: bool = False,
        intraday_limit: int = 30,
        use_service_account: bool = False,
        batch_size: int = 5,
        request_delay: float = 0.0,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not (1 <= batch_size <= 5):
            raise ValueError("batch_size must be an integer between 1 and 5.")
        if request_delay < 0:
            raise ValueError("request_delay must be non-negative.")

        self.default_api_name = "analyticsdata"
        self.default_version = "v1beta"
        self.client_secret = client_secret
        self.debug_status = debug
        self.auto_paginate = auto_paginate
        self.extract_sampling = extract_sampling
        self.intraday_limit = intraday_limit * 100000
        self.batch_size = batch_size
        self.request_delay = float(request_delay)
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

    def set_batch_size(self, batch_size: int = 5):
        """
        Sets the number of daily requests grouped per batchRunReports HTTP call (1..5).
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not (1 <= batch_size <= 5):
            raise ValueError("batch_size must be an integer between 1 and 5.")
        self.batch_size = batch_size
        return self.batch_size

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

    def _report_to_df(self, report_data: dict) -> pd.DataFrame:
        """
        Transforms a single GA4 report dict into a Pandas DataFrame.
        """
        cols = []
        for dimensions_cols in report_data.get("dimensionHeaders", []):
            cols.append(dimensions_cols.get("name"))
        for metrics_cols in report_data.get("metricHeaders", []):
            cols.append(metrics_cols.get("name"))

        results = []
        for row in report_data.get("rows", []):
            row_array = []
            for dimension in row.get("dimensionValues", []):
                row_array.append(dimension.get("value"))
            for metrics in row.get("metricValues", []):
                row_array.append(metrics.get("value"))
            results.append(row_array)

        return pd.DataFrame(results, columns=cols)

    def _to_df(self, raw_server_response: dict) -> pd.DataFrame:
        """
        Transforms the raw GA4 API response into a Pandas DataFrame.
        """
        if not raw_server_response.get("reports"):
            return pd.DataFrame()

        return self._report_to_df(raw_server_response.get("reports")[0])

    def _attach_sampling_info(
        self, df_report: pd.DataFrame, report_data: dict
    ) -> pd.DataFrame:
        """
        Attaches GA4 sampling metadata columns to a non-empty DataFrame.
        """
        sampling_info = self._extract_sampling_info(report_data)
        df_report["samplesReadCounts"] = sampling_info["samplesReadCounts"]
        df_report["samplingSpaceSizes"] = sampling_info["samplingSpaceSizes"]
        df_report["sampling_percentage"] = sampling_info["sampling_percentage"]
        df_report["sampled"] = sampling_info["sampled"]
        df_report["dataLossFromOtherRow"] = sampling_info["dataLossFromOtherRow"]
        return df_report

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
                if status_code in [429, 500, 502, 503, 504]:
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
            self._attach_sampling_info(df_report, report_data)

        return df_report

    def _get_paginated_report(
        self, property_id: str, query: dict, extract_sampling: bool = False
    ):
        """
        Obtains a report with automatic pagination, iterating day by day to avoid sampling and API limits.
        When batch_size > 1, groups up to batch_size independent 1-day requests per batchRunReports call.
        """
        original_date_ranges = query["requests"][0]["dateRanges"]
        start_date = original_date_ranges[0]["startDate"]
        end_date = original_date_ranges[0]["endDate"]

        self.debug(f"Paging from {start_date} to {end_date}")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        day_strings = []
        current_date = start
        while current_date <= end:
            day_strings.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        all_dataframes = []

        if self.batch_size == 1:
            for idx, day_str in enumerate(day_strings):
                if idx > 0 and self.request_delay > 0:
                    time.sleep(self.request_delay)
                self.debug(f"Consulting day: {day_str}")
                day_dataframes = self._get_all_rows_for_day(
                    property_id, query, day_str, extract_sampling
                )
                if day_dataframes:
                    all_dataframes.extend(day_dataframes)
        else:
            chunks = [
                day_strings[i : i + self.batch_size]
                for i in range(0, len(day_strings), self.batch_size)
            ]
            for idx, chunk in enumerate(chunks):
                if idx > 0 and self.request_delay > 0:
                    time.sleep(self.request_delay)
                self.debug(f"Consulting batch: {chunk[0]}..{chunk[-1]} ({len(chunk)} days)")
                batch_dataframes = self._get_batch_rows_for_days(
                    property_id, query, chunk, extract_sampling
                )
                if batch_dataframes:
                    all_dataframes.extend(batch_dataframes)

        # Combinar resultados
        if all_dataframes:
            result_df = pd.concat(all_dataframes, ignore_index=True)
            self.debug(f"Total of rows obtained: {len(result_df)}")
            return result_df
        else:
            self.debug("No data found for the specified period.")
            return pd.DataFrame()

    def _get_batch_rows_for_days(
        self,
        property_id: str,
        query: dict,
        day_strings: list[str],
        extract_sampling: bool = False,
    ) -> list[pd.DataFrame]:
        """
        Fetches up to batch_size independent 1-day reports in a single batchRunReports call,
        falling back to per-day offset pagination only for days that reach LIMIT_PER_REQUEST.
        """
        batch_query = self._create_batch_query(query, day_strings, offset=0)
        raw_response = self._get_report_raw(property_id, batch_query)
        reports = raw_response.get("reports", [])

        batch_dataframes = []
        limit_per_request = self.LIMIT_PER_REQUEST

        for day_str, report_data in zip(day_strings, reports):
            df_day = self._report_to_df(report_data)
            if df_day.empty:
                continue

            if extract_sampling:
                self._attach_sampling_info(df_day, report_data)

            batch_dataframes.append(df_day)

            if len(df_day) >= limit_per_request:
                overflow_dfs = self._get_all_rows_for_day(
                    property_id,
                    query,
                    day_str,
                    extract_sampling,
                    start_offset=limit_per_request,
                )
                if overflow_dfs:
                    batch_dataframes.extend(overflow_dfs)

        return batch_dataframes

    def _get_all_rows_for_day(
        self,
        property_id: str,
        query: dict,
        day_str: str,
        extract_sampling: bool = False,
        start_offset: int = 0,
    ):
        """
        Obtains all rows for a specific day using offset
        """
        day_dataframes = []
        offset = start_offset
        limit_per_request = self.LIMIT_PER_REQUEST

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

    def _create_batch_query(
        self, original_query: dict, day_strings: list[str], offset: int = 0
    ) -> dict:
        """
        Builds a batchRunReports payload containing one independent 1-day request per date.
        """
        base_request = original_query["requests"][0]
        requests = []
        for day_str in day_strings:
            req = copy.deepcopy(base_request)
            req["dateRanges"] = [{"startDate": day_str, "endDate": day_str}]
            req["offset"] = offset
            requests.append(req)
        return {"requests": requests}

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
