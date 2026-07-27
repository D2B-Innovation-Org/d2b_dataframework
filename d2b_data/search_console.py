from __future__ import annotations

import copy
import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

import d2b_data.Google_Token_MNG
from d2b_data.workflow_logger import WorkflowLogger


class GoogleSearchConsole:
    """
    Wrapper for the Google Search Console API.

    Handles authentication through the internal Google_Token_MNG wrapper,
    report execution, pagination and response conversion to pandas DataFrames.
    """

    def __init__(
        self,
        client_secret: str,
        token_json: str | None,
        verbose_logger: (WorkflowLogger | None) = None,
        auto_paginate: bool = True,
        row_limit: int = 25_000,
        use_service_account: bool = False,
    ) -> None:
        """

        Initializes the Google Search Console API wrapper.
        Args:
            client_secret:
                OAuth client secret path or Service Account JSON path.
            token_json:
                OAuth token path. It can be None when using a Service Account.
            verbose_logger:
                WorkflowLogger instance used during execution. If omitted,
                an internal logger with alerts disabled is created.
            auto_paginate:
                Enables automatic pagination.
            row_limit:
                Maximum number of rows requested per API call.
            use_service_account:
                Indicates whether Service Account authentication should be used.
        """
        self.default_api_name: str = "searchconsole"
        self.default_version: str = "v1"
        self.client_secret: str = client_secret
        self.token_json: str | None = token_json
        self.auto_paginate: bool = auto_paginate
        self.row_limit: int = row_limit
        self.use_service_account: bool = use_service_account
        self.logger: WorkflowLogger = verbose_logger or self._build_default_logger()

        self.logger.info(
            "--- EXECUTING Google_Search_Console Class v1.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )

        self.service: Resource = self.create_service(
            secrets=self.client_secret,
            credentials=self.token_json,
            use_service_account=self.use_service_account,
        )

    def get_service(self) -> Resource:
        """
        Returns the Search Console API service object.
        Returns:
            Authenticated Search Console API resource.
        """
        return self.service

    def get_token(self) -> str | None:
        """
        Returns the configured OAuth token path.
        Returns:
            OAuth token path or None.
        """
        return self.token_json

    def set_auto_paginate(
        self,
        auto_paginate: bool = True,
    ) -> bool:
        """
        Enables or disables automatic pagination.
        Args:
            auto_paginate:
                New automatic pagination status.
        Returns:
            Current automatic pagination status.
        Raises:
            TypeError:
                If auto_paginate is not a boolean.
        """
        if not isinstance(auto_paginate, bool):
            self.logger.critical("auto_paginate must be a boolean")
            raise TypeError("auto_paginate must be a boolean")

        self.auto_paginate = auto_paginate

        return self.auto_paginate

    def create_service(
        self,
        secrets: str,
        credentials: str | None,
        use_service_account: bool = False,
    ) -> Resource:
        """
        Creates the Google Search Console API service object.
        Authentication is delegated to the internal Google_Token_MNG wrapper.
        Args:
            secrets:
                OAuth client secret path or Service Account JSON path.
            credentials:
                OAuth token path. It can be None when using a Service Account.
            use_service_account:
                Indicates whether Service Account authentication should be used.
        Returns:
            Authenticated Search Console API resource.
        """
        token_mng = d2b_data.Google_Token_MNG.Google_Token_MNG(
            client_secret=secrets,
            token=credentials,
            scopes=[
                "https://www.googleapis.com/auth/webmasters.readonly",
            ],
            api_version=self.default_version,
            api_name=self.default_api_name,
            use_service_account=use_service_account,
        )

        service: Resource = token_mng.get_service()

        self.logger.info("Connected to Google Search Console")

        return service

    def get_report_df(
        self,
        property_uri: str,
        start_date: str,
        end_date: str,
        dimensions: list[str],
        dimension_filter_groups: list[dict[str, Any]] | None = None,
        search_type: str = "web",
        data_state: str = "final",
    ) -> pd.DataFrame:
        """
        Retrieves a Search Console report as a pandas DataFrame.

        Args:
            property_uri:
                Search Console property URI, such as
                "sc-domain:example.com" or "https://example.com/".
            start_date:
                Report start date in YYYY-MM-DD format.
            end_date:
                Report end date in YYYY-MM-DD format.
            dimensions:
                Dimensions included in the report.
            dimension_filter_groups:
                Optional Search Console dimension filters.
            search_type:
                Search type, such as web, image, video or news.
            data_state:
                Data state requested from Search Console.

        Returns:
            Search Console report as a pandas DataFrame.
        """
        self._validate_report_parameters(
            property_uri=property_uri,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
        )

        report_dimensions: list[str] = dimensions.copy()

        if "date" not in report_dimensions:
            report_dimensions.append("date")

        query: dict[str, Any] = self._create_query(
            start_date=start_date,
            end_date=end_date,
            dimensions=report_dimensions,
            dimension_filter_groups=dimension_filter_groups,
            search_type=search_type,
            data_state=data_state,
        )

        if not self.auto_paginate:
            response: dict[str, Any] = self._get_report_raw(
                property_uri=property_uri,
                query=query,
            )

            return self._to_df(
                raw_server_response=response,
                dimensions=report_dimensions,
            )

        return self._get_paginated_report(
            property_uri=property_uri,
            query=query,
            dimensions=report_dimensions,
        )

    def _create_query(
        self,
        start_date: str,
        end_date: str,
        dimensions: list[str],
        dimension_filter_groups: list[dict[str, Any]] | None = None,
        search_type: str = "web",
        data_state: str = "final",
    ) -> dict[str, Any]:
        """
        Creates the request body for the Search Console API.
        Args:
            start_date:
                Report start date.
            end_date:
                Report end date.
            dimensions:
                Dimensions included in the report.
            dimension_filter_groups:
                Optional dimension filters.
            search_type:
                Search result type.
            data_state:
                Requested data state.

        Returns:
            Search Console request body.
        """
        query: dict[str, Any] = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions,
            "rowLimit": self.row_limit,
            "startRow": 0,
            "type": search_type,
            "dataState": data_state,
        }

        if dimension_filter_groups:
            query["dimensionFilterGroups"] = dimension_filter_groups

        return query

    def _get_report_raw(
        self,
        property_uri: str,
        query: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Executes a Search Console API request with retry logic.

        Args:
            property_uri:
                Search Console property URI.
            query:
                Search Console request body.
        Returns:
            Raw Search Console API response.
        Raises:
            HttpError:
                If the API returns a non-recoverable error or the maximum
                retry count is exceeded.
        """
        max_retries: int = 5
        retry_count: int = 0

        while True:
            try:
                response: dict[str, Any] = (
                    self.service.searchanalytics()
                    .query(
                        siteUrl=property_uri,
                        body=query,
                    )
                    .execute()
                )

                return response

            except HttpError as error:
                status_code: int = error.resp.status
                reason: str = error._get_reason()

                if status_code not in {429, 500, 502, 503, 504}:
                    self.logger.critical(
                        f"Non-recoverable error {status_code}: {reason}"
                    )
                    raise

                if retry_count >= max_retries:
                    self.logger.critical(
                        f"Error {status_code} ({reason}): maximum retries exceeded."
                    )
                    raise

                sleep_time: float = (2**retry_count) + random.uniform(0, 1)

                self.logger.info(
                    f"Error {status_code}. "
                    f"Retry {retry_count + 1}/{max_retries}. "
                    f"Waiting {sleep_time:.2f} seconds."
                )

                time.sleep(sleep_time)
                retry_count += 1

    def _get_paginated_report(
        self,
        property_uri: str,
        query: dict[str, Any],
        dimensions: list[str],
    ) -> pd.DataFrame:
        """
        Retrieves all available rows using startRow pagination.

        Args:
            property_uri:
                Search Console property URI.

            query:
                Search Console request body.

            dimensions:
                Dimensions included in the report.

        Returns:
            Complete Search Console report as a pandas DataFrame.
        """
        all_dataframes: list[pd.DataFrame] = []
        start_row: int = 0

        while True:
            paginated_query: dict[str, Any] = copy.deepcopy(query)
            paginated_query["startRow"] = start_row
            paginated_query["rowLimit"] = self.row_limit

            self.logger.info(
                f"Querying from row {start_row} with limit {self.row_limit}"
            )

            response: dict[str, Any] = self._get_report_raw(
                property_uri=property_uri,
                query=paginated_query,
            )

            page_df: pd.DataFrame = self._to_df(
                raw_server_response=response,
                dimensions=dimensions,
            )

            if page_df.empty:
                break

            all_dataframes.append(page_df)

            rows_received: int = len(page_df)

            if rows_received < self.row_limit:
                break

            start_row += self.row_limit

        if not all_dataframes:
            self.logger.info("No data found for the specified period.")
            return self._empty_df(dimensions)

        result_df: pd.DataFrame = pd.concat(
            all_dataframes,
            ignore_index=True,
        )

        self.logger.info(f"Total rows obtained: {len(result_df)}")

        return result_df

    def _to_df(
        self,
        raw_server_response: dict[str, Any],
        dimensions: list[str],
    ) -> pd.DataFrame:
        """
        Transforms a raw Search Console response into a DataFrame.

        Args:
            raw_server_response:
                Raw response returned by Search Console.

            dimensions:
                Dimensions requested in the report.

        Returns:
            Search Console report as a pandas DataFrame.
        """
        rows: list[dict[str, Any]] = raw_server_response.get(
            "rows",
            [],
        )

        if not rows:
            return self._empty_df(dimensions)

        results: list[dict[str, Any]] = []

        for row in rows:
            keys: list[str] = row.get("keys", [])

            record: dict[str, Any] = {
                dimension: keys[index] if index < len(keys) else None
                for index, dimension in enumerate(dimensions)
            }

            record.update(
                {
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                    "ctr": row.get("ctr", 0),
                    "position": row.get("position", 0),
                }
            )

            results.append(record)

        return pd.DataFrame(results)

    def _empty_df(
        self,
        dimensions: list[str],
    ) -> pd.DataFrame:
        """
        Creates an empty DataFrame with the expected report columns.

        Args:
            dimensions:
                Dimensions requested in the report.

        Returns:
            Empty pandas DataFrame.
        """
        columns: list[str] = dimensions + [
            "clicks",
            "impressions",
            "ctr",
            "position",
        ]

        return pd.DataFrame(columns=columns)

    def _validate_report_parameters(
        self,
        property_uri: str,
        start_date: str,
        end_date: str,
        dimensions: list[str],
    ) -> None:
        """
        Validates the main Search Console report parameters.

        Args:
            property_uri:
                Search Console property URI.
            start_date:
                Report start date.
            end_date:
                Report end date.
            dimensions:
                Dimensions requested in the report.

        Raises:
            ValueError:
                If a required parameter is missing or invalid.
        """
        if not property_uri:
            raise ValueError("property_uri is required")

        if not dimensions:
            raise ValueError("dimensions is required")

        try:
            parsed_start_date: datetime = datetime.strptime(
                start_date,
                "%Y-%m-%d",
            )

            parsed_end_date: datetime = datetime.strptime(
                end_date,
                "%Y-%m-%d",
            )

        except ValueError as error:
            raise ValueError(
                "start_date and end_date must use YYYY-MM-DD format"
            ) from error

        if parsed_start_date > parsed_end_date:
            raise ValueError("start_date cannot be greater than end_date")

    @staticmethod
    def _build_default_logger() -> logging.Logger:
        logger = logging.getLogger("GoogleSearchConsole")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(console_handler)

        return logger
