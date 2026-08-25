"""LinkedIn Community Management API — marketing data extraction.


Provides a clean DataFrame-based interface for the team, with raw
dict-returning methods (prefixed with _) available for debugging.
"""

import pandas as pd
import json
import requests
import time
from datetime import UTC, datetime

from urllib.parse import quote
from typing import Optional
import logging


class QuotaExhaustedError(Exception):
    """Raised when LinkedIn returns 429 due to daily quota exhaustion.

    Unlike transient rate limits, this resets at midnight UTC and
    retrying immediately is pointless.
    """

    pass


class LinkedinMarketing:
    """Abstraction layer for the LinkedIn Marketing Reporting API.

    Handles OAuth authentication, advertising analytics reporting,
    report pivoting, and campaign metadata enrichment.

    Atributes:
        token_path: Path to JSON file containing access token.

    Author: D2B Data Engineering Team.
    Version: 2.0.0
    license: Propietary / Internal Use only.
    """

    def __init__(
        self,
        token_path: Optional[str] = None,
        logger: Optional[object] = None,
    ) -> None:
        """Initialize the LinkedinMarketing client,

        Args:

        """
        self.logger = logger or self._build_default_logger()
        self.linkedin_version = "202607"
        self.logger.info(
            f"--- EXECUTING: LinkedinMarketing, Linkedin-Version:{self.linkedin_version}"
            f"- Initialized at {datetime.now(UTC).isoformat()}---"
        )
        self.token_path: Optional[str] = token_path or None
        self.headers = None
        self.token: Optional[str] = None

        if token_path:
            self._load_token_from_file()
            if self.token:
                self._set_headers()
                self.logger.info(
                    f"LinkedinMarketing instantiated with token from {self.token_path}."
                )
            else:
                self.logger.info(f"No token found in {self.token_path}.")
        else:
            self.logger.info("Token file not specified. set_token() to load a token.")

    @staticmethod
    def _build_default_logger() -> object:
        """Build a stdlib-based fallback logger that matches the verbose interface.

        Returns:
            An object with .info() and .critical() methods.
        """

        logger = logging.getLogger("LinkedinMarketing")
        logger.setLevel(logging.INFO)

        class _StdlibAdapter:
            def info(self, message: str) -> None:
                logger.info(message)

            def critical(self, message: str) -> None:
                logger.error(message)

            def debug(self, message: str) -> None:
                logger.debug(message)

        return _StdlibAdapter()

    def _load_token_from_file(self) -> Optional[dict]:
        """Read token JSON from disk and set self.token.

        Any previously loaded credential is discarded first, so a failed
        read never leaves the client authenticated with a stale token.
        """
        self.token = None

        try:
            with open(self.token_path, "r") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.logger.info(f"Error loading token from file: {exc}")
            return None

        if "access_token" not in data:
            self.logger.critical("Token file is missing 'access_token' field.")
            return None

        self.token = data["access_token"]
        return data

    def _set_headers(self) -> None:
        """Build the headers dict required by the LinkedIn REST API."""
        if not self.token:
            self.headers = None
            self.logger.critical("Cannot set headers: access token is missing.")
            return

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Linkedin-Version": self.linkedin_version,
            "Content-Type": "application/json",
        }
        self.logger.info("LinkedIn API headers configured.")

    def _request_get(self, url: str, max_retries: int = 3) -> dict:
        """Execute an authenticated GET and return the parsed JSON.

        Retries on transient server errors (500, 502, 503) with
        exponential backoff. Raises QuotaExhaustedError on 429
        without retrying — LinkedIn daily quotas reset at midnight
        UTC, so retrying is pointless.

        Args:
            url: Fully-formed LinkedIn API URL.
            max_retries: Number of retry attempts for transient errors.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            QuotaExhaustedError: On 429 (daily quota exceeded).
            requests.exceptions.RequestException: On non-retryable
                HTTP errors.
        """
        if not self.headers:
            raise RuntimeError("Headers not set. Authenticate first.")

        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")

        transient_codes = {500, 502, 503}

        session = requests.Session()
        prepared = requests.Request("GET", url, headers=self.headers).prepare()
        prepared.url = url  # override to prevent re-encoding

        for attempt in range(max_retries + 1):
            response = session.send(prepared)

            if response.status_code == 429:
                raise QuotaExhaustedError(
                    "LinkedIn daily quota exhausted (429). "
                    "Resets at midnight UTC. "
                    f"URL: {url[:80]}..."
                )

            if response.status_code in transient_codes and attempt < max_retries:
                wait = 2**attempt
                self.logger.info(
                    f"Transient {response.status_code}, "
                    f"retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

    def _fetch_report(self, url: str) -> list[dict]:
        """Fetch simple not-paginated response for a LinkedIn analytics report.

        Args:
            url: Fully formed LinkedIn Analytics API URL.

        Returns:
            Report rows collected across all fetched pages.
        """
        page_url = f"{url}&count={page_size}&start={start_index}"

        try:
            data = self._request_get(page_url)

        except requests.exceptions.RequestException as exc:
            self.logger.critical(f"LinkedIn API Error during pagination: {exc}")
            raise

        elements = data.get("elements")
        self.logger.info(f"Retrieved {len(elements)} rows.")

    return elements

    def _fetch_paginated_report(self, url: str) -> list[dict]:
        """Fetch all available pages for a LinkedIn analytics report.

        Args:
            url: Fully formed LinkedIn Analytics API URL.

        Returns:
            Report rows collected across all fetched pages.
        """
        collected: list[dict] = []
        start_index = 0
        page_size = 50

        while True:
            page_url = f"{url}&count={page_size}&start={start_index}"

            try:
                data = self._request_get(page_url)
            except requests.exceptions.RequestException as exc:
                self.logger.critical(f"LinkedIn API Error during pagination: {exc}")
                raise

            elements = data.get("elements")
            if not elements:
                self.logger.info("No more pages available.")
                break

            collected.extend(elements)

            self.logger.info(
                f"Processing batch of {len(elements)} rows (offset {start_index})..."
            )

            if len(elements) < page_size:
                self.logger.info("No more pages available.")
                break

            start_index += page_size

        return collected

    def _get_campaign_names(
        self, campaign_ids: set[str], account_id: str
    ) -> dict[str, str]:
        """Extracts campaign names for a campaign name id list.

        Args:
            campaign_ids: A list of campaign ids extracted with get_report.
            account_id: A string containing the account id for the queried account.

        Return
        """
        if not campaign_ids:
            self.logger.info("No campaign id's provided. returning empty dict")
            return {}

        campaign_ids_str = ",".join(campaign_ids)
        url = f"https://api.linkedin.com/rest/adAccounts/{account_id}/adCampaigns?ids=List({campaign_ids_str})"

        res = self._request_get(url)
        self.logger.debug("Campaign name information retrieved")

        results = res.get("results", {})
        if not results:
            self.logger("No campaign information returned.")
            return {}

        self.logger.debug(f"This is raw results: {results}")

        campaign_name_map = {}

        for key, value in results.items():
            campaign_name_map[key] = value["name"]

        return campaign_name_map

    def _get_campaign_group_names(
        self, campaign_group_ids: set[str], account_id: str
    ) -> dict[str, str]:
        """Extracts campaign group names for a campaign group id list

        Args:
           campaign_group_ids: a list containing the campaign group ids to be mapped.
           account_id: A string containing the account id for the queried account.

        Return:
            A dictionary containing the ids mapped to campaign group names.
        """
        if not campaign_group_ids:
            self.logger.info("No campaign id's provided. returning empty dict")
            return {}

        campaign_group_str = ",".join(campaign_group_ids)
        url = f"https://api.linkedin.com/rest/adAccounts/{account_id}/adCampaignGroups?ids=List({campaign_group_str})"

        res = self._request_get(url)
        self.logger.debug("Campaign name information retrieved")

        results = res.get("results", {})
        if not results:
            self.logger("No campaign information returned.")
            return {}

        self.logger.debug(f"This is raw results: {results}")

        campaign_group_name_map = {}

        for key, value in results.items():
            campaign_group_name_map[key] = value["name"]

        return campaign_group_name_map

    def set_token(self, token_path: str) -> None:
        """Load an access token from a file and update the request headers."""

        self.token_path = token_path
        self._load_token_from_file()
        self._set_headers()

    def get_report(
        self,
        account_id: str,
        start: str,
        end: str,
        metrics: list[str],
        pivot: list[str] = None,
        time_granularity: str = "DAILY",
    ) -> list[dict]:
        """Fetch an analytics report from the LinkedIn Marketing API.

        Args:
            account_id: LinkedIn sponsored account ID.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            metrics: Metrics to include in the report.
            pivot: Dimension used to group the report.
            time_granularity: Time granularity for the report.

        Returns:
            Report rows returned by the LinkedIn API.

        Raises:
            ValueError: If pivot is not provided or the dates are malformed.
            requests.exceptions.RequestException: If the API request fails.
        """
        if not pivot:
            raise ValueError("pivot is required for statistics reports")

        if len(pivot) > 3:
            raise ValueError("Only 3 pivot values can be passed for each query")

        if len(metrics) > 20:
            raise ValueError("Only 20 metrics can be passed for each query")

        metrics = metrics.copy()

        for required in ("dateRange", "pivotValues"):
            if required not in metrics:
                metrics.append(required)

        urn_encoded = quote(f"urn:li:sponsoredAccount:{account_id}")
        accounts = f"List({urn_encoded})"

        try:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"start and end must be YYYY-MM-DD dates (got {start!r}, {end!r})"
            ) from exc

        if start_date > end_date:
            raise ValueError(f"start ({start}) must not be after end ({end})")

        date_range = (
            f"(start:(year:{start_date.year},month:{start_date.month},"
            f"day:{start_date.day}),"
            f"end:(year:{end_date.year},month:{end_date.month},"
            f"day:{end_date.day}))"
        )

        pivot_values = ",".join(pivot)
        fields = ",".join(metrics)

        url = (
            f"https://api.linkedin.com/rest/adAnalytics"
            f"?q=statistics"
            f"&pivots=List({pivot_values})"
            f"&timeGranularity={time_granularity}"
            f"&dateRange={date_range}"
            f"&accounts={accounts}"
            f"&fields={fields}"
        )

        self.logger.info(f"GET analytics report for account: {account_id}")

        try:
            data = self._fetch_report(url)
            self.logger.info(f"Data extraction successfull: {len(data)} rows")

        except requests.exceptions.RequestException as exc:
            self.logger.critical(f"LinkedIn API Error: {exc}")
            raise

        return data

    def get_report_dataframe(
        self,
        account_id: str,
        start: str,
        end: str,
        metrics: list[str],
        pivot: list[str],
        time_granularity: str = "DAILY",
        get_campaign_information: bool = True,
    ) -> pd.DataFrame:
        """Fetch and transform a LinkedIn analytics report into a DataFrame.

        Retrieves report data using get_report() and normalizes the response
        into a pandas DataFrame. Optionally enriches campaign data with campaign
        and campaign group IDs and names, and normalizes the report date.

        Args:
            account_id: LinkedIn sponsored account ID.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            metrics: Metrics to include in the report.
            pivot: Dimensions used to group the report.
            time_granularity: Time granularity for the report. Defaults to "DAILY".
            get_campaign_information: Whether to enrich the report with campaign
                and campaign group information. Defaults to True.

        Returns:
            A normalized and optionally enriched pandas DataFrame.
        """
        if get_campaign_information:
            raw_data = self.get_report(
                account_id,
                start,
                end,
                metrics,
                pivot,
                time_granularity,
            )

            self.logger.debug("Getting campaign name information")

            campaign_names_ids = set()
            campaign_group_names_ids = set()

            for row in raw_data:
                pivot_values = row.get("pivotValues")

                if pivot_values:
                    campaign_name_id = pivot_values[1].split(":")[3]
                    campaign_group_id = pivot_values[0].split(":")[3]

                    campaign_names_ids.add(campaign_name_id)
                    campaign_group_names_ids.add(campaign_group_id)

            try:
                campaign_name_map = self._get_campaign_names(
                    campaign_names_ids, account_id
                )
                self.logger.debug(f"Campaign name maps: {campaign_name_map}")
                campaign_group_name_map = self._get_campaign_group_names(
                    campaign_group_names_ids, account_id
                )
                self.logger.debug(
                    f"Campaign group name maps: {campaign_group_name_map}"
                )

                self.logger.debug("Campaign name and group information retrieved")
            except Exception as e:
                self.logger.critical(f"Error during campaign name extraction: {e}")
                raise

            df = pd.json_normalize(raw_data, sep="_")

            self.logger.debug("Starting DataFrame transformation...")

            df["campaign_group_id"] = df["pivotValues"].apply(
                lambda x: x[0].split(":")[3]
            )
            df["campaign_id"] = df["pivotValues"].apply(lambda x: x[1].split(":")[3])

            df["campaign_group_name"] = df["pivotValues"].apply(
                lambda x: campaign_group_name_map.get(x[0].split(":")[3])
            )

            df["campaign_name"] = df["pivotValues"].apply(
                lambda x: campaign_name_map.get(x[1].split(":")[3])
            )

            df["date"] = pd.to_datetime(
                {
                    "year": df["dateRange_start_year"],
                    "month": df["dateRange_start_month"],
                    "day": df["dateRange_start_day"],
                }
            )

            columns_to_drop = [
                "dateRange_start_year",
                "dateRange_start_month",
                "dateRange_start_day",
                "dateRange_end_year",
                "dateRange_end_month",
                "dateRange_end_day",
                "pivotValues",
            ]

            df = df.drop(columns=columns_to_drop)

            return df

        raw_data = self.get_report(
            account_id,
            start,
            end,
            metrics,
            pivot,
            time_granularity,
        )
        return pd.json_normalize(raw_data, sep="_")
