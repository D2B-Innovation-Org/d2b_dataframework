"""LinkedIn Community Management API — marketing data extraction.


Provides a clean DataFrame-based interface for the team, with raw
dict-returning methods (prefixed with _) available for debugging.
"""

import pandas as pd
import json
import requests
import time
from datetime import UTC, datetime, timezone, timedelta

from os.path import isfile
from requests.structures import CaseInsensitiveDict
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
            self.logger.info(
                "Token file not specified. set__token to save a new token."
            )

    @staticmethod
    def _build_default_logger() -> object:
        """Build a stdlib-based fallback logger that matches the verbose interface.

        Returns:
            An object with .info() and .critical() methods.
        """

        logger = logging.getLogger("LinkedinMarketing")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

        class _StdlibAdapter:
            def info(self, message: str) -> None:
                logger.info(message)

            def critical(self, message: str) -> None:
                logger.error(message)

        return _StdlibAdapter()

    def _load_token_from_file(self) -> Optional[dict]:
        """Read token JSON from disk and set self.token."""
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
                f"Processing batch of {len(elements)} posts (offset {start_index})..."
            )

            if page_size > len(elements):
                self.logger.info("No more pages available.")
                break

            start_index += page_size

        return collected

    def set_token(self, token_path: str) -> None:
        """Calls private method _load_token_from_file"""
        self.token_path = token_path
        self._load_token_from_file()
        self._set_headers()

    def get_report(
        self,
        account_id: str,
        start: str,
        end: str,
        metrics: str,
        pivot: Optional[str] = None,
        time_granularity: str = "DAILY",
    ) -> list[dict]:
        """Fetch an analytics report from the LinkedIn Marketing API.

        Args:
            account_id: LinkedIn sponsored account ID.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            metrics: Comma-separated metrics to include in the report.
            pivot: Dimension used to group the report.
            time_granularity: Time granularity for the report.

        Returns:
            Report rows returned by the LinkedIn API.

        Raises:
            ValueError: If pivot is not provided.
            requests.exceptions.RequestException: If the API request fails.
        """
        if not pivot:
            raise ValueError("pivot is required for statistics reports")

        urn_encoded = quote(f"urn:li:sponsoredAccount:{account_id}")
        accounts = f"List({urn_encoded})"
        start_parts = start.split("-")
        end_parts = end.split("-")

        start_year, start_month, start_day = start_parts
        end_year, end_month, end_day = end_parts

        date_range = (
            f"(start:(year:{start_year},month:{int(start_month)},day:{int(start_day)}),"
            f"end:(year:{end_year},month:{int(end_month)},day:{int(end_day)}))"
        )

        pivot_values = ",".join(val.strip() for val in pivot.split(","))

        fields = f"{metrics},pivotValues"

        url = (
            f"https://api.linkedin.com/rest/adAnalytics"
            f"?q=statistics"
            f"&pivots=List({pivot_values})"
            f"&timeGranularity={time_granularity}"
            f"&dateRange={date_range}"
            f"&accounts={accounts}"
            f"&fields={fields}"
        )

        self.logger.info(f"GET campaign information for org: {account_id}")

        try:
            data = self._fetch_paginated_report(url)
            self.logger.info(f"Data extraction successfull")
            return data
        except requests.exceptions.RequestException as exc:
            self.logger.critical(f"LinkedIn API Error{exc}")
            raise

    def get_report_dataframe(
        self, account_id, start, end, metrics, pivot=None, time_granularity="DAILY"
    ) -> pd.DataFrame:
        """Gets data from get_report and transforms to pd.DataFrame."""

        raw_data = self.get_report(
            account_id,
            start,
            end,
            metrics,
            pivot,
            time_granularity,
        )
        self.logger.info(f"Data extraction successfull")
        return pd.json_normalize(raw_data, sep="_")

    def get_campaign_names(self, campaign_ids):
        """Dev Pending"""
        pass

    def get_campaign_group_names(self, group_ids):
        """Dev Pending"""
        pass
