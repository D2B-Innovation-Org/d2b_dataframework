"""TikTok API for Business (TikTok Accounts) - organic account data extraction.

Produces three raw, analyst-friendly tables per TikTok Business account:

* daily   -> ``get_daily_stats``: one row per day with the account metrics
             TikTok reports daily (views, profile views, followers, clicks...).
* videos  -> ``get_videos``: one row per video with lifetime metrics as of
             the extraction (snapshot).
* profile -> ``get_profile_snapshot``: one row with the profile counters and
             the audience breakdowns (snapshot).

Data is extracted raw: metrics TikTok derives from other fields (averages,
engagement or growth rates) are not requested, and nested lists (audience
breakdowns, impression sources, retention curves) are kept as JSON strings.

API limits that shape the design:

* the access token lasts 24 hours and every refresh returns a new refresh
  token, so callers must refresh and persist the token file on each run;
* daily metrics are only served for the last 60 days;
* ``/business/get/`` returns all daily rows of a range in one call and
  ``/business/video/list/`` pages 20 videos per call.
"""

import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests
from d2b_data.workflow_logger import resolve_logger

API_BASE = "https://business-api.tiktok.com/open_api/v1.3/"
TOKEN_URL = API_BASE + "tt_user/oauth2/refresh_token/"
REQUEST_TIMEOUT = 60  # seconds
VIDEOS_PAGE_SIZE = 20  # max_count allowed by /business/video/list/
MAX_DAYS_BACK = 60  # /business/get/ rejects older start dates

# TikTok answers HTTP 200 with an error code in the body. 40100 is the
# rate limit; 5xxxx are internal errors worth retrying.
RATE_LIMIT_CODES = {40100}
TRANSIENT_HTTP = {500, 502, 503, 504}

# /business/get/ fields returned inside metrics[] (one item per day).
DAILY_FIELDS = [
    "video_views",
    "unique_video_views",
    "profile_views",
    "likes",
    "comments",
    "shares",
    "followers_count",
    "daily_new_followers",
    "daily_lost_followers",
    "daily_total_followers",
    "engaged_audience",
    "audience_activity",
    "bio_link_clicks",
    "email_clicks",
    "phone_number_clicks",
    "address_clicks",
    "app_download_clicks",
    "lead_submissions",
]

# /business/get/ account-level fields. Derived metrics (average_*,
# engagement_rate, followers_growth_rate, completion_rate) and the expiring
# profile_image URL are left out on purpose.
PROFILE_FIELDS = [
    "username",
    "display_name",
    "bio_description",
    "profile_deep_link",
    "is_business_account",
    "is_verified",
    "has_advanced_access",
    "followers_count",
    "following_count",
    "videos_count",
    "total_likes",
    "audience_countries",
    "audience_cities",
    "audience_genders",
    "audience_ages",
]

# /business/video/list/ fields. average_time_watched (total_time_watched /
# video_views) and the expiring thumbnail_url are left out on purpose.
VIDEO_FIELDS = [
    "item_id",
    "create_time",
    "media_type",
    "caption",
    "share_url",
    "embed_url",
    "is_ad",
    "is_ai_generated",
    "video_duration",
    "video_views",
    "reach",
    "likes",
    "comments",
    "shares",
    "favorites",
    "new_followers",
    "profile_views",
    "total_time_watched",
    "full_video_watched_rate",
    "website_clicks",
    "email_clicks",
    "phone_number_clicks",
    "address_clicks",
    "app_download_clicks",
    "lead_submissions",
    "engagement_likes",
    "impression_sources",
    "audience_types",
    "audience_countries",
    "audience_cities",
    "audience_genders",
    "video_view_retention",
]


class QuotaExhaustedError(Exception):
    """Raised when TikTok throttles the app (HTTP 429 or code 40100).

    Retrying right away only burns more of the quota, so callers must stop.
    """


class TikTokApiError(Exception):
    """Raised when TikTok answers with a non-zero error code.

    Attributes:
        code: TikTok error code from the response body.
    """

    def __init__(self, code: int, message: str, endpoint: str) -> None:
        super().__init__(f"TikTok API error {code} on {endpoint}: {message}")
        self.code = code


def _to_json(value: Any) -> Optional[str]:
    """Serialize nested lists/dicts to a JSON string; leave scalars as is."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


class TikTokOrganic:
    """Extract organic account data from the TikTok API for Business.

    The token file must hold the ``data`` object returned by
    ``/tt_user/oauth2/token/`` (``access_token``, ``refresh_token``,
    ``open_id``, ...). ``open_id`` identifies the authorized account and is
    sent as ``business_id`` on every call.

    Attributes:
        token_path: Path to the token JSON file.
        token: Current access token, or None if it could not be loaded.
        open_id: TikTok account identifier from the token file.
    """

    def __init__(
        self,
        token_path: Optional[str] = None,
        logger: Optional[object] = None,
        *,
        verbose_logger=None,
    ) -> None:
        logger = resolve_logger(logger, verbose_logger, "TikTokOrganic")
        self.token_path = token_path or "token_tiktok_organic.json"
        self.logger = logger or self._build_default_logger()
        self.logger.log(
            "--- EXECUTING TikTokOrganic Class v1.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )
        self.token: Optional[str] = None
        self.open_id: Optional[str] = None
        self.session = requests.Session()

        if token_path:
            self._load_token_from_file()
            if self.token:
                self.logger.log(
                    f"TikTokOrganic instantiated with token from {self.token_path}."
                )
            else:
                self.logger.log(f"No usable token found in {self.token_path}.")
        else:
            self.logger.log(
                "Token file not specified. Pass token_path pointing to a "
                "TikTok account holder token JSON."
            )

    @staticmethod
    def _build_default_logger() -> object:
        """Build a stdlib-based fallback logger with .log() and .critical()."""
        logger = logging.getLogger("TikTokOrganic")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

        class _Adapter:
            def log(self, message: str) -> None:
                logger.info(message)

            def warning(self, message: str) -> None:
                logger.warning(message)

            def critical(self, message: str) -> None:
                logger.error(message)

        return _Adapter()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _load_token_from_file(self) -> Optional[dict]:
        """Read the token JSON from disk and set token and open_id.

        Returns:
            The token data, or None if the file is unreadable or incomplete.
        """
        try:
            with open(self.token_path, "r") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.logger.log(f"Error loading token from file: {exc}")
            return None

        missing = [k for k in ("access_token", "open_id") if not data.get(k)]
        if missing:
            self.logger.log(f"Token file is missing fields: {missing}.")
            return None

        self.token = data["access_token"]
        self.open_id = data["open_id"]
        return data

    def refresh_access_token(self, client_id: str, client_secret: str) -> bool:
        """Exchange the stored refresh_token for a new access token.

        TikTok rotates the refresh token on every refresh, so the whole
        response is written back to ``token_path``; callers must persist
        that file (e.g. as a new secret version) or the next refresh fails.
        Does not consume API quota.

        Args:
            client_id: App ID of the TikTok developer app.
            client_secret: Secret of the same app.

        Returns:
            True if the token was refreshed and saved, False otherwise.
        """
        data = self._load_token_from_file() or {}
        if not data.get("refresh_token"):
            self.logger.critical("No refresh_token available in token file.")
            return False

        try:
            resp = self.session.post(
                TOKEN_URL,
                json={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": data["refresh_token"],
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self.logger.critical(f"TikTok token refresh failed: {exc}")
            return False

        new = body.get("data") or {}
        if body.get("code") != 0 or not new.get("access_token"):
            self.logger.critical(
                f"TikTok token refresh failed: code={body.get('code')} "
                f"message={body.get('message')!r}"
            )
            return False

        # Keep fields TikTok may omit on refresh (e.g. open_id) from the old file.
        merged = {**data, **new}
        with open(self.token_path, "w") as fh:
            json.dump(merged, fh)

        self.token = merged["access_token"]
        self.open_id = merged["open_id"]
        self.logger.log("TikTok access token refreshed and saved.")
        return True

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _request_get(
        self, endpoint: str, params: dict[str, Any], max_retries: int = 3
    ) -> dict:
        """Execute an authenticated GET and return the ``data`` object.

        List parameters are sent JSON-encoded, as TikTok expects. Transient
        HTTP errors are retried with exponential backoff.

        Args:
            endpoint: Path under API_BASE, e.g. 'business/get/'.
            params: Query parameters; ``business_id`` is added automatically.
            max_retries: Retry attempts for transient HTTP errors.

        Returns:
            The ``data`` object of the response (empty dict if absent).

        Raises:
            RuntimeError: If no token is loaded.
            QuotaExhaustedError: On HTTP 429 or TikTok rate-limit codes.
            TikTokApiError: On any other non-zero TikTok code.
            requests.exceptions.RequestException: On non-retryable HTTP errors.
        """
        if not self.token or not self.open_id:
            raise RuntimeError("TikTok token not loaded. Authenticate first.")

        query = {"business_id": self.open_id}
        for key, value in params.items():
            query[key] = json.dumps(value) if isinstance(value, list) else value

        for attempt in range(max_retries + 1):
            response = self.session.get(
                API_BASE + endpoint,
                params=query,
                headers={"Access-Token": self.token},
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                raise QuotaExhaustedError(f"TikTok throttled {endpoint} (HTTP 429).")

            if response.status_code in TRANSIENT_HTTP and attempt < max_retries:
                wait = 2**attempt
                self.logger.log(
                    f"Transient {response.status_code} on {endpoint}, retrying in "
                    f"{wait}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            body = response.json()
            code = body.get("code")
            if code in RATE_LIMIT_CODES:
                raise QuotaExhaustedError(
                    f"TikTok rate limit on {endpoint}: {body.get('message')}"
                )
            if code != 0:
                raise TikTokApiError(code, str(body.get("message")), endpoint)
            return body.get("data") or {}

        # Only reached if the last attempt was transient and retries ran out.
        response.raise_for_status()
        return {}

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def get_daily_stats(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Daily account metrics, one row per day.

        TikTok rejects start dates more than 60 days back, so an older
        ``start_date`` is moved to the oldest allowed day (with a warning).

        Args:
            start_date: First day, 'YYYY-MM-DD' (UTC).
            end_date: Last day, 'YYYY-MM-DD' (UTC), inclusive.

        Returns:
            DataFrame [date (datetime.date), open_id, <DAILY_FIELDS>] sorted
            by date; ``audience_activity`` is a JSON string. Empty DataFrame
            with those columns when TikTok returns no days.

        Raises:
            ValueError: If the dates are malformed or start_date > end_date.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("start_date must be <= end_date")

        oldest = _today() - timedelta(days=MAX_DAYS_BACK - 1)
        if start < oldest:
            self.logger.log(
                f"start_date {start} is beyond TikTok's {MAX_DAYS_BACK}-day "
                f"window; using {oldest} instead."
            )
            start = oldest
            if start > end:
                return pd.DataFrame(columns=["date", "open_id", *DAILY_FIELDS])

        data = self._request_get(
            "business/get/",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "fields": DAILY_FIELDS,
            },
        )
        rows = data.get("metrics") or []
        if not rows:
            return pd.DataFrame(columns=["date", "open_id", *DAILY_FIELDS])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.insert(1, "open_id", self.open_id)
        if "audience_activity" in df.columns:
            df["audience_activity"] = df["audience_activity"].map(_to_json)
        others = [c for c in df.columns if c not in ("date", "open_id")]
        ordered = ["date", "open_id", *others]
        return df[ordered].sort_values("date").reset_index(drop=True)

    def get_profile_snapshot(self) -> pd.DataFrame:
        """Profile counters and audience breakdowns as of now, one row.

        Returns:
            DataFrame [snapshot_date, extracted_at, open_id, <PROFILE_FIELDS>]
            with the audience_* lists as JSON strings.
        """
        data = self._request_get("business/get/", {"fields": PROFILE_FIELDS})
        now = datetime.now(UTC)
        profile = {k: v for k, v in data.items() if k != "metrics"}
        row = {key: _to_json(value) for key, value in profile.items()}
        df = pd.DataFrame([row])
        df.insert(0, "open_id", self.open_id)
        df.insert(0, "extracted_at", pd.Timestamp(now))
        df.insert(0, "snapshot_date", now.date())
        return df

    def get_videos(self) -> pd.DataFrame:
        """All videos of the account with lifetime metrics, one row each.

        Pages through ``/business/video/list/`` (20 videos per call) until
        TikTok reports no more results.

        Returns:
            DataFrame [snapshot_date, extracted_at, open_id, <VIDEO_FIELDS>];
            ``create_time`` is a UTC timestamp and nested lists are JSON
            strings. Empty DataFrame with those columns if there are no videos.
        """
        videos: list[dict] = []
        cursor = 0
        while True:
            data = self._request_get(
                "business/video/list/",
                {
                    "fields": VIDEO_FIELDS,
                    "max_count": VIDEOS_PAGE_SIZE,
                    "cursor": cursor,
                },
            )
            page = data.get("videos") or []
            videos.extend(page)
            # An empty page with has_more=true would loop forever on the same cursor.
            if not data.get("has_more") or not page:
                break
            cursor = data.get("cursor", 0)

        now = datetime.now(UTC)
        base_cols = ["snapshot_date", "extracted_at", "open_id"]
        if not videos:
            return pd.DataFrame(columns=base_cols + VIDEO_FIELDS)

        df = pd.DataFrame(videos).map(_to_json)
        if "create_time" in df.columns:
            df["create_time"] = pd.to_datetime(
                pd.to_numeric(df["create_time"]), unit="s", utc=True
            )
        df.insert(0, "open_id", self.open_id)
        df.insert(0, "extracted_at", pd.Timestamp(now))
        df.insert(0, "snapshot_date", now.date())
        self.logger.log(f"Fetched {len(df)} TikTok videos.")
        return df


def _today() -> date:
    """UTC today; isolated so tests can patch it."""
    return datetime.now(UTC).date()
