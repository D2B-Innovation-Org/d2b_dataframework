"""LinkedIn Community Management API — organic page data extraction.

Produces three raw, analyst-friendly tables per organization:

* posts     -> ``get_posts_with_statistics``: one row per post, metadata
               plus lifetime statistics (impressions, clicks, reactions...).
* daily     -> ``get_daily_stats``: one row per day with page share
               statistics, follower gains and page views.
* followers -> ``get_follower_stats``: follower snapshot, total plus every
               demographic breakdown LinkedIn exposes, in long format.

Quota is limited on restricted (development tier) apps, so every method is
designed around the minimum number of calls:

* posts are paged 100 at a time (API max) sorted by creation date, so
  paging stops as soon as a post older than ``start_date`` appears;
* post statistics are fetched in batches through
  ``organizationalEntityShareStatistics`` (one call per 50 posts) instead
  of one ``socialActions`` call per post;
* daily series use one call per source for the whole date range.

Private ``_*_raw`` methods return the API JSON for debugging.
"""

import json
import logging
import time
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd
import requests
from d2b_data.workflow_logger import resolve_logger

API_BASE = "https://api.linkedin.com/rest"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
API_VERSION = "202606"
POSTS_PAGE_SIZE = 100  # max allowed by the posts finder
STATS_BATCH_SIZE = 50  # post URNs per share-statistics call
REQUEST_TIMEOUT = 60  # seconds

# Follower breakdown key -> field inside each element holding the value.
FOLLOWER_BREAKDOWNS = {
    "followerCountsByAssociationType": "associationType",
    "followerCountsBySeniority": "seniority",
    "followerCountsByIndustry": "industry",
    "followerCountsByFunction": "function",
    "followerCountsByStaffCountRange": "staffCountRange",
    "followerCountsByGeoCountry": "geo",
    "followerCountsByGeo": "geo",
}

SHARE_STAT_FIELDS = [
    "impressionCount",
    "uniqueImpressionsCount",
    "clickCount",
    "likeCount",
    "commentCount",
    "shareCount",
    "engagement",
]


class QuotaExhaustedError(Exception):
    """Raised when LinkedIn returns 429 (throttle or daily quota).

    The daily quota resets at midnight UTC, so retrying immediately is
    pointless and only wastes calls.
    """


def _snake(name: str) -> str:
    """Convert camelCase API field names to snake_case column names."""
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out).lstrip("_")


def _ms_to_ts(ms: Optional[int]) -> Optional[pd.Timestamp]:
    """Epoch milliseconds to a UTC pandas Timestamp (None-safe)."""
    if ms is None:
        return None
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def _parse_date(value: str) -> datetime:
    """Parse 'YYYY-MM-DD' into a UTC midnight datetime."""
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class LinkedinOrganic:
    """Extract organic page data from the LinkedIn Community Management API.

    Attributes:
        token_path: Path to the JSON file containing the access token.
    """

    def __init__(
        self,
        token_path: Optional[str] = None,
        logger: Optional[object] = None,
        *,
        verbose_logger=None,
    ) -> None:
        logger = resolve_logger(logger, verbose_logger, "LinkedinOrganic")
        self.token_path = token_path or "token_linkedin_organic.json"
        self.logger = logger or self._build_default_logger()
        self.logger.log(
            "--- EXECUTING LinkedinOrganic Class v3.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )
        self.token: Optional[str] = None
        self.headers: Optional[dict[str, str]] = None
        self.session = requests.Session()

        if token_path:
            self._load_token_from_file()
            if self.token:
                self._set_headers()
                self.logger.log(
                    f"LinkedinOrganic instantiated with token from {self.token_path}."
                )
            else:
                self.logger.log(f"No token found in {self.token_path}.")
        else:
            self.logger.log(
                "Token file not specified. Pass token_path pointing to a "
                "Community Management API token JSON."
            )

    @staticmethod
    def _build_default_logger() -> object:
        """Build a stdlib-based fallback logger with .log() and .critical()."""
        logger = logging.getLogger("LinkedInOrganic")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

        class _Adapter:
            def log(self, message: str) -> None:
                logger.info(message)

            def critical(self, message: str) -> None:
                logger.error(message)

        return _Adapter()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _load_token_from_file(self) -> Optional[dict]:
        """Read token JSON from disk and set self.token."""
        try:
            with open(self.token_path, "r") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.logger.log(f"Error loading token from file: {exc}")
            return None

        if "access_token" not in data:
            self.logger.log("Token file is missing 'access_token' field.")
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
            "LinkedIn-Version": API_VERSION,
            "Content-Type": "application/json",
        }
        self.logger.log("LinkedIn API headers configured.")

    def refresh_access_token(self, client_id: str, client_secret: str) -> bool:
        """Exchange the stored refresh_token for a new access token.

        Writes the new token JSON back to ``token_path`` (keeping the
        refresh token if LinkedIn does not return a new one) and resets
        the request headers. Does not consume API quota.

        Args:
            client_id: Client ID of the Community Management app.
            client_secret: Client secret of the same app.

        Returns:
            True if the token was refreshed and saved, False otherwise.
        """
        data = self._load_token_from_file()
        if not data or not data.get("refresh_token"):
            self.logger.critical("No refresh_token available in token file.")
            return False

        try:
            resp = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": data["refresh_token"],
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            self.logger.critical(f"Token refresh failed: {exc}")
            return False

        new = resp.json()
        new.setdefault("refresh_token", data["refresh_token"])
        with open(self.token_path, "w") as fh:
            json.dump(new, fh)

        self.token = new["access_token"]
        self._set_headers()
        self.logger.log("Access token refreshed and saved.")
        return True

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _request_get(self, url: str, max_retries: int = 3) -> dict:
        """Execute an authenticated GET and return the parsed JSON.

        Retries transient server errors (500, 502, 503) with exponential
        backoff. Raises QuotaExhaustedError on 429 without retrying.

        Args:
            url: Fully-formed LinkedIn API URL (already encoded; Rest.li
                List()/tuple syntax must reach the server untouched).
            max_retries: Number of retry attempts for transient errors.

        Raises:
            QuotaExhaustedError: On 429.
            requests.exceptions.RequestException: On non-retryable errors.
        """
        if not self.headers:
            raise RuntimeError("Headers not set. Authenticate first.")

        transient_codes = {500, 502, 503}

        # PreparedRequest so requests does not re-encode Rest.li syntax.
        prepared = requests.Request("GET", url, headers=self.headers).prepare()
        prepared.url = url

        for attempt in range(max_retries + 1):
            response = self.session.send(prepared, timeout=REQUEST_TIMEOUT)

            if response.status_code == 429:
                raise QuotaExhaustedError(
                    "LinkedIn quota exhausted or throttled (429). "
                    "Daily quota resets at midnight UTC. "
                    f"URL: {url[:80]}..."
                )

            if response.status_code in transient_codes and attempt < max_retries:
                wait = 2**attempt
                self.logger.log(
                    f"Transient {response.status_code}, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        response.raise_for_status()
        return response.json()

    def _safe_get(self, url: str, what: str) -> Optional[dict]:
        """_request_get that logs and returns None on HTTP errors.

        QuotaExhaustedError is not caught: callers must stop, not continue
        burning calls.
        """
        try:
            return self._request_get(url)
        except requests.exceptions.RequestException as exc:
            self.logger.critical(f"LinkedIn API Error ({what}): {exc}")
            return None

    @staticmethod
    def _org_urn(org_id: str) -> str:
        return quote(f"urn:li:organization:{org_id}")

    @staticmethod
    def _time_intervals(start_date: str, end_date: str) -> str:
        """Rest.li DAY timeIntervals covering start..end (both inclusive)."""
        start = _to_ms(_parse_date(start_date))
        end = _to_ms(_parse_date(end_date) + timedelta(days=1))
        return f"(timeRange:(start:{start},end:{end}),timeGranularityType:DAY)"

    # ------------------------------------------------------------------
    # RAW (private) — return API dicts, useful for debugging
    # ------------------------------------------------------------------

    def _get_managed_pages_raw(self, role: Optional[str] = None) -> Optional[dict]:
        """Organizations the token can access (1 call)."""
        url = f"{API_BASE}/organizationAcls?q=roleAssignee"
        if role:
            url += f"&role={role}"
        self.logger.log("GET managed organizations...")
        return self._safe_get(url, "organizationAcls")

    def _get_follower_stats_raw(self, org_id: str) -> Optional[dict]:
        """Lifetime follower breakdowns (1 call)."""
        url = (
            f"{API_BASE}/organizationalEntityFollowerStatistics"
            f"?q=organizationalEntity&organizationalEntity={self._org_urn(org_id)}"
        )
        self.logger.log(f"GET follower stats for org {org_id}...")
        return self._safe_get(url, "followerStatistics")

    def _get_follower_total_raw(self, org_id: str) -> Optional[dict]:
        """Total follower count via networkSizes (1 call).

        Breakdowns only count members with that attribute set, so their sum
        undercounts; this is the real total shown on the page.
        """
        url = (
            f"{API_BASE}/networkSizes/{self._org_urn(org_id)}"
            "?edgeType=COMPANY_FOLLOWED_BY_MEMBER"
        )
        self.logger.log(f"GET follower total for org {org_id}...")
        return self._safe_get(url, "networkSizes")

    def _fetch_paginated_posts(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[list[dict]]:
        """Page through posts (newest created first) within a date range.

        Both dates are inclusive and compared against ``createdAt``.
        ``sortBy=CREATED`` guarantees order, so paging stops at the first
        post older than ``start_date``.

        Returns:
            List of raw post dicts, or None on date-parse or API failure.
        """
        try:
            start_dt = _parse_date(start_date)
            end_excl = _parse_date(end_date) + timedelta(days=1)
        except ValueError as exc:
            self.logger.critical(f"Date format error (use YYYY-MM-DD): {exc}")
            return None

        collected: list[dict] = []
        start_index = 0
        self.logger.log(
            f"Fetching posts for org {org_id} between {start_date} and {end_date}..."
        )

        while True:
            url = (
                f"{API_BASE}/posts?q=author&author={self._org_urn(org_id)}"
                f"&count={POSTS_PAGE_SIZE}&start={start_index}&sortBy=CREATED"
            )
            data = self._safe_get(url, "posts")
            if data is None:
                return None

            elements = data.get("elements", [])
            if not elements:
                break

            stop_paging = False
            for post in elements:
                created_ms = post.get("createdAt")
                if not created_ms:
                    continue
                post_dt = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
                if post_dt >= end_excl:
                    continue
                if post_dt >= start_dt:
                    collected.append(post)
                else:
                    stop_paging = True
                    break

            if stop_paging or len(elements) < POSTS_PAGE_SIZE:
                break
            start_index += POSTS_PAGE_SIZE

        self.logger.log(f"Collected {len(collected)} posts in date range.")
        return collected

    def _get_post_statistics_raw(
        self, org_id: str, post_urns: list[str]
    ) -> Optional[list[dict]]:
        """Lifetime share statistics for posts, batched (1 call per 50).

        Returns:
            List of statistics elements, or None if any batch fails.
        """
        elements: list[dict] = []
        for i in range(0, len(post_urns), STATS_BATCH_SIZE):
            batch = post_urns[i : i + STATS_BATCH_SIZE]
            shares = [u for u in batch if u.startswith("urn:li:share:")]
            ugcs = [u for u in batch if u.startswith("urn:li:ugcPost:")]
            params = ""
            if shares:
                params += "&shares=List(" + ",".join(map(quote, shares)) + ")"
            if ugcs:
                params += "&ugcPosts=List(" + ",".join(map(quote, ugcs)) + ")"
            if not params:
                continue

            url = (
                f"{API_BASE}/organizationalEntityShareStatistics"
                f"?q=organizationalEntity&organizationalEntity={self._org_urn(org_id)}"
                f"{params}"
            )
            data = self._safe_get(url, "postStatistics")
            if data is None:
                return None
            elements.extend(data.get("elements", []))

        self.logger.log(
            f"Retrieved statistics for {len(elements)}/{len(post_urns)} posts."
        )
        return elements

    def _get_daily_raw(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[dict[str, list[dict]]]:
        """Daily share stats, follower gains and page stats (3 calls)."""
        ti = self._time_intervals(start_date, end_date)
        org = self._org_urn(org_id)
        urls = {
            "share": (
                f"{API_BASE}/organizationalEntityShareStatistics"
                f"?q=organizationalEntity&organizationalEntity={org}&timeIntervals={ti}"
            ),
            "follower": (
                f"{API_BASE}/organizationalEntityFollowerStatistics"
                f"?q=organizationalEntity&organizationalEntity={org}&timeIntervals={ti}"
            ),
            "page": (
                f"{API_BASE}/organizationPageStatistics"
                f"?q=organization&organization={org}&timeIntervals={ti}"
            ),
        }
        out: dict[str, list[dict]] = {}
        for key, url in urls.items():
            self.logger.log(f"GET daily {key} statistics for org {org_id}...")
            data = self._safe_get(url, f"daily {key}")
            if data is None:
                return None
            out[key] = data.get("elements", [])
        return out

    # ------------------------------------------------------------------
    # PUBLIC — DataFrames
    # ------------------------------------------------------------------

    def get_managed_pages(self, role: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Organizations the token can manage.

        Returns:
            DataFrame [org_id, role, state].
        """
        raw = self._get_managed_pages_raw(role)
        if raw is None:
            return None
        rows = [
            {
                "org_id": el["organization"].split(":")[-1],
                "role": el.get("role"),
                "state": el.get("state"),
            }
            for el in raw.get("elements", [])
        ]
        return pd.DataFrame(rows, columns=["org_id", "role", "state"])

    def get_posts(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Posts created between start_date and end_date (inclusive).

        Returns:
            DataFrame with one row per post: post_urn, org_id, author,
            created_at, published_at, last_modified_at, commentary,
            content_type, visibility, lifecycle_state, reshare_parent,
            content_json (raw content object as JSON string).
        """
        posts = self._fetch_paginated_posts(org_id, start_date, end_date)
        if posts is None:
            return None

        rows = []
        for p in posts:
            content = p.get("content") or {}
            # Content shapes: article, media, multiImage, poll, celebration...
            content_type = next(iter(content), "none").upper() if content else "NONE"
            rows.append(
                {
                    "post_urn": p.get("id"),
                    "org_id": org_id,
                    "author": p.get("author"),
                    "created_at": _ms_to_ts(p.get("createdAt")),
                    "published_at": _ms_to_ts(p.get("publishedAt")),
                    "last_modified_at": _ms_to_ts(p.get("lastModifiedAt")),
                    "commentary": p.get("commentary", ""),
                    "content_type": content_type,
                    "visibility": p.get("visibility"),
                    "lifecycle_state": p.get("lifecycleState"),
                    "reshare_parent": (p.get("reshareContext") or {}).get("parent"),
                    "content_json": json.dumps(content, ensure_ascii=False)
                    if content
                    else None,
                }
            )
        return pd.DataFrame(rows)

    def get_post_statistics(
        self, org_id: str, post_urns: list[str]
    ) -> Optional[pd.DataFrame]:
        """Lifetime statistics per post (1 API call per 50 posts).

        Requires ADMINISTRATOR / ANALYST / CONTENT_ADMINISTRATOR on the org.

        Returns:
            DataFrame [post_urn, impression_count, unique_impressions_count,
            click_count, like_count, comment_count, share_count, engagement].
        """
        cols = ["post_urn"] + [_snake(f) for f in SHARE_STAT_FIELDS]
        if not post_urns:
            return pd.DataFrame(columns=cols)

        elements = self._get_post_statistics_raw(org_id, post_urns)
        if elements is None:
            return None

        rows = []
        for el in elements:
            stats = el.get("totalShareStatistics", {})
            row: dict[str, Any] = {"post_urn": el.get("share") or el.get("ugcPost")}
            row.update({_snake(f): stats.get(f) for f in SHARE_STAT_FIELDS})
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def get_posts_with_statistics(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Posts table: metadata + lifetime statistics, as of extraction.

        Statistics are cumulative to ``extracted_at``; re-extracting a post
        later yields updated numbers, so keep ``extracted_at`` when loading.

        Returns:
            get_posts columns + statistics columns + extracted_at. Posts with
            no statistics returned by LinkedIn keep nulls.
        """
        df_posts = self.get_posts(org_id, start_date, end_date)
        if df_posts is None or df_posts.empty:
            return df_posts

        df_stats = self.get_post_statistics(org_id, df_posts["post_urn"].tolist())
        if df_stats is None:
            return None

        merged = df_posts.merge(df_stats, on="post_urn", how="left")
        int_cols = [_snake(f) for f in SHARE_STAT_FIELDS if f != "engagement"]
        merged[int_cols] = merged[int_cols].astype("Int64")
        merged["extracted_at"] = pd.Timestamp.now(tz="UTC")
        return merged

    def get_daily_stats(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Daily page metrics between start_date and end_date (inclusive).

        3 API calls regardless of range length. LinkedIn only serves these
        series for roughly the last 12 months.

        Returns:
            One row per day: date, org_id, share statistics columns
            (impression_count, click_count, ... engagement),
            organic_follower_gain, paid_follower_gain and every
            totalPageStatistics.views metric flattened as
            ``page_<view>_<metric>`` (e.g. page_all_page_views_page_views).
        """
        try:
            self._time_intervals(start_date, end_date)
        except ValueError as exc:
            self.logger.critical(f"Date format error (use YYYY-MM-DD): {exc}")
            return None

        raw = self._get_daily_raw(org_id, start_date, end_date)
        if raw is None:
            return None

        days: dict[int, dict[str, Any]] = {}

        def row_for(el: dict) -> dict[str, Any]:
            start = el["timeRange"]["start"]
            return days.setdefault(
                start,
                {"date": _ms_to_ts(start).date(), "org_id": org_id},
            )

        for el in raw["share"]:
            stats = el.get("totalShareStatistics", {})
            row_for(el).update({_snake(f): stats.get(f) for f in SHARE_STAT_FIELDS})

        for el in raw["follower"]:
            gains = el.get("followerGains", {})
            row_for(el).update(
                {
                    "organic_follower_gain": gains.get("organicFollowerGain"),
                    "paid_follower_gain": gains.get("paidFollowerGain"),
                }
            )

        for el in raw["page"]:
            views = el.get("totalPageStatistics", {}).get("views", {})
            row = row_for(el)
            for view_name, metrics in views.items():
                for metric, value in (metrics or {}).items():
                    row[f"page_{_snake(view_name)}_{_snake(metric)}"] = value

        if not days:
            return pd.DataFrame(columns=["date", "org_id"])

        df = pd.DataFrame([days[k] for k in sorted(days)])
        # Sources can miss a day (e.g. follower series lags one day);
        # keep counts as nullable ints instead of letting NaN make floats.
        count_cols = [
            c for c in df.columns if c not in ("date", "org_id", "engagement")
        ]
        df[count_cols] = df[count_cols].astype("Int64")
        df["extracted_at"] = pd.Timestamp.now(tz="UTC")
        return df

    def get_follower_stats(self, org_id: str) -> Optional[pd.DataFrame]:
        """Follower snapshot: total + every demographic breakdown (2 calls).

        Long format, one row per (dimension, dimension_value). The row with
        dimension='total' comes from networkSizes and is the real follower
        count; breakdown rows only include members with that attribute, so
        they do not add up to the total. Values are LinkedIn URNs/enums
        (e.g. urn:li:function:15); analysts map them to names.

        Returns:
            DataFrame [snapshot_date, org_id, dimension, dimension_value,
            organic_follower_count, paid_follower_count, extracted_at].
        """
        raw = self._get_follower_stats_raw(org_id)
        if raw is None:
            return None
        total = self._get_follower_total_raw(org_id)
        if total is None:
            return None

        now = pd.Timestamp.now(tz="UTC")
        rows: list[dict[str, Any]] = [
            {
                "dimension": "total",
                "dimension_value": None,
                "organic_follower_count": total.get("firstDegreeSize"),
                "paid_follower_count": None,
            }
        ]

        elements = raw.get("elements", [])
        inner = elements[0] if elements else {}
        for key, value_field in FOLLOWER_BREAKDOWNS.items():
            dimension = _snake(key.removeprefix("followerCountsBy"))
            for item in inner.get(key, []):
                counts = item.get("followerCounts", {})
                rows.append(
                    {
                        "dimension": dimension,
                        "dimension_value": item.get(value_field),
                        "organic_follower_count": counts.get("organicFollowerCount"),
                        "paid_follower_count": counts.get("paidFollowerCount"),
                    }
                )

        df = pd.DataFrame(rows)
        df.insert(0, "org_id", org_id)
        df.insert(0, "snapshot_date", now.date())
        df[["organic_follower_count", "paid_follower_count"]] = df[
            ["organic_follower_count", "paid_follower_count"]
        ].astype("Int64")
        df["extracted_at"] = now
        return df
