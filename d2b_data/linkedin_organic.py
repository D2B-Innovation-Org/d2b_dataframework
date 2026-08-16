"""LinkedIn Community Management API — organic data extraction.

Provides a clean DataFrame-based interface for the team, with raw
dict-returning methods (prefixed with _) available for debugging.
"""

import json
import logging
import time
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests


class QuotaExhaustedError(Exception):
    """Raised when LinkedIn returns 429 due to daily quota exhaustion.

    Unlike transient rate limits, this resets at midnight UTC and
    retrying immediately is pointless.
    """

    pass


class LinkedinOrganic:
    """Extract organic metrics from the LinkedIn Community Management API.

    Public methods return pd.DataFrame ready for analysis.
    Private _raw methods return raw API dicts for debugging.

    Attributes:
        token_path: Path to the JSON file containing the access token.
    """

    def __init__(
        self,
        token_path: Optional[str] = None,
        verbose_logger: Optional[object] = None,
    ) -> None:
        self.token_path = token_path or "token_linkedin_organic.json"
        self.verbose = verbose_logger or self._build_default_logger()
        self.verbose.log(
            "--- EXECUTING LinkedinOrganic Class v2.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )
        self.token: Optional[str] = None
        self.headers: Optional[dict[str, str]] = None

        if token_path:
            self._load_token_from_file()
            if self.token:
                self._set_headers()
                self.verbose.log(
                    f"LinkedinOrganic instantiated with token from {self.token_path}."
                )
            else:
                self.verbose.log(f"No token found in {self.token_path}.")
        else:
            self.verbose.log(
                "Token file not specified. Use get_access_token to "
                "generate and save a new token."
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

    def _load_token_from_file(self) -> Optional[dict]:
        """Read token JSON from disk and set self.token."""
        try:
            with open(self.token_path, "r") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.verbose.log(f"Error loading token from file: {exc}")
            return None

        if "access_token" not in data:
            self.verbose.log("Token file is missing 'access_token' field.")
            return None

        self.token = data["access_token"]
        return data

    def _set_headers(self) -> None:
        """Build the headers dict required by the LinkedIn REST API."""
        if not self.token:
            self.verbose.critical("Cannot set headers: access token is missing.")
            return

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": "202606",
            "Content-Type": "application/json",
        }
        self.verbose.log("LinkedIn API headers configured.")

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

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

        # Use a PreparedRequest so requests sends the URL exactly as
        # provided, without re-encoding brackets or other characters
        # that are already correctly encoded (or intentionally literal).
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
                self.verbose.log(
                    f"Transient {response.status_code}, "
                    f"retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        # Should not reach here, but just in case
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # RAW (private) — return dicts, useful for debugging
    # ------------------------------------------------------------------

    def _get_managed_pages_raw(self, role: Optional[str] = None) -> Optional[dict]:
        """Fetch organizations the token has access to (raw JSON).

        Args:
            role: Optional filter — 'ADMINISTRATOR', 'ANALYST', etc.

        Returns:
            Full API response dict, or None on failure.
        """
        url = "https://api.linkedin.com/rest/organizationAcls?q=roleAssignee"
        if role:
            url += f"&role={role}"

        self.verbose.log("GET managed organizations...")
        try:
            data = self._request_get(url)
            self.verbose.log(
                f"Retrieved {len(data.get('elements', []))} organization ACLs."
            )
            return data
        except requests.exceptions.RequestException as exc:
            self.verbose.critical(f"LinkedIn API Error: {exc}")
            return None

    def _get_follower_stats_raw(self, org_id: str) -> Optional[dict]:
        """Fetch follower statistics for an organization (raw JSON).

        Args:
            org_id: Numeric organization ID, e.g. '1234567'.

        Returns:
            Full API response dict, or None on failure.
        """
        urn_encoded = quote(f"urn:li:organization:{org_id}")
        url = (
            "https://api.linkedin.com/rest/"
            "organizationalEntityFollowerStatistics"
            f"?q=organizationalEntity"
            f"&organizationalEntity={urn_encoded}"
        )

        self.verbose.log(f"GET follower stats for org {org_id}...")
        try:
            data = self._request_get(url)
            self.verbose.log(
                f"Retrieved {len(data.get('elements', []))} follower stat elements."
            )
            return data
        except requests.exceptions.RequestException as exc:
            self.verbose.critical(f"LinkedIn API Error: {exc}")
            return None

    def _fetch_paginated_posts(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[list[dict]]:
        """Paginate through posts and return only those within the date range.

        LinkedIn returns posts newest-first, so we stop as soon as
        we hit a post older than start_date.

        Args:
            org_id: Numeric organization ID.
            start_date: Inclusive lower bound, 'YYYY-MM-DD'.
            end_date: Inclusive upper bound, 'YYYY-MM-DD'.

        Returns:
            List of raw post dicts, or None on date-parse failure.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            self.verbose.critical(f"Date format error (use YYYY-MM-DD): {exc}")
            return None

        urn_encoded = quote(f"urn:li:organization:{org_id}")
        collected: list[dict] = []
        start_index = 0
        page_size = 50

        self.verbose.log(
            f"Fetching posts for org {org_id} between {start_date} and {end_date}..."
        )

        while True:
            url = (
                "https://api.linkedin.com/rest/posts"
                f"?q=author&author={urn_encoded}"
                f"&count={page_size}&start={start_index}"
            )

            try:
                data = self._request_get(url)
            except requests.exceptions.RequestException as exc:
                self.verbose.critical(f"LinkedIn API Error during pagination: {exc}")
                break

            elements = data.get("elements", [])
            if not elements:
                self.verbose.log("No more posts available.")
                break

            self.verbose.log(
                f"Processing batch of {len(elements)} posts (offset {start_index})..."
            )

            stop_paging = False
            for post in elements:
                created_ms = post.get("createdAt")
                if not created_ms:
                    continue

                post_dt = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)

                if post_dt > end_dt:
                    # Newer than range — keep paging
                    continue
                elif start_dt <= post_dt <= end_dt:
                    collected.append(post)
                else:
                    # Older than range — everything after is older too
                    self.verbose.log(
                        f"Reached post from {post_dt:%Y-%m-%d}, "
                        "older than start_date. Stopping."
                    )
                    stop_paging = True
                    break

            if stop_paging:
                break

            start_index += page_size

        self.verbose.log(f"Collected {len(collected)} posts in date range.")
        return collected

    def _get_posts_raw(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[dict]:
        """Fetch posts within a date range (raw JSON wrapper).

        Args:
            org_id: Numeric organization ID.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        Returns:
            Dict with key 'elements' containing the list of post dicts.
        """
        posts = self._fetch_paginated_posts(org_id, start_date, end_date)
        if posts is None:
            return None
        return {"elements": posts}

    def _get_engagement_raw(self, post_urns: list[str]) -> Optional[dict]:
        """Fetch social actions for a list of posts (raw JSON).

        Uses GET /socialActions/{encoded_urn} individually per post.
        The batch ?ids=List() endpoint returns 400 regardless of
        encoding — individual path calls are the working approach.

        Args:
            post_urns: List of post URNs to query.

        Returns:
            Dict keyed by URN with engagement data, or None on failure.
        """
        if not post_urns:
            self.verbose.log("No post URNs provided.")
            return {}

        results = {}
        for urn in post_urns:
            url = f"https://api.linkedin.com/rest/socialActions/{quote(urn)}"
            try:
                data = self._request_get(url)
                results[urn] = data
            except QuotaExhaustedError:
                raise
            except requests.exceptions.RequestException as exc:
                self.verbose.critical(f"Social Actions API Error for {urn}: {exc}")

        self.verbose.log(
            f"Retrieved engagement for {len(results)}/{len(post_urns)} posts."
        )
        return results

    # ------------------------------------------------------------------
    # PUBLIC — return DataFrames, ready for analysis
    # ------------------------------------------------------------------

    def get_managed_pages(self, role: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get organizations the token can manage.

        Args:
            role: Optional filter — 'ADMINISTRATOR', 'ANALYST', etc.

        Returns:
            DataFrame with columns [org_id, role, state].
        """
        raw = self._get_managed_pages_raw(role)
        if raw is None:
            return None

        rows = []
        for el in raw.get("elements", []):
            rows.append(
                {
                    "org_id": el["organization"].split(":")[-1],
                    "role": el.get("role"),
                    "state": el.get("state"),
                }
            )

        return pd.DataFrame(rows)

    def get_follower_stats(self, org_id: str) -> Optional[pd.DataFrame]:
        """Get follower statistics for an organization.

        Flattens followerCountsByFunction and returns totals per
        extraction date.

        Args:
            org_id: Numeric organization ID.

        Returns:
            DataFrame with columns [extraction_date,
            organizational_entity, dimension,
            organic_followers, paid_followers].
        """
        raw = self._get_follower_stats_raw(org_id)
        if raw is None:
            return None

        elements = raw.get("elements", [])
        if not elements:
            self.verbose.log("No follower stat elements returned.")
            return pd.DataFrame()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        inner = elements[0]
        resolved_org = inner["organizationalEntity"].split(":")[-1]

        rows = []
        for item in inner.get("followerCountsByFunction", []):
            counts = item.get("followerCounts", {})
            rows.append(
                {
                    "extraction_date": today,
                    "organizational_entity": resolved_org,
                    "dimension": "function",
                    "dimension_id": item["function"],
                    "organic_followers": counts.get("organicFollowerCount", 0),
                    "paid_followers": counts.get("paidFollowerCount", 0),
                }
            )

        df = pd.DataFrame(rows)

        # ponytail: pivot only if rows exist; empty DF pivot raises
        if df.empty:
            return df

        return df.pivot_table(
            index=[
                "extraction_date",
                "organizational_entity",
                "dimension",
            ],
            values=["organic_followers", "paid_followers"],
            aggfunc="sum",
        ).reset_index()

    def get_posts(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Get organic posts within a date range.

        Args:
            org_id: Numeric organization ID.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        Returns:
            DataFrame with columns [post_urn, created_at, commentary,
            content_type, visibility, lifecycle_state].
        """
        posts = self._fetch_paginated_posts(org_id, start_date, end_date)
        if posts is None:
            return None

        rows = []
        for p in posts:
            created_ms = p.get("createdAt")
            created_at = (
                datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if created_ms
                else None
            )

            # ponytail: extract content type from whichever content
            # key is present; LinkedIn uses different shapes
            content = p.get("content", {})
            if "article" in content:
                content_type = "ARTICLE"
            elif "media" in content:
                content_type = "MEDIA"
            elif "multiImage" in content:
                content_type = "MULTI_IMAGE"
            else:
                content_type = "NONE"

            rows.append(
                {
                    "post_urn": p.get("id", ""),
                    "created_at": created_at,
                    "commentary": p.get("commentary", ""),
                    "content_type": content_type,
                    "visibility": p.get("visibility", ""),
                    "lifecycle_state": p.get("lifecycleState", ""),
                }
            )

        return pd.DataFrame(rows)

    def get_engagement(self, post_urns: list[str]) -> Optional[pd.DataFrame]:
        """Get engagement metrics for a list of posts.

        Uses socialActions endpoint. Requires ANALYST role or higher.
        Note: quota is limited — check LinkedIn developer portal for
        current limits. Use get_posts_with_engagement_safe if quota
        is a concern.

        Args:
            post_urns: List of post URNs.

        Returns:
            DataFrame with columns [post_urn, likes, comments, shares].

        Raises:
            QuotaExhaustedError: If LinkedIn daily quota is exceeded.
        """
        raw = self._get_engagement_raw(post_urns)
        if raw is None:
            return None

        rows = []
        for urn, actions in raw.items():
            rows.append(
                {
                    "post_urn": urn,
                    "likes": actions.get("likesSummary", {}).get("totalLikes", 0),
                    "comments": actions.get("commentsSummary", {}).get(
                        "totalFirstLevelComments", 0
                    ),
                    "shares": actions.get("sharesSummary", {}).get("totalShares", 0),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CONVENIENCE — common multi-step workflows
    # ------------------------------------------------------------------

    def get_posts_with_engagement(
        self, org_id: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Get posts merged with their engagement metrics in one call.

        This is the typical team workflow: fetch posts for a date range
        and join the engagement data automatically. If the engagement
        quota is exhausted, returns posts without engagement columns
        and logs a warning.

        Args:
            org_id: Numeric organization ID.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        Returns:
            DataFrame with post metadata + likes, comments, shares.
            If quota is hit, returns posts only (no engagement cols).
        """
        df_posts = self.get_posts(org_id, start_date, end_date)
        if df_posts is None or df_posts.empty:
            return df_posts

        urns = df_posts["post_urn"].tolist()

        try:
            df_eng = self.get_engagement(urns)
        except QuotaExhaustedError:
            self.verbose.critical(
                "Engagement quota exhausted. Returning posts without engagement data."
            )
            return df_posts

        if df_eng is None or df_eng.empty:
            self.verbose.log("No engagement data returned; returning posts only.")
            return df_posts

        merged = df_posts.merge(df_eng, on="post_urn", how="left")
        for col in ["likes", "comments", "shares"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)

        self.verbose.log(f"Merged {len(merged)} posts with engagement data.")
        return merged

    # ------------------------------------------------------------------
    # PATCH — cached engagement for low-quota scenarios
    # Remove this section once quota upgrade is approved.
    # ------------------------------------------------------------------

    def get_posts_with_engagement_safe(
        self,
        org_id: str,
        start_date: str,
        end_date: str,
        cache_path: str = "engagement_cache.json",
    ) -> Optional[pd.DataFrame]:
        """Get posts + engagement with disk-based cache for quota limits.

        Saves engagement results to a JSON file keyed by post URN.
        On subsequent runs, only fetches engagement for URNs not
        already in the cache. If quota is hit mid-fetch, returns
        whatever was cached so far merged with the posts.

        Remove this method once the quota upgrade is approved.

        Args:
            org_id: Numeric organization ID.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.
            cache_path: Path to the JSON cache file.

        Returns:
            DataFrame with post metadata + likes, comments, shares.
            Engagement columns may be partial if quota was hit.
        """
        df_posts = self.get_posts(org_id, start_date, end_date)
        if df_posts is None or df_posts.empty:
            return df_posts

        # Load existing cache
        cache_file = Path(cache_path)
        cache: dict[str, dict] = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
                self.verbose.log(f"Loaded {len(cache)} cached engagement entries.")
            except (json.JSONDecodeError, OSError) as exc:
                self.verbose.log(f"Cache read failed, starting fresh: {exc}")

        # Find URNs that still need fetching
        all_urns = df_posts["post_urn"].tolist()
        missing_urns = [u for u in all_urns if u not in cache]

        if missing_urns:
            self.verbose.log(
                f"{len(missing_urns)} URNs not in cache, fetching from API..."
            )
            try:
                raw = self._get_engagement_raw(missing_urns)
                if raw:
                    # raw is a dict keyed by URN from socialActions
                    cache.update(raw)
                    cache_file.write_text(json.dumps(cache, indent=2))
                    self.verbose.log(f"Cache updated: {len(cache)} total entries.")
            except QuotaExhaustedError:
                self.verbose.critical(
                    "Quota hit. Returning partial results from "
                    f"cache ({len(cache)} entries). "
                    "Re-run after midnight UTC for the rest."
                )
        else:
            self.verbose.log("All URNs found in cache, no API call needed.")

        # Build engagement DataFrame from cache
        rows = []
        for urn in all_urns:
            actions = cache.get(urn)
            if actions:
                rows.append(
                    {
                        "post_urn": urn,
                        "likes": actions.get("likesSummary", {}).get("totalLikes", 0),
                        "comments": actions.get("commentsSummary", {}).get(
                            "totalFirstLevelComments", 0
                        ),
                        "shares": actions.get("sharesSummary", {}).get(
                            "totalShares", 0
                        ),
                    }
                )

        if not rows:
            self.verbose.log("No engagement data available yet. Returning posts only.")
            return df_posts

        df_eng = pd.DataFrame(rows)
        merged = df_posts.merge(df_eng, on="post_urn", how="left")
        for col in ["likes", "comments", "shares"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)

        cached_count = len(rows)
        total_count = len(all_urns)
        self.verbose.log(
            f"Merged {cached_count}/{total_count} posts with engagement data."
        )
        return merged
