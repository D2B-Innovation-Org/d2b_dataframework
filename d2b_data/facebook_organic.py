import logging
from datetime import UTC, datetime

import pandas as pd
import requests
from d2b_data.workflow_logger import resolve_logger


class FacebookOrganic:
    """Abstraction layer for the Facebook Graph API Page Insights (organic).

    Handles authentication, pagination, post retrieval, and post-level
    insights extraction. Designed to mirror the interface of Facebook_Marketing
    so it can be dropped into the same ETL patterns.

    Attributes:
        page_id: Facebook Page ID to query.
        access_token: Page Access Token with pages_read_engagement
            and read_insights permissions.
        base_url: Base URL for the Graph API, pinned to v25.0.
    """

    def __init__(
        self,
        access_token: str,
        logger=None,
        *,
        verbose_logger=None,
    ) -> None:
        """Initialize the Facebook_Organic client.

        Args:
            page_id: The numeric Facebook Page ID.
            access_token: A valid Page Access Token.
            logger: Optional logger instance. Must expose a .log()
                and .critical() method. If None, falls back to stdlib logging.
        """
        logger = resolve_logger(logger, verbose_logger, "FacebookOrganic")
        self.access_token = access_token
        self.POST_FIELDS = "id,message,created_time,like_count,comments_count,shares,comments.summary(true),reactions.summary(true)"
        self.BASE_URL = "https://graph.facebook.com/v25.0"
        self.logger = (
            logger if logger else self._build_default_logger()
        )
        self.logger.log(
            "--- EXECUTING FacebookOrganic Class v1.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_default_logger(self):
        """Build a stdlib-based fallback logger that matches the verbose interface.

        Returns:
            An object with .log() and .critical() methods.
        """
        logger = logging.getLogger("Facebook_Organic")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

        class _StdlibAdapter:
            def log(self, message: str) -> None:
                logger.info(message)

            def critical(self, message: str) -> None:
                logger.error(message)

        return _StdlibAdapter()

    def _get(self, endpoint: str, params: dict) -> dict:
        """Execute a single GET request against the Graph API.

        Injects the access token automatically. Raises on HTTP errors
        or API-level error responses.

        Args:
            endpoint: Path relative to BASE_URL, e.g. '/302064363144019/posts'.
            params: Query parameters to include in the request.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            requests.HTTPError: If the HTTP status code indicates failure.
            ValueError: If the API returns an error payload.
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = {**params, "access_token": self.access_token}

        response = requests.get(url, params=params, timeout=30)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            self.logger.critical(
                f"_get | HTTP error for {endpoint}: {exc} — "
                f"Response body: {response.text[:500]}"
            )
            raise

        payload = response.json()

        if "error" in payload:
            error = payload["error"]
            message = (
                f"_get | Graph API error for {endpoint}: "
                f"[{error.get('code')}] {error.get('message')}"
            )
            self.logger.critical(message)
            raise ValueError(message)

        return payload

    def _paginate(self, endpoint: str, params: dict) -> list[dict]:
        """Fetch all pages of a paginated Graph API endpoint.

        Iterates through cursor-based pagination until no next page exists.

        Args:
            endpoint: Path relative to BASE_URL.
            params: Initial query parameters. 'after' cursor is injected
                automatically on subsequent pages.

        Returns:
            Flat list of all records across all pages.
        """
        results: list[dict] = []
        page_count = 0

        while True:
            response = self._get(endpoint, params)
            data = response.get("data", [])
            results.extend(data)
            page_count += 1

            self.logger.log(
                f"_paginate | Page {page_count}: fetched {len(data)} records "
                f"(total so far: {len(results)})"
            )

            paging = response.get("paging", {})
            cursors = paging.get("cursors", {})
            after_cursor = cursors.get("after")

            # Stop if there is no next page or the current page returned nothing
            if not paging.get("next") or not after_cursor or len(data) == 0:
                break

            params = {**params, "after": after_cursor}

        self.logger.log(
            f"_paginate | Completed: {len(results)} total records across "
            f"{page_count} page(s) for {endpoint}"
        )
        return results

    def _flatten_insights(self, data: list[dict]) -> dict:
        """Convert a raw insights response into a flat key-value dictionary.

        Extracts the 'lifetime' value for each metric. For
        post_reactions_by_type_total, expands the nested reaction counts
        into individual keys prefixed with 'reactions_'.

        Args:
            data: The 'data' array from a /insights API response.

        Returns:
            Flat dictionary mapping metric names to their lifetime values.
        """
        flat: dict = {}

        for metric in data:
            name = metric.get("name", "")
            values = metric.get("values", [])

            # Prefer the lifetime period entry; fall back to the first value
            lifetime_entry = next(
                (v for v in values if metric.get("period") == "lifetime"),
                values[0] if values else None,
            )

            if lifetime_entry is None:
                flat[name] = None
                continue

            value = lifetime_entry.get("value")

            if name == "post_reactions_by_type_total" and isinstance(value, dict):
                # Expand reaction types into individual columns
                for reaction_type, count in value.items():
                    flat[f"reactions_{reaction_type}"] = count
            else:
                flat[name] = value

        return flat

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_posts(self, page_id: str, since: str, until: str) -> list[dict]:
        """Retrieve all posts published within a date range.

        Fetches id, message, created_time, like_count, comments_count,
        and shares for every post. Handles pagination automatically.

        Args:
            page_id: The ID of the Facebook page.
            since: Start date in 'YYYY-MM-DD' format (inclusive).
            until: End date in 'YYYY-MM-DD' format (inclusive).

        Returns:
            List of post dictionaries as returned by the Graph API,
            with shares normalized to an integer count.
        """
        self.logger.log(f"get_posts | Fetching posts from {since} to {until}")

        params = {
            "fields": self.POST_FIELDS,
            "since": since,
            "until": until,
            "limit": 100,
        }

        posts = self._paginate(f"/{page_id}/posts", params)

        # Normalize shares: API returns {"count": N} or is absent
        for post in posts:
            raw_shares = post.get("shares")
            if isinstance(raw_shares, dict):
                post["shares"] = raw_shares.get("count", 0)
            elif raw_shares is None:
                post["shares"] = 0

            raw_comments = post.get("comments")
            if isinstance(raw_comments, dict):
                post["comments"] = raw_comments.get("summary", {}).get("total_count", 0)
            elif raw_comments is None:
                post["comments"] = 0

            raw_reactions = post.get("reactions")
            if isinstance(raw_reactions, dict):
                post["reactions"] = raw_reactions.get("summary", {}).get(
                    "total_count", 0
                )
            elif raw_reactions is None:
                post["reactions"] = 0

        self.logger.log(f"get_posts | Retrieved {len(posts)} posts")
        return posts

    def get_post_insights(self, post_id: str, metrics: list[str]) -> dict:
        """Retrieve lifetime insight metrics for a single post.

        Args:
            post_id: The full post ID in the format '{page_id}_{post_id}'.
            metrics: List of metric names to request.

        Returns:
            Flat dictionary of metric names to their lifetime values.
            Returns an empty dict if the request fails, so callers can
            continue processing remaining posts.
        """

        self.logger.log(
            f"get_post_insights | Fetching {len(metrics)} metrics for {post_id}"
        )

        params = {
            "metric": ",".join(metrics),
            "period": "lifetime",
        }

        try:
            response = self._get(f"/{post_id}/insights", params)
            flat = self._flatten_insights(response.get("data", []))
            self.logger.log(
                f"get_post_insights | {post_id} — {len(flat)} metrics extracted"
            )
            return flat
        except (ValueError, requests.HTTPError) as exc:
            self.logger.critical(
                f"get_post_insights | Failed for {post_id}: {exc}. Skipping."
            )
            return {}

    def get_report_dataframe(
        self, page_id: str, start_date: str, end_date: str, metrics: list[str]
    ) -> pd.DataFrame:
        """Build a combined DataFrame of posts and their insight metrics.

        Args:
            page_id: The ID of the Facebook page.
            start_date: Start date in 'YYYY-MM-DD' format or 'YYYYMMDD' format.
            end_date: End date in 'YYYY-MM-DD' format or 'YYYYMMDD' format.
            metrics: List of insight metric names to request per post.

        Returns:
            DataFrame where each row is a post, with columns for post
            metadata (message, created_time, etc.) and all insight metrics.
            Returns an empty DataFrame if no posts are found.
        """

        if start_date:
            try:
                since = datetime.strptime(start_date, "%Y-%m-%d")
                since = since.strftime("%Y-%m-%d")
            except ValueError:
                try:
                    since = datetime.strptime(start_date, "%Y%m%d")
                    since = since.strftime("%Y-%m-%d")
                except ValueError:
                    raise ValueError(
                        "Expected 'start_date' value\n"
                        "accepted values: 'AAAA-MM-DD' o 'AAAAMMDD'."
                    )
        else:
            raise ValueError("'start date' argument must be provided")

        if end_date:
            try:
                until = datetime.strptime(end_date, "%Y-%m-%d")
                until = until.strftime("%Y-%m-%d")
            except ValueError:
                try:
                    until = datetime.strptime(end_date, "%Y%m%d")
                    until = until.strftime("%Y-%m-%d")
                except ValueError:
                    raise ValueError(
                        "Expected 'end_date' value\n"
                        "accepted values: 'AAAA-MM-DD' o 'AAAAMMDD'."
                    )
        else:
            raise ValueError("'end date' argument must be provided")

        self.logger.log(
            f"get_report_dataframe | Starting report for page {page_id} "
            f"from {since} to {until}"
        )

        posts = self.get_posts(page_id, since, until)

        if not posts:
            self.logger.log(
                "get_report_dataframe | No posts found for the given date range. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        records: list[dict] = []

        for post in posts:
            post_id = post.get("id", "")
            insights = self.get_post_insights(post_id, metrics=metrics)

            record = {
                "post_id": post_id,
                "page_id": page_id,
                "message": post.get("message", ""),
                "created_time": post.get("created_time", ""),
                "like_count": post.get("like_count", 0),
                "comments": post.get("comments", 0),
                "reactions": post.get("reactions", 0),
                "shares": post.get("shares", 0),
                **insights,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)

        self.logger.log(f"get_report_dataframe | Done. DataFrame shape: {df.shape}")
        return df

    def __repr__(self):
        return f"<FacebookOrganic Client version: {self.BASE_URL}>"
