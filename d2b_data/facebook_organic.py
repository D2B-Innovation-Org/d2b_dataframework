from datetime import UTC, datetime
from typing import Optional

import pandas as pd
from d2b_data.meta_graph import GRAPH_API_VERSION, MetaGraphClient, MetaGraphError
from d2b_data.workflow_logger import resolve_logger


class FacebookOrganic:
    """Abstraction layer for the Facebook Graph API Page Insights (organic).

    Handles post retrieval and post-level insights extraction. HTTP, auth,
    retries and pagination live in ``MetaGraphClient`` (``self.graph``).

    Data is raw: a value the API does not return is None, never a made-up 0
    or empty string.

    Attributes:
        graph: Shared Graph API client (token, session, API version).
        logger: Logger with .log() and .critical().
        POST_FIELDS: Fields requested for every post.
    """

    def __init__(
        self,
        access_token: str,
        logger=None,
        *,
        verbose_logger=None,
        api_version: str = GRAPH_API_VERSION,
    ) -> None:
        """Initialize the FacebookOrganic client.

        The Page ID is not a constructor argument: pass it to each method so
        one instance can serve several Pages.

        Args:
            access_token: A valid Page Access Token with pages_read_engagement
                and read_insights permissions.
            logger: Optional logger instance. Must expose a .log()
                and .critical() method. If None, falls back to stdlib logging.
            verbose_logger: Deprecated alias of ``logger``.
            api_version: Graph API version; defaults to GRAPH_API_VERSION.
        """
        logger = resolve_logger(logger, verbose_logger, "FacebookOrganic")
        self.POST_FIELDS = "id,message,created_time,like_count,comments_count,shares,comments.summary(true),reactions.summary(true)"
        self.graph = MetaGraphClient(access_token, logger, api_version)
        self.logger = self.graph.logger
        self.logger.log(
            "--- EXECUTING FacebookOrganic Class v2.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )

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

    @staticmethod
    def _nested_count(value: object, *path: str) -> Optional[int]:
        """Read a count from a nested dict, or None when it is not there.

        Args:
            value: The raw field, e.g. ``{"count": 7}``; None when absent.
            *path: Keys to follow inside ``value``.

        Returns:
            The count, or None if the field or any key is missing.
        """
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value if isinstance(value, int) else None

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
            List of post dictionaries as returned by the Graph API, with
            shares, comments and reactions flattened to integer counts (None
            when the API did not return them).

        Raises:
            MetaGraphError: On Graph API errors.
        """
        self.logger.log(f"get_posts | Fetching posts from {since} to {until}")

        params = {
            "fields": self.POST_FIELDS,
            "since": since,
            "until": until,
            "limit": 100,
        }

        posts = self.graph.paginate(f"/{page_id}/posts", params)

        # Flatten the nested counters. A counter the API did not return is
        # None: Facebook omits them when it has nothing to report, and a
        # silent 0 would be indistinguishable from a real zero.
        for post in posts:
            post["shares"] = self._nested_count(post.get("shares"), "count")
            post["comments"] = self._nested_count(
                post.get("comments"), "summary", "total_count"
            )
            post["reactions"] = self._nested_count(
                post.get("reactions"), "summary", "total_count"
            )

        self.logger.log(f"get_posts | Retrieved {len(posts)} posts")
        return posts

    def get_post_insights(self, post_id: str, metrics: list[str]) -> dict:
        """Retrieve lifetime insight metrics for a single post.

        Args:
            post_id: The full post ID in the format '{page_id}_{post_id}'.
            metrics: List of metric names to request.

        Returns:
            Flat dictionary of metric names to their lifetime values.
            Returns an empty dict if the request fails with an error that is
            specific to this post, so callers can continue with the rest.

        Raises:
            MetaGraphError: On auth, permission or quota errors, which affect
                every post and must stop the run.
        """

        self.logger.log(
            f"get_post_insights | Fetching {len(metrics)} metrics for {post_id}"
        )

        params = {
            "metric": ",".join(metrics),
            "period": "lifetime",
        }

        try:
            response = self.graph.get(f"/{post_id}/insights", params)
        except MetaGraphError as exc:
            if exc.must_propagate:
                raise
            self.logger.log(
                f"get_post_insights | Skipping metrics for {post_id}: "
                f"API code {exc.api_code}, HTTP {exc.http_status}"
            )
            return {}

        flat = self._flatten_insights(response.get("data", []))
        self.logger.log(
            f"get_post_insights | {post_id} - {len(flat)} metrics extracted"
        )
        return flat

    def get_report_dataframe(
        self, page_id: str, start_date: str, end_date: str, metrics: list[str]
    ) -> pd.DataFrame:
        """Build a combined DataFrame of posts and their insight metrics.

        Posts whose insights could not be read keep their row, with the
        metric columns missing (NaN/None).

        Args:
            page_id: The ID of the Facebook page.
            start_date: Start date in 'YYYY-MM-DD' format or 'YYYYMMDD' format.
            end_date: End date in 'YYYY-MM-DD' format or 'YYYYMMDD' format.
            metrics: List of insight metric names to request per post.

        Returns:
            DataFrame where each row is a post, with columns for post
            metadata (message, created_time, etc.) and all insight metrics.
            Returns an empty DataFrame if no posts are found.

        Raises:
            ValueError: If a date is missing or has an unsupported format.
            MetaGraphError: On auth, permission or quota errors.
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
            post_id = post.get("id")
            insights = self.get_post_insights(post_id, metrics=metrics)

            record = {
                "post_id": post_id,
                "page_id": page_id,
                "message": post.get("message"),
                "created_time": post.get("created_time"),
                "like_count": post.get("like_count"),
                "comments": post.get("comments"),
                "reactions": post.get("reactions"),
                "shares": post.get("shares"),
                **insights,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)

        self.logger.log(f"get_report_dataframe | Done. DataFrame shape: {df.shape}")
        return df

    def __repr__(self) -> str:
        return f"<FacebookOrganic api_version={self.graph.api_version}>"
