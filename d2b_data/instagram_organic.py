from datetime import UTC, datetime, timedelta
from typing import Optional

import pandas as pd
from d2b_data.meta_graph import GRAPH_API_VERSION, MetaGraphClient, MetaGraphError
from d2b_data.workflow_logger import resolve_logger


class InstagramOrganic:
    """Abstraction layer for the Instagram Graph API Media Insights (organic).

    Handles media retrieval and media-level insights extraction for FEED
    posts, REELS, and STORIES. HTTP, auth, retries and pagination live in
    ``MetaGraphClient`` (``self.graph``).

    Designed to mirror the interface of FacebookOrganic so it can be
    dropped into the same ETL patterns. Data is raw: a value the API does
    not return is None, never a made-up empty string.

    Attributes:
        graph: Shared Graph API client (token, session, API version).
        logger: Logger with .log() and .critical().
        MEDIA_FIELDS: Fields requested for every media object.
    """

    def __init__(
        self,
        access_token: str,
        logger=None,
        *,
        verbose_logger=None,
        api_version: str = GRAPH_API_VERSION,
    ) -> None:
        """Initialize the InstagramOrganic client.

        The Instagram account ID is not a constructor argument: pass it to
        each method so one instance can serve several accounts.

        Args:
            access_token: A valid Page Access Token with instagram_basic
                and instagram_manage_insights permissions.
            logger: Optional logger instance. Must expose a .log()
                and .critical() method. If None, falls back to stdlib logging.
            verbose_logger: Deprecated alias of ``logger``.
            api_version: Graph API version; defaults to GRAPH_API_VERSION.
        """
        logger = resolve_logger(logger, verbose_logger, "InstagramOrganic")
        self.MEDIA_FIELDS = (
            "id,caption,timestamp,media_type,media_product_type,permalink"
        )
        self.graph = MetaGraphClient(access_token, logger, api_version)
        self.logger = self.graph.logger
        self.logger.log(
            "--- EXECUTING InstagramOrganic Class v2.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )

    def _list_media(
        self,
        ig_account_id: str,
        since: str,
        until: str,
    ) -> list[dict]:
        """Fetch every media object of the account within a date range.

        One paginated listing of /{ig_account_id}/media, with no type filter.

        Args:
            ig_account_id: The Instagram Business Account ID.
            since: Start date in 'YYYY-MM-DD' format (inclusive).
            until: End date in 'YYYY-MM-DD' format (inclusive).

        Returns:
            List of media dictionaries of all product types.
        """
        params = {
            "fields": self.MEDIA_FIELDS,
            "since": since,
            "until": until,
            "limit": 100,
        }
        return self.graph.paginate(f"/{ig_account_id}/media", params)

    @staticmethod
    def _filter_by_type(media: list[dict], media_product_type: str) -> list[dict]:
        """Keep the media items of one media_product_type."""
        return [m for m in media if m.get("media_product_type") == media_product_type]

    def _get_media(
        self,
        ig_account_id: str,
        since: str,
        until: str,
        media_product_type: str,
    ) -> list[dict]:
        """Fetch all media of a specific type within a date range.

        Lists /{ig_account_id}/media, then filters client side by
        media_product_type. Pagination is handled automatically.

        Args:
            ig_account_id: The Instagram Business Account ID.
            since: Start date in 'YYYY-MM-DD' format (inclusive).
            until: End date in 'YYYY-MM-DD' format (inclusive).
            media_product_type: One of 'FEED', 'REELS'.

        Returns:
            List of media dictionaries matching the requested type.
        """
        self.logger.log(
            f"_get_media | Fetching {media_product_type} media for "
            f"{ig_account_id} from {since} to {until}"
        )

        all_media = self._list_media(ig_account_id, since, until)
        filtered = self._filter_by_type(all_media, media_product_type)

        self.logger.log(
            f"_get_media | {len(filtered)} {media_product_type} items "
            f"out of {len(all_media)} total media"
        )
        return filtered

    def _get_stories(self, ig_account_id: str) -> list[dict]:
        """Fetch all active stories for the account.

        Stories use a dedicated endpoint and are only available for 24 hours.
        The since/until parameters are not supported for stories.

        Args:
            ig_account_id: The Instagram Business Account ID.

        Returns:
            List of story dictionaries.
        """
        self.logger.log(f"_get_stories | Fetching active stories for {ig_account_id}")

        params = {
            "fields": self.MEDIA_FIELDS,
            "limit": 100,
        }

        stories = self.graph.paginate(f"/{ig_account_id}/stories", params)

        self.logger.log(f"_get_stories | {len(stories)} active stories found")
        return stories

    def _get_media_insights(
        self,
        media_id: str,
        metrics: list[str],
    ) -> dict:
        """Retrieve lifetime insight metrics for a single media object.

        Args:
            media_id: The Instagram Media ID.
            metrics: List of metric names to request.

        Returns:
            Flat dictionary of metric names to their lifetime values.
            Returns an empty dict if the request fails with an error that is
            specific to this media item, so callers can continue with the rest.

        Raises:
            MetaGraphError: On auth, permission or quota errors, which affect
                every item and must stop the run.
        """
        self.logger.log(
            f"_get_media_insights | Fetching {len(metrics)} metrics for {media_id}"
        )

        params = {
            "metric": ",".join(metrics),
            "period": "lifetime",
        }

        try:
            response = self.graph.get(f"/{media_id}/insights", params)
        except MetaGraphError as exc:
            if exc.must_propagate:
                raise
            self.logger.log(
                f"_get_media_insights | Skipping metrics for {media_id}: "
                f"API code {exc.api_code}, HTTP {exc.http_status}"
            )
            return {}

        flat = self._flatten_insights(response.get("data", []))
        self.logger.log(
            f"_get_media_insights | {media_id} - {len(flat)} metrics extracted"
        )
        return flat

    def _flatten_insights(self, data: list[dict]) -> dict:
        """Convert a raw insights response into a flat key-value dictionary.

        Extracts the lifetime value for each metric.

        Args:
            data: The 'data' array from a /insights API response.

        Returns:
            Flat dictionary mapping metric names to their lifetime values.
        """
        flat: dict = {}

        for metric in data:
            name = metric.get("name", "")
            values = metric.get("values", [])

            lifetime_entry = next(
                (v for v in values if metric.get("period") == "lifetime"),
                values[0] if values else None,
            )

            if lifetime_entry is None:
                flat[name] = None
                continue

            flat[name] = lifetime_entry.get("value")

        return flat

    def _build_dataframe(
        self,
        media_list: list[dict],
        ig_account_id: str,
        metrics: list[str],
    ) -> pd.DataFrame:
        """Build a DataFrame from a list of media objects and their insights.

        Fetches insights for each media item and merges them with
        the media metadata into a single row per media object.

        Args:
            media_list: List of media dicts from _get_media or _get_stories.
            ig_account_id: The Instagram Business Account ID.
            metrics: List of insight metrics to fetch per media item.

        Returns:
            DataFrame where each row is a media object with metadata
            and insight metrics as columns. Missing values are None/NaN.
            Returns empty DataFrame if media_list is empty.

        Raises:
            MetaGraphError: On auth, permission or quota errors.
        """
        if not media_list:
            self.logger.log(
                "_build_dataframe | No media to process, returning empty DataFrame"
            )
            return pd.DataFrame()

        records: list[dict] = []

        for media in media_list:
            media_id = media.get("id")
            insights = self._get_media_insights(media_id, metrics)

            record = {
                "media_id": media_id,
                "ig_account_id": ig_account_id,
                "media_type": media.get("media_type"),
                "media_product_type": media.get("media_product_type"),
                "caption": media.get("caption"),
                "timestamp": media.get("timestamp"),
                "permalink": media.get("permalink"),
                **insights,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        self.logger.log(f"_build_dataframe | DataFrame shape: {df.shape}")
        return df

    def get_feed(
        self,
        ig_account_id: str,
        metrics: list[str],
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve FEED posts and their organic insights.

        Args:
            ig_account_id: The Instagram Business Account ID.
            metrics: List of metric names to request per media item.
            since: Start date in 'YYYY-MM-DD' format. Defaults to 30 days ago.
            until: End date in 'YYYY-MM-DD' format. Defaults to yesterday.

        Returns:
            DataFrame with one row per FEED post and columns for
            metadata and insight metrics.
        """
        since, until = self._resolve_dates(since, until)
        self.logger.log(f"get_feed | {ig_account_id} | {since} → {until}")

        media = self._get_media(ig_account_id, since, until, "FEED")
        return self._build_dataframe(media, ig_account_id, metrics)

    def get_reels(
        self,
        ig_account_id: str,
        metrics: list[str],
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve REELS and their organic insights.

        Args:
            ig_account_id: The Instagram Business Account ID.
            metrics: List of metric names to request per media item.
            since: Start date in 'YYYY-MM-DD' format. Defaults to 30 days ago.
            until: End date in 'YYYY-MM-DD' format. Defaults to yesterday.

        Returns:
            DataFrame with one row per Reel and columns for
            metadata and insight metrics.
        """
        since, until = self._resolve_dates(since, until)
        self.logger.log(f"get_reels | {ig_account_id} | {since} → {until}")

        media = self._get_media(ig_account_id, since, until, "REELS")
        return self._build_dataframe(media, ig_account_id, metrics)

    def get_stories(
        self,
        ig_account_id: str,
        metrics: list[str],
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve active STORIES and their organic insights.

        Note: Stories are only available for 24 hours. The since/until
        parameters are accepted for API consistency but are not applied —
        the Instagram API only returns currently active stories.

        Args:
            ig_account_id: The Instagram Business Account ID.
            metrics: List of metric names to request per story.
            since: Accepted for interface consistency. Not applied to stories.
            until: Accepted for interface consistency. Not applied to stories.

        Returns:
            DataFrame with one row per active Story and columns for
            metadata and insight metrics.
        """
        self.logger.log(
            f"get_stories | {ig_account_id} | active stories only (last 24hs)"
        )

        stories = self._get_stories(ig_account_id)
        return self._build_dataframe(stories, ig_account_id, metrics)

    def get_all(
        self,
        ig_account_id: str,
        feed_metrics: list[str],
        reels_metrics: list[str],
        story_metrics: list[str],
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Retrieve FEED, REELS, and STORIES in a single call.

        Lists /media once and splits FEED and REELS client side, instead of
        paginating the whole listing twice. Stories use their own endpoint.
        Each DataFrame is returned separately to preserve atomicity.

        Args:
            ig_account_id: The Instagram Business Account ID.
            feed_metrics: List of metric names to request for FEED posts.
            reels_metrics: List of metric names to request for REELS.
            story_metrics: List of metric names to request for STORIES.
            since: Start date in 'YYYY-MM-DD' format. Defaults to 30 days ago.
            until: End date in 'YYYY-MM-DD' format. Defaults to yesterday.

        Returns:
            Dictionary with keys 'feed', 'reels', 'stories', each mapping
            to its corresponding DataFrame.
        """
        since, until = self._resolve_dates(since, until)
        self.logger.log(f"get_all | {ig_account_id} | {since} → {until}")

        media = self._list_media(ig_account_id, since, until)
        feed = self._filter_by_type(media, "FEED")
        reels = self._filter_by_type(media, "REELS")
        self.logger.log(
            f"get_all | {len(feed)} FEED and {len(reels)} REELS out of "
            f"{len(media)} media"
        )

        return {
            "feed": self._build_dataframe(feed, ig_account_id, feed_metrics),
            "reels": self._build_dataframe(reels, ig_account_id, reels_metrics),
            "stories": self.get_stories(ig_account_id, story_metrics),
        }

    def _resolve_dates(
        self,
        since: Optional[str],
        until: Optional[str],
    ) -> tuple[str, str]:
        """Resolve default date range if since/until are not provided.

        Defaults to the last 30 days up to yesterday.

        Args:
            since: Start date string or None.
            until: End date string or None.

        Returns:
            Tuple of (since, until) as 'YYYY-MM-DD' strings.
        """
        yesterday = datetime.now(UTC).date() - timedelta(days=1)
        default_since = yesterday - timedelta(days=30)

        since = since or default_since.strftime("%Y-%m-%d")
        until = until or yesterday.strftime("%Y-%m-%d")

        return since, until

    def __repr__(self) -> str:
        return f"<InstagramOrganic api_version={self.graph.api_version}>"
