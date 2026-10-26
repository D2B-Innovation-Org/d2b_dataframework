import logging
from datetime import UTC, datetime
from typing import Any, Optional, Union

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from d2b_data.workflow_logger import resolve_logger


class YouTubePublic:
    """YouTube Data API v3 connector for public data.

    Authenticates exclusively via an API key — no OAuth, no service account.
    Provides access to publicly available channel, video, and comment data.

    Attributes:
        api_key: Developer API key used to authenticate requests.
        verbose: Logger object exposing ``.log()`` and ``.critical()`` methods.
        service: Authenticated YouTube Data API v3 service object.
    """

    def __init__(
        self,
        api_key: str,
        logger: Optional[Any] = None,
        *,
        verbose_logger=None,
    ) -> None:
        """Initialise the public YouTube API connector.

        Args:
            api_key: Google Cloud developer API key with YouTube Data API v3
                enabled.
            logger: Logger object with ``.log()`` and ``.critical()``
                methods. Defaults to a stdlib-based adapter when omitted.
        """
        logger = resolve_logger(logger, verbose_logger, "YouTubePublic")
        self.api_key = api_key
        self.logger = (
            logger if logger else self._build_default_logger()
        )
        self.service = build(
            "youtube", "v3", developerKey=api_key, cache_discovery=False
        )
        self.logger.log(
            "--- EXECUTING YouTubePublic Class v1.0 "
            f"- Initialized at {datetime.now(UTC).isoformat()} ---"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_default_logger(self) -> Any:
        """Build a stdlib-based fallback logger matching the verbose interface.

        Returns:
            An object with ``.log()`` and ``.critical()`` methods backed by
            Python's standard ``logging`` module.
        """
        logger = logging.getLogger("YouTubePublic")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

        class _StdlibAdapter:
            def log(self, message: str) -> None:
                logger.info(message)

            def critical(self, message: str) -> None:
                logger.error(message)

        return _StdlibAdapter()

    def _normalize_df_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace dots with underscores in column names for BigQuery compatibility.

        Args:
            df: DataFrame whose column names may contain dots from
                ``pd.json_normalize``.

        Returns:
            Same DataFrame with dots replaced by underscores in all column names.
        """
        df.columns = [col.replace(".", "_") for col in df.columns]
        return df

    def _filter_columns(
        self,
        df: pd.DataFrame,
        columns: Union[list[str], None],
    ) -> pd.DataFrame:
        """Filter DataFrame columns based on the caller's selection.

        Passing ``None`` returns the full DataFrame unchanged. Passing a list
        returns only those columns. Raises ``ValueError`` if any requested
        column is absent from the API response, listing both the invalid
        columns and all available columns so the caller can correct the request.

        The check runs after extraction so the error reflects the real API
        response rather than a hardcoded schema — if YouTube adds or removes
        fields, the error message stays accurate.

        Args:
            df: Source DataFrame, already normalised and cast.
            columns: List of column names to keep, or ``None`` to keep all.

        Returns:
            Filtered DataFrame, or the original if ``columns`` is ``None``.

        Raises:
            ValueError: When one or more requested columns are not present in
                the API response.
        """
        if columns is None:
            return df

        missing = [c for c in columns if c not in df.columns]
        if missing:
            available = sorted(df.columns.tolist())
            raise ValueError(
                f"Invalid columns requested: {missing}\nAvailable columns: {available}"
            )

        return df[columns]

    def _add_extracted_at(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend an ``extracted_at`` column with the current UTC timestamp.

        Args:
            df: Source DataFrame.

        Returns:
            Same DataFrame with ``extracted_at`` as the first column.
        """
        df.insert(0, "extracted_at", datetime.now(UTC))
        return df

    # ------------------------------------------------------------------
    # Public API — Channel data
    # ------------------------------------------------------------------

    def list_channels(
        self,
        part: str = "snippet,contentDetails,statistics",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch channels matching the given parameters.

        Args:
            part: Comma-separated channel resource parts to retrieve.
                Defaults to ``'snippet,contentDetails,statistics'``.
            **kwargs: Lookup parameters — ``id='CHANNEL_ID'`` or
                ``forUsername='USERNAME'``. Note: ``mine=True`` requires
                OAuth and is not supported here.

        Returns:
            DataFrame with one row per channel.

        Raises:
            ValueError: When called without an explicit channel identifier.
        """
        explicit_keys = {"id", "forUsername", "categoryId"}
        if not any(k in kwargs for k in explicit_keys):
            raise ValueError(
                "Must specify the channel with id='CHANNEL_ID' or "
                "forUsername='USERNAME'. The mine=True parameter requires "
                "OAuth authentication and is not supported by YouTubePublic."
            )

        try:
            response = self.service.channels().list(part=part, **kwargs).execute()
            items = response.get("items", [])
            if not items:
                self.logger.log("No channels found with the given parameters.")
                return pd.DataFrame()

            return pd.json_normalize(items)

        except Exception as e:
            self.logger.critical(f"Error listing channels: {e}")
            raise e

    # ------------------------------------------------------------------
    # Public API — Playlist and video data
    # ------------------------------------------------------------------

    def get_playlist_videos(self, playlist_id: str) -> list[str]:
        """Fetch all video IDs from a playlist, handling pagination automatically.

        Args:
            playlist_id: YouTube playlist ID.

        Returns:
            List of video ID strings.
        """
        video_ids: list[str] = []
        next_page: Optional[str] = None

        while True:
            response = (
                self.service.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page,
                )
                .execute()
            )

            video_ids += [
                item["contentDetails"]["videoId"] for item in response.get("items", [])
            ]
            next_page = response.get("nextPageToken")
            self.logger.log(f"  Page processed. Accumulated videos: {len(video_ids)}")

            if not next_page:
                break

        self.logger.log(f"Total videos in playlist '{playlist_id}': {len(video_ids)}")
        return video_ids

    def get_video_statistics(
        self,
        video_ids: list[str],
        part: str = "snippet,statistics",
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch statistics for a list of videos in batches of 50.

        Args:
            video_ids: List of YouTube video ID strings.
            part: Comma-separated resource parts to retrieve.
                Defaults to ``'snippet,statistics'``.
            columns: List of column names to return. Pass ``None`` to return
                all available columns.

        Returns:
            DataFrame with one row per video.
        """
        if not video_ids:
            self.logger.log("Empty video_ids list.")
            return pd.DataFrame()

        all_items: list[dict[str, Any]] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            response = (
                self.service.videos().list(part=part, id=",".join(batch)).execute()
            )
            all_items += response.get("items", [])
            self.logger.log(f"  Batch {i // 50 + 1}: {len(batch)} videos queried.")

        if not all_items:
            self.logger.log("No video data found.")
            return pd.DataFrame()

        df = self._normalize_df_columns(pd.json_normalize(all_items))
        return self._filter_columns(df, columns)

    def _get_channel_videos_raw(
        self,
        channel_id: str,
        part: str = "snippet,statistics",
    ) -> list[dict[str, Any]]:
        """Fetch raw video items (with statistics) for an entire channel.

        Chains: channel lookup -> uploads playlist -> video stats in
        batches of 50. Does not transform the data.

        Args:
            channel_id: YouTube channel ID (e.g. ``'UCxxxxxxxxxxxxxx'``).
            part: Resource parts forwarded to ``videos().list``.
                Defaults to ``'snippet,statistics'``.

        Returns:
            List of raw video item dicts as returned by the API.
        """
        channel_response = (
            self.service.channels().list(part="contentDetails", id=channel_id).execute()
        )

        items = channel_response.get("items", [])
        if not items:
            self.logger.critical(f"Channel '{channel_id}' not found.")
            return []

        uploads_id: str = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        self.logger.log(f"Uploads playlist: {uploads_id}")

        video_ids = self.get_playlist_videos(uploads_id)
        self.logger.log(f"Total videos found: {len(video_ids)}")

        all_items: list[dict[str, Any]] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            response = (
                self.service.videos().list(part=part, id=",".join(batch)).execute()
            )
            all_items += response.get("items", [])
            self.logger.log(f"  Batch {i // 50 + 1}: {len(batch)} videos queried.")

        return all_items

    def get_channel_videos_df(
        self,
        channel_id: str,
        part: str = "snippet,statistics",
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return all video statistics for a channel as a BigQuery-compatible DataFrame.

        Column names use underscores (no dots), numeric counters are cast to
        ``Int64``, the publish date is parsed to a timezone-aware datetime, and
        an ``extracted_at`` timestamp is prepended to every row.

        Args:
            channel_id: YouTube channel ID (e.g. ``'UCxxxxxxxxxxxxxx'``).
            part: Resource parts forwarded to ``videos().list``.
                Defaults to ``'snippet,statistics'``.
            columns: List of column names to return. Pass ``None`` to return
                all available columns.

        Returns:
            DataFrame with one row per video, an ``extracted_at`` column, and
            BigQuery-friendly column names.
        """
        raw = self._get_channel_videos_raw(channel_id, part=part)

        if not raw:
            self.logger.log("No videos found for the channel.")
            return pd.DataFrame()

        df = self._normalize_df_columns(pd.json_normalize(raw))

        # Cast numeric counters — returned as strings by the API.
        int_cols = [
            "statistics_viewCount",
            "statistics_likeCount",
            "statistics_commentCount",
            "statistics_favoriteCount",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        if "snippet_publishedAt" in df.columns:
            df["snippet_publishedAt"] = pd.to_datetime(
                df["snippet_publishedAt"], utc=True, errors="coerce"
            )

        df = self._filter_columns(df, columns)
        df = self._add_extracted_at(df)

        return df

    # ------------------------------------------------------------------
    # Public API — Comment data
    # ------------------------------------------------------------------

    def _get_video_comments_raw(
        self,
        video_id: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch all comment threads for a video with automatic pagination.

        Videos with comments disabled are silently skipped (empty list
        returned). All other errors are re-raised.

        Args:
            video_id: YouTube video ID string.
            max_results: Maximum results per API page (max 100).
                Defaults to 100.

        Returns:
            List of raw ``commentThread`` dicts as returned by the API.

        Raises:
            HttpError: For 403 errors other than ``commentsDisabled``, or
                any other HTTP error.
        """
        all_comments: list[dict[str, Any]] = []
        next_page: Optional[str] = None

        while True:
            try:
                response = (
                    self.service.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=max_results,
                        pageToken=next_page,
                        textFormat="plainText",
                    )
                    .execute()
                )
            except HttpError as e:
                if e.resp.status == 403:
                    reason = (
                        e.error_details[0].get("reason", "") if e.error_details else ""
                    )
                    if reason == "commentsDisabled":
                        self.logger.log(
                            f"  Video '{video_id}': comments disabled, skipping."
                        )
                        break
                    # insufficientPermissions or any other 403 — propagate.
                self.logger.critical(
                    f"  Video '{video_id}': HTTP error {e.resp.status}. Re-raising."
                )
                raise e

            all_comments += response.get("items", [])
            next_page = response.get("nextPageToken")
            self.logger.log(
                f"  Video '{video_id}': {len(all_comments)} comments accumulated."
            )

            if not next_page:
                break

        return all_comments

    def get_video_comments_df(
        self,
        video_id: str,
        max_results: int = 100,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return all comments for a video as a BigQuery-friendly DataFrame.

        Args:
            video_id: YouTube video ID string.
            max_results: Maximum results per API page (max 100).
                Defaults to 100.
            columns: List of column names to return. Pass ``None`` to return
                all available columns.

        Returns:
            DataFrame with one row per comment thread.
        """
        raw = self._get_video_comments_raw(video_id, max_results=max_results)

        if not raw:
            return pd.DataFrame()

        df = self._normalize_df_columns(pd.json_normalize(raw))

        date_col = "snippet_topLevelComment_snippet_publishedAt"
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

        int_cols = [
            "snippet_totalReplyCount",
            "snippet_topLevelComment_snippet_likeCount",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        df = self._filter_columns(df, columns)
        df = self._add_extracted_at(df)

        return df

    def get_channel_comments_df(
        self,
        channel_id: str,
        part: str = "snippet,statistics",
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return all comments across every video in a channel as a DataFrame.

        Videos with comments disabled are silently skipped.

        Args:
            channel_id: YouTube channel ID string.
            part: Resource parts used when fetching the channel's videos.
                Defaults to ``'snippet,statistics'``.
            columns: List of column names to return. Pass ``None`` to return
                all available columns.

        Returns:
            DataFrame with one row per comment thread, a leading ``'video_id'``
            column, and an ``extracted_at`` timestamp.
        """
        raw_videos = self._get_channel_videos_raw(channel_id, part=part)
        if not raw_videos:
            self.logger.critical("No videos found for the channel.")
            return pd.DataFrame()

        video_ids: list[str] = [v["id"] for v in raw_videos if "id" in v]
        self.logger.log(f"Fetching comments from {len(video_ids)} videos...")

        frames: list[pd.DataFrame] = []
        for video_id in video_ids:
            df_comments = self.get_video_comments_df(video_id, columns=columns)
            self.logger.log(
                f"Video '{video_id}': {len(df_comments)} comments retrieved."
            )
            if not df_comments.empty:
                df_comments.insert(1, "video_id", video_id)
                frames.append(df_comments)

        if not frames:
            self.logger.log("No videos had available comments.")
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
