from unittest.mock import MagicMock

import pandas as pd
import pytest
from googleapiclient.errors import HttpError


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_instantiation_sets_api_key(yt):
    assert yt.api_key == "fake_api_key"


def test_instantiation_creates_service(yt):
    assert yt.service is not None


def test_instantiation_uses_custom_logger(mocker):
    mocker.patch("d2b_data.youtube_public.build", return_value=MagicMock())
    from d2b_data.youtube_public import YouTubePublic

    custom_logger = MagicMock()
    yt = YouTubePublic(api_key="key", logger=custom_logger)
    assert yt.logger is custom_logger


def test_instantiation_uses_default_logger_when_none(yt):
    assert hasattr(yt.logger, "log")
    assert hasattr(yt.logger, "critical")


# ---------------------------------------------------------------------------
# _normalize_df_columns
# ---------------------------------------------------------------------------


def test_normalize_df_columns_replaces_dots(yt):
    df = pd.DataFrame({"snippet.title": ["a"], "statistics.viewCount": [1]})
    result = yt._normalize_df_columns(df)
    assert list(result.columns) == ["snippet_title", "statistics_viewCount"]


def test_normalize_df_columns_no_dots_unchanged(yt):
    df = pd.DataFrame({"title": ["a"], "count": [1]})
    result = yt._normalize_df_columns(df)
    assert list(result.columns) == ["title", "count"]


# ---------------------------------------------------------------------------
# _filter_columns
# ---------------------------------------------------------------------------


def test_filter_columns_none_returns_all(yt):
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = yt._filter_columns(df, None)
    assert list(result.columns) == ["a", "b", "c"]


def test_filter_columns_valid_subset(yt):
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = yt._filter_columns(df, ["a", "c"])
    assert list(result.columns) == ["a", "c"]


def test_filter_columns_raises_on_invalid(yt):
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Invalid columns"):
        yt._filter_columns(df, ["a", "nonexistent"])


# ---------------------------------------------------------------------------
# _add_extracted_at
# ---------------------------------------------------------------------------


def test_add_extracted_at_prepends_column(yt):
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = yt._add_extracted_at(df)
    assert result.columns[0] == "extracted_at"
    assert pd.api.types.is_datetime64_any_dtype(result["extracted_at"])


# ---------------------------------------------------------------------------
# list_channels
# ---------------------------------------------------------------------------


def test_list_channels_requires_identifier(yt):
    with pytest.raises(ValueError, match="Must specify"):
        yt.list_channels()


def test_list_channels_by_id(yt):
    yt.service.channels.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "UC123", "snippet": {"title": "Test Channel"}}]
    }
    df = yt.list_channels(id="UC123")
    assert not df.empty
    assert "id" in df.columns


def test_list_channels_returns_empty_when_no_items(yt):
    yt.service.channels.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    df = yt.list_channels(id="UC_nonexistent")
    assert df.empty


def test_list_channels_re_raises_api_error(yt):
    yt.service.channels.return_value.list.return_value.execute.side_effect = (
        Exception("quota exceeded")
    )
    with pytest.raises(Exception, match="quota exceeded"):
        yt.list_channels(id="UC123")


# ---------------------------------------------------------------------------
# get_playlist_videos
# ---------------------------------------------------------------------------


def test_get_playlist_videos_single_page(yt):
    yt.service.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [
            {"contentDetails": {"videoId": "v1"}},
            {"contentDetails": {"videoId": "v2"}},
        ],
    }
    result = yt.get_playlist_videos("PL123")
    assert result == ["v1", "v2"]


def test_get_playlist_videos_pagination(yt):
    responses = [
        {
            "items": [{"contentDetails": {"videoId": "v1"}}],
            "nextPageToken": "page2",
        },
        {
            "items": [{"contentDetails": {"videoId": "v2"}}],
        },
    ]
    yt.service.playlistItems.return_value.list.return_value.execute.side_effect = (
        responses
    )
    result = yt.get_playlist_videos("PL123")
    assert result == ["v1", "v2"]


def test_get_playlist_videos_empty_playlist(yt):
    yt.service.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [],
    }
    result = yt.get_playlist_videos("PL_empty")
    assert result == []


# ---------------------------------------------------------------------------
# get_video_statistics
# ---------------------------------------------------------------------------


def test_get_video_statistics_returns_df(yt):
    yt.service.videos.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "v1",
                "snippet": {"title": "Video 1"},
                "statistics": {"viewCount": "100"},
            }
        ]
    }
    df = yt.get_video_statistics(["v1"])
    assert len(df) == 1
    assert "id" in df.columns


def test_get_video_statistics_empty_ids(yt):
    df = yt.get_video_statistics([])
    assert df.empty


def test_get_video_statistics_batches_over_50(yt):
    ids = [f"v{i}" for i in range(75)]
    call_count = 0

    def fake_execute():
        nonlocal call_count
        call_count += 1
        return {"items": [{"id": f"v{call_count}", "snippet": {"title": "T"}}]}

    yt.service.videos.return_value.list.return_value.execute.side_effect = (
        fake_execute
    )
    df = yt.get_video_statistics(ids)
    assert call_count == 2


def test_get_video_statistics_no_items_returns_empty(yt):
    yt.service.videos.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    df = yt.get_video_statistics(["v1"])
    assert df.empty


def test_get_video_statistics_filters_columns(yt):
    yt.service.videos.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "v1",
                "snippet": {"title": "Video 1"},
                "statistics": {"viewCount": "100"},
            }
        ]
    }
    df = yt.get_video_statistics(["v1"], columns=["id"])
    assert list(df.columns) == ["id"]


# ---------------------------------------------------------------------------
# get_channel_videos_df
# ---------------------------------------------------------------------------


def _setup_channel_with_videos(yt, video_items):
    yt.service.channels.return_value.list.return_value.execute.return_value = {
        "items": [
            {"contentDetails": {"relatedPlaylists": {"uploads": "UU123"}}}
        ]
    }
    yt.service.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [
            {"contentDetails": {"videoId": item["id"]}} for item in video_items
        ],
    }
    yt.service.videos.return_value.list.return_value.execute.return_value = {
        "items": video_items,
    }


def test_get_channel_videos_df_returns_dataframe(yt):
    _setup_channel_with_videos(yt, [
        {
            "id": "v1",
            "snippet": {
                "title": "Test",
                "publishedAt": "2024-01-01T10:00:00Z",
            },
            "statistics": {
                "viewCount": "500",
                "likeCount": "10",
                "commentCount": "2",
                "favoriteCount": "0",
            },
        }
    ])
    df = yt.get_channel_videos_df("UC123")
    assert len(df) == 1
    assert "extracted_at" in df.columns


def test_get_channel_videos_df_casts_int_columns(yt):
    _setup_channel_with_videos(yt, [
        {
            "id": "v1",
            "snippet": {"title": "T", "publishedAt": "2024-01-01T10:00:00Z"},
            "statistics": {
                "viewCount": "500",
                "likeCount": "10",
                "commentCount": "2",
                "favoriteCount": "0",
            },
        }
    ])
    df = yt.get_channel_videos_df("UC123")
    assert df["statistics_viewCount"].dtype.name == "Int64"
    assert df["statistics_likeCount"].dtype.name == "Int64"


def test_get_channel_videos_df_parses_publish_date(yt):
    _setup_channel_with_videos(yt, [
        {
            "id": "v1",
            "snippet": {"title": "T", "publishedAt": "2024-01-01T10:00:00Z"},
            "statistics": {"viewCount": "1"},
        }
    ])
    df = yt.get_channel_videos_df("UC123")
    assert pd.api.types.is_datetime64_any_dtype(df["snippet_publishedAt"])


def test_get_channel_videos_df_empty_channel(yt):
    yt.service.channels.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    df = yt.get_channel_videos_df("UC_empty")
    assert df.empty


def test_get_channel_videos_df_filters_columns(yt):
    _setup_channel_with_videos(yt, [
        {
            "id": "v1",
            "snippet": {"title": "T", "publishedAt": "2024-01-01T10:00:00Z"},
            "statistics": {"viewCount": "1"},
        }
    ])
    df = yt.get_channel_videos_df("UC123", columns=["id", "statistics_viewCount"])
    assert "extracted_at" in df.columns
    assert "id" in df.columns
    assert "snippet_title" not in df.columns


# ---------------------------------------------------------------------------
# get_video_comments_df
# ---------------------------------------------------------------------------


def test_get_video_comments_df_returns_dataframe(yt):
    yt.service.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "c1",
                "snippet": {
                    "totalReplyCount": "3",
                    "topLevelComment": {
                        "snippet": {
                            "textDisplay": "Nice!",
                            "likeCount": "5",
                            "publishedAt": "2024-01-15T10:00:00Z",
                        }
                    },
                },
            }
        ],
    }
    df = yt.get_video_comments_df("v1")
    assert len(df) == 1
    assert "extracted_at" in df.columns


def test_get_video_comments_df_empty_when_no_comments(yt):
    yt.service.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [],
    }
    df = yt.get_video_comments_df("v1")
    assert df.empty


def test_get_video_comments_df_handles_comments_disabled(yt):
    resp = MagicMock()
    resp.status = 403
    error = HttpError(resp=resp, content=b"comments disabled")
    error.error_details = [{"reason": "commentsDisabled"}]
    yt.service.commentThreads.return_value.list.return_value.execute.side_effect = error

    df = yt.get_video_comments_df("v1")
    assert df.empty


def test_get_video_comments_df_re_raises_other_403(yt):
    resp = MagicMock()
    resp.status = 403
    error = HttpError(resp=resp, content=b"forbidden")
    error.error_details = [{"reason": "insufficientPermissions"}]
    yt.service.commentThreads.return_value.list.return_value.execute.side_effect = error

    with pytest.raises(HttpError):
        yt.get_video_comments_df("v1")


def test_get_video_comments_df_casts_int_columns(yt):
    yt.service.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "c1",
                "snippet": {
                    "totalReplyCount": "3",
                    "topLevelComment": {
                        "snippet": {
                            "textDisplay": "Hi",
                            "likeCount": "5",
                            "publishedAt": "2024-01-15T10:00:00Z",
                        }
                    },
                },
            }
        ],
    }
    df = yt.get_video_comments_df("v1")
    assert df["snippet_totalReplyCount"].dtype.name == "Int64"


def test_get_video_comments_df_filters_columns(yt):
    yt.service.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "c1",
                "snippet": {
                    "totalReplyCount": "1",
                    "topLevelComment": {
                        "snippet": {
                            "textDisplay": "Hi",
                            "likeCount": "0",
                            "publishedAt": "2024-01-15T10:00:00Z",
                        }
                    },
                },
            }
        ],
    }
    df = yt.get_video_comments_df("v1", columns=["id"])
    assert "extracted_at" in df.columns
    assert "id" in df.columns


# ---------------------------------------------------------------------------
# get_channel_comments_df
# ---------------------------------------------------------------------------


def test_get_channel_comments_df_returns_combined(yt, mocker):
    mocker.patch.object(
        yt,
        "_get_channel_videos_raw",
        return_value=[{"id": "v1"}, {"id": "v2"}],
    )
    def _make_comment_df(*args, **kwargs):
        return pd.DataFrame({
            "extracted_at": [pd.Timestamp.now()],
            "id": ["c1"],
            "text": ["hello"],
        })

    mocker.patch.object(yt, "get_video_comments_df", side_effect=_make_comment_df)

    df = yt.get_channel_comments_df("UC123")
    assert not df.empty
    assert "video_id" in df.columns


def test_get_channel_comments_df_empty_when_no_videos(yt, mocker):
    mocker.patch.object(yt, "_get_channel_videos_raw", return_value=[])
    df = yt.get_channel_comments_df("UC_empty")
    assert df.empty


def test_get_channel_comments_df_skips_videos_with_no_comments(yt, mocker):
    mocker.patch.object(
        yt, "_get_channel_videos_raw", return_value=[{"id": "v1"}, {"id": "v2"}]
    )
    mocker.patch.object(yt, "get_video_comments_df", return_value=pd.DataFrame())

    df = yt.get_channel_comments_df("UC123")
    assert df.empty
