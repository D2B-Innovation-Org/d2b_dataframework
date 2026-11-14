from unittest.mock import MagicMock

import pandas as pd
import pytest

from d2b_data.instagram_organic import InstagramOrganic
from d2b_data.meta_graph import GRAPH_API_VERSION, MetaErrorCategory, MetaGraphError


def graph_error(category, code=None, http_status=400):
    return MetaGraphError(
        f"error {code}", category, http_status=http_status, api_code=code
    )


def api_resp(status=200, payload=None):
    """Fake requests.Response for Session.get."""
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    return r


MEDIA = [
    {"id": "1", "media_product_type": "FEED"},
    {"id": "2", "media_product_type": "REELS"},
    {"id": "3", "media_product_type": "FEED"},
    {"id": "4", "media_product_type": "AD"},
]


# ---------------------------------------------------------------------------
# Instantiation and repr
# ---------------------------------------------------------------------------


def test_default_api_version(ig):
    assert ig.graph.api_version == GRAPH_API_VERSION == "v25.0"
    assert ig.graph.base_url == "https://graph.facebook.com/v25.0"


def test_api_version_override():
    ig = InstagramOrganic(access_token="fake_token", api_version="v26.0")
    assert ig.graph.base_url.endswith("/v26.0")


def test_repr_does_not_need_page_id_nor_leak_token():
    ig = InstagramOrganic(access_token="secret_token")
    assert repr(ig) == "<InstagramOrganic api_version=v25.0>"
    assert "secret_token" not in repr(ig)


def test_requests_go_through_session_with_bearer_header(ig, mocker):
    get = mocker.patch.object(
        ig.graph.session, "get", return_value=api_resp(payload={"data": [], "paging": {}})
    )

    ig._get_stories("acc123")

    args, kwargs = get.call_args
    assert args[0] == "https://graph.facebook.com/v25.0/acc123/stories"
    assert kwargs["headers"] == {"Authorization": "Bearer fake_token"}
    assert "access_token" not in kwargs["params"]


# ---------------------------------------------------------------------------
# _get_media — media fetch + type filtering
# ---------------------------------------------------------------------------


def test_get_media_filters_by_product_type(ig, mocker):
    """Only media matching media_product_type is returned."""
    all_media = [
        {"id": "1", "media_product_type": "FEED"},
        {"id": "2", "media_product_type": "REELS"},
        {"id": "3", "media_product_type": "FEED"},
    ]
    mocker.patch.object(ig.graph, "paginate", return_value=all_media)

    result = ig._get_media("acc123", "2024-01-01", "2024-01-31", "FEED")
    assert len(result) == 2
    assert all(m["media_product_type"] == "FEED" for m in result)


def test_get_media_returns_empty_when_no_match(ig, mocker):
    """Returns empty list when no media matches the requested type."""
    mocker.patch.object(
        ig.graph, "paginate", return_value=[{"id": "1", "media_product_type": "REELS"}]
    )

    result = ig._get_media("acc123", "2024-01-01", "2024-01-31", "FEED")
    assert result == []


# ---------------------------------------------------------------------------
# _get_stories
# ---------------------------------------------------------------------------


def test_get_stories_uses_stories_endpoint(ig, mocker):
    """Stories are fetched from /{ig_account_id}/stories."""
    mock_paginate = mocker.patch.object(ig.graph, "paginate", return_value=[{"id": "s1"}])

    ig._get_stories("acc123")

    call_args = mock_paginate.call_args
    assert "/acc123/stories" in call_args[0][0]


def test_get_stories_returns_all_stories(ig, mocker):
    stories = [{"id": "s1"}, {"id": "s2"}]
    mocker.patch.object(ig.graph, "paginate", return_value=stories)

    result = ig._get_stories("acc123")
    assert result == stories


# ---------------------------------------------------------------------------
# _flatten_insights
# ---------------------------------------------------------------------------


def test_flatten_insights_extracts_lifetime_values(ig):
    data = [
        {"name": "views", "period": "lifetime", "values": [{"value": 500}]},
        {"name": "reach", "period": "lifetime", "values": [{"value": 300}]},
    ]
    result = ig._flatten_insights(data)
    assert result == {"views": 500, "reach": 300}


def test_flatten_insights_returns_none_for_empty_values(ig):
    data = [{"name": "views", "period": "lifetime", "values": []}]
    result = ig._flatten_insights(data)
    assert result["views"] is None


def test_flatten_insights_empty_data(ig):
    assert ig._flatten_insights([]) == {}


# ---------------------------------------------------------------------------
# _get_media_insights
# ---------------------------------------------------------------------------


def test_get_media_insights_returns_flat_dict(ig, mocker):
    """Successful insights call returns flattened metrics."""
    mocker.patch.object(
        ig.graph,
        "get",
        return_value={
            "data": [{"name": "likes", "period": "lifetime", "values": [{"value": 42}]}]
        },
    )
    result = ig._get_media_insights("media123", ["likes"])
    assert result == {"likes": 42}


@pytest.mark.parametrize(
    "category",
    [MetaErrorCategory.AUTH, MetaErrorCategory.PERMISSION, MetaErrorCategory.QUOTA],
)
def test_get_media_insights_propagates_account_wide_errors(ig, mocker, category):
    mocker.patch.object(ig.graph, "get", side_effect=graph_error(category, 190))

    with pytest.raises(MetaGraphError) as info:
        ig._get_media_insights("media123", ["likes"])
    assert info.value.category is category


@pytest.mark.parametrize(
    "category", [MetaErrorCategory.OTHER, MetaErrorCategory.TRANSIENT]
)
def test_get_media_insights_other_errors_are_logged_and_skipped(ig, mocker, category):
    mocker.patch.object(ig.graph, "get", side_effect=graph_error(category, 100))
    log = mocker.patch.object(ig.logger, "log")

    assert ig._get_media_insights("media123", ["likes"]) == {}

    messages = [c.args[0] for c in log.call_args_list]
    assert any("media123" in m and "100" in m for m in messages)


def test_get_media_insights_end_to_end_quota_error(ig, mocker):
    mocker.patch.object(
        ig.graph.session,
        "get",
        return_value=api_resp(400, {"error": {"code": 4, "message": "limit"}}),
    )
    with pytest.raises(MetaGraphError) as info:
        ig._get_media_insights("media123", ["likes"])
    assert info.value.category is MetaErrorCategory.QUOTA


# ---------------------------------------------------------------------------
# _build_dataframe
# ---------------------------------------------------------------------------


def test_build_dataframe_returns_empty_df_for_empty_media(ig, feed_metrics):
    df = ig._build_dataframe([], "acc123", feed_metrics)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_build_dataframe_one_row_per_media(ig, feed_metrics, mocker):
    media_list = [
        {
            "id": "m1",
            "media_type": "IMAGE",
            "media_product_type": "FEED",
            "caption": "hello",
            "timestamp": "2024-01-01T10:00:00+0000",
            "permalink": "https://ig.com/p/abc",
        },
        {
            "id": "m2",
            "media_type": "VIDEO",
            "media_product_type": "FEED",
            "caption": "world",
            "timestamp": "2024-01-02T10:00:00+0000",
            "permalink": "https://ig.com/p/def",
        },
    ]
    mocker.patch.object(ig, "_get_media_insights", return_value={"views": 100})

    df = ig._build_dataframe(media_list, "acc123", feed_metrics)

    assert len(df) == 2
    assert list(df["media_id"]) == ["m1", "m2"]
    assert "views" in df.columns


def test_build_dataframe_timestamp_is_datetime(ig, feed_metrics, mocker):
    media_list = [
        {
            "id": "m1",
            "media_type": "IMAGE",
            "media_product_type": "FEED",
            "caption": "",
            "timestamp": "2024-01-01T10:00:00+0000",
            "permalink": "",
        }
    ]
    mocker.patch.object(ig, "_get_media_insights", return_value={})

    df = ig._build_dataframe(media_list, "acc123", feed_metrics)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_build_dataframe_insight_failure_still_produces_row(ig, feed_metrics, mocker):
    """A failed insights fetch doesn't drop the media row."""
    media_list = [
        {
            "id": "m1",
            "media_type": "IMAGE",
            "media_product_type": "FEED",
            "caption": "",
            "timestamp": "2024-01-01T10:00:00+0000",
            "permalink": "",
        }
    ]
    mocker.patch.object(ig, "_get_media_insights", return_value={})

    df = ig._build_dataframe(media_list, "acc123", feed_metrics)
    assert len(df) == 1
    assert df.iloc[0]["media_id"] == "m1"


def test_build_dataframe_missing_fields_are_none_not_defaults(ig, feed_metrics, mocker):
    mocker.patch.object(ig, "_get_media_insights", return_value={})

    df = ig._build_dataframe([{"id": "m1", "timestamp": "2024-01-01T10:00:00+0000"}], "acc123", feed_metrics)

    for column in ("caption", "permalink", "media_type", "media_product_type"):
        assert column in df.columns
        assert pd.isna(df.iloc[0][column])


# ---------------------------------------------------------------------------
# get_feed / get_reels / get_stories — public API
# ---------------------------------------------------------------------------


def test_get_feed_calls_get_media_with_feed_type(ig, feed_metrics, mocker):
    mock_get_media = mocker.patch.object(ig, "_get_media", return_value=[])
    mocker.patch.object(ig, "_build_dataframe", return_value=pd.DataFrame())

    ig.get_feed("acc123", feed_metrics, since="2024-01-01", until="2024-01-31")

    mock_get_media.assert_called_once_with("acc123", "2024-01-01", "2024-01-31", "FEED")


def test_get_reels_calls_get_media_with_reels_type(ig, feed_metrics, mocker):
    mock_get_media = mocker.patch.object(ig, "_get_media", return_value=[])
    mocker.patch.object(ig, "_build_dataframe", return_value=pd.DataFrame())

    ig.get_reels("acc123", feed_metrics, since="2024-01-01", until="2024-01-31")

    mock_get_media.assert_called_once_with(
        "acc123", "2024-01-01", "2024-01-31", "REELS"
    )


def test_get_stories_calls_get_stories(ig, story_metrics, mocker):
    mock_get_stories = mocker.patch.object(ig, "_get_stories", return_value=[])
    mocker.patch.object(ig, "_build_dataframe", return_value=pd.DataFrame())

    ig.get_stories("acc123", story_metrics)

    mock_get_stories.assert_called_once_with("acc123")


def test_get_feed_returns_dataframe(ig, feed_metrics, mocker):
    mocker.patch.object(ig, "_get_media", return_value=[])
    mocker.patch.object(ig, "_build_dataframe", return_value=pd.DataFrame({"x": [1]}))

    df = ig.get_feed("acc123", feed_metrics)
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


def test_get_all_lists_media_once_and_splits_client_side(
    ig, feed_metrics, story_metrics, mocker
):
    paginate = mocker.patch.object(
        ig.graph, "paginate", side_effect=[MEDIA, [{"id": "s1"}]]
    )
    mocker.patch.object(ig, "_get_media_insights", return_value={"views": 1})

    result = ig.get_all(
        "acc123",
        feed_metrics=feed_metrics,
        reels_metrics=feed_metrics,
        story_metrics=story_metrics,
        since="2024-01-01",
        until="2024-01-31",
    )

    endpoints = [c.args[0] for c in paginate.call_args_list]
    assert endpoints == ["/acc123/media", "/acc123/stories"]
    assert set(result.keys()) == {"feed", "reels", "stories"}
    assert list(result["feed"]["media_id"]) == ["1", "3"]
    assert list(result["reels"]["media_id"]) == ["2"]
    assert list(result["stories"]["media_id"]) == ["s1"]
    assert all(isinstance(v, pd.DataFrame) for v in result.values())


def test_get_all_uses_each_metric_list(ig, mocker):
    mocker.patch.object(ig.graph, "paginate", side_effect=[MEDIA, []])
    insights = mocker.patch.object(ig, "_get_media_insights", return_value={})

    ig.get_all("acc123", ["f"], ["r"], ["s"], "2024-01-01", "2024-01-31")

    used = {c.args[0]: c.args[1] for c in insights.call_args_list}
    assert used["1"] == ["f"]
    assert used["2"] == ["r"]


def test_get_all_empty_account(ig, feed_metrics, mocker):
    mocker.patch.object(ig.graph, "paginate", side_effect=[[], []])

    result = ig.get_all("acc123", feed_metrics, feed_metrics, feed_metrics)

    assert all(df.empty for df in result.values())


# ---------------------------------------------------------------------------
# _resolve_dates
# ---------------------------------------------------------------------------


def test_resolve_dates_uses_provided_values(ig):
    since, until = ig._resolve_dates("2024-01-01", "2024-01-31")
    assert since == "2024-01-01"
    assert until == "2024-01-31"


def test_resolve_dates_defaults_when_none(ig):
    since, until = ig._resolve_dates(None, None)
    assert since is not None
    assert until is not None
    assert since < until
