from unittest.mock import MagicMock

import pandas as pd
import pytest

from d2b_data.facebook_organic import FacebookOrganic
from d2b_data.meta_graph import GRAPH_API_VERSION, MetaErrorCategory, MetaGraphError


PAGE_ID = "302064363144019"


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


# ---------------------------------------------------------------------------
# Instantiation and repr
# ---------------------------------------------------------------------------


def test_instantiation_without_page_id(fb):
    """FacebookOrganic can be instantiated with only access_token."""
    assert not hasattr(fb, "page_id")
    assert fb.graph.api_version == GRAPH_API_VERSION == "v25.0"


def test_api_version_override():
    fb = FacebookOrganic(access_token="fake_token", api_version="v26.0")
    assert fb.graph.base_url.endswith("/v26.0")


def test_repr_shows_version_not_token():
    fb = FacebookOrganic(access_token="secret_token")
    assert repr(fb) == "<FacebookOrganic api_version=v25.0>"
    assert "secret_token" not in repr(fb)


def test_same_instance_works_for_multiple_page_ids(metrics, mocker):
    """One instance can serve multiple pages without re-instantiating."""
    fb = FacebookOrganic(access_token="fake_token")
    mocker.patch.object(fb, "get_posts", return_value=[])

    df1 = fb.get_report_dataframe("page_aaa", "2024-01-01", "2024-01-31", metrics)
    df2 = fb.get_report_dataframe("page_bbb", "2024-01-01", "2024-01-31", metrics)

    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)


def test_requests_go_through_session_with_bearer_header(fb, mocker):
    get = mocker.patch.object(
        fb.graph.session, "get", return_value=api_resp(payload={"data": [], "paging": {}})
    )

    fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")

    args, kwargs = get.call_args
    assert args[0] == f"https://graph.facebook.com/v25.0/{PAGE_ID}/posts"
    assert kwargs["headers"] == {"Authorization": "Bearer fake_token"}
    assert "access_token" not in kwargs["params"]


# ---------------------------------------------------------------------------
# _flatten_insights
# ---------------------------------------------------------------------------


def test_flatten_insights_basic(fb):
    data = [
        {"name": "post_media_view", "period": "lifetime", "values": [{"value": 1000}]},
        {"name": "post_clicks", "period": "lifetime", "values": [{"value": 50}]},
    ]
    result = fb._flatten_insights(data)
    assert result == {"post_media_view": 1000, "post_clicks": 50}


def test_flatten_insights_expands_reactions(fb):
    data = [
        {
            "name": "post_reactions_by_type_total",
            "period": "lifetime",
            "values": [{"value": {"like": 10, "love": 3, "haha": 1}}],
        }
    ]
    result = fb._flatten_insights(data)
    assert result == {"reactions_like": 10, "reactions_love": 3, "reactions_haha": 1}
    assert "post_reactions_by_type_total" not in result


def test_flatten_insights_returns_none_for_empty_values(fb):
    data = [{"name": "post_media_view", "period": "lifetime", "values": []}]
    result = fb._flatten_insights(data)
    assert result["post_media_view"] is None


def test_flatten_insights_empty_data(fb):
    assert fb._flatten_insights([]) == {}


# ---------------------------------------------------------------------------
# get_posts — normalization of shares / comments / reactions
# ---------------------------------------------------------------------------


def test_get_posts_normalizes_shares(fb, mocker):
    raw = [{"id": "p1", "shares": {"count": 7}}]
    mocker.patch.object(fb.graph, "paginate", return_value=raw)

    posts = fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")
    assert posts[0]["shares"] == 7


def test_get_posts_leaves_missing_counters_as_none(fb, mocker):
    mocker.patch.object(fb.graph, "paginate", return_value=[{"id": "p1"}])

    post = fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")[0]
    assert post["shares"] is None
    assert post["comments"] is None
    assert post["reactions"] is None


def test_get_posts_keeps_real_zero(fb, mocker):
    raw = [{"id": "p1", "shares": {"count": 0}, "comments": {"summary": {"total_count": 0}}}]
    mocker.patch.object(fb.graph, "paginate", return_value=raw)

    post = fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")[0]
    assert post["shares"] == 0
    assert post["comments"] == 0


def test_get_posts_malformed_counter_is_none(fb, mocker):
    raw = [{"id": "p1", "shares": {}, "reactions": {"summary": {}}}]
    mocker.patch.object(fb.graph, "paginate", return_value=raw)

    post = fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")[0]
    assert post["shares"] is None
    assert post["reactions"] is None


def test_get_posts_normalizes_comments(fb, mocker):
    raw = [{"id": "p1", "comments": {"summary": {"total_count": 4}}}]
    mocker.patch.object(fb.graph, "paginate", return_value=raw)

    posts = fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")
    assert posts[0]["comments"] == 4


def test_get_posts_normalizes_reactions(fb, mocker):
    raw = [{"id": "p1", "reactions": {"summary": {"total_count": 12}}}]
    mocker.patch.object(fb.graph, "paginate", return_value=raw)

    posts = fb.get_posts(PAGE_ID, "2024-01-01", "2024-01-31")
    assert posts[0]["reactions"] == 12


def test_get_posts_uses_page_id_in_endpoint(fb, mocker):
    mock_paginate = mocker.patch.object(fb.graph, "paginate", return_value=[])

    fb.get_posts("my_page_id", "2024-01-01", "2024-01-31")

    endpoint = mock_paginate.call_args[0][0]
    assert "/my_page_id/posts" in endpoint


# ---------------------------------------------------------------------------
# get_post_insights
# ---------------------------------------------------------------------------


def test_get_post_insights_returns_flat_dict(fb, mocker):
    mocker.patch.object(
        fb.graph,
        "get",
        return_value={
            "data": [
                {"name": "post_media_view", "period": "lifetime", "values": [{"value": 999}]}
            ]
        },
    )
    result = fb.get_post_insights("p1", ["post_media_view"])
    assert result == {"post_media_view": 999}


@pytest.mark.parametrize(
    "category",
    [MetaErrorCategory.AUTH, MetaErrorCategory.PERMISSION, MetaErrorCategory.QUOTA],
)
def test_get_post_insights_propagates_account_wide_errors(fb, mocker, category):
    mocker.patch.object(fb.graph, "get", side_effect=graph_error(category, 190))
    with pytest.raises(MetaGraphError) as info:
        fb.get_post_insights("p1", ["post_media_view"])
    assert info.value.category is category


@pytest.mark.parametrize(
    "category", [MetaErrorCategory.OTHER, MetaErrorCategory.TRANSIENT]
)
def test_get_post_insights_other_errors_are_logged_and_skipped(fb, mocker, category):
    mocker.patch.object(fb.graph, "get", side_effect=graph_error(category, 100))
    log = mocker.patch.object(fb.logger, "log")

    assert fb.get_post_insights("p1", ["post_media_view"]) == {}

    messages = [c.args[0] for c in log.call_args_list]
    assert any("p1" in m and "100" in m for m in messages)


def test_get_post_insights_end_to_end_auth_error(fb, mocker):
    mocker.patch.object(
        fb.graph.session,
        "get",
        return_value=api_resp(400, {"error": {"code": 190, "message": "expired"}}),
    )
    with pytest.raises(MetaGraphError) as info:
        fb.get_post_insights("p1", ["post_media_view"])
    assert info.value.category is MetaErrorCategory.AUTH
    assert "fake_token" not in str(info.value)


# ---------------------------------------------------------------------------
# get_report_dataframe
# ---------------------------------------------------------------------------


def test_get_report_dataframe_returns_empty_df_when_no_posts(fb, metrics, mocker):
    mocker.patch.object(fb, "get_posts", return_value=[])

    df = fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_report_dataframe_one_row_per_post(fb, metrics, mocker):
    posts = [
        {
            "id": "p1",
            "message": "Hello",
            "created_time": "2024-01-10T12:00:00+0000",
            "like_count": 5,
            "comments": 2,
            "reactions": 7,
            "shares": 1,
        },
        {
            "id": "p2",
            "message": "World",
            "created_time": "2024-01-11T12:00:00+0000",
            "like_count": 3,
            "comments": 0,
            "reactions": 3,
            "shares": 0,
        },
    ]
    mocker.patch.object(fb, "get_posts", return_value=posts)
    mocker.patch.object(fb, "get_post_insights", return_value={"post_media_view": 100})

    df = fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)

    assert len(df) == 2
    assert list(df["post_id"]) == ["p1", "p2"]
    assert "post_media_view" in df.columns


def test_get_report_dataframe_created_time_is_datetime(fb, metrics, mocker):
    posts = [
        {
            "id": "p1",
            "message": "",
            "created_time": "2024-01-10T12:00:00+0000",
            "like_count": 0,
            "comments": 0,
            "reactions": 0,
            "shares": 0,
        }
    ]
    mocker.patch.object(fb, "get_posts", return_value=posts)
    mocker.patch.object(fb, "get_post_insights", return_value={})

    df = fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)
    assert pd.api.types.is_datetime64_any_dtype(df["created_time"])


def test_get_report_dataframe_accepts_yyyymmdd_format(fb, metrics, mocker):
    mocker.patch.object(fb, "get_posts", return_value=[])

    df = fb.get_report_dataframe(PAGE_ID, "20240101", "20240131", metrics)
    assert df.empty


def test_get_report_dataframe_raises_on_invalid_start_date(fb, metrics):
    with pytest.raises(ValueError, match="start_date"):
        fb.get_report_dataframe(PAGE_ID, "01-01-2024", "2024-01-31", metrics)


def test_get_report_dataframe_raises_on_missing_start_date(fb, metrics):
    with pytest.raises(ValueError, match="start date"):
        fb.get_report_dataframe(PAGE_ID, "", "2024-01-31", metrics)


def test_get_report_dataframe_raises_on_invalid_end_date(fb, metrics):
    with pytest.raises(ValueError, match="end_date"):
        fb.get_report_dataframe(PAGE_ID, "2024-01-01", "31-01-2024", metrics)


def test_get_report_dataframe_raises_on_missing_end_date(fb, metrics):
    with pytest.raises(ValueError, match="end date"):
        fb.get_report_dataframe(PAGE_ID, "2024-01-01", "", metrics)


def test_get_report_dataframe_includes_page_id_column(fb, metrics, mocker):
    posts = [
        {
            "id": "p1",
            "message": "",
            "created_time": "2024-01-10T12:00:00+0000",
            "like_count": 0,
            "comments": 0,
            "reactions": 0,
            "shares": 0,
        }
    ]
    mocker.patch.object(fb, "get_posts", return_value=posts)
    mocker.patch.object(fb, "get_post_insights", return_value={})

    df = fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)
    assert df.iloc[0]["page_id"] == PAGE_ID


def test_get_report_dataframe_insight_failure_still_produces_row(fb, metrics, mocker):
    """A failed insights call doesn't drop the post row."""
    posts = [
        {
            "id": "p1",
            "created_time": "2024-01-10T12:00:00+0000",
        }
    ]
    mocker.patch.object(fb, "get_posts", return_value=posts)
    mocker.patch.object(fb, "get_post_insights", return_value={})

    df = fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)
    assert len(df) == 1


def test_get_report_dataframe_missing_values_are_not_defaults(fb, metrics, mocker):
    """No message / counters -> missing, not "" or 0; columns stay present."""
    posts = [{"id": "p1", "created_time": "2024-01-10T12:00:00+0000"}]
    mocker.patch.object(fb, "get_posts", return_value=posts)
    mocker.patch.object(fb, "get_post_insights", return_value={})

    df = fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)

    for column in ("message", "like_count", "comments", "reactions", "shares"):
        assert column in df.columns
        assert pd.isna(df.iloc[0][column])


def test_get_report_dataframe_propagates_auth_error(fb, metrics, mocker):
    mocker.patch.object(
        fb, "get_posts", return_value=[{"id": "p1", "created_time": "2024-01-10T12:00:00+0000"}]
    )
    mocker.patch.object(
        fb.graph, "get", side_effect=graph_error(MetaErrorCategory.AUTH, 190)
    )
    with pytest.raises(MetaGraphError):
        fb.get_report_dataframe(PAGE_ID, "2024-01-01", "2024-01-31", metrics)
