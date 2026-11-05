"""Tests for LinkedinOrganic. Payload shapes mirror real API responses."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

ORG = "35521656"
DAY_MS = 86_400_000
OCT1 = 1790812800000  # 2026-10-01 00:00 UTC


def resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(str(status))
    return r


def post(urn, created_ms, content=None, **extra):
    p = {
        "id": urn,
        "author": f"urn:li:organization:{ORG}",
        "createdAt": created_ms,
        "publishedAt": created_ms,
        "lastModifiedAt": created_ms,
        "commentary": "hola",
        "visibility": "PUBLIC",
        "lifecycleState": "PUBLISHED",
        "content": content if content is not None else {"media": {"id": "x"}},
    }
    p.update(extra)
    return p


def stats(**kw):
    base = {
        "impressionCount": 100,
        "uniqueImpressionsCount": 60,
        "clickCount": 5,
        "likeCount": 3,
        "commentCount": 1,
        "shareCount": 2,
        "engagement": 0.11,
    }
    base.update(kw)
    return base


def route(mocker, linkedin, handlers):
    """Patch session.send; handlers maps URL substring -> response(s)."""
    sent = []

    def send(prepared, **kwargs):
        sent.append(prepared.url)
        assert kwargs.get("timeout"), "requests must carry a timeout"
        for key, value in handlers.items():
            if key in prepared.url:
                return value.pop(0) if isinstance(value, list) else value
        raise AssertionError(f"unexpected URL {prepared.url}")

    mocker.patch.object(linkedin.session, "send", side_effect=send)
    return sent


# ------------------------------------------------------------------
# Init & auth
# ------------------------------------------------------------------


def test_instance_with_valid_token(linkedin):
    assert linkedin.token == "fake_token_123"
    assert linkedin.headers["Authorization"] == "Bearer fake_token_123"
    assert linkedin.headers["X-Restli-Protocol-Version"] == "2.0.0"


def test_instance_no_file_specified(linkedin_no_file):
    assert linkedin_no_file.token is None
    assert linkedin_no_file.headers is None


def test_instance_bad_file(linkedin_bad_file):
    assert linkedin_bad_file.token is None
    assert linkedin_bad_file.headers is None


def test_load_token_missing_access_token_key(mocker):
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.load", return_value={"refresh_token": "x"})
    from d2b_data.linkedin_organic import LinkedinOrganic

    li = LinkedinOrganic(token_path="incomplete.json")
    assert li.token is None and li.headers is None


def test_refresh_access_token_saves_and_keeps_refresh(tmp_path, mocker):
    import json

    from d2b_data.linkedin_organic import LinkedinOrganic

    path = tmp_path / "token.json"
    path.write_text(json.dumps({"access_token": "old", "refresh_token": "r1"}))
    li = LinkedinOrganic(token_path=str(path))
    post_mock = mocker.patch(
        "d2b_data.linkedin_organic.requests.post",
        return_value=resp(200, {"access_token": "new", "expires_in": 5183999}),
    )

    assert li.refresh_access_token("cid", "secret") is True
    saved = json.loads(path.read_text())
    assert saved["access_token"] == "new"
    assert saved["refresh_token"] == "r1"
    assert li.headers["Authorization"] == "Bearer new"
    assert post_mock.call_args.kwargs["data"]["grant_type"] == "refresh_token"


def test_refresh_access_token_without_refresh_token(tmp_path):
    import json

    from d2b_data.linkedin_organic import LinkedinOrganic

    path = tmp_path / "token.json"
    path.write_text(json.dumps({"access_token": "old"}))
    li = LinkedinOrganic(token_path=str(path))
    assert li.refresh_access_token("cid", "secret") is False


def test_refresh_access_token_http_error(tmp_path, mocker):
    import json

    from d2b_data.linkedin_organic import LinkedinOrganic

    path = tmp_path / "token.json"
    path.write_text(json.dumps({"access_token": "old", "refresh_token": "r1"}))
    li = LinkedinOrganic(token_path=str(path))
    mocker.patch("d2b_data.linkedin_organic.requests.post", return_value=resp(401))
    assert li.refresh_access_token("cid", "bad") is False
    assert json.loads(path.read_text())["access_token"] == "old"


# ------------------------------------------------------------------
# HTTP helper
# ------------------------------------------------------------------


def test_request_get_no_headers_raises(linkedin_no_file):
    with pytest.raises(RuntimeError, match="Headers not set"):
        linkedin_no_file._request_get("https://example.com")


def test_request_get_success_uses_timeout_and_raw_url(linkedin, mocker):
    url = "https://api.linkedin.com/rest/x?shares=List(urn%3Ali%3Ashare%3A1)"
    sent = route(mocker, linkedin, {"rest/x": resp(200, {"elements": []})})
    assert linkedin._request_get(url) == {"elements": []}
    assert sent == [url]  # Rest.li syntax not re-encoded


def test_request_get_429_raises_quota_error(linkedin, mocker):
    from d2b_data.linkedin_organic import QuotaExhaustedError

    route(mocker, linkedin, {"rest": resp(429)})
    with pytest.raises(QuotaExhaustedError):
        linkedin._request_get("https://api.linkedin.com/rest/test")


def test_request_get_transient_error_retries(linkedin, mocker):
    route(mocker, linkedin, {"rest": [resp(500), resp(200, {"ok": True})]})
    sleep = mocker.patch("d2b_data.linkedin_organic.time.sleep")
    assert linkedin._request_get("https://api.linkedin.com/rest/t") == {"ok": True}
    sleep.assert_called_once_with(1)


def test_request_get_exhausts_retries(linkedin, mocker):
    route(mocker, linkedin, {"rest": [resp(500), resp(500), resp(500)]})
    mocker.patch("d2b_data.linkedin_organic.time.sleep")
    with pytest.raises(requests.exceptions.HTTPError):
        linkedin._request_get("https://api.linkedin.com/rest/t", max_retries=2)


def test_request_get_non_retryable_error(linkedin, mocker):
    route(mocker, linkedin, {"rest": resp(403)})
    with pytest.raises(requests.exceptions.HTTPError):
        linkedin._request_get("https://api.linkedin.com/rest/t")


# ------------------------------------------------------------------
# get_managed_pages
# ------------------------------------------------------------------


def test_get_managed_pages(linkedin, mocker):
    payload = {
        "elements": [
            {
                "organization": f"urn:li:organization:{ORG}",
                "role": "CONTENT_ADMINISTRATOR",
                "state": "APPROVED",
            }
        ]
    }
    sent = route(mocker, linkedin, {"organizationAcls": resp(200, payload)})
    df = linkedin.get_managed_pages(role="ADMINISTRATOR")
    assert df.to_dict("records") == [
        {"org_id": ORG, "role": "CONTENT_ADMINISTRATOR", "state": "APPROVED"}
    ]
    assert "role=ADMINISTRATOR" in sent[0]


def test_get_managed_pages_empty_and_failure(linkedin, mocker):
    route(mocker, linkedin, {"organizationAcls": [resp(200, {}), resp(403)]})
    assert list(linkedin.get_managed_pages().columns) == ["org_id", "role", "state"]
    assert linkedin.get_managed_pages() is None


# ------------------------------------------------------------------
# Posts
# ------------------------------------------------------------------


def test_fetch_posts_uses_sort_by_created_and_max_page(linkedin, mocker):
    sent = route(mocker, linkedin, {"/posts": resp(200, {"elements": []})})
    linkedin._fetch_paginated_posts(ORG, "2026-10-01", "2026-10-31")
    assert "sortBy=CREATED" in sent[0]
    assert "count=100" in sent[0]


def test_fetch_posts_end_date_is_inclusive(linkedin, mocker):
    last_day_noon = OCT1 + 30 * DAY_MS + 12 * 3_600_000  # 2026-10-31 12:00
    next_day = OCT1 + 31 * DAY_MS  # 2026-11-01 00:00
    payload = {
        "elements": [
            post("urn:li:share:3", next_day),
            post("urn:li:share:2", last_day_noon),
            post("urn:li:share:1", OCT1),
        ]
    }
    route(mocker, linkedin, {"/posts": resp(200, payload)})
    out = linkedin._fetch_paginated_posts(ORG, "2026-10-01", "2026-10-31")
    assert [p["id"] for p in out] == ["urn:li:share:2", "urn:li:share:1"]


def test_fetch_posts_stops_at_older_post_without_extra_calls(linkedin, mocker):
    page = {"elements": [post(f"urn:li:share:{i}", OCT1 + DAY_MS) for i in range(99)]}
    page["elements"].append(post("urn:li:share:old", OCT1 - DAY_MS))
    sent = route(mocker, linkedin, {"/posts": resp(200, page)})
    out = linkedin._fetch_paginated_posts(ORG, "2026-10-01", "2026-10-31")
    assert len(out) == 99
    assert len(sent) == 1


def test_fetch_posts_short_page_ends_paging(linkedin, mocker):
    sent = route(
        mocker,
        linkedin,
        {"/posts": resp(200, {"elements": [post("urn:li:share:1", OCT1)]})},
    )
    linkedin._fetch_paginated_posts(ORG, "2026-01-01", "2026-12-31")
    assert len(sent) == 1


def test_fetch_posts_paginates_full_pages(linkedin, mocker):
    full = {"elements": [post(f"urn:li:share:{i}", OCT1) for i in range(100)]}
    sent = route(
        mocker, linkedin, {"/posts": [resp(200, full), resp(200, {"elements": []})]}
    )
    out = linkedin._fetch_paginated_posts(ORG, "2026-01-01", "2026-12-31")
    assert len(out) == 100
    assert "start=100" in sent[1]


def test_fetch_posts_bad_dates_make_no_calls(linkedin, mocker):
    sent = route(mocker, linkedin, {})
    assert linkedin._fetch_paginated_posts(ORG, "01-10-2026", "2026-10-31") is None
    assert sent == []


def test_fetch_posts_api_error_returns_none(linkedin, mocker):
    route(mocker, linkedin, {"/posts": resp(500)})
    mocker.patch("d2b_data.linkedin_organic.time.sleep")
    assert linkedin._fetch_paginated_posts(ORG, "2026-10-01", "2026-10-31") is None


def test_get_posts_columns(linkedin, mocker):
    payload = {
        "elements": [
            post("urn:li:share:1", OCT1, {"article": {"source": "https://d2b.cl"}}),
            post(
                "urn:li:share:2",
                OCT1,
                {},
                reshareContext={"parent": "urn:li:share:0"},
            ),
            post("urn:li:ugcPost:3", OCT1, {"multiImage": {"images": []}}),
        ]
    }
    route(mocker, linkedin, {"/posts": resp(200, payload)})
    df = linkedin.get_posts(ORG, "2026-10-01", "2026-10-01")
    assert df["content_type"].tolist() == ["ARTICLE", "NONE", "MULTIIMAGE"]
    assert df.loc[1, "reshare_parent"] == "urn:li:share:0"
    assert pd.isna(df.loc[1, "content_json"])
    assert "d2b.cl" in df.loc[0, "content_json"]
    assert df.loc[0, "created_at"] == pd.Timestamp("2026-10-01", tz="UTC")
    assert (df["org_id"] == ORG).all()


# ------------------------------------------------------------------
# Post statistics
# ------------------------------------------------------------------


def test_post_statistics_batches_shares_and_ugcposts(linkedin, mocker):
    from d2b_data import linkedin_organic

    mocker.patch.object(linkedin_organic, "STATS_BATCH_SIZE", 2)
    urns = ["urn:li:share:1", "urn:li:ugcPost:2", "urn:li:share:3"]
    sent = route(
        mocker,
        linkedin,
        {
            "ShareStatistics": [
                resp(
                    200,
                    {
                        "elements": [
                            {"share": urns[0], "totalShareStatistics": stats()},
                            {"ugcPost": urns[1], "totalShareStatistics": stats()},
                        ]
                    },
                ),
                resp(
                    200,
                    {"elements": [{"share": urns[2], "totalShareStatistics": stats()}]},
                ),
            ]
        },
    )
    df = linkedin.get_post_statistics(ORG, urns)
    assert len(sent) == 2
    assert "shares=List(urn%3Ali%3Ashare%3A1)" in sent[0]
    assert "ugcPosts=List(urn%3Ali%3AugcPost%3A2)" in sent[0]
    assert df["post_urn"].tolist() == urns
    assert list(df.columns) == [
        "post_urn",
        "impression_count",
        "unique_impressions_count",
        "click_count",
        "like_count",
        "comment_count",
        "share_count",
        "engagement",
    ]


def test_post_statistics_empty_makes_no_calls(linkedin, mocker):
    sent = route(mocker, linkedin, {})
    assert linkedin.get_post_statistics(ORG, []).empty
    assert sent == []


def test_post_statistics_failure_returns_none(linkedin, mocker):
    route(mocker, linkedin, {"ShareStatistics": resp(403)})
    assert linkedin.get_post_statistics(ORG, ["urn:li:share:1"]) is None


def test_post_statistics_quota_propagates(linkedin, mocker):
    from d2b_data.linkedin_organic import QuotaExhaustedError

    route(mocker, linkedin, {"ShareStatistics": resp(429)})
    with pytest.raises(QuotaExhaustedError):
        linkedin.get_post_statistics(ORG, ["urn:li:share:1"])


def test_posts_with_statistics_merge(linkedin, mocker):
    payload = {"elements": [post("urn:li:share:1", OCT1), post("urn:li:share:2", OCT1)]}
    sent = route(
        mocker,
        linkedin,
        {
            "/posts": resp(200, payload),
            "ShareStatistics": resp(
                200,
                {
                    "elements": [
                        {
                            "share": "urn:li:share:1",
                            "totalShareStatistics": stats(likeCount=7),
                        }
                    ]
                },
            ),
        },
    )
    df = linkedin.get_posts_with_statistics(ORG, "2026-10-01", "2026-10-01")
    assert len(sent) == 2  # 1 page + 1 stats batch
    assert df.loc[0, "like_count"] == 7
    assert pd.isna(df.loc[1, "like_count"])  # no stats -> null, not 0
    assert str(df["like_count"].dtype) == "Int64"
    assert "extracted_at" in df.columns


def test_posts_with_statistics_no_posts_skips_stats(linkedin, mocker):
    sent = route(mocker, linkedin, {"/posts": resp(200, {"elements": []})})
    assert linkedin.get_posts_with_statistics(ORG, "2026-10-01", "2026-10-02").empty
    assert len(sent) == 1


def test_posts_with_statistics_stats_failure(linkedin, mocker):
    route(
        mocker,
        linkedin,
        {
            "/posts": resp(200, {"elements": [post("urn:li:share:1", OCT1)]}),
            "ShareStatistics": resp(403),
        },
    )
    assert linkedin.get_posts_with_statistics(ORG, "2026-10-01", "2026-10-01") is None


# ------------------------------------------------------------------
# Daily stats
# ------------------------------------------------------------------


def daily_payloads(days=2, follower_days=None):
    follower_days = days if follower_days is None else follower_days
    tr = lambda i: {"start": OCT1 + i * DAY_MS, "end": OCT1 + (i + 1) * DAY_MS}
    share = {
        "elements": [
            {"timeRange": tr(i), "totalShareStatistics": stats(impressionCount=10 + i)}
            for i in range(days)
        ]
    }
    follower = {
        "elements": [
            {
                "timeRange": tr(i),
                "followerGains": {"organicFollowerGain": 2, "paidFollowerGain": 0},
            }
            for i in range(follower_days)
        ]
    }
    page = {
        "elements": [
            {
                "timeRange": tr(i),
                "totalPageStatistics": {
                    "views": {"allPageViews": {"pageViews": 5, "uniquePageViews": 4}},
                    "clicks": {"mobileCustomButtonClickCounts": [{"clicks": 0}]},
                },
            }
            for i in range(days)
        ]
    }
    return share, follower, page


def test_daily_stats_three_calls_one_row_per_day(linkedin, mocker):
    share, follower, page = daily_payloads(days=2, follower_days=1)
    sent = route(
        mocker,
        linkedin,
        {
            "ShareStatistics": resp(200, share),
            "FollowerStatistics": resp(200, follower),
            "organizationPageStatistics": resp(200, page),
        },
    )
    df = linkedin.get_daily_stats(ORG, "2026-10-01", "2026-10-02")
    assert len(sent) == 3
    assert df["date"].astype(str).tolist() == ["2026-10-01", "2026-10-02"]
    assert df["impression_count"].tolist() == [10, 11]
    assert df.loc[0, "page_all_page_views_page_views"] == 5
    assert df.loc[0, "page_all_page_views_unique_page_views"] == 4
    assert df.loc[0, "organic_follower_gain"] == 2
    assert pd.isna(df.loc[1, "organic_follower_gain"])
    assert str(df["organic_follower_gain"].dtype) == "Int64"


def test_daily_stats_time_range_end_is_inclusive(linkedin, mocker):
    share, follower, page = daily_payloads()
    sent = route(
        mocker,
        linkedin,
        {
            "ShareStatistics": resp(200, share),
            "FollowerStatistics": resp(200, follower),
            "organizationPageStatistics": resp(200, page),
        },
    )
    linkedin.get_daily_stats(ORG, "2026-10-01", "2026-10-02")
    assert f"start:{OCT1},end:{OCT1 + 2 * DAY_MS}" in sent[0]
    assert "timeGranularityType:DAY" in sent[0]


def test_daily_stats_bad_dates_make_no_calls(linkedin, mocker):
    sent = route(mocker, linkedin, {})
    assert linkedin.get_daily_stats(ORG, "2026/10/01", "2026-10-02") is None
    assert sent == []


def test_daily_stats_partial_failure_returns_none(linkedin, mocker):
    share, _, _ = daily_payloads()
    sent = route(
        mocker,
        linkedin,
        {"ShareStatistics": resp(200, share), "FollowerStatistics": resp(403)},
    )
    assert linkedin.get_daily_stats(ORG, "2026-10-01", "2026-10-02") is None
    assert len(sent) == 2  # stops before page stats


def test_daily_stats_empty(linkedin, mocker):
    route(
        mocker,
        linkedin,
        {
            "ShareStatistics": resp(200, {"elements": []}),
            "FollowerStatistics": resp(200, {"elements": []}),
            "organizationPageStatistics": resp(200, {"elements": []}),
        },
    )
    assert linkedin.get_daily_stats(ORG, "2026-10-01", "2026-10-02").empty


# ------------------------------------------------------------------
# Follower stats
# ------------------------------------------------------------------


def test_follower_stats_total_and_breakdowns(linkedin, mocker):
    fc = lambda n: {"organicFollowerCount": n, "paidFollowerCount": 0}
    breakdowns = {
        "elements": [
            {
                "organizationalEntity": f"urn:li:organization:{ORG}",
                "followerCountsByFunction": [
                    {"function": "urn:li:function:15", "followerCounts": fc(510)}
                ],
                "followerCountsByGeoCountry": [
                    {"geo": "urn:li:geo:104621616", "followerCounts": fc(1939)}
                ],
                "followerCountsByAssociationType": [
                    {"associationType": "EMPLOYEE", "followerCounts": fc(27)}
                ],
            }
        ]
    }
    sent = route(
        mocker,
        linkedin,
        {
            "FollowerStatistics": resp(200, breakdowns),
            "networkSizes": resp(200, {"firstDegreeSize": 3749}),
        },
    )
    df = linkedin.get_follower_stats(ORG)
    assert len(sent) == 2
    total = df[df["dimension"] == "total"].iloc[0]
    assert total["organic_follower_count"] == 3749
    rows = df.set_index("dimension")["dimension_value"].to_dict()
    assert rows["function"] == "urn:li:function:15"
    assert rows["geo_country"] == "urn:li:geo:104621616"
    assert rows["association_type"] == "EMPLOYEE"
    assert list(df.columns) == [
        "snapshot_date",
        "org_id",
        "dimension",
        "dimension_value",
        "organic_follower_count",
        "paid_follower_count",
        "extracted_at",
    ]


def test_follower_stats_breakdown_failure_skips_total_call(linkedin, mocker):
    sent = route(mocker, linkedin, {"FollowerStatistics": resp(403)})
    assert linkedin.get_follower_stats(ORG) is None
    assert len(sent) == 1


def test_follower_stats_total_failure(linkedin, mocker):
    route(
        mocker,
        linkedin,
        {
            "FollowerStatistics": resp(200, {"elements": []}),
            "networkSizes": resp(403),
        },
    )
    assert linkedin.get_follower_stats(ORG) is None


def test_follower_stats_no_breakdowns_keeps_total(linkedin, mocker):
    route(
        mocker,
        linkedin,
        {
            "FollowerStatistics": resp(200, {"elements": []}),
            "networkSizes": resp(200, {"firstDegreeSize": 10}),
        },
    )
    df = linkedin.get_follower_stats(ORG)
    assert df["dimension"].tolist() == ["total"]
