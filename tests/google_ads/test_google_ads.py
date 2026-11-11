"""Tests for GoogleAds. No network: the API client is always a fake."""

import importlib
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from d2b_data.google_ads import (
    GoogleAds,
    GoogleAdsQuotaError,
    GoogleAdsRequestError,
    normalize_customer_id,
)

from .conftest import (
    FakeClient,
    FakeCustomerService,
    ads_exception,
    make_batch,
    make_row,
    requires_google_ads,
    rpc_error,
)

FIELDS = ["segments.date", "campaign.id", "metrics.cost_micros"]


# ---------------------------------------------------------------------- #
# Constructor
# ---------------------------------------------------------------------- #
def test_constructor_uses_explicit_arguments(credentials):
    ads = GoogleAds(**credentials, login_customer_id="098-765-4321")
    assert ads.developer_token == "dev-token-secret"
    assert ads.client_id == "client-id-secret"
    assert ads.client_secret == "client-secret-secret"
    assert ads.refresh_token == "refresh-token-secret"
    assert ads.login_customer_id == "0987654321"
    assert ads.api_version is None


def test_constructor_does_not_build_the_client(credentials, mocker):
    build = mocker.patch.object(GoogleAds, "_build_client")
    ads = GoogleAds(**credentials)
    build.assert_not_called()
    assert ads._client is None


def test_constructor_reads_token_path(credentials, token_file):
    path = token_file({**credentials, "login_customer_id": 1234567890})
    ads = GoogleAds(token_path=path)
    assert ads.refresh_token == "refresh-token-secret"
    assert ads.login_customer_id == "1234567890"


def test_explicit_arguments_take_precedence_over_token_path(credentials, token_file):
    path = token_file(credentials)
    ads = GoogleAds(refresh_token="explicit-token", token_path=path)
    assert ads.refresh_token == "explicit-token"
    assert ads.developer_token == "dev-token-secret"


def test_missing_credentials_are_named_without_values(credentials):
    with pytest.raises(ValueError) as info:
        GoogleAds(developer_token="dev-token-secret", client_id="client-id-secret")
    message = str(info.value)
    assert "client_secret" in message
    assert "refresh_token" in message
    assert "developer_token" not in message
    assert "secret" not in message.replace("client_secret", "")


def test_missing_token_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GoogleAds(token_path=str(tmp_path / "missing.json"))


def test_token_file_must_be_a_json_object(token_file):
    with pytest.raises(ValueError, match="JSON object"):
        GoogleAds(token_path=token_file(["not", "an", "object"]))


def test_invalid_login_customer_id_raises(credentials):
    with pytest.raises(ValueError, match="customer_id"):
        GoogleAds(**credentials, login_customer_id="12345")


def test_init_log_does_not_leak_secrets(credentials, mocker):
    logger = mocker.MagicMock()
    GoogleAds(**credentials, logger=logger)
    logged = " ".join(str(call) for call in logger.mock_calls)
    for value in credentials.values():
        assert value not in logged


def test_default_logger_has_alerts_disabled(ads):
    assert ads.logger.alerts_enabled is False


# ---------------------------------------------------------------------- #
# Customer ids
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        ("123-456-7890", "1234567890"),
        ("1234567890", "1234567890"),
        (" 1234567890 ", "1234567890"),
        (1234567890, "1234567890"),
        (123456789, "0123456789"),
    ],
)
def test_normalize_customer_id(value, expected):
    assert normalize_customer_id(value) == expected


@pytest.mark.parametrize(
    "value", ["12-3456-7890", "123456789", "12345678901", "abc", "", 0, -1, True, 1.5]
)
def test_normalize_customer_id_rejects_invalid(value):
    with pytest.raises(ValueError):
        normalize_customer_id(value)


# ---------------------------------------------------------------------- #
# build_query
# ---------------------------------------------------------------------- #
def test_build_query_basic():
    query = GoogleAds.build_query("campaign", FIELDS, "2026-11-01", "2026-11-10")
    assert query == (
        "SELECT segments.date, campaign.id, metrics.cost_micros FROM campaign "
        "WHERE segments.date BETWEEN '2026-11-01' AND '2026-11-10'"
    )


def test_build_query_with_where():
    query = GoogleAds.build_query(
        "ad_group_ad",
        ["campaign.id"],
        "2026-11-01",
        "2026-11-01",
        where=["campaign.status != 'REMOVED'", "metrics.impressions > 0"],
    )
    assert query.endswith(
        "WHERE segments.date BETWEEN '2026-11-01' AND '2026-11-01' "
        "AND campaign.status != 'REMOVED' AND metrics.impressions > 0"
    )


def test_build_query_rejects_empty_fields():
    with pytest.raises(ValueError, match="at least one field"):
        GoogleAds.build_query("campaign", [], "2026-11-01", "2026-11-10")


@pytest.mark.parametrize(
    "resource, fields",
    [
        ("campaign", ["campaign.id", "metrics.cost micros"]),
        ("campaign", ["Campaign.Id"]),
        ("campaign", ["campaign.id,campaign.name"]),
        ("campaign", ["campaign..id"]),
        ("campaign; DROP", ["campaign.id"]),
        ("", ["campaign.id"]),
    ],
)
def test_build_query_rejects_invalid_identifiers(resource, fields):
    with pytest.raises(ValueError, match="Invalid GAQL identifier"):
        GoogleAds.build_query(resource, fields, "2026-11-01", "2026-11-10")


@pytest.mark.parametrize("bad", ["2026/11/01", "2026-13-01", "2026-02-30", "20261101"])
def test_build_query_rejects_malformed_dates(bad):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        GoogleAds.build_query("campaign", FIELDS, bad, "2026-11-10")


def test_build_query_rejects_start_after_end():
    with pytest.raises(ValueError, match="greater than end_date"):
        GoogleAds.build_query("campaign", FIELDS, "2026-11-10", "2026-11-01")


def test_build_query_rejects_semicolon_in_where():
    with pytest.raises(ValueError, match="';'"):
        GoogleAds.build_query(
            "campaign", FIELDS, "2026-11-01", "2026-11-10", where=["a = 1; b"]
        )


# ---------------------------------------------------------------------- #
# Optional dependency
# ---------------------------------------------------------------------- #
def test_module_imports_without_google_ads(monkeypatch):
    for name in list(sys.modules):
        if name == "google.ads" or name.startswith("google.ads."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "google.ads", None)
    monkeypatch.delitem(sys.modules, "d2b_data.google_ads", raising=False)

    module = importlib.import_module("d2b_data.google_ads")

    assert hasattr(module, "GoogleAds")
    # The import has to leave google.ads untouched for real.
    assert sys.modules["google.ads"] is None


def test_missing_library_raises_clear_import_error(ads, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.ads.googleads.client", None)
    with pytest.raises(ImportError, match=r"pip install .*google_ads"):
        ads.search("1234567890", "SELECT campaign.id FROM campaign")


@requires_google_ads
def test_build_client_passes_credentials_and_version(credentials, mocker):
    from google.ads.googleads.client import GoogleAdsClient

    load = mocker.patch.object(GoogleAdsClient, "load_from_dict")
    ads = GoogleAds(**credentials, login_customer_id="0987654321", api_version="v25")

    assert ads._get_client() is load.return_value
    assert ads._get_client() is load.return_value  # cached
    load.assert_called_once()
    config = load.call_args.args[0]
    assert config == {
        **credentials,
        "use_proto_plus": True,
        "login_customer_id": "0987654321",
    }
    assert load.call_args.kwargs == {"version": "v25"}


@requires_google_ads
def test_build_client_config_passes_library_validation(credentials, mocker):
    # The library refreshes the OAuth token while building the client, which
    # would hit the network; only that step is replaced.
    get_credentials = mocker.patch(
        "google.ads.googleads.oauth2.get_installed_app_credentials"
    )
    ads = GoogleAds(**credentials)

    client = ads._get_client()

    get_credentials.assert_called_once()
    assert get_credentials.call_args.args[:3] == (
        "client-id-secret",
        "client-secret-secret",
        "refresh-token-secret",
    )
    assert client.developer_token == "dev-token-secret"
    assert client.login_customer_id is None
    assert client.use_proto_plus is True
    assert client.version is None


# ---------------------------------------------------------------------- #
# search and get_report_df
# ---------------------------------------------------------------------- #
@requires_google_ads
def test_search_reads_several_batches_in_field_mask_order(ads, with_stream):
    paths = ["segments.date", "campaign.id", "metrics.cost_micros"]
    service = with_stream(
        [
            make_batch(
                paths,
                [
                    make_row(
                        segments__date="2026-11-01",
                        campaign__id=1,
                        metrics__cost_micros=5,
                    )
                ],
            ),
            make_batch(
                paths,
                [
                    make_row(
                        segments__date="2026-11-02",
                        campaign__id=2,
                        metrics__cost_micros=7,
                    )
                ],
            ),
        ]
    )

    rows = ads.search("123-456-7890", "SELECT ...")

    assert service.calls == [{"customer_id": "1234567890", "query": "SELECT ..."}]
    assert rows == [
        {"segments.date": "2026-11-01", "campaign.id": 1, "metrics.cost_micros": 5},
        {"segments.date": "2026-11-02", "campaign.id": 2, "metrics.cost_micros": 7},
    ]
    assert list(rows[0]) == paths


@requires_google_ads
def test_search_keeps_zero_and_empty_values(ads, with_stream):
    paths = ["campaign.id", "campaign.name", "metrics.cost_micros", "metrics.ctr"]
    with_stream([make_batch(paths, [make_row(campaign__id=0, campaign__name="")])])

    row = ads.search("1234567890", "q")[0]

    assert row == {
        "campaign.id": 0,
        "campaign.name": "",
        "metrics.cost_micros": 0,
        "metrics.ctr": 0.0,
    }


@requires_google_ads
def test_search_returns_enums_by_name(ads, with_stream):
    from google.ads.googleads.v26.enums.types.campaign_status import (
        CampaignStatusEnum,
    )

    with_stream(
        [
            make_batch(
                ["campaign.status", "ad_group_criterion.type"],
                [
                    make_row(
                        campaign__status=CampaignStatusEnum.CampaignStatus.ENABLED,
                        ad_group_criterion__type_=2,
                    )
                ],
            )
        ]
    )

    row = ads.search("1234567890", "q")[0]

    assert row == {"campaign.status": "ENABLED", "ad_group_criterion.type": "KEYWORD"}


@requires_google_ads
def test_search_returns_repeated_fields_as_lists(ads, with_stream):
    with_stream(
        [
            make_batch(
                [
                    "ad_group_ad.ad.final_urls",
                    "ad_group_ad.ad.responsive_search_ad.headlines",
                ],
                [
                    make_row(
                        ad_group_ad__ad__final_urls=["https://a.cl", "https://b.cl"],
                    )
                ],
            )
        ]
    )
    row = ads.search("1234567890", "q")[0]
    assert row["ad_group_ad.ad.final_urls"] == ["https://a.cl", "https://b.cl"]
    assert row["ad_group_ad.ad.responsive_search_ad.headlines"] == []


@requires_google_ads
def test_search_returns_nested_messages_as_dicts(ads, with_stream):
    from google.ads.googleads.v26.common.types.ad_asset import AdTextAsset

    row = make_row()
    row.ad_group_ad.ad.responsive_search_ad.headlines.append(AdTextAsset(text="Hola"))
    row.campaign.network_settings.target_search_network = False
    with_stream(
        [
            make_batch(
                [
                    "ad_group_ad.ad.responsive_search_ad.headlines",
                    "campaign.network_settings",
                ],
                [row],
            )
        ]
    )

    result = ads.search("1234567890", "q")[0]

    headline = result["ad_group_ad.ad.responsive_search_ad.headlines"][0]
    assert headline["text"] == "Hola"
    assert result["campaign.network_settings"]["target_search_network"] is False


@requires_google_ads
def test_search_without_rows_returns_empty_list(ads, with_stream):
    with_stream([make_batch(FIELDS, [])])
    assert ads.search("1234567890", "q") == []


@requires_google_ads
def test_search_with_empty_stream_returns_empty_list(ads, with_stream):
    # The stream may send no batch at all when nothing matches.
    with_stream([])
    assert ads.search("1234567890", "q") == []


def test_search_rejects_invalid_customer_id(ads, with_stream):
    service = with_stream()
    with pytest.raises(ValueError):
        ads.search("12345", "q")
    assert service.calls == []


@requires_google_ads
def test_get_report_df_keeps_field_mask_order_and_values(ads, with_stream):
    # field_mask order wins over the order of the requested fields.
    mask = ["metrics.cost_micros", "segments.date", "campaign.id"]
    service = with_stream(
        [
            make_batch(
                mask,
                [
                    make_row(
                        segments__date="2026-11-01",
                        campaign__id=1,
                        metrics__cost_micros=0,
                    ),
                    make_row(
                        segments__date="2026-11-01",
                        campaign__id=2,
                        metrics__cost_micros=1500000,
                    ),
                ],
            )
        ]
    )

    df = ads.get_report_df(
        "1234567890", "campaign", FIELDS, "2026-11-01", "2026-11-01", where=["x = 1"]
    )

    assert list(df.columns) == mask
    assert df["metrics.cost_micros"].tolist() == [0, 1500000]
    assert df["campaign.id"].tolist() == [1, 2]
    assert service.calls[0]["query"] == GoogleAds.build_query(
        "campaign", FIELDS, "2026-11-01", "2026-11-01", ["x = 1"]
    )


@requires_google_ads
def test_get_report_df_empty(ads, with_stream):
    with_stream([make_batch(FIELDS, [])])
    df = ads.get_report_df("1234567890", "campaign", FIELDS, "2026-11-01", "2026-11-01")
    assert df.empty
    assert list(df.columns) == FIELDS


def test_get_report_df_validates_before_calling_api(ads, with_stream):
    service = with_stream()
    with pytest.raises(ValueError):
        ads.get_report_df("1234567890", "campaign", [], "2026-11-01", "2026-11-01")
    assert service.calls == []


# ---------------------------------------------------------------------- #
# Retries and errors
# ---------------------------------------------------------------------- #
@requires_google_ads
@pytest.mark.parametrize(
    "transient",
    [
        lambda: rpc_error("UNAVAILABLE", "connection reset"),
        lambda: rpc_error("DEADLINE_EXCEEDED"),
        lambda: rpc_error("INTERNAL"),
        lambda: ads_exception([("internal_error", "TRANSIENT_ERROR", "try again")]),
        lambda: ads_exception([("internal_error", "INTERNAL_ERROR", "oops")]),
    ],
)
def test_transient_error_then_success(ads, with_stream, no_sleep, mocker, transient):
    ads.logger = mocker.MagicMock()
    batch = make_batch(["campaign.id"], [make_row(campaign__id=9)])
    service = with_stream(transient(), [batch])

    rows = ads.search("1234567890", "q")

    assert rows == [{"campaign.id": 9}]
    assert len(service.calls) == 2
    assert no_sleep.call_count == 1
    assert ads.logger.warning.call_count == 1
    assert "Retry 1/3" in ads.logger.warning.call_args.args[0]


@requires_google_ads
def test_retry_restarts_stream_without_duplicating_rows(ads, with_stream, no_sleep):
    first = make_batch(["campaign.id"], [make_row(campaign__id=1)])

    def broken_stream():
        yield first
        raise rpc_error("UNAVAILABLE")

    service = with_stream(None, [first])
    service.outcomes[0] = broken_stream()

    assert ads.search("1234567890", "q") == [{"campaign.id": 1}]


@requires_google_ads
def test_backoff_is_exponential(ads, with_stream, no_sleep, mocker):
    mocker.patch("d2b_data.google_ads.random.uniform", return_value=0.5)
    with_stream(*[rpc_error("UNAVAILABLE")] * 3, [])
    ads.search("1234567890", "q")
    assert [c.args[0] for c in no_sleep.call_args_list] == [5.5, 10.5, 20.5]


@requires_google_ads
def test_transient_error_gives_up_after_three_retries(ads, with_stream, no_sleep):
    service = with_stream(*[rpc_error("UNAVAILABLE", "down", "req-9")] * 4)

    with pytest.raises(GoogleAdsRequestError) as info:
        ads.search("1234567890", "q")

    assert len(service.calls) == 4
    assert no_sleep.call_count == 3
    assert "gave up after 3 retries" in str(info.value)
    assert "req-9" in str(info.value)
    assert info.value.status == "UNAVAILABLE"


@requires_google_ads
@pytest.mark.parametrize(
    "quota",
    [
        lambda: rpc_error("RESOURCE_EXHAUSTED", "daily limit", "req-q"),
        lambda: ads_exception(
            [("quota_error", "RESOURCE_TEMPORARILY_EXHAUSTED", "slow down")],
            request_id="req-q",
        ),
    ],
)
def test_quota_error_is_not_retried(ads, with_stream, no_sleep, quota):
    service = with_stream(quota(), [])

    with pytest.raises(GoogleAdsQuotaError, match="req-q"):
        ads.search("1234567890", "q")

    assert len(service.calls) == 1
    no_sleep.assert_not_called()


@requires_google_ads
def test_query_error_propagates_with_request_id_and_messages(
    ads, with_stream, no_sleep
):
    error = ads_exception(
        [
            ("query_error", "UNRECOGNIZED_FIELD", "Unrecognized field 'campaign.foo'."),
            ("query_error", "PROHIBITED_FIELD_IN_SELECT_CLAUSE", "Second problem."),
        ],
        request_id="req-abc",
    )
    service = with_stream(error)

    with pytest.raises(GoogleAdsRequestError) as info:
        ads.search("1234567890", "q")

    message = str(info.value)
    assert "request_id=req-abc" in message
    assert "status=INVALID_ARGUMENT" in message
    assert "query_error=UNRECOGNIZED_FIELD" in message
    assert "Unrecognized field 'campaign.foo'." in message
    assert "Second problem." in message
    assert info.value.request_id == "req-abc"
    assert info.value.__cause__ is error
    assert len(service.calls) == 1
    no_sleep.assert_not_called()


@requires_google_ads
def test_mixed_transient_and_query_errors_are_not_retried(ads, with_stream, no_sleep):
    error = ads_exception(
        [
            ("internal_error", "TRANSIENT_ERROR", "hiccup"),
            ("query_error", "UNRECOGNIZED_FIELD", "bad"),
        ]
    )
    with_stream(error)
    with pytest.raises(GoogleAdsRequestError):
        ads.search("1234567890", "q")
    no_sleep.assert_not_called()


@requires_google_ads
@pytest.mark.parametrize(
    "error",
    [
        lambda: ads_exception(
            [("authentication_error", "OAUTH_TOKEN_REVOKED", "revoked")],
            code_name="UNAUTHENTICATED",
        ),
        lambda: ads_exception(
            [("authorization_error", "USER_PERMISSION_DENIED", "no access")],
            code_name="PERMISSION_DENIED",
        ),
        lambda: rpc_error("UNAUTHENTICATED", "bad token"),
    ],
)
def test_auth_errors_propagate_immediately(ads, with_stream, no_sleep, error):
    service = with_stream(error())
    with pytest.raises(GoogleAdsRequestError):
        ads.search("1234567890", "q")
    assert len(service.calls) == 1
    no_sleep.assert_not_called()


@requires_google_ads
def test_non_api_errors_are_left_untouched(ads, with_stream, no_sleep):
    from google.auth.exceptions import RefreshError

    with_stream(RefreshError("invalid_grant"))
    with pytest.raises(RefreshError):
        ads.search("1234567890", "q")
    no_sleep.assert_not_called()


# ---------------------------------------------------------------------- #
# list_accessible_customers
# ---------------------------------------------------------------------- #
def test_list_accessible_customers(ads):
    ads._client = FakeClient(
        {
            "CustomerService": FakeCustomerService(
                ["customers/3618724513", "customers/5308411360"]
            )
        }
    )
    assert ads.list_accessible_customers() == ["3618724513", "5308411360"]


@requires_google_ads
def test_list_accessible_customers_errors(ads):
    ads._client = FakeClient(
        {"CustomerService": FakeCustomerService(rpc_error("RESOURCE_EXHAUSTED"))}
    )
    with pytest.raises(GoogleAdsQuotaError):
        ads.list_accessible_customers()

    ads._client = FakeClient(
        {
            "CustomerService": FakeCustomerService(
                ads_exception([("authentication_error", "NOT_ADS_USER", "x")])
            )
        }
    )
    with pytest.raises(GoogleAdsRequestError, match="NOT_ADS_USER"):
        ads.list_accessible_customers()

    ads._client = FakeClient({"CustomerService": FakeCustomerService(KeyError("x"))})
    with pytest.raises(KeyError):
        ads.list_accessible_customers()


# ---------------------------------------------------------------------- #
# generate_refresh_token
# ---------------------------------------------------------------------- #
def _patch_flow(mocker, refresh_token="new-refresh-secret"):
    credentials = SimpleNamespace(
        client_id="cid", client_secret="csecret", refresh_token=refresh_token
    )
    flow = mocker.MagicMock()
    flow.run_local_server.return_value = credentials
    factory = mocker.patch(
        "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
        return_value=flow,
    )
    return factory, flow


def test_generate_refresh_token_uses_offline_consent(mocker, tmp_path):
    factory, flow = _patch_flow(mocker)
    token_path = tmp_path / "token.json"

    token = GoogleAds.generate_refresh_token("client.json", str(token_path))

    assert token == "new-refresh-secret"
    factory.assert_called_once_with(
        "client.json", scopes=["https://www.googleapis.com/auth/adwords"]
    )
    flow.run_local_server.assert_called_once_with(
        port=0, access_type="offline", prompt="consent"
    )
    assert json.loads(token_path.read_text()) == {
        "client_id": "cid",
        "client_secret": "csecret",
        "refresh_token": "new-refresh-secret",
    }
    if os.name == "posix":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_generate_refresh_token_without_token_path(mocker, tmp_path):
    _patch_flow(mocker)
    assert GoogleAds.generate_refresh_token("client.json", port=8080) == (
        "new-refresh-secret"
    )


def test_generate_refresh_token_without_refresh_token(mocker):
    _patch_flow(mocker, refresh_token=None)
    with pytest.raises(RuntimeError, match="refresh token"):
        GoogleAds.generate_refresh_token("client.json")


def test_saved_token_file_can_be_used_by_constructor(mocker, tmp_path):
    _patch_flow(mocker)
    token_path = tmp_path / "token.json"
    GoogleAds.generate_refresh_token("client.json", str(token_path))

    ads = GoogleAds(developer_token="dev", token_path=str(token_path))

    assert ads.refresh_token == "new-refresh-secret"
    assert ads.login_customer_id is None
