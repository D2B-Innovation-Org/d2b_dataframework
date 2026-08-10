import pandas as pd
import pytest

from d2b_data.shopify_api import ShopifyAPI
from d2b_data.utc_converter import UTCConverter


SHOP = "mitienda"
TOKEN = "shpat_fake"
ORDERS_URL = f"https://{SHOP}.myshopify.com/admin/api/2024-01/orders.json"


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_instance_builds_the_admin_url(shop):
    """The base URL targets the shop's Admin API at the default version."""
    assert shop.base_url == f"https://{SHOP}.myshopify.com/admin/api/2024-01"
    assert shop.api_version == "2024-01"


def test_custom_api_version_is_used():
    """A pinned API version shows up in the URL."""
    client = ShopifyAPI(SHOP, TOKEN, api_version="2025-07")
    assert client.base_url.endswith("/admin/api/2025-07")


def test_headers_carry_the_access_token(shop):
    """Shopify authenticates with the X-Shopify-Access-Token header."""
    assert shop.headers["X-Shopify-Access-Token"] == TOKEN
    assert shop.headers["Content-Type"] == "application/json"


def test_utc_converter_is_available(shop):
    """The client exposes the date helper it was built with."""
    assert isinstance(shop.utc_converter, UTCConverter)


def test_verbose_is_off_by_default(shop, capsys):
    """Nothing is printed unless verbose was requested."""
    shop.verbose("no debería verse")
    assert capsys.readouterr().out == ""


def test_verbose_prints_when_enabled(capsys):
    """verbose=True turns the helper into a print."""
    ShopifyAPI(SHOP, TOKEN, verbose=True).verbose("hola")
    assert "hola" in capsys.readouterr().out


# --------------------------------------------------------------------- #
# get_orders
# --------------------------------------------------------------------- #
def test_get_orders_returns_all_orders(shop, page, order, mocker):
    """A single page of results is returned as a list."""
    mocker.patch("requests.get", return_value=page([order(), order(id=2)]))
    result = shop.get_orders()
    assert len(result) == 2


def test_get_orders_sends_the_expected_params(shop, page, mocker):
    """Status, limit and sort order are sent on the first request."""
    get = mocker.patch("requests.get", return_value=page([]))
    shop.get_orders(date_start="2024-01-01", date_end="2024-01-31")

    assert get.call_args.args[0] == ORDERS_URL
    params = get.call_args.kwargs["params"]
    assert params == {
        "status": "any",
        "limit": 250,
        "order": "created_at asc",
        "created_at_min": "2024-01-01",
        "created_at_max": "2024-01-31",
    }


def test_get_orders_caps_the_limit_at_250(shop, page, mocker):
    """Shopify's hard limit is enforced client-side."""
    get = mocker.patch("requests.get", return_value=page([]))
    shop.get_orders(limit=1000)
    assert get.call_args.kwargs["params"]["limit"] == 250


def test_get_orders_omits_date_filters_when_not_given(shop, page, mocker):
    """Without dates no created_at filters are sent."""
    get = mocker.patch("requests.get", return_value=page([]))
    shop.get_orders()
    params = get.call_args.kwargs["params"]
    assert "created_at_min" not in params
    assert "created_at_max" not in params


def test_get_orders_follows_the_link_header(shop, page, order, mocker):
    """Pagination follows the 'next' link until it disappears."""
    next_url = ORDERS_URL + "?page_info=abc"
    get = mocker.patch("requests.get", side_effect=[
        page([order(id=1)], next_url=next_url),
        page([order(id=2)]),
    ])
    result = shop.get_orders()

    assert len(result) == 2
    assert get.call_count == 2
    assert get.call_args_list[1].args[0] == next_url


def test_get_orders_sends_params_only_on_the_first_page(shop, page, order, mocker):
    """Follow-up requests must not re-send params; the link carries them."""
    get = mocker.patch("requests.get", side_effect=[
        page([order()], next_url=ORDERS_URL + "?page_info=abc"),
        page([]),
    ])
    shop.get_orders()

    assert get.call_args_list[0].kwargs["params"] is not None
    assert get.call_args_list[1].kwargs["params"] is None


def test_get_orders_returns_none_on_http_error(shop, page, mocker, capsys):
    """An HTTP error is reported and surfaced as None."""
    mocker.patch("requests.get", return_value=page([], status_ok=False, text="unauthorized"))

    assert shop.get_orders() is None
    assert "Error HTTP" in capsys.readouterr().out


def test_get_orders_handles_a_response_without_orders(shop, page, mocker):
    """A payload with no 'orders' key yields an empty list."""
    response = page([])
    response.json.return_value = {}
    mocker.patch("requests.get", return_value=response)
    assert shop.get_orders() == []


# --------------------------------------------------------------------- #
# orders_to_dataframe
# --------------------------------------------------------------------- #
def test_orders_to_dataframe_maps_the_core_fields(shop, order):
    """Money fields are cast to float and identifiers preserved."""
    df = shop.orders_to_dataframe([order()])
    row = df.iloc[0]

    assert row["orders"] == "#1001"
    assert row["order_id"] == 123456789
    assert row["order_number"] == 1001
    assert row["gross_sales"] == 100000.0
    assert row["discounts"] == 5000.0
    assert row["taxes"] == 19000.0
    assert row["total_sales"] == 114000.0
    assert row["currency"] == "CLP"
    assert row["customer_email"] == "cliente@example.cl"


def test_orders_to_dataframe_empty_input():
    """No orders yields an empty DataFrame."""
    df = ShopifyAPI(SHOP, TOKEN).orders_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_pending_shipping_when_fulfillment_is_none(shop, order):
    """A null fulfillment_status maps to 'Pending Shipping'."""
    assert shop.orders_to_dataframe([order()]).iloc[0]["estado_envio"] == "Pending Shipping"


def test_fulfillment_status_is_capitalised(shop, order):
    """An existing status is title-cased for reporting."""
    df = shop.orders_to_dataframe([order(fulfillment_status="fulfilled")])
    assert df.iloc[0]["estado_envio"] == "Fulfilled"


@pytest.mark.parametrize("closed_at,expected", [
    (None, "Open"),
    ("2024-01-20T10:00:00-03:00", "Closed"),
])
def test_order_lifecycle_status(shop, order, closed_at, expected):
    """closed_at decides whether the order is Open or Closed."""
    df = shop.orders_to_dataframe([order(closed_at=closed_at)])
    assert df.iloc[0]["estado_ciclo"] == expected


def test_shipping_charges_are_read_from_the_money_set(shop, order):
    """Shipping comes from total_shipping_price_set.shop_money.amount."""
    df = shop.orders_to_dataframe([order(
        total_shipping_price_set={"shop_money": {"amount": "3990"}},
    )])
    assert df.iloc[0]["shipping_charges"] == 3990.0


def test_missing_money_sets_default_to_zero(shop, order):
    """Absent shipping, fees and duties do not break the mapping."""
    row = shop.orders_to_dataframe([order()]).iloc[0]
    assert row["shipping_charges"] == 0.0
    assert row["additional_fees"] == 0.0
    assert row["duties"] == 0.0


def test_additional_fees_and_duties_are_mapped(shop, order):
    """Fees and duties come from their own money sets."""
    df = shop.orders_to_dataframe([order(
        current_total_additional_fees_set={"shop_money": {"amount": "1000"}},
        current_total_duties_set={"shop_money": {"amount": "500"}},
    )])
    row = df.iloc[0]
    assert row["additional_fees"] == 1000.0
    assert row["duties"] == 500.0


def test_net_sales_adds_fees_duties_shipping_and_taxes(shop, order):
    """net_sales is the sum of the non-product charges."""
    df = shop.orders_to_dataframe([order(
        total_shipping_price_set={"shop_money": {"amount": "3000"}},
        current_total_additional_fees_set={"shop_money": {"amount": "1000"}},
        current_total_duties_set={"shop_money": {"amount": "500"}},
    )])
    assert df.iloc[0]["net_sales"] == 3000 + 1000 + 500 + 19000


def test_refund_line_items_are_netted_of_tax(shop, order):
    """A returned product counts as subtotal minus its tax."""
    df = shop.orders_to_dataframe([order(refunds=[{
        "refund_line_items": [{"subtotal": "10000", "total_tax": "1331"}],
    }])])
    assert df.iloc[0]["returns"] == pytest.approx(8669.0)


def test_shipping_refunds_are_kept_out_of_returns(shop, order):
    """A shipping_refund adjustment does not inflate 'returns'."""
    df = shop.orders_to_dataframe([order(refunds=[{
        "order_adjustments": [{"kind": "shipping_refund", "amount": "-3156.20"}],
    }])])
    assert df.iloc[0]["returns"] == 0.0


def test_other_adjustments_count_as_returns(shop, order):
    """A manual adjustment is absorbed into 'returns' as a positive amount."""
    df = shop.orders_to_dataframe([order(refunds=[{
        "order_adjustments": [{"kind": "refund_discrepancy", "amount": "-2000"}],
    }])])
    assert df.iloc[0]["returns"] == 2000.0


def test_custom_total_sales_subtracts_returns(shop, order):
    """custom_total_sales is total_price minus what came back."""
    df = shop.orders_to_dataframe([order(refunds=[{
        "refund_line_items": [{"subtotal": "10000", "total_tax": "0"}],
    }])])
    row = df.iloc[0]
    assert row["custom_total_sales"] == 114000.0 - 10000.0


def test_refunds_input_uses_updated_at_and_negates_the_total(shop, order):
    """In refund mode the row is dated by update and the total flips sign."""
    df = shop.orders_to_dataframe([order()], refunds_input=True)
    row = df.iloc[0]
    assert row["date"] == "2024-01-20T10:00:00-03:00"
    assert row["total_sales"] == -114000.0


def test_default_mode_uses_created_at(shop, order):
    """Sales rows are dated by creation."""
    assert shop.orders_to_dataframe([order()]).iloc[0]["date"] == "2024-01-15T10:00:00-03:00"


def test_cancel_reason_defaults_to_empty(shop, order):
    """A missing cancel_reason does not become NaN."""
    payload = order()
    del payload["cancel_reason"]
    assert shop.orders_to_dataframe([payload]).iloc[0]["cancel_reason"] == ""


# --------------------------------------------------------------------- #
# get_orders_as_df
# --------------------------------------------------------------------- #
def test_get_orders_as_df_returns_a_dataframe(shop, order, mocker):
    """The convenience wrapper maps the fetched orders."""
    mocker.patch.object(shop, "get_orders", return_value=[order()])
    df = shop.get_orders_as_df()
    assert len(df) == 1
    assert df.iloc[0]["orders"] == "#1001"


def test_get_orders_as_df_forwards_its_arguments(shop, order, mocker):
    """Dates, status and limit are passed straight through."""
    get_orders = mocker.patch.object(shop, "get_orders", return_value=[])
    shop.get_orders_as_df(date_start="2024-01-01", date_end="2024-01-31", status="open", limit=50)

    assert get_orders.call_args.kwargs == {
        "date_start": "2024-01-01",
        "date_end": "2024-01-31",
        "status": "open",
        "limit": 50,
    }


def test_get_orders_as_df_empty_on_failure(shop, mocker):
    """A failed fetch degrades to an empty DataFrame, never None."""
    mocker.patch.object(shop, "get_orders", return_value=None)
    df = shop.get_orders_as_df()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# --------------------------------------------------------------------- #
# get_refunds / get_partially_refundeds
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("method,financial_status", [
    ("get_refunds", "refunded"),
    ("get_partially_refundeds", "partially_refunded"),
])
def test_refund_methods_filter_by_financial_status(shop, page, mocker, method, financial_status):
    """Each method targets its own financial_status and sorts by update."""
    get = mocker.patch("requests.get", return_value=page([]))
    getattr(shop, method)(date_start="2024-01-01", date_end="2024-01-31")

    params = get.call_args.kwargs["params"]
    assert params["financial_status"] == financial_status
    assert params["status"] == "any"
    assert params["order"] == "updated_at asc"
    assert params["updated_at_min"] == "2024-01-01"
    assert params["updated_at_max"] == "2024-01-31"


@pytest.mark.parametrize("method", ["get_refunds", "get_partially_refundeds"])
def test_refund_methods_keep_only_orders_with_refunds(shop, page, order, mocker, method):
    """Orders without a refunds block are filtered out."""
    mocker.patch("requests.get", return_value=page([
        order(id=1, refunds=[{"id": 99}]),
        order(id=2),
        order(id=3, refunds=[]),
    ]))
    result = getattr(shop, method)()
    assert [o["id"] for o in result] == [1]


@pytest.mark.parametrize("method", ["get_refunds", "get_partially_refundeds"])
def test_refund_methods_cap_the_limit(shop, page, mocker, method):
    """The 250 cap applies here too."""
    get = mocker.patch("requests.get", return_value=page([]))
    getattr(shop, method)(limit=9999)
    assert get.call_args.kwargs["params"]["limit"] == 250


@pytest.mark.parametrize("method", ["get_refunds", "get_partially_refundeds"])
def test_refund_methods_omit_absent_date_filters(shop, page, mocker, method):
    """No dates means no updated_at filters."""
    get = mocker.patch("requests.get", return_value=page([]))
    getattr(shop, method)()
    params = get.call_args.kwargs["params"]
    assert "updated_at_min" not in params
    assert "updated_at_max" not in params


@pytest.mark.parametrize("method", ["get_refunds", "get_partially_refundeds"])
def test_refund_methods_paginate(shop, page, order, mocker, method):
    """Pagination works the same as in get_orders."""
    next_url = ORDERS_URL + "?page_info=xyz"
    get = mocker.patch("requests.get", side_effect=[
        page([order(id=1, refunds=[{"id": 1}])], next_url=next_url),
        page([order(id=2, refunds=[{"id": 2}])]),
    ])
    result = getattr(shop, method)()

    assert [o["id"] for o in result] == [1, 2]
    assert get.call_args_list[1].kwargs["params"] is None


@pytest.mark.parametrize("method", ["get_refunds", "get_partially_refundeds"])
def test_refund_methods_return_none_on_http_error(shop, page, mocker, capsys, method):
    """HTTP failures surface as None."""
    mocker.patch("requests.get", return_value=page([], status_ok=False, text="boom"))
    assert getattr(shop, method)() is None
    assert "Error HTTP" in capsys.readouterr().out
