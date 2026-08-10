import pytest
import requests

from d2b_data.shopify_api import ShopifyAPI


SHOP = "mitienda"
TOKEN = "shpat_fake"


@pytest.fixture
def shop():
    return ShopifyAPI(shop_name=SHOP, access_token=TOKEN)


@pytest.fixture
def page(mocker):
    """Builds a fake Shopify page response, optionally linking to a next one."""

    def _build(orders, next_url=None, status_ok=True, text=""):
        response = mocker.MagicMock()
        response.json.return_value = {"orders": orders}
        response.links = {"next": {"url": next_url}} if next_url else {}
        response.text = text
        if not status_ok:
            response.raise_for_status.side_effect = requests.exceptions.HTTPError("401")
        return response

    return _build


@pytest.fixture
def order():
    """A realistic order payload with the fields the mapper reads."""

    def _build(**overrides):
        base = {
            "name": "#1001",
            "id": 123456789,
            "order_number": 1001,
            "created_at": "2024-01-15T10:00:00-03:00",
            "updated_at": "2024-01-20T10:00:00-03:00",
            "closed_at": None,
            "fulfillment_status": None,
            "financial_status": "paid",
            "total_line_items_price": "100000",
            "total_discounts": "5000",
            "total_tax": "19000",
            "total_price": "114000",
            "currency": "CLP",
            "email": "cliente@example.cl",
            "cancel_reason": None,
        }
        base.update(overrides)
        return base

    return _build
