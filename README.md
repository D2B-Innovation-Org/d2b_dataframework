# D2B Data Framework 🚀

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/D2b-Innovation/d2b_dataframework/actions/workflows/tests.yml/badge.svg)](https://github.com/D2b-Innovation/d2b_dataframework/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Organization](https://img.shields.io/badge/Org-D2b--Innovation-black.svg)](https://github.com/D2b-Innovation)

A robust Python framework designed for managing and integrating data operations across multiple marketing and analytics platforms. D2B Data Framework provides modular, reusable classes to streamline ETL processes and data handling.

---

## 🛠 Installation

Requires **Python 3.11+**.

### 1. Install via `pip` (Directly from GitHub)

```bash
pip install git+https://github.com/D2b-Innovation/d2b_dataframework.git
```

### 2. Optional extras

Forecasting is optional because Prophet compiles Stan and takes a while to install:

```bash
pip install "d2b_data[forecasting] @ git+https://github.com/D2b-Innovation/d2b_dataframework.git"
```

Google Ads is optional too, because the `google-ads` wheel is large:

```bash
pip install "d2b_data[google_ads] @ git+https://github.com/D2b-Innovation/d2b_dataframework.git"
```

### 3. Local development

```bash
git clone https://github.com/D2b-Innovation/d2b_dataframework.git
cd d2b_dataframework
pip install -e ".[dev]"
pytest
```

---

## 📦 Modules and Capabilities

| Module | Main Class | Description |
| --- | --- | --- |
| 🔵 **Facebook Ads** | `Facebook_Marketing` | Manage Ads Insights and campaign data. |
| 🌿 **Facebook Organic** | `FacebookOrganic` | Page Insights via Graph API — posts, reactions, impressions, reach. |
| 📊 **GA4** | `Google_GA4` | Fetch reports and real-time analytics data. |
| ☁️ **BigQuery** | `Google_Bigquery` | Streamline SQL queries and data uploads. |
| 📝 **Sheets** | `Google_Spreadsheet` | Read/Write operations on Google Sheets. |
| 🔑 **Auth** | `Google_Token_MNG` | Manage OAuth2 tokens and credentials. |
| **Google Ads** | `GoogleAds` | Raw GAQL reports via `search_stream` (optional: `pip install "d2b_data[google_ads]"`). |
| 💼 **LinkedIn Ads** | `LinkedinMarketing` | Extract B2B marketing performance metrics from the Ads Reporting API. |
| 💼 **LinkedIn Organic** | `LinkedinOrganic` | Page posts, follower stats and engagement via Community Management API. |
| 🎵 **TikTok** | `TikTokMarketing` | Integration with TikTok Ads API. |
| 🐦 **X (Twitter)** | `X_ads` | Handle X Ads data reporting. |
| 🔮 **Forecasting** | `ProphetForecaster` | Time-series forecasting with Meta Prophet (optional: `pip install prophet`). |

---

## 🚀 Quick Start (Usage)

```python
from d2b_data.Google_GA4 import Google_GA4

ga4_client = Google_GA4('client_secret.json', 'token.json')

property_id = 'properties/YOUR_PROPERTY_ID'
query = { "dimensions": [{"name": "city"}], "metrics": [{"name": "activeUsers"}] }

df = ga4_client.get_report_df(property_id, query, realtime=True)
print(df.head())
```

### Facebook Organic (Page Insights)

```python
from d2b_data.facebook_organic import FacebookOrganic

client = FacebookOrganic(
    page_id="YOUR_PAGE_ID",
    access_token="YOUR_PAGE_ACCESS_TOKEN",
)

# Pull a consolidated DataFrame with post metrics + insights
df = client.get_report_dataframe(since="2024-01-01", until="2024-01-31")
print(df.head())
```

Requires a Page Access Token with `pages_read_engagement` and `read_insights` permissions.

### Google Ads

```python
from d2b_data.google_ads import GoogleAds, GoogleAdsQuotaError

ads = GoogleAds(
    developer_token="...",
    client_id="...",
    client_secret="...",
    refresh_token="...",             # scope https://www.googleapis.com/auth/adwords
    login_customer_id="123-456-7890",  # manager (MCC) account; optional
)
# Or read what is missing from a JSON with the same keys:
# GoogleAds(developer_token="...", token_path="google_ads.json")

print(ads.list_accessible_customers())  # connection test

df = ads.get_report_df(
    customer_id="1234567890",
    resource="campaign",
    fields=["segments.date", "campaign.id", "campaign.name", "metrics.cost_micros"],
    start_date="2026-11-01",
    end_date="2026-11-10",
    where=["campaign.status != 'REMOVED'"],
)
```

Data is raw: columns are the GAQL field paths (`metrics.cost_micros`), costs stay
in micros, and enums come back by name (`"ENABLED"`). Transient errors are retried
up to 3 times; quota errors raise `GoogleAdsQuotaError` without retrying; any other
API error raises `GoogleAdsRequestError` with the request id.

To get a refresh token (local and interactive only, it opens a browser):

```python
GoogleAds.generate_refresh_token("client_secret.json", token_path="google_ads.json")
```

### Forecasting (Prophet)

Basic usage — one line, no configuration. Every numeric column is forecasted:

```python
from d2b_data.prophet_forecaster import ProphetForecaster

# df: a 'date' (or 'fecha') column + numeric metric columns
fc = ProphetForecaster(df)
predictions = fc.get_forecast(30)   # 30 days ahead
```

Advanced usage — `forecast()` exposes the full Prophet configuration:

```python
fc = ProphetForecaster(df, verbose=False)

predictions = fc.forecast(
    days=12,
    metrics=['sessions'],            # forecast only these columns
    freq='W',                        # weekly periods instead of daily
    include_history=False,           # only future dates
    include_intervals=True,          # adds sessions_lower / sessions_upper
    interval_width=0.95,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.5,     # more reactive trend
    country_holidays='CL',
    custom_seasonalities=[{'name': 'monthly', 'period': 30.5, 'fourier_order': 5}],
    regressors=['spend'],            # external driver, not forecasted
    future_regressors=planned_spend, # its future values ('date' + 'spend')
    non_negative=True,               # clip negative predictions to 0
)
```

Other options: `growth='logistic'` with `cap`/`floor`, a custom `holidays` DataFrame,
`round_decimals`, and `prophet_kwargs` for anything else in the Prophet constructor.

Models can be reused: `save_models()`, `load_models()` and `predict_from_loaded_models()`.

---

## 📄 License

This project is licensed under the **MIT License**.

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---