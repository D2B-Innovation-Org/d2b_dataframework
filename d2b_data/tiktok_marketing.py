import copy
import json
import random
import time
import urllib.parse
import webbrowser

import pandas as pd
import requests

from d2b_data.verbose_logger import Verbose

# Campos por defecto de get_ads. tiktok_item_id e identity_id identifican el post
# de TikTok en los Spark Ads; el reporte de métricas no los entrega.
AD_FIELDS = [
    "advertiser_id",
    "campaign_id",
    "adgroup_id",
    "ad_id",
    "ad_name",
    "operation_status",
    "secondary_status",
    "identity_type",
    "identity_id",
    "tiktok_item_id",
    "video_id",
    "create_time",
    "modify_time",
]


class TikTokMarketing:
    """
    TikTok Marketing API Wrapper
    ============================

    A robust Python client for interacting with the TikTok Business API (v1.3),
    specifically designed for automated advertising performance reporting.

    Key Technical Features:
    ----------------------
    * **OAuth2 Lifecycle Management:** Handles the full authentication flow,
        storing and loading credentials from local JSON files.
    * **Rate Limit Resiliency:** Implements an exponential backoff strategy
        to handle HTTP 429 (Too Many Requests) errors gracefully.
    * **Smart Date Chunking:** Automatically bypasses TikTok's 30-day limit
        on time-series queries by segmenting requests into valid date ranges.
    * **Pagination Handling:** Transparently manages multi-page API responses
        to ensure complete data retrieval.
    * **Pandas Integration:** Returns clean, analysis-ready DataFrames with
        automated column normalization and sorting.

    Basic Implementation:
    --------------------
        >>> tt = TikTokMarketing(token_path="tiktok_credentials.json")
        >>> df = tt.get_report_dataframe(
        ...     advertiser_id="7289151914388865025",
        ...     start_date="20260101",
        ...     end_date="20260423",
        ...     dimensions=["ad_id", "stat_time_day"],
        ...     metrics=["spend", "impressions", "clicks"]
        ... )

    Author: D2B Data Engineering Team
    Version: 1.0.0
    License: Proprietary / Internal Use Only
    """

    def __init__(self, token_path: str | None = None, verbose: bool = True):
        self.endpoint_base = "https://business-api.tiktok.com/open_api/v1.3/"
        self.token_path = (
            token_path if token_path is not None else "token_tiktok.json"
        )  # Cambio aquí
        self.token = None
        self.app_id = None
        self.secret = None
        self.headers = {"Content-Type": "application/json"}

        self.verbose = Verbose(
            active=verbose, alerts_enabled=False, workflow_name="TikTokMarketing class"
        )

        if token_path:
            self._load_token_from_file()
            if self.token:
                self.headers["Access-Token"] = self.token
                self.verbose.log(
                    f"TikTok Class instantiated with token from {self.token_path}."
                )
        elif token_path:
            self.verbose.log(
                f"No token file found at {self.token_path}. Use get_access_token method to generate and save a new token"
            )
        else:
            self.verbose.log(
                "Token file not specified. Use get_access_token method to generate and save a new token"
            )

    def _load_token_from_file(self):
        """Reads token and credentials from file"""
        try:
            with open(self.token_path, "r") as file:
                token_data = json.load(file)

            if not all(k in token_data for k in ("app_id", "secret", "access_token")):
                self.verbose.log(
                    "Token file is missing required fields (app_id, secret, access_token)"
                )
                return None
            self.app_id = token_data.get("app_id")
            self.secret = token_data.get("secret")
            self.token = token_data.get("access_token")
            return token_data
        except Exception as e:
            self.verbose.log(f"Error loading token from file: {e}")
            return None

    def _token_test_connection(self) -> bool:
        """
        Validates the token against the API.
        """
        url = f"{self.endpoint_base}oauth2/advertiser/get/"
        if not self.app_id or not self.secret or not self.token:
            return False

        params = {"app_id": self.app_id, "secret": self.secret}

        test_headers = {
            "Access-Token": self.token,
            "Content-Type": "application/json",
        }

        try:
            connection_test_response = requests.get(
                url, headers=test_headers, params=params
            )
            connection_test_response_json = connection_test_response.json()
            return connection_test_response_json.get("code") == 0
        except Exception as e:
            self.verbose.log(f"Error during connection test: {e}")
            return False

    def _get_report_raw(self, params: dict, max_retries: int = 5):
        """Low level internal API call handler"""
        return self._get_raw("report/integrated/get/", params, max_retries)

    def _get_raw(self, endpoint: str, params: dict, max_retries: int = 5):
        """GET an endpoint under endpoint_base, with backoff on HTTP 429.

        Returns the parsed JSON when TikTok answers code 0, otherwise None.
        """
        url = f"{self.endpoint_base}{endpoint}"
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params)

                if response.status_code == 429:
                    wait_time = (2**attempt) + random.random()
                    self.verbose.log(
                        f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if data.get("code") != 0:
                    self.verbose.log(
                        f"TikTok API Error: {data.get('message')} (Code: {data.get('code')})"
                    )
                    return None

                return data

            except Exception as e:
                self.verbose.log(f"Request failed: {e}")
                return None
        return None

    def validate_connection(self) -> bool:
        """Public method to validate current token connection"""
        if not self.token:
            self.verbose.log("No token available to validate connection")
            return False
        return self._token_test_connection()

    def get_access_token(
        self,
        app_id: str,
        secret: str,
        auth_code: str | None = None,
        redirect_uri: str = "https://tiktok.cl",
    ):
        """Exchanges auth_code for access_token and saves to JSON. Interactively prompts if auth_code is missing."""
        if not auth_code:
            auth_url = f"https://business-api.tiktok.com/portal/auth?app_id={app_id}&state=auth_request&redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
            self.verbose.log(
                "Se requiere autorización manual. Mostrando instrucciones al usuario..."
            )
            print("\n" + "=" * 60)
            print("TIKTOK AUTHENTICATION REQUIRED")
            print("1. Open the following URL in your browser to authorize the app:")
            print(f"\n{auth_url}\n")
            try:
                webbrowser.open(auth_url)
            except Exception as e:
                self.verbose.log(f"Couldn't open browser {e}")

            print(
                "\n2. Authorize the application. You will be redirected to a new URL (your callback)."
            )
            print(
                "3. Look for the parameter '?auth_code=XXXXXXXXX' in the address bar of that new URL."
            )
            auth_code = input(
                "\nPaste only the 'auth_code' value here and press Enter: "
            ).strip()
            print("=" * 60 + "\n")

            if not auth_code:
                self.verbose.log("No code provided, aborting authentication process")
                return False  # Cambio aquí

        url = f"{self.endpoint_base}oauth2/access_token/"
        payload = {"app_id": app_id, "secret": secret, "auth_code": auth_code}

        response = requests.post(url, json=payload)
        res_json = response.json()

        if res_json.get("code") == 0:
            token_data = res_json.get("data", {})

            save_data = {
                "access_token": token_data.get("access_token"),
                "app_id": app_id,
                "secret": secret,
                "scope": token_data.get("scope"),
            }

            self.app_id = app_id
            self.secret = secret
            self.token = token_data.get("access_token")
            self.headers["Access-Token"] = self.token

            try:
                with open(self.token_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                self.verbose.log(f"Token saved successfully as {self.token_path}")
            except Exception as e:
                self.verbose.log(f"Error saving token to file: {e}")

            if self._token_test_connection():
                self.verbose.log("Token test connection succesfull")
                return True  # Cambio Aqui
            else:
                self.verbose.log(
                    "New token generated but failed connection test. Must be checked"
                )
            return True  # Cambio Aqui
        else:
            self.verbose.log(f"Error obtaining token: {res_json.get('message')}")
            return False  # Cambio Aqui

    def get_authorized_advertisers(
        self, app_id: str | None = None, secret: str | None = None
    ):
        """Public method that returns a list of advertisers accessible by the current token"""
        url = f"{self.endpoint_base}oauth2/advertiser/get/"

        # Update credentials if provided
        if app_id and secret:
            self.app_id = app_id
            self.secret = secret

        if not self.app_id or not self.secret:
            self.verbose.log("Missing app_id or secret. Cannot retrieve accounts")
            return []

        params = {"app_id": self.app_id, "secret": self.secret}

        response = requests.get(url, headers=self.headers, params=params)
        data = response.json()
        if data.get("code") == 0:
            return data.get("data", {}).get("list", [])
        return []

    def get_ads(
        self,
        advertiser_id: str,
        fields: list[str] | None = None,
        filtering: dict | None = None,
        page_size: int = 1000,
    ) -> pd.DataFrame:
        """Lists the ads of an advertiser with their attributes, as returned by TikTok.

        Uses the ad/get/ endpoint. Values are not transformed: for Spark Ads,
        tiktok_item_id and identity_id identify the promoted TikTok post.

        Args:
            advertiser_id: TikTok advertiser ID.
            fields: Ad fields to request. Defaults to AD_FIELDS.
            filtering: Raw ad/get/ filtering object, for example
                {"primary_status": "STATUS_ALL"}. If None, TikTok's default
                filter applies (it leaves deleted ads out).
            page_size: Ads per page, 1 to 1000. Defaults to 1000.

        Returns:
            DataFrame with one row per ad and one column per requested field.
            Empty, but with those columns, when the advertiser has no ads.

        Raises:
            RuntimeError: If TikTok returns an error or the request fails.
        """
        fields = list(fields) if fields else list(AD_FIELDS)
        params = {
            "advertiser_id": advertiser_id,
            "fields": json.dumps(fields),
            "page_size": page_size,
        }
        if filtering:
            params["filtering"] = json.dumps(filtering)

        records = []
        page = 1
        while True:
            params["page"] = page
            data = self._get_raw("ad/get/", params)
            if data is None:
                raise RuntimeError(
                    f"Critical error found while listing ads for {advertiser_id}"
                )

            records.extend(data.get("data", {}).get("list") or [])
            total_page = data.get("data", {}).get("page_info", {}).get("total_page", 1)
            if page >= total_page:
                break
            page += 1

        self.verbose.log(f"get_ads | {advertiser_id} | {len(records)} ads")
        return pd.DataFrame(records, columns=fields)

    def get_report_json(self, params: dict, max_retries: int = 5):
        """Public method to retrieve raw JSON data with date chunking and pagination for debugging
        **Any complex query (like using filters) must be done through this method."""

        # Evitar json.dumps en cada iteración, se asume que el usuario pasa los parámetros correctamente formateados.
        # Si no hay fechas en los parámetros, hacemos la llamada directa sin iterar

        if "start_date" not in params or "end_date" not in params:
            self.verbose.log(
                "No start_date or end_date found in params. Making a direct request. IT MUST NOT EXCEED 365 DAYS PERIOD"
            )
            return self._get_report_raw(params, max_retries)

        start_dt = pd.to_datetime(params["start_date"])
        end_dt = pd.to_datetime(params["end_date"])
        current_start = start_dt
        all_records = []

        # Usamos una copia para no mutar el diccionario original que pasa el usuario
        request_params = copy.deepcopy(params)

        for key in ["metrics", "dimensions", "filtering"]:
            if key in request_params and isinstance(request_params[key], list):
                request_params[key] = json.dumps(request_params[key])

        while current_start <= end_dt:
            # Límite de 30 días por petición (current_start + 29 días)
            current_end = min(current_start + pd.Timedelta(days=29), end_dt)
            self.verbose.log(
                f"Extracting JSON {current_start.date()} to {current_end.date()}"
            )

            request_params["start_date"] = current_start.strftime("%Y-%m-%d")
            request_params["end_date"] = current_end.strftime("%Y-%m-%d")

            page = 1
            while True:
                request_params["page"] = page

                # Pasamos tanto los parámetros actualizados como max_retries
                data = self._get_report_raw(request_params, max_retries)

                if not data or "list" not in data.get("data", {}):
                    break

                # Acumulamos los registros de esta página/bloque de fechas
                all_records.extend(data["data"]["list"])

                total_page = (
                    data.get("data", {}).get("page_info", {}).get("total_page", 1)
                )
                if page < total_page:
                    page += 1
                else:
                    break

            # Avanzamos al siguiente bloque de fechas
            current_start = current_end + pd.Timedelta(days=1)

        if not all_records:
            self.verbose.log("No raw JSON data found for the specified period")
            return {}

        self.verbose.log(f"Successfully extracted {len(all_records)} total raw records")

        # Reconstruimos la estructura JSON original consolidando todos los registros
        return {"data": {"list": all_records}}

    def get_report_dataframe(
        self,
        advertiser_id: str,
        start_date: str,
        end_date: str,
        dimensions: list,
        metrics: list,
        data_level: str = "AUCTION_AD",
    ):
        """Extracts data and converts it into a formatted Pandas DataFrame"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days_diff = (end_dt - start_dt).days

        # Whenever we want more complex queries, by default we use get_report_json
        if "stat_time_day" not in dimensions:
            if days_diff > 365:
                raise ValueError(
                    "Cannot retrieve more than 365 days when query acumulated data"
                )

            all_records = []
            page = 1
            while True:
                params = {
                    "advertiser_id": advertiser_id,
                    "service_type": "AUCTION",
                    "report_type": "BASIC",
                    "data_level": data_level,
                    "start_date": start_dt.strftime("%Y-%m-%d"),
                    "end_date": end_dt.strftime("%Y-%m-%d"),
                    "metrics": json.dumps(metrics),
                    "dimensions": json.dumps(dimensions),
                    "page_size": 1000,
                    "page": page,
                }

                data = self._get_report_raw(params)

                if data == None:
                    raise RuntimeError("Critical error found while downloading data")

                if not data or "list" not in data.get("data", {}):
                    break

                all_records.extend(data["data"]["list"])

                total_page = (
                    data.get("data", {}).get("page_info", {}).get("total_page", 1)
                )
                if page < total_page:
                    page += 1
                else:
                    break

            if all_records:
                df = pd.json_normalize(all_records)
                df.columns = [col.split(".")[-1] for col in df.columns]
                self.verbose.log(f"Successfully extracted {len(df)} rows")
                return df

            self.verbose.log("No data found for this advertiser in the selected range.")
            return pd.DataFrame()

        else:
            all_records = []
            current_start = start_dt
            while current_start <= end_dt:
                current_end = min(current_start + pd.Timedelta(days=29), end_dt)
                self.verbose.log(
                    f"Extracting {current_start.date()} to {current_end.date()}"
                )

                page = 1
                while True:
                    params = {
                        "advertiser_id": advertiser_id,
                        "service_type": "AUCTION",
                        "report_type": "BASIC",
                        "data_level": data_level,
                        "start_date": current_start.strftime("%Y-%m-%d"),
                        "end_date": current_end.strftime("%Y-%m-%d"),
                        "metrics": json.dumps(metrics),
                        "dimensions": json.dumps(dimensions),
                        "page_size": 1000,
                        "page": page,
                        "query_lifetime": False,
                    }

                    data = self._get_report_raw(params)

                    if data == None:
                        raise RuntimeError(
                            "Critical error found while downloading data"
                        )

                    if not data or "list" not in data.get("data", {}):
                        break

                    all_records.extend(data["data"]["list"])

                    total_page = (
                        data.get("data", {}).get("page_info", {}).get("total_page", 1)
                    )
                    if page < total_page:
                        page += 1
                    else:
                        break

                current_start = current_end + pd.Timedelta(days=1)

            if all_records:
                df = pd.json_normalize(all_records)
                df.columns = [col.split(".")[-1] for col in df.columns]
                # Revisar si necesitamos eliminar los 0s
                df = df.sort_values(by=["stat_time_day"]).reset_index(drop=True)
                self.verbose.log(f"Successfully extracted {len(df)} rows")
                return df

            self.verbose.log("No data found for the specified period")
            return pd.DataFrame()
