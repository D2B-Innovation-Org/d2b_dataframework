import json
import os
import time
import webbrowser

import google.auth
import httplib2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from oauth2client import client


class Google_Token_MNG:
    """
    Manages authentication and creates authenticated Google API service objects.

    This class centralizes all supported authentication mechanisms used across
    the project, including:

    - OAuth2 using a stored user token.
    - Interactive OAuth2 flow to generate a new token when one does not exist.
    - Service Account authentication.
    - Application Default Credentials (ADC) for Google Cloud environments.

    The resulting authenticated service can be retrieved through `get_service()`
    and reused by API wrapper classes such as Google_GA4 and
    GoogleSearchConsole.
    """

    def __init__(
        self,
        client_secret: str | None,
        token: str | None,
        api_name: str,
        api_version: str,
        scopes: list[str] | None,
        use_service_account: bool = False,
    ):
        self.scopes = scopes
        self.client_secret = client_secret
        self.token = token
        self.api_name = api_name
        self.version = api_version
        self.use_sa = use_service_account
        self.service = self.create_api(
            api_name=self.api_name,
            api_version=self.version,
            secrets=self.client_secret,
            credentials=self.token,
            scopes=self.scopes,
            use_sa=self.use_sa,
        )

    def save_json(self, filename: str, content: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)

    def open_json(self, filename: str) -> str:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()

        try:
            decoded_content = json.loads(content)
        except json.JSONDecodeError:
            return content

        if isinstance(decoded_content, str):
            return decoded_content

        return content

    def get_credentials(
        self,
        secrets: str | None,
        credentials: str,
        scopes: list[str],
    ):
        """
        Retrieves OAuth2 credentials required to authenticate with a Google API.

        The method follows this authentication order:

        1. Loads an existing OAuth token when the credentials file exists.
        2. Starts an interactive OAuth flow when no token file exists.
        3. Saves the newly generated token at the credentials path.

        Args:
            secrets (str | None):
                Path to the OAuth client secret JSON file. Required only when
                the token file does not already exist.

            credentials (str):
                Path where the OAuth token is loaded from or saved.

            scopes (list[str]):
                OAuth scopes required to access the Google API.

        Returns:
            oauth2client.client.Credentials:
                Credentials that can authorize requests to Google APIs.

        Raises:
            ValueError:
                If the token does not exist and no client secret file is provided.
        """
        if os.path.isfile(credentials):
            return client.Credentials.new_from_json(self.open_json(credentials))

        print("OAuth token not found. Starting authentication flow...")

        if not secrets:
            raise ValueError(
                "A client secret file is required because the OAuth token does not exist."
            )

        flow = client.flow_from_clientsecrets(
            secrets, scope=scopes, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
        )
        auth_uri = flow.step1_get_authorize_url()
        print(f"Please, visit url and authorize token:\n{auth_uri}")

        if not webbrowser.open(auth_uri):
            print("Could not open the web browser correctly")

        time.sleep(3)
        auth_code = input("\nIngresa el código de autorización: ")
        time.sleep(3)

        creds = flow.step2_exchange(auth_code)

        self.save_json(credentials, creds.to_json())

        return creds

    def create_api(
        self,
        api_name: str,
        api_version: str,
        scopes: list[str] | None = None,
        secrets: str | None = None,
        credentials: str | None = None,
        use_sa: bool = False,
    ):
        """
        Creates an authenticated Google API service.

        Depending on the authentication parameters, the method supports the
        following authentication flows:

        1. Service Account or Application Default Credentials (ADC).
        2. OAuth2 using an existing user token or generating a new one if needed.
        3. Public APIs that do not require authentication.

        Args:
            api_name (str):
                Name of the Google API to connect to.

            api_version (str):
                Version of the Google API.

            scopes (list[str] | None):
                OAuth scopes required by the API.

            secrets (str | None):
                Path to the client secret JSON file for OAuth authentication or
                the Service Account key file when using a Service Account.

            credentials (str | None):
                Path to the OAuth token file.

            use_sa (bool, optional):
                Whether to authenticate using a Service Account or Application
                Default Credentials. Defaults to False.

        Returns:
            googleapiclient.discovery.Resource:
                An authenticated Google API service object.
        """

        if use_sa:
            if secrets and os.path.exists(secrets):
                creds = service_account.Credentials.from_service_account_file(
                    secrets, scopes=scopes
                )
            else:
                creds, project = google.auth.default(scopes=scopes)
                print(f"Using ADC Credentials. Project detected: {project}")

            return build(
                api_name, api_version, credentials=creds, cache_discovery=False
            )

        if credentials:
            if not scopes:
                raise ValueError("scopes are required when using OAuth authentication.")

            creds = self.get_credentials(
                secrets=secrets,
                credentials=credentials,
                scopes=scopes,
            )

            http_auth = creds.authorize(httplib2.Http())

            return build(
                api_name,
                api_version,
                http=http_auth,
                cache_discovery=False,
            )

        return build(
            api_name,
            api_version,
            cache_discovery=False,
        )

    def get_service(self):
        return self.service
