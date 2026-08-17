# d2b_data/HubSpot_Api.py

import requests
import pandas as pd
from typing import List, Dict, Any

class HubSpot_API:
    """
    Una clase de ayuda para interactuar con la API de HubSpot v3/v4.
    """
    BASE_URL = "https://api.hubapi.com/"

    def __init__(self, token: str, verbose_logger=None):
        if not token:
            raise ValueError("Se requiere un token de HubSpot.")
        self.token = token
        self.verbose = verbose_logger
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })

    def _log(self, message: str, level: str = "info"):
        """Método de logging interno."""
        if self.verbose:
            if level == "error":
                self.verbose.error(message)
            else:
                self.verbose.log(message)
        else:
            print(f"[{level.upper()}] {message}")

    def call_api(self, method: str, endpoint: str, params: Dict = None, json_data: Dict = None) -> Dict:
        """
        Realiza una llamada genérica a la API de HubSpot.

        Args:
            method (str): Método HTTP (GET, POST, etc.).
            endpoint (str): El endpoint de la API (ej: 'crm/v3/objects/contacts').
            params (Dict, optional): Parámetros de la URL.
            json_data (Dict, optional): Cuerpo de la solicitud en formato JSON.

        Returns:
            Dict: La respuesta de la API en formato JSON, o un diccionario vacío si falla.
        """
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.request(method, url, params=params, json=json_data, timeout=30)
            response.raise_for_status()  # Lanza una excepción para errores HTTP (4xx o 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            self._log(f"Error HTTP en {method} {url}: {e.response.status_code} - {e.response.text}", "error")
        except requests.exceptions.RequestException as e:
            self._log(f"Error de conexión en {method} {url}: {e}", "error")
        
        return {} # Retorna un diccionario vacío en caso de error

    def test_connection(self) -> bool:
        """Verifica si la conexión y el token son válidos."""
        self._log("Probando conexión con HubSpot...")
        try:
            # Pide un solo contacto para verificar que el token funciona
            response = self.call_api("GET", "crm/v3/objects/contacts", params={"limit": 1})
            if "results" in response:
                self._log("✓ Conexión a HubSpot exitosa.")
                return True
            self._log(f"La conexión a HubSpot falló. Respuesta: {response}", "error")
            return False
        except Exception as e:
            self._log(f"Excepción al probar la conexión: {e}", "error")
            return False

    def to_dataframe(self, records: List[Dict], properties: List[str] = None) -> pd.DataFrame:
        """
        Convierte una lista de registros de HubSpot (con 'properties') a un DataFrame de Pandas.
        """
        if not records:
            return pd.DataFrame()

        flat_data = []
        for record in records:
            row = {}
            # Copiar campos de nivel superior como 'id', 'createdAt', 'updatedAt', 'archived'
            for key, value in record.items():
                if key != 'properties' and not isinstance(value, (dict, list)):
                    row[key] = value

            # Aplanar el diccionario 'properties'
            props = record.get('properties', {})
            if props:
                row.update(props)
            
            flat_data.append(row)
        
        df = pd.DataFrame(flat_data)

        # Renombrar 'id' a 'hs_object_id' si existe para consistencia
        if 'id' in df.columns and 'hs_object_id' not in df.columns:
            df.rename(columns={'id': 'hs_object_id'}, inplace=True)
            
        return df