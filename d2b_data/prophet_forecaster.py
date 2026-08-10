import pandas as pd
import pickle
import os
import logging
from pathlib import Path


# Aquí agregamos los imports que vamos a usar
class ProphetForecaster:
    """Clase para generar pronósticos usando Prophet.

    Uso básico (sin configuración):
        >>> fc = ProphetForecaster(df)
        >>> df_pred = fc.get_forecast(30)

    Uso avanzado (control fino sobre el modelo):
        >>> fc = ProphetForecaster(df, verbose=False)
        >>> df_pred = fc.forecast(
        ...     days=90,
        ...     freq='W',
        ...     seasonality_mode='multiplicative',
        ...     country_holidays='CL',
        ...     include_intervals=True,
        ...     non_negative=True,
        ... )

    ARGS:
        df_topredict (DataFrame): DataFrame con una columna de fecha
            ('date' o 'fecha') y una o más columnas numéricas a predecir.
        verbose (bool): Si es False silencia los logs de la clase y de
            Prophet/cmdstanpy. Por defecto True.
    RETURNS:
        DataFrame con las predicciones para las métricas especificadas.
    """

    DATE_OPTIONS = ['date', 'fecha']

    def __init__(self, df_topredict, verbose=True):
        self.df_beforepredict = df_topredict
        # añadir validar
        self.verbose = verbose
        self.models = {}
        self.forecasts = {}          # salida completa de Prophet por métrica
        self.last_params = {}        # parámetros usados en el último forecast
        self.display = None
        self.df_ready = None
        self.df_postpredict = None

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #
    def _log(self, msg):
        """Imprime solo si la instancia está en modo verbose."""
        if self.verbose:
            print(msg)

    def _data_validation(self, df_topredict, metrics=None, regressors=None):
        """Validates the DataFrame data.

        Args:
            df_topredict (DataFrame): The DataFrame to be validated.
            metrics (list, optional): Subconjunto de columnas a predecir.
                Si es None se usan todas las columnas numéricas.
            regressors (list, optional): Columnas que son regresores externos
                y por lo tanto NO deben tratarse como métricas a predecir.
        """
        date_options = self.DATE_OPTIONS
        regressors = list(regressors or [])

        date_in_df = [col for col in df_topredict.columns if col in date_options]
        if not date_in_df:
            raise ValueError("No date columns found in the DataFrame. Please upload a 'date' or 'fecha' column.")
        date_to_predict = date_in_df[0]

        candidate_cols = [col for col in df_topredict.columns if col not in date_options]

        missing_reg = [r for r in regressors if r not in candidate_cols]
        if missing_reg:
            raise ValueError(f"Regressor columns not found in the DataFrame: {missing_reg}")

        if metrics is None:
            metrics_in_df = [col for col in candidate_cols if col not in regressors]
        else:
            missing_metrics = [m for m in metrics if m not in candidate_cols]
            if missing_metrics:
                raise ValueError(f"Metric columns not found in the DataFrame: {missing_metrics}")
            metrics_in_df = list(metrics)

        if not metrics_in_df:
            raise ValueError("No numeric metric columns left to forecast.")

        prophet_dataframe = df_topredict.copy()

        # 3. Strict Type Checking
        for col in metrics_in_df + regressors:
            if not pd.api.types.is_numeric_dtype(prophet_dataframe[col]):
                raise TypeError(
                    f"\n[INTEGRITY ERROR]: Column '{col}' contains string or non-numeric data.\n"
                    f"Prophet requires numeric values to forecast. Please remove dimensions "
                    f"(like keywords, pages, or categories) and only pass date and numeric metrics."
                )

        prophet_dataframe[date_to_predict] = pd.to_datetime(prophet_dataframe[date_to_predict], format='%Y-%m-%d')
        prophet_dataframe.rename(columns={date_to_predict: 'ds'}, inplace=True)
        self._log(f" Using {date_to_predict} as date column")

        for col in metrics_in_df + regressors:
            prophet_dataframe[col] = pd.to_numeric(prophet_dataframe[col], errors='coerce')
            if prophet_dataframe[col].isnull().any():
                self._log(f" Column {col} has Null values, check for data integrity")

        new_date = ['ds']
        df_final_cols = new_date + metrics_in_df + regressors
        prophet_dataframe = prophet_dataframe[df_final_cols]
        self.df_ready = prophet_dataframe
        return metrics_in_df

    def _import_prophet(self):
        """Importa Prophet y ajusta el nivel de log según self.verbose."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required for forecasting: pip install prophet")

        if not self.verbose:
            for logger_name in ('prophet', 'cmdstanpy', 'fbprophet'):
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        return Prophet

    @staticmethod
    def _normalize_regressors(regressors):
        """Acepta ['col'] o [{'name': 'col', 'mode': 'multiplicative'}] y normaliza a dicts."""
        normalized = []
        for reg in (regressors or []):
            if isinstance(reg, str):
                normalized.append({'name': reg})
            elif isinstance(reg, dict) and 'name' in reg:
                normalized.append(dict(reg))
            else:
                raise ValueError(
                    "Each regressor must be a column name (str) or a dict with a 'name' key."
                )
        return normalized

    @staticmethod
    def _value_for(param, metric):
        """Permite pasar un valor único o un dict {metrica: valor}."""
        if isinstance(param, dict):
            return param.get(metric)
        return param

    def _build_model(self, Prophet, params, custom_seasonalities, country_holidays,
                     regressor_specs):
        """Instancia Prophet con los parámetros dados y le agrega los extras."""
        m = Prophet(**params)

        for season in (custom_seasonalities or []):
            m.add_seasonality(**season)
            self._log(f" Custom seasonality '{season.get('name')}' added")

        if country_holidays:
            m.add_country_holidays(country_name=country_holidays)
            self._log(f" Country holidays '{country_holidays}' added")

        for reg in regressor_specs:
            m.add_regressor(**reg)
            self._log(f" Regressor '{reg['name']}' added")

        return m

    def _normalize_future_regressors(self, future_regressors, regressor_names):
        """Valida el DataFrame de regresores futuros y normaliza su columna de fecha."""
        extra = future_regressors.copy()
        date_col = next((c for c in extra.columns if c in self.DATE_OPTIONS + ['ds']), None)
        if date_col is None:
            raise ValueError("future_regressors must contain a 'ds', 'date' or 'fecha' column.")
        extra.rename(columns={date_col: 'ds'}, inplace=True)
        extra['ds'] = pd.to_datetime(extra['ds'])
        missing_cols = [r for r in regressor_names if r not in extra.columns]
        if missing_cols:
            raise ValueError(f"future_regressors is missing the columns: {missing_cols}")
        return extra[['ds'] + regressor_names]

    def _attach_regressors(self, future, regressor_names, future_regressors, history_source):
        """Rellena los valores de los regresores en el DataFrame futuro.

        Los valores históricos salen de `history_source` (el DataFrame validado
        o el `history` del modelo cargado) y los futuros de `future_regressors`.
        """
        if not regressor_names:
            return future

        history = history_source[['ds'] + regressor_names]

        if future_regressors is not None:
            extra = self._normalize_future_regressors(future_regressors, regressor_names)
            history = pd.concat([history, extra], ignore_index=True)

        history = history.drop_duplicates(subset='ds', keep='last')
        future = future.merge(history, on='ds', how='left')

        incomplete = [r for r in regressor_names if future[r].isnull().any()]
        if incomplete:
            raise ValueError(
                f"Missing future values for the regressors {incomplete}. "
                f"Pass them through 'future_regressors' (a DataFrame with 'ds' + regressor columns)."
            )
        return future

    @staticmethod
    def _apply_growth_bounds(df, growth, cap, floor):
        """Agrega las columnas cap/floor que exige el crecimiento logístico."""
        if growth == 'logistic':
            if cap is None:
                raise ValueError("growth='logistic' requires a 'cap' value (number or {metric: value}).")
            df['cap'] = cap
        if floor is not None:
            df['floor'] = floor
        return df

    def _format_forecast(self, forecast, metric, include_intervals, non_negative):
        """Deja la salida de Prophet con el formato estándar de la clase."""
        cols = ['ds', 'yhat']
        renames = {'ds': 'date', 'yhat': metric}
        if include_intervals:
            cols += ['yhat_lower', 'yhat_upper']
            renames.update({'yhat_lower': f'{metric}_lower', 'yhat_upper': f'{metric}_upper'})

        clean = forecast[cols].rename(columns=renames)

        if non_negative:
            value_cols = [c for c in clean.columns if c != 'date']
            clean[value_cols] = clean[value_cols].clip(lower=0)

        return clean

    @staticmethod
    def _merge_results(results, forecast_clean):
        if results.empty:
            return forecast_clean
        return pd.merge(forecast_clean, results, on='date', how='outer')

    def _finalize(self, results, round_decimals):
        results = results.sort_values('date')
        if round_decimals is not None:
            results = results.round(round_decimals)
        self.df_postpredict = results
        return self.df_postpredict

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def get_forecast(self, days):
        """Generates forecasts for the specified metrics using Prophet.

        Uso básico: entrena un Prophet con los parámetros por defecto para
        cada métrica numérica del DataFrame. Para control fino usa forecast().

        Args:
            days (int): Number of days to generate the forecast for.

        Returns:
            DataFrame: A DataFrame containing the predictions for the specified metrics.
        """
        return self.forecast(days=days)

    def forecast(self,
                 days,
                 metrics=None,
                 freq='D',
                 include_history=True,
                 include_intervals=False,
                 interval_width=0.80,
                 growth='linear',
                 cap=None,
                 floor=None,
                 seasonality_mode='additive',
                 yearly_seasonality='auto',
                 weekly_seasonality='auto',
                 daily_seasonality='auto',
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0,
                 n_changepoints=25,
                 changepoint_range=0.8,
                 holidays=None,
                 country_holidays=None,
                 custom_seasonalities=None,
                 regressors=None,
                 future_regressors=None,
                 non_negative=False,
                 round_decimals=0,
                 prophet_kwargs=None):
        """Versión configurable de get_forecast().

        Args:
            days (int): Períodos a predecir (en la unidad de `freq`).
            metrics (list, optional): Columnas a predecir. None = todas las numéricas.
            freq (str): Frecuencia de las fechas futuras ('D', 'W', 'MS', 'H'...).
            include_history (bool): Si False solo devuelve las fechas futuras.
            include_intervals (bool): Agrega columnas `<metrica>_lower` / `_upper`.
            interval_width (float): Ancho del intervalo de confianza (0-1).
            growth (str): 'linear', 'logistic' o 'flat'.
            cap (float|dict): Techo requerido por growth='logistic'. Acepta
                un número o {metrica: valor}.
            floor (float|dict): Piso opcional para growth='logistic'.
            seasonality_mode (str): 'additive' o 'multiplicative'.
            yearly_seasonality / weekly_seasonality / daily_seasonality:
                'auto', True, False o un entero (orden de Fourier).
            changepoint_prior_scale (float): Flexibilidad de la tendencia
                (sube para tendencias más reactivas, baja para suavizar).
            seasonality_prior_scale (float): Fuerza de la estacionalidad.
            holidays_prior_scale (float): Fuerza del efecto de feriados.
            n_changepoints (int): Cantidad de puntos de cambio potenciales.
            changepoint_range (float): Proporción del histórico donde buscarlos.
            holidays (DataFrame, optional): DataFrame de feriados custom de Prophet
                (columnas 'holiday' y 'ds').
            country_holidays (str, optional): Código de país para feriados
                automáticos, por ejemplo 'CL', 'US', 'MX'.
            custom_seasonalities (list, optional): Lista de dicts para
                add_seasonality(), ej. [{'name': 'monthly', 'period': 30.5,
                'fourier_order': 5}].
            regressors (list, optional): Columnas del DataFrame a usar como
                regresores externos (no se predicen). Acepta nombres o dicts
                con los argumentos de add_regressor().
            future_regressors (DataFrame, optional): Valores futuros de los
                regresores ('ds' + columnas de los regresores).
            non_negative (bool): Recorta las predicciones negativas a 0.
            round_decimals (int|None): Decimales del resultado. None = sin redondear.
            prophet_kwargs (dict, optional): Cualquier otro argumento del
                constructor de Prophet (mcmc_samples, uncertainty_samples, etc.).

        Returns:
            DataFrame: predicciones por fecha, una columna por métrica.
        """
        regressor_specs = self._normalize_regressors(regressors)
        regressor_names = [r['name'] for r in regressor_specs]

        metrics_to_predict = self._data_validation(
            self.df_beforepredict, metrics=metrics, regressors=regressor_names
        )

        Prophet = self._import_prophet()

        base_params = {
            'growth': growth,
            'interval_width': interval_width,
            'seasonality_mode': seasonality_mode,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'n_changepoints': n_changepoints,
            'changepoint_range': changepoint_range,
        }
        if holidays is not None:
            base_params['holidays'] = holidays
        base_params.update(prophet_kwargs or {})
        self.last_params = dict(base_params)

        self.models = {}
        self.forecasts = {}
        results = pd.DataFrame()

        for metric in metrics_to_predict:
            self._log(f" Predicting: {metric}...")

            metric_cap = self._value_for(cap, metric)
            metric_floor = self._value_for(floor, metric)

            df_train = self.df_ready[['ds', metric] + regressor_names].rename(columns={metric: 'y'})
            df_train = self._apply_growth_bounds(df_train, growth, metric_cap, metric_floor)
            self._log(f" Column {metric} renamed succesfully to 'y'")

            m = self._build_model(Prophet, base_params, custom_seasonalities,
                                  country_holidays, regressor_specs)
            self._log(" Class Prophet instanciated")

            m.fit(df_train)
            self._log(" Training successull")

            future = m.make_future_dataframe(periods=days, freq=freq,
                                             include_history=include_history)
            future = self._apply_growth_bounds(future, growth, metric_cap, metric_floor)
            future = self._attach_regressors(future, regressor_names, future_regressors,
                                             self.df_ready)

            forecast = m.predict(future)
            self._log(" future dates calculated")

            forecast_clean = self._format_forecast(forecast, metric, include_intervals, non_negative)

            self.models[metric] = m
            self.forecasts[metric] = forecast

            results = self._merge_results(results, forecast_clean)
            self._log(f" Forecast para {metric} por {days} períodos ({freq}) listo")

        self._finalize(results, round_decimals)
        self._log(f" Forecast ready for {self.df_postpredict.columns.to_list()}")
        return self.df_postpredict

    def save_models(self, directory='prophet_models'):
        """Guarda todos los modelos entrenados en archivos pickle.

        Args:
            directory (str): Directorio donde se guardarán los modelos.
                           Por defecto crea una carpeta 'prophet_models'.

        Returns:
            dict: Diccionario con las rutas de los archivos guardados.
        """
        if not self.models:
            raise ValueError("No hay modelos entrenados. Ejecuta get_forecast() primero.")

        # Crear directorio si no existe
        Path(directory).mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for metric, model in self.models.items():
            filename = f"{metric}_model.pkl"
            filepath = os.path.join(directory, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

            saved_files[metric] = filepath
            self._log(f" Modelo para '{metric}' guardado en: {filepath}")

        self._log(f"\n✓ {len(saved_files)} modelos guardados exitosamente")
        return saved_files

    def load_models(self, directory='prophet_models', metrics=None):
        """Carga modelos previamente guardados desde archivos pickle.

        Args:
            directory (str): Directorio donde están guardados los modelos.
            metrics (list, optional): Lista de métricas específicas a cargar.
                                    Si es None, carga todos los modelos disponibles.

        Returns:
            dict: Diccionario con los modelos cargados.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"El directorio '{directory}' no existe.")

        # Si no se especifican métricas, cargar todos los archivos .pkl
        if metrics is None:
            model_files = [f for f in os.listdir(directory) if f.endswith('_model.pkl')]
            metrics = [f.replace('_model.pkl', '') for f in model_files]

        loaded_models = {}

        for metric in metrics:
            filename = f"{metric}_model.pkl"
            filepath = os.path.join(directory, filename)

            if not os.path.exists(filepath):
                print(f"⚠ Advertencia: No se encontró el modelo para '{metric}' en {filepath}")
                continue

            with open(filepath, 'rb') as f:
                model = pickle.load(f)

            loaded_models[metric] = model
            self._log(f" Modelo para '{metric}' cargado desde: {filepath}")

        self.models = loaded_models
        self._log(f"\n✓ {len(loaded_models)} modelos cargados exitosamente")
        return loaded_models

    def predict_from_loaded_models(self, days, freq='D', include_history=True,
                                   include_intervals=False, non_negative=False,
                                   round_decimals=0, future_regressors=None):
        """Genera predicciones usando modelos previamente cargados.

        Args:
            days (int): Número de períodos a predecir.
            freq (str): Frecuencia de las fechas futuras ('D', 'W', 'MS'...).
            include_history (bool): Si False solo devuelve las fechas futuras.
            include_intervals (bool): Agrega columnas `<metrica>_lower` / `_upper`.
            non_negative (bool): Recorta las predicciones negativas a 0.
            round_decimals (int|None): Decimales del resultado. None = sin redondear.
            future_regressors (DataFrame, optional): Valores futuros de los
                regresores si los modelos guardados los usan.

        Returns:
            DataFrame: DataFrame con las predicciones.
        """
        if not self.models:
            raise ValueError("No hay modelos cargados. Usa load_models() primero.")

        results = pd.DataFrame()

        for metric, model in self.models.items():
            self._log(f" Prediciendo con modelo cargado: {metric}...")

            future = model.make_future_dataframe(periods=days, freq=freq,
                                                 include_history=include_history)

            # El modelo guardado ya conoce sus propios cap/floor y regresores
            if getattr(model, 'growth', 'linear') == 'logistic':
                future['cap'] = model.history['cap'].max()
                if 'floor' in model.history.columns:
                    future['floor'] = model.history['floor'].min()

            regressor_names = list(getattr(model, 'extra_regressors', {}).keys())
            future = self._attach_regressors(future, regressor_names, future_regressors,
                                             model.history)

            forecast = model.predict(future)
            forecast_clean = self._format_forecast(forecast, metric, include_intervals, non_negative)

            self.forecasts[metric] = forecast
            results = self._merge_results(results, forecast_clean)

            self._log(f" Predicción para {metric} completada")

        self._finalize(results, round_decimals)
        self._log(f"\n✓ Predicciones listas para {self.df_postpredict.columns.to_list()}")
        return self.df_postpredict
