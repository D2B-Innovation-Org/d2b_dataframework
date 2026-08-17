import sys
import types

import numpy as np
import pandas as pd
import pytest


class FakeProphet:
    """Minimal stand-in for prophet.Prophet.

    Prophet (and its cmdstan backend) is a heavy optional dependency, so the
    test suite injects this stub instead. It mirrors the parts of the API the
    forecaster uses and records everything it receives so the tests can assert
    on how the model was configured, trained and queried.
    """

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.growth = kwargs.get('growth', 'linear')
        self.seasonalities = []
        self.countries = []
        self.extra_regressors = {}
        self.history = None
        self.future_seen = None
        FakeProphet.instances.append(self)

    def add_seasonality(self, **kwargs):
        self.seasonalities.append(kwargs)

    def add_country_holidays(self, country_name):
        self.countries.append(country_name)

    def add_regressor(self, name, **kwargs):
        self.extra_regressors[name] = kwargs

    def fit(self, df):
        self.history = df.copy()
        return self

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        last = self.history['ds'].max()
        future_dates = pd.date_range(start=last, periods=periods + 1, freq=freq)[1:]
        dates = list(self.history['ds']) + list(future_dates) if include_history else list(future_dates)
        return pd.DataFrame({'ds': pd.to_datetime(dates)})

    def predict(self, future):
        self.future_seen = future.copy()
        out = future[['ds']].copy()
        # Empieza en negativo a propósito para poder probar non_negative,
        # y con decimales para poder probar el redondeo.
        out['yhat'] = np.linspace(-5.5, 100.5, len(out))
        out['yhat_lower'] = out['yhat'] - 10
        out['yhat_upper'] = out['yhat'] + 10
        out['trend'] = out['yhat']
        return out


@pytest.fixture(autouse=True)
def fake_prophet(monkeypatch):
    """Injects the Prophet stub into sys.modules for every test."""
    FakeProphet.instances = []
    module = types.ModuleType('prophet')
    module.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, 'prophet', module)
    return FakeProphet


@pytest.fixture
def df_metrics():
    """60 daily rows with a 'date' column and three numeric metrics."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=60, freq='D').strftime('%Y-%m-%d'),
        'sessions': np.arange(60, dtype=float),
        'conversions': np.arange(60, dtype=float) * 0.3,
        'spend': np.arange(60, dtype=float) * 2,
    })


@pytest.fixture
def forecaster(df_metrics):
    """Silent ProphetForecaster over the sample DataFrame."""
    from d2b_data.prophet_forecaster import ProphetForecaster

    return ProphetForecaster(df_metrics, verbose=False)


@pytest.fixture
def future_regressors():
    """Future values for the 'spend' regressor."""
    return pd.DataFrame({
        'date': pd.date_range('2024-03-01', periods=10, freq='D'),
        'spend': [200.0] * 10,
    })
