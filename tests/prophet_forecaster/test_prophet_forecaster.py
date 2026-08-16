import os
import sys

import numpy as np
import pandas as pd
import pytest

from d2b_data.prophet_forecaster import ProphetForecaster


# --------------------------------------------------------------------- #
# Construcción y validación de datos
# --------------------------------------------------------------------- #
def test_instance_is_created_correctly(df_metrics):
    """The object starts empty and keeps a reference to the source DataFrame."""
    fc = ProphetForecaster(df_metrics)
    assert fc.df_beforepredict is df_metrics
    assert fc.verbose is True
    assert fc.models == {}
    assert fc.forecasts == {}
    assert fc.last_params == {}
    assert fc.df_ready is None
    assert fc.df_postpredict is None


def test_data_validation_renames_date_column(forecaster):
    """The 'date' column is renamed to 'ds' and parsed as datetime."""
    metrics = forecaster._data_validation(forecaster.df_beforepredict)
    assert list(forecaster.df_ready.columns) == ['ds', 'sessions', 'conversions', 'spend']
    assert pd.api.types.is_datetime64_any_dtype(forecaster.df_ready['ds'])
    assert metrics == ['sessions', 'conversions', 'spend']


def test_data_validation_accepts_spanish_date_column(df_metrics):
    """A 'fecha' column works exactly like 'date'."""
    fc = ProphetForecaster(df_metrics.rename(columns={'date': 'fecha'}), verbose=False)
    fc._data_validation(fc.df_beforepredict)
    assert 'ds' in fc.df_ready.columns
    assert 'fecha' not in fc.df_ready.columns


def test_data_validation_raises_without_date_column(df_metrics):
    """A DataFrame without 'date'/'fecha' is rejected."""
    fc = ProphetForecaster(df_metrics.rename(columns={'date': 'dia'}), verbose=False)
    with pytest.raises(ValueError, match="No date columns found"):
        fc._data_validation(fc.df_beforepredict)


def test_data_validation_raises_on_non_numeric_metric(df_metrics):
    """String dimensions are rejected before hitting Prophet."""
    df = df_metrics.assign(keyword='zapatillas')
    fc = ProphetForecaster(df, verbose=False)
    with pytest.raises(TypeError, match="INTEGRITY ERROR"):
        fc._data_validation(fc.df_beforepredict)


def test_data_validation_raises_on_unknown_metric(forecaster):
    """Asking for a metric that is not in the DataFrame fails fast."""
    with pytest.raises(ValueError, match="Metric columns not found"):
        forecaster._data_validation(forecaster.df_beforepredict, metrics=['ghost'])


def test_data_validation_raises_on_unknown_regressor(forecaster):
    """Asking for a regressor that is not in the DataFrame fails fast."""
    with pytest.raises(ValueError, match="Regressor columns not found"):
        forecaster._data_validation(forecaster.df_beforepredict, regressors=['ghost'])


def test_data_validation_raises_when_no_metrics_left(df_metrics):
    """Using every numeric column as a regressor leaves nothing to forecast."""
    fc = ProphetForecaster(df_metrics, verbose=False)
    with pytest.raises(ValueError, match="No numeric metric columns"):
        fc._data_validation(fc.df_beforepredict, regressors=['sessions', 'conversions', 'spend'])


def test_data_validation_warns_on_nulls(df_metrics, capsys):
    """Null values are reported but do not stop the run."""
    df = df_metrics.copy()
    df.loc[3, 'sessions'] = np.nan
    ProphetForecaster(df)._data_validation(df)
    assert "has Null values" in capsys.readouterr().out


# --------------------------------------------------------------------- #
# get_forecast: uso básico / retro-compatibilidad
# --------------------------------------------------------------------- #
def test_get_forecast_returns_all_metrics(forecaster):
    """The basic entry point forecasts every numeric column."""
    result = forecaster.get_forecast(7)
    assert set(result.columns) == {'date', 'sessions', 'conversions', 'spend'}
    assert len(result) == 67  # 60 históricos + 7 futuros
    assert result['date'].is_monotonic_increasing


def test_get_forecast_rounds_by_default(forecaster):
    """Default output keeps the historical rounding behaviour."""
    result = forecaster.get_forecast(5)
    values = result.drop(columns='date').to_numpy()
    assert np.allclose(values, np.round(values))


def test_get_forecast_stores_models_and_output(forecaster, fake_prophet):
    """One fitted model per metric is kept, plus the raw Prophet output."""
    forecaster.get_forecast(5)
    assert sorted(forecaster.models) == ['conversions', 'sessions', 'spend']
    assert sorted(forecaster.forecasts) == ['conversions', 'sessions', 'spend']
    assert 'trend' in forecaster.forecasts['sessions'].columns
    assert len(fake_prophet.instances) == 3
    assert forecaster.df_postpredict is not None


def test_get_forecast_trains_with_ds_and_y(forecaster, fake_prophet):
    """Each metric is renamed to 'y' before fitting."""
    forecaster.get_forecast(3)
    history = fake_prophet.instances[0].history
    assert list(history.columns) == ['ds', 'y']


def test_get_forecast_uses_prophet_defaults(forecaster, fake_prophet):
    """The basic path keeps Prophet's own defaults."""
    forecaster.get_forecast(3)
    kwargs = fake_prophet.instances[0].kwargs
    assert kwargs['growth'] == 'linear'
    assert kwargs['seasonality_mode'] == 'additive'
    assert kwargs['interval_width'] == 0.80
    assert kwargs['changepoint_prior_scale'] == 0.05
    assert 'holidays' not in kwargs


def test_get_forecast_is_verbose_by_default(df_metrics, capsys):
    """verbose=True (default) keeps the original logging."""
    ProphetForecaster(df_metrics).get_forecast(2)
    assert "Predicting: sessions" in capsys.readouterr().out


def test_verbose_false_silences_logs(forecaster, capsys):
    """verbose=False produces no output at all."""
    forecaster.get_forecast(2)
    assert capsys.readouterr().out == ""


def test_prophet_import_error_is_explicit(forecaster, monkeypatch):
    """A missing prophet install produces an actionable message."""
    monkeypatch.setitem(sys.modules, 'prophet', None)
    with pytest.raises(ImportError, match="pip install prophet"):
        forecaster.get_forecast(2)


# --------------------------------------------------------------------- #
# forecast(): parámetros avanzados
# --------------------------------------------------------------------- #
def test_forecast_selects_subset_of_metrics(forecaster):
    """Only the requested metrics are forecasted."""
    result = forecaster.forecast(days=5, metrics=['sessions'])
    assert list(result.columns) == ['date', 'sessions']
    assert list(forecaster.models) == ['sessions']


def test_forecast_include_history_false(forecaster):
    """include_history=False returns only future dates."""
    result = forecaster.forecast(days=5, metrics=['sessions'], include_history=False)
    assert len(result) == 5
    assert result['date'].min() > pd.Timestamp('2024-02-29')


def test_forecast_respects_frequency(forecaster):
    """freq changes the spacing of the generated future dates."""
    result = forecaster.forecast(days=4, metrics=['sessions'], freq='W', include_history=False)
    deltas = result['date'].diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(days=7)]


def test_forecast_include_intervals(forecaster):
    """include_intervals adds the lower/upper columns per metric."""
    result = forecaster.forecast(days=3, metrics=['sessions'], include_intervals=True)
    assert list(result.columns) == ['date', 'sessions', 'sessions_lower', 'sessions_upper']
    assert (result['sessions_lower'] <= result['sessions']).all()
    assert (result['sessions_upper'] >= result['sessions']).all()


def test_forecast_intervals_for_multiple_metrics(forecaster):
    """Interval columns are namespaced per metric so nothing collides."""
    result = forecaster.forecast(days=3, metrics=['sessions', 'spend'], include_intervals=True)
    assert set(result.columns) == {
        'date', 'sessions', 'sessions_lower', 'sessions_upper',
        'spend', 'spend_lower', 'spend_upper',
    }


def test_forecast_non_negative_clips_values(forecaster):
    """non_negative removes impossible negative predictions."""
    result = forecaster.forecast(days=3, metrics=['sessions'],
                                 include_intervals=True, non_negative=True)
    assert (result['sessions'] >= 0).all()
    assert (result['sessions_lower'] >= 0).all()


def test_forecast_keeps_negatives_by_default(forecaster):
    """Without non_negative the raw Prophet output is preserved."""
    result = forecaster.forecast(days=3, metrics=['sessions'])
    assert (result['sessions'] < 0).any()


def test_forecast_round_decimals(forecaster):
    """round_decimals controls the precision of the output."""
    two = forecaster.forecast(days=3, metrics=['sessions'], round_decimals=2)
    assert np.allclose(two['sessions'].to_numpy(), two['sessions'].round(2).to_numpy())


def test_forecast_round_decimals_none_keeps_precision(forecaster):
    """round_decimals=None returns the untouched float values."""
    raw = forecaster.forecast(days=3, metrics=['sessions'], round_decimals=None)
    assert not np.allclose(raw['sessions'].to_numpy(), raw['sessions'].round().to_numpy())


def test_forecast_passes_tuning_params_to_prophet(forecaster, fake_prophet):
    """Tuning arguments reach the Prophet constructor untouched."""
    forecaster.forecast(
        days=3,
        metrics=['sessions'],
        seasonality_mode='multiplicative',
        interval_width=0.95,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=5.0,
        holidays_prior_scale=1.0,
        n_changepoints=10,
        changepoint_range=0.9,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=12,
    )
    kwargs = fake_prophet.instances[0].kwargs
    assert kwargs['seasonality_mode'] == 'multiplicative'
    assert kwargs['interval_width'] == 0.95
    assert kwargs['changepoint_prior_scale'] == 0.5
    assert kwargs['seasonality_prior_scale'] == 5.0
    assert kwargs['holidays_prior_scale'] == 1.0
    assert kwargs['n_changepoints'] == 10
    assert kwargs['changepoint_range'] == 0.9
    assert kwargs['yearly_seasonality'] is True
    assert kwargs['weekly_seasonality'] is False
    assert kwargs['daily_seasonality'] == 12


def test_forecast_stores_last_params(forecaster):
    """The parameters used are exposed for reproducibility."""
    forecaster.forecast(days=3, metrics=['sessions'], seasonality_mode='multiplicative')
    assert forecaster.last_params['seasonality_mode'] == 'multiplicative'


def test_forecast_prophet_kwargs_passthrough(forecaster, fake_prophet):
    """prophet_kwargs forwards anything not covered by the signature."""
    forecaster.forecast(days=3, metrics=['sessions'],
                        prophet_kwargs={'mcmc_samples': 100, 'uncertainty_samples': 50})
    kwargs = fake_prophet.instances[0].kwargs
    assert kwargs['mcmc_samples'] == 100
    assert kwargs['uncertainty_samples'] == 50


def test_forecast_custom_holidays_dataframe(forecaster, fake_prophet):
    """A custom holidays DataFrame is handed to Prophet."""
    holidays = pd.DataFrame({'holiday': ['cyber'], 'ds': pd.to_datetime(['2024-02-01'])})
    forecaster.forecast(days=3, metrics=['sessions'], holidays=holidays)
    assert fake_prophet.instances[0].kwargs['holidays'] is holidays


def test_forecast_country_holidays(forecaster, fake_prophet):
    """country_holidays triggers add_country_holidays."""
    forecaster.forecast(days=3, metrics=['sessions'], country_holidays='CL')
    assert fake_prophet.instances[0].countries == ['CL']


def test_forecast_custom_seasonalities(forecaster, fake_prophet):
    """custom_seasonalities are registered through add_seasonality."""
    seasonality = {'name': 'monthly', 'period': 30.5, 'fourier_order': 5}
    forecaster.forecast(days=3, metrics=['sessions'], custom_seasonalities=[seasonality])
    assert fake_prophet.instances[0].seasonalities == [seasonality]


# --------------------------------------------------------------------- #
# Crecimiento logístico
# --------------------------------------------------------------------- #
def test_logistic_growth_sets_cap_on_train_and_future(forecaster, fake_prophet):
    """growth='logistic' adds the cap column to both frames."""
    forecaster.forecast(days=3, metrics=['sessions'], growth='logistic', cap=500)
    model = fake_prophet.instances[0]
    assert (model.history['cap'] == 500).all()
    assert (model.future_seen['cap'] == 500).all()


def test_logistic_growth_accepts_cap_per_metric(forecaster, fake_prophet):
    """cap can be a {metric: value} mapping."""
    forecaster.forecast(days=3, metrics=['sessions', 'spend'], growth='logistic',
                        cap={'sessions': 500, 'spend': 900})
    caps = [model.history['cap'].iloc[0] for model in fake_prophet.instances]
    assert caps == [500, 900]


def test_logistic_growth_requires_cap(forecaster):
    """Forgetting the cap raises an explicit error instead of a Prophet crash."""
    with pytest.raises(ValueError, match="requires a 'cap'"):
        forecaster.forecast(days=3, metrics=['sessions'], growth='logistic')


def test_floor_is_applied_when_provided(forecaster, fake_prophet):
    """floor is added to the training and future frames."""
    forecaster.forecast(days=3, metrics=['sessions'], growth='logistic', cap=500, floor=10)
    model = fake_prophet.instances[0]
    assert (model.history['floor'] == 10).all()
    assert (model.future_seen['floor'] == 10).all()


# --------------------------------------------------------------------- #
# Regresores externos
# --------------------------------------------------------------------- #
def test_regressor_is_excluded_from_metrics(forecaster, future_regressors):
    """A column used as a regressor is not forecasted."""
    result = forecaster.forecast(days=5, metrics=['sessions'], regressors=['spend'],
                                 future_regressors=future_regressors)
    assert list(result.columns) == ['date', 'sessions']


def test_regressor_reaches_model_and_frames(forecaster, fake_prophet, future_regressors):
    """The regressor is registered and present in train and future data."""
    forecaster.forecast(days=5, metrics=['sessions'], regressors=['spend'],
                        future_regressors=future_regressors, include_history=False)
    model = fake_prophet.instances[0]
    assert 'spend' in model.extra_regressors
    assert 'spend' in model.history.columns
    assert (model.future_seen['spend'] == 200.0).all()


def test_regressor_accepts_dict_spec(forecaster, fake_prophet, future_regressors):
    """Regressors can be dicts with add_regressor options."""
    forecaster.forecast(days=5, metrics=['sessions'],
                        regressors=[{'name': 'spend', 'mode': 'multiplicative', 'prior_scale': 2}],
                        future_regressors=future_regressors)
    assert fake_prophet.instances[0].extra_regressors['spend'] == {
        'mode': 'multiplicative', 'prior_scale': 2,
    }


def test_regressor_uses_history_when_no_future_values_needed(forecaster, fake_prophet):
    """With include_history=True and days=0 the historical values are enough."""
    forecaster.forecast(days=0, metrics=['sessions'], regressors=['spend'])
    assert not fake_prophet.instances[0].future_seen['spend'].isnull().any()


def test_regressor_without_future_values_raises(forecaster):
    """Missing future regressor values raise a descriptive error."""
    with pytest.raises(ValueError, match="Missing future values"):
        forecaster.forecast(days=5, metrics=['sessions'], regressors=['spend'])


def test_future_regressors_without_date_column_raises(forecaster):
    """future_regressors must carry a date column."""
    bad = pd.DataFrame({'spend': [1.0, 2.0]})
    with pytest.raises(ValueError, match="must contain a 'ds'"):
        forecaster.forecast(days=2, metrics=['sessions'], regressors=['spend'],
                            future_regressors=bad)


def test_future_regressors_missing_column_raises(forecaster):
    """future_regressors must include every declared regressor."""
    bad = pd.DataFrame({'date': pd.date_range('2024-03-01', periods=5, freq='D')})
    with pytest.raises(ValueError, match="missing the columns"):
        forecaster.forecast(days=5, metrics=['sessions'], regressors=['spend'],
                            future_regressors=bad)


def test_invalid_regressor_spec_raises(forecaster):
    """Regressors must be strings or dicts with a 'name'."""
    with pytest.raises(ValueError, match="must be a column name"):
        forecaster.forecast(days=5, metrics=['sessions'], regressors=[123])


# --------------------------------------------------------------------- #
# Persistencia de modelos
# --------------------------------------------------------------------- #
def test_save_models_writes_one_file_per_metric(forecaster, tmp_path):
    """Every trained model is pickled into the target directory."""
    forecaster.get_forecast(3)
    saved = forecaster.save_models(str(tmp_path))
    assert sorted(saved) == ['conversions', 'sessions', 'spend']
    for metric, path in saved.items():
        assert (tmp_path / f'{metric}_model.pkl').exists()
        assert os.path.exists(path)


def test_save_models_without_training_raises(forecaster, tmp_path):
    """Saving before forecasting is rejected."""
    with pytest.raises(ValueError, match="No hay modelos entrenados"):
        forecaster.save_models(str(tmp_path))


def test_save_models_creates_directory(forecaster, tmp_path):
    """The target directory is created if it does not exist."""
    target = tmp_path / 'nested' / 'models'
    forecaster.forecast(days=2, metrics=['sessions'])
    forecaster.save_models(str(target))
    assert target.is_dir()


def test_load_models_roundtrip(forecaster, df_metrics, tmp_path):
    """Models saved by one instance can be loaded by another."""
    forecaster.forecast(days=2, metrics=['sessions', 'spend'])
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    loaded = fresh.load_models(str(tmp_path))
    assert sorted(loaded) == ['sessions', 'spend']
    assert sorted(fresh.models) == ['sessions', 'spend']


def test_load_models_subset(forecaster, df_metrics, tmp_path):
    """Only the requested metrics are loaded."""
    forecaster.get_forecast(2)
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    fresh.load_models(str(tmp_path), metrics=['sessions'])
    assert list(fresh.models) == ['sessions']


def test_load_models_missing_directory_raises(forecaster, tmp_path):
    """A non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        forecaster.load_models(str(tmp_path / 'nope'))


def test_load_models_warns_on_missing_metric(forecaster, df_metrics, tmp_path, capsys):
    """A missing model file warns and keeps going."""
    forecaster.forecast(days=2, metrics=['sessions'])
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    fresh.load_models(str(tmp_path), metrics=['sessions', 'ghost'])
    assert "No se encontró el modelo para 'ghost'" in capsys.readouterr().out
    assert list(fresh.models) == ['sessions']


# --------------------------------------------------------------------- #
# predict_from_loaded_models
# --------------------------------------------------------------------- #
def test_predict_from_loaded_models(forecaster, df_metrics, tmp_path):
    """Loaded models produce the same output shape as a fresh forecast."""
    forecaster.forecast(days=2, metrics=['sessions'])
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    fresh.load_models(str(tmp_path))
    result = fresh.predict_from_loaded_models(3, include_history=False)
    assert list(result.columns) == ['date', 'sessions']
    assert len(result) == 3


def test_predict_from_loaded_models_supports_extras(forecaster, df_metrics, tmp_path):
    """The loaded-model path accepts intervals, clipping and rounding too."""
    forecaster.forecast(days=2, metrics=['sessions'])
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    fresh.load_models(str(tmp_path))
    result = fresh.predict_from_loaded_models(
        4, freq='W', include_history=False, include_intervals=True,
        non_negative=True, round_decimals=1,
    )
    assert list(result.columns) == ['date', 'sessions', 'sessions_lower', 'sessions_upper']
    assert (result['sessions_lower'] >= 0).all()
    assert result['date'].diff().dropna().eq(pd.Timedelta(days=7)).all()


def test_predict_from_loaded_models_with_regressor(forecaster, df_metrics, tmp_path,
                                                   future_regressors):
    """Regressors declared at training time are refilled at prediction time."""
    forecaster.forecast(days=2, metrics=['sessions'], regressors=['spend'],
                        future_regressors=future_regressors)
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    fresh.load_models(str(tmp_path))
    with pytest.raises(ValueError, match="Missing future values"):
        fresh.predict_from_loaded_models(5, include_history=False)

    result = fresh.predict_from_loaded_models(
        5, include_history=False, future_regressors=future_regressors,
    )
    assert len(result) == 5


def test_predict_from_loaded_models_logistic(forecaster, df_metrics, tmp_path):
    """A logistic model reuses its own cap/floor when re-predicting."""
    forecaster.forecast(days=2, metrics=['sessions'], growth='logistic', cap=500, floor=10)
    forecaster.save_models(str(tmp_path))

    fresh = ProphetForecaster(df_metrics, verbose=False)
    fresh.load_models(str(tmp_path))
    fresh.predict_from_loaded_models(3, include_history=False)
    model = fresh.models['sessions']
    assert (model.future_seen['cap'] == 500).all()
    assert (model.future_seen['floor'] == 10).all()


def test_predict_from_loaded_models_without_models_raises(forecaster):
    """Predicting without loading models first is rejected."""
    with pytest.raises(ValueError, match="No hay modelos cargados"):
        forecaster.predict_from_loaded_models(5)
