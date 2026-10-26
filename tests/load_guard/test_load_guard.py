from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from d2b_data.load_guard import (
    STATUS_ERROR,
    STATUS_OK,
    LoadGuard,
    normalize_account,
)


TARGET = "bice-data-management.facebook_BM.act"
# 26-oct-2026 10:00 en Santiago (UTC-3 en horario de verano).
NOW = datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc)
SINCE = date(2026, 10, 10)
UNTIL = date(2026, 10, 25)


def build(rows=None, query_error=None, insert_result=None, insert_error=None):
    client = MagicMock()
    if query_error is not None:
        client.query.side_effect = query_error
    else:
        client.query.return_value.result.return_value = [
            {"account": account} for account in (rows or [])
        ]
    if insert_error is not None:
        client.insert_rows_json.side_effect = insert_error
    else:
        client.insert_rows_json.return_value = insert_result or []
    logger = MagicMock()
    return LoadGuard(client=client, logger=logger), client, logger


def query_params(client):
    job_config = client.query.call_args.kwargs["job_config"]
    return {p.name: p for p in job_config.query_parameters}


# --------------------------------------------------------------------- #
# normalize_account
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("act_123", "123"),
        ("123", "123"),
        (123, "123"),
        (" act_123 ", "123"),
    ],
)
def test_normalize_account(raw, expected):
    """Ids end up as plain strings without the Meta 'act_' prefix."""
    assert normalize_account(raw) == expected


# --------------------------------------------------------------------- #
# freshness_cutoff
# --------------------------------------------------------------------- #
def test_cutoff_in_chilean_summer_time():
    """06:00 Santiago in October (UTC-3) is 09:00 UTC."""
    guard, _, _ = build()
    assert guard.freshness_cutoff(6, NOW) == datetime(
        2026, 10, 26, 9, 0, tzinfo=timezone.utc
    )


def test_cutoff_in_chilean_winter_time():
    """06:00 Santiago in July (UTC-4) is 10:00 UTC."""
    guard, _, _ = build()
    july = datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc)
    assert guard.freshness_cutoff(6, july) == datetime(
        2026, 7, 15, 10, 0, tzinfo=timezone.utc
    )


def test_cutoff_uses_local_date_not_utc_date():
    """At 22:00 Santiago the UTC date is already tomorrow; the cutoff is not."""
    guard, _, _ = build()
    late_night = datetime(2026, 10, 27, 1, 0, tzinfo=timezone.utc)  # 26-oct 22:00
    assert guard.freshness_cutoff(6, late_night) == datetime(
        2026, 10, 26, 9, 0, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("hour", [-1, 24])
def test_cutoff_rejects_invalid_hour(hour):
    """Hours outside 0-23 raise ValueError."""
    guard, _, _ = build()
    with pytest.raises(ValueError):
        guard.freshness_cutoff(hour, NOW)


# --------------------------------------------------------------------- #
# already_loaded
# --------------------------------------------------------------------- #
def test_already_loaded_when_every_account_is_fresh():
    """All requested accounts present means the load can be skipped."""
    guard, _, _ = build(rows=["111", "222"])
    assert guard.already_loaded("facebook", TARGET, ["111", "222"], SINCE, UNTIL, now=NOW)


def test_not_loaded_when_an_account_is_missing():
    """One missing account forces the full load."""
    guard, _, logger = build(rows=["111"])
    assert not guard.already_loaded(
        "facebook", TARGET, ["111", "222"], SINCE, UNTIL, now=NOW
    )
    assert "['222']" in logger.log.call_args.args[0]


def test_not_loaded_without_accounts_and_no_query():
    """An empty account list never skips and never queries."""
    guard, client, _ = build()
    assert not guard.already_loaded("facebook", TARGET, [], SINCE, UNTIL, now=NOW)
    client.query.assert_not_called()


def test_query_failure_means_load_anyway():
    """If the control table cannot be read, the function must still load."""
    guard, _, logger = build(query_error=RuntimeError("403 Access Denied"))
    assert not guard.already_loaded("facebook", TARGET, ["111"], SINCE, UNTIL, now=NOW)
    assert "se carga igual" in logger.log.call_args.args[0]


def test_query_parameters():
    """The query filters by source, target, normalized accounts, status and dates."""
    guard, client, _ = build(rows=["111"])
    guard.already_loaded("facebook", TARGET, ["act_111", 111], SINCE, UNTIL, now=NOW)

    params = query_params(client)
    assert params["source"].value == "facebook"
    assert params["target"].value == TARGET
    assert params["accounts"].values == ["111"]
    assert params["status"].value == STATUS_OK
    assert params["since"].value == SINCE
    assert params["until"].value == UNTIL
    assert "date_since <= @since" in client.query.call_args.args[0]
    assert params["cutoff"].value == datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)
    assert "`d2b-data-management.ops.load_log`" in client.query.call_args.args[0]


def test_custom_fresh_after_hour_changes_cutoff():
    """fresh_after_hour moves the cutoff."""
    guard, client, _ = build(rows=["111"])
    guard.already_loaded(
        "facebook", TARGET, ["111"], SINCE, UNTIL, fresh_after_hour=4, now=NOW
    )
    assert query_params(client)["cutoff"].value == datetime(
        2026, 10, 26, 7, 0, tzinfo=timezone.utc
    )


# --------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------- #
def record_kwargs(**overrides):
    kwargs = {
        "source": "facebook",
        "target": TARGET,
        "client_name": "BICE",
        "account": "act_111",
        "date_since": SINCE,
        "date_until": UNTIL,
        "rows": 250,
        "status": STATUS_OK,
        "now": NOW,
    }
    kwargs.update(overrides)
    return kwargs


def test_record_writes_expected_row():
    """A successful load inserts one row with normalized values."""
    guard, client, _ = build()
    assert guard.record(**record_kwargs()) is True

    table_id, rows = client.insert_rows_json.call_args.args
    assert table_id == "d2b-data-management.ops.load_log"
    assert rows == [
        {
            "loaded_at": "2026-10-26T13:00:00+00:00",
            "source": "facebook",
            "target": TARGET,
            "client": "BICE",
            "account": "111",
            "date_since": "2026-10-10",
            "date_until": "2026-10-25",
            "rows": 250,
            "status": "ok",
            "error": None,
        }
    ]


def test_record_accepts_zero_rows():
    """A day without data is still a successful load."""
    guard, client, _ = build()
    assert guard.record(**record_kwargs(rows=0)) is True
    assert client.insert_rows_json.call_args.args[1][0]["rows"] == 0


def test_record_error_status_keeps_message():
    """Failed loads keep the error text."""
    guard, client, _ = build()
    guard.record(**record_kwargs(status=STATUS_ERROR, error="2 de 16 tablas fallaron"))
    row = client.insert_rows_json.call_args.args[1][0]
    assert row["status"] == "error"
    assert row["error"] == "2 de 16 tablas fallaron"


def test_record_rejects_unknown_status():
    """Only ok and error are valid statuses."""
    guard, _, _ = build()
    with pytest.raises(ValueError):
        guard.record(**record_kwargs(status="partial"))


def test_record_returns_false_on_insert_errors():
    """BigQuery row errors are logged, not raised."""
    guard, _, logger = build(insert_result=[{"index": 0, "errors": ["bad"]}])
    assert guard.record(**record_kwargs()) is False
    assert "no se pudo registrar" in logger.log.call_args.args[0]


def test_record_returns_false_on_exception():
    """A network or permission error never breaks the finished load."""
    guard, _, logger = build(insert_error=RuntimeError("403"))
    assert guard.record(**record_kwargs()) is False
    assert "403" in logger.log.call_args.args[0]


# --------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------- #
def test_client_is_created_lazily_with_table_project(mocker):
    """Without a client, one is created on first use in the table's project."""
    factory = mocker.patch("d2b_data.load_guard.bigquery.Client")
    guard = LoadGuard(table_id="otro-proyecto.ops.load_log")
    factory.assert_not_called()

    guard.client
    factory.assert_called_once_with(project="otro-proyecto")


def test_works_without_logger():
    """The default null logger accepts calls silently."""
    client = MagicMock()
    client.query.side_effect = RuntimeError("x")
    guard = LoadGuard(client=client)
    assert guard.already_loaded("facebook", TARGET, ["1"], SINCE, UNTIL, now=NOW) is False
