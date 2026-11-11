"""Tests for d2b_data.bq_loader. BigQuery is always a mock: no network calls."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.api_core import exceptions as google_exceptions
from google.cloud import bigquery

from d2b_data.bq_loader import ensure_partitioned_table, load_df_to_partitioned_table

TABLE = "proj.dataset.ga4_123"


def make_table(partition_field: str | None = "date") -> bigquery.Table:
    """Build a table object partitioned by the given field (None = not partitioned)."""
    table = bigquery.Table(TABLE)
    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(field=partition_field)
    return table


@pytest.fixture
def bq_client() -> MagicMock:
    """BigQuery client whose table exists and whose jobs succeed by default."""
    client = MagicMock(spec=bigquery.Client)
    client.get_table.return_value = make_table()
    return client


@pytest.fixture
def logger() -> MagicMock:
    """Stand-in for WorkflowLogger that records calls."""
    return MagicMock()


@pytest.fixture(autouse=True)
def no_sleep():
    """Skip real waits between retries."""
    with patch("d2b_data.bq_loader.time.sleep") as mocked_sleep:
        yield mocked_sleep


def make_df(dates: list) -> pd.DataFrame:
    """Build a minimal DataFrame with one row per given date."""
    return pd.DataFrame({"date": dates, "sessions": [1.0] * len(dates)})


class TestLoadDfToPartitionedTable:
    """Tests for the partition replacement flow of load_df_to_partitioned_table."""

    def test_single_date_runs_one_load_and_no_delete(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A single date runs one load job, no DML, and reports no failures."""
        df = make_df([date(2026, 10, 1)])

        failed = load_df_to_partitioned_table(bq_client, df, TABLE, logger)

        assert failed == []
        bq_client.query.assert_not_called()
        assert bq_client.load_table_from_dataframe.call_count == 1
        logger.critical.assert_not_called()

    def test_load_targets_the_partition_decorator(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """The destination is 'table$YYYYMMDD' for the date being loaded."""
        load_df_to_partitioned_table(
            bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
        )

        destination = bq_client.load_table_from_dataframe.call_args.args[1]
        assert destination == f"{TABLE}$20261001"

    def test_load_truncates_partition_and_allows_field_addition(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """The load job replaces the partition and allows new columns."""
        load_df_to_partitioned_table(
            bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
        )

        job_config = bq_client.load_table_from_dataframe.call_args.kwargs[
            "job_config"
        ]
        assert (
            job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        assert job_config.schema_update_options == [
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        ]

    def test_schema_uses_table_types_for_existing_columns(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """Table columns take its type, matched ignoring case; new ones are left out."""
        table = make_table()
        table.schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("sessionsourcemedium", "STRING"),
            bigquery.SchemaField("sessions", "FLOAT64"),
            bigquery.SchemaField("only_in_table", "STRING"),
        ]
        bq_client.get_table.return_value = table
        df = make_df([date(2026, 10, 1)])
        df["sessionSourceMedium"] = ["google / cpc"]
        df["new_metric"] = [1.0]

        load_df_to_partitioned_table(bq_client, df, TABLE, logger)

        job_config = bq_client.load_table_from_dataframe.call_args.kwargs[
            "job_config"
        ]
        assert [(f.name, f.field_type, f.mode) for f in job_config.schema] == [
            ("date", "DATE", "REQUIRED"),
            ("sessions", "FLOAT64", "NULLABLE"),
            ("sessionSourceMedium", "STRING", "NULLABLE"),
        ]

    def test_multiple_dates_load_one_partition_each(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """Each distinct date is loaded to its own partition with only its rows."""
        dates = [date(2026, 10, 1), date(2026, 10, 1), date(2026, 10, 2)]

        load_df_to_partitioned_table(bq_client, make_df(dates), TABLE, logger)

        calls = bq_client.load_table_from_dataframe.call_args_list
        assert [len(call.args[0]) for call in calls] == [2, 1]
        assert [call.args[1] for call in calls] == [
            f"{TABLE}$20261001",
            f"{TABLE}$20261002",
        ]

    def test_custom_partition_column(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A non-default partition column is used to split the partitions."""
        bq_client.get_table.return_value = make_table("event_date")
        df = pd.DataFrame(
            {"event_date": [date(2026, 10, 1), date(2026, 10, 2)], "value": [1.0, 2.0]}
        )

        load_df_to_partitioned_table(
            bq_client, df, TABLE, logger, partition_column="event_date"
        )

        destinations = [
            call.args[1] for call in bq_client.load_table_from_dataframe.call_args_list
        ]
        assert destinations == [f"{TABLE}$20261001", f"{TABLE}$20261002"]

    def test_empty_dataframe_does_nothing(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """An empty DataFrame triggers no BigQuery calls."""
        failed = load_df_to_partitioned_table(bq_client, make_df([]), TABLE, logger)

        assert failed == []
        bq_client.get_table.assert_not_called()
        bq_client.load_table_from_dataframe.assert_not_called()


class TestTransientErrors:
    """Transient errors are retried; exhausted retries skip the date."""

    @pytest.mark.parametrize(
        "error",
        [
            google_exceptions.ServiceUnavailable("Backend Unavailable"),
            google_exceptions.InternalServerError("backendError"),
            google_exceptions.TooManyRequests("rateLimitExceeded"),
        ],
    )
    def test_transient_error_retries_then_succeeds(
        self,
        bq_client: MagicMock,
        logger: MagicMock,
        no_sleep: MagicMock,
        error: Exception,
    ) -> None:
        """A 5xx or 429 on the first attempt is retried and the second succeeds."""
        bq_client.load_table_from_dataframe.side_effect = [error, MagicMock()]

        failed = load_df_to_partitioned_table(
            bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
        )

        assert failed == []
        assert bq_client.load_table_from_dataframe.call_count == 2
        no_sleep.assert_called_once_with(10)
        logger.warning.assert_called_once()
        logger.critical.assert_not_called()

    def test_transient_error_exhausts_retries(
        self, bq_client: MagicMock, logger: MagicMock, no_sleep: MagicMock
    ) -> None:
        """Persistent 503s stop after max_retries, alert, and report the date."""
        target = date(2026, 10, 1)
        bq_client.load_table_from_dataframe.side_effect = (
            google_exceptions.ServiceUnavailable("503")
        )

        failed = load_df_to_partitioned_table(
            bq_client, make_df([target]), TABLE, logger, max_retries=3
        )

        assert failed == [target]
        assert bq_client.load_table_from_dataframe.call_count == 3
        assert [c.args[0] for c in no_sleep.call_args_list] == [10, 20]
        logger.critical.assert_called_once()

    def test_exhausted_date_does_not_stop_next_date(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A date that exhausts retries is reported and the next date still loads."""
        first, second = date(2026, 10, 1), date(2026, 10, 2)
        unavailable = google_exceptions.ServiceUnavailable("503")
        bq_client.load_table_from_dataframe.side_effect = [unavailable] * 3 + [
            MagicMock()
        ]

        failed = load_df_to_partitioned_table(
            bq_client, make_df([first, second]), TABLE, logger
        )

        assert failed == [first]
        assert bq_client.load_table_from_dataframe.call_args.args[1] == (
            f"{TABLE}$20261002"
        )


class TestLogicalErrors:
    """Non-transient errors stop the load instead of repeating on every date."""

    def test_non_transient_error_raises_without_retry(
        self, bq_client: MagicMock, logger: MagicMock, no_sleep: MagicMock
    ) -> None:
        """A 400 is raised on the first date and no other date is attempted."""
        bq_client.load_table_from_dataframe.side_effect = (
            google_exceptions.BadRequest("Field sessions has changed type")
        )
        df = make_df([date(2026, 10, 1), date(2026, 10, 2)])

        with pytest.raises(google_exceptions.BadRequest):
            load_df_to_partitioned_table(bq_client, df, TABLE, logger)

        assert bq_client.load_table_from_dataframe.call_count == 1
        no_sleep.assert_not_called()
        logger.error.assert_called_once()

    def test_503_in_message_of_logical_error_is_not_retried(
        self, bq_client: MagicMock, logger: MagicMock, no_sleep: MagicMock
    ) -> None:
        """A 400 whose message contains '503' (e.g. in a table id) is not retried."""
        bq_client.load_table_from_dataframe.side_effect = (
            google_exceptions.BadRequest("Invalid value in proj.dataset.ga4_125034567")
        )

        with pytest.raises(google_exceptions.BadRequest):
            load_df_to_partitioned_table(
                bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
            )

        assert bq_client.load_table_from_dataframe.call_count == 1
        no_sleep.assert_not_called()

    def test_non_google_error_raises(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """Errors outside BigQuery (e.g. DataFrame conversion) are raised too."""
        bq_client.load_table_from_dataframe.side_effect = TypeError("bad dtype")

        with pytest.raises(TypeError):
            load_df_to_partitioned_table(
                bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
            )


class TestValidation:
    """Input checks that run before anything is written."""

    @pytest.mark.parametrize(
        "bad_dates",
        [
            ["2026-10-01"],
            [datetime(2026, 10, 1)],
            list(pd.to_datetime(["2026-10-01"])),
        ],
        ids=["string", "datetime", "timestamp"],
    )
    def test_partition_values_must_be_plain_dates(
        self, bq_client: MagicMock, logger: MagicMock, bad_dates: list
    ) -> None:
        """Strings, datetimes and Timestamps are rejected before touching BigQuery."""
        with pytest.raises(ValueError, match="must hold datetime.date"):
            load_df_to_partitioned_table(bq_client, make_df(bad_dates), TABLE, logger)

        bq_client.load_table_from_dataframe.assert_not_called()

    def test_datetime64_column_is_rejected(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A datetime64 column (pd.to_datetime without .dt.date) is rejected."""
        df = make_df([date(2026, 10, 1)])
        df["date"] = pd.to_datetime(df["date"])

        with pytest.raises(ValueError, match="must hold datetime.date"):
            load_df_to_partitioned_table(bq_client, df, TABLE, logger)

    def test_null_partition_values_are_rejected(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """Null dates are rejected."""
        with pytest.raises(ValueError, match="null values"):
            load_df_to_partitioned_table(
                bq_client, make_df([date(2026, 10, 1), None]), TABLE, logger
            )

    def test_missing_partition_column_is_rejected(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A DataFrame without the partition column is rejected."""
        df = pd.DataFrame({"sessions": [1.0]})

        with pytest.raises(ValueError, match="not found in DataFrame"):
            load_df_to_partitioned_table(bq_client, df, TABLE, logger)

    def test_missing_table_is_rejected(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A table that does not exist fails once, not once per date."""
        bq_client.get_table.side_effect = google_exceptions.NotFound("table")
        df = make_df([date(2026, 10, 1), date(2026, 10, 2)])

        with pytest.raises(ValueError, match="does not exist"):
            load_df_to_partitioned_table(bq_client, df, TABLE, logger)

        bq_client.load_table_from_dataframe.assert_not_called()

    @pytest.mark.parametrize("partition_field", [None, "event_date"])
    def test_table_partitioned_by_other_column_is_rejected(
        self, bq_client: MagicMock, logger: MagicMock, partition_field: str | None
    ) -> None:
        """A table not partitioned by the partition column is rejected."""
        bq_client.get_table.return_value = make_table(partition_field)

        with pytest.raises(ValueError, match="must be partitioned by 'date'"):
            load_df_to_partitioned_table(
                bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
            )

        bq_client.load_table_from_dataframe.assert_not_called()

    def test_table_with_monthly_partitions_is_rejected(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """Only daily partitions can be targeted with 'table$YYYYMMDD'."""
        table = make_table()
        table.time_partitioning.type_ = bigquery.TimePartitioningType.MONTH
        bq_client.get_table.return_value = table

        with pytest.raises(ValueError, match="daily partitions"):
            load_df_to_partitioned_table(
                bq_client, make_df([date(2026, 10, 1)]), TABLE, logger
            )

    def test_dbdate_column_from_bigquery_is_accepted(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """The dbdate dtype returned by to_dataframe() passes validation."""
        df = make_df([date(2026, 10, 1)])
        df["date"] = df["date"].astype("dbdate")

        failed = load_df_to_partitioned_table(bq_client, df, TABLE, logger)

        assert failed == []


class TestCreateIfMissing:
    """Optional creation of the destination table before loading."""

    @pytest.fixture
    def missing_table_client(self, bq_client: MagicMock) -> MagicMock:
        """Client whose table does not exist and whose create_table echoes back."""
        bq_client.get_table.side_effect = google_exceptions.NotFound("table")
        bq_client.create_table.side_effect = lambda table, exists_ok: table
        return bq_client

    def test_missing_table_without_flag_is_rejected_and_not_created(
        self, missing_table_client: MagicMock, logger: MagicMock
    ) -> None:
        """By default a missing table is still a ValueError and nothing is created."""
        with pytest.raises(ValueError, match="does not exist"):
            load_df_to_partitioned_table(
                missing_table_client, make_df([date(2026, 10, 1)]), TABLE, logger
            )

        missing_table_client.create_table.assert_not_called()
        missing_table_client.load_table_from_dataframe.assert_not_called()

    def test_missing_table_is_created_with_only_the_partition_column(
        self, missing_table_client: MagicMock, logger: MagicMock
    ) -> None:
        """The new table is partitioned by day on the column, with expiration."""
        df = pd.DataFrame(
            {"segments_date": [date(2026, 10, 1)], "metrics_clicks": [3]}
        )

        failed = load_df_to_partitioned_table(
            missing_table_client,
            df,
            TABLE,
            logger,
            partition_column="segments_date",
            create_if_missing=True,
            partition_expiration_days=30,
        )

        assert failed == []
        created = missing_table_client.create_table.call_args.args[0]
        assert missing_table_client.create_table.call_args.kwargs == {
            "exists_ok": True
        }
        assert created.reference.table_id == "ga4_123"
        assert [(f.name, f.field_type) for f in created.schema] == [
            ("segments_date", "DATE")
        ]
        assert created.time_partitioning.field == "segments_date"
        assert created.time_partitioning.type_ == bigquery.TimePartitioningType.DAY
        assert created.time_partitioning.expiration_ms == 30 * 24 * 60 * 60 * 1000

    def test_new_table_load_adds_the_other_columns(
        self, missing_table_client: MagicMock, logger: MagicMock
    ) -> None:
        """The load truncates the partition, allows field addition and only
        pins the partition column type, so the client infers the rest."""
        df = pd.DataFrame(
            {"segments_date": [date(2026, 10, 1)], "metrics_clicks": [3]}
        )

        load_df_to_partitioned_table(
            missing_table_client,
            df,
            TABLE,
            logger,
            partition_column="segments_date",
            create_if_missing=True,
        )

        call = missing_table_client.load_table_from_dataframe.call_args
        job_config = call.kwargs["job_config"]
        assert call.args[1] == f"{TABLE}$20261001"
        assert (
            job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        assert job_config.schema_update_options == [
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        ]
        assert [(f.name, f.field_type) for f in job_config.schema] == [
            ("segments_date", "DATE")
        ]

    def test_default_expiration_is_1096_days(
        self, missing_table_client: MagicMock, logger: MagicMock
    ) -> None:
        """Without partition_expiration_days the new table keeps about 3 years."""
        load_df_to_partitioned_table(
            missing_table_client,
            make_df([date(2026, 10, 1)]),
            TABLE,
            logger,
            create_if_missing=True,
        )

        created = missing_table_client.create_table.call_args.args[0]
        assert created.time_partitioning.expiration_ms == 1096 * 24 * 60 * 60 * 1000

    def test_existing_table_is_not_touched(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """With the flag on, an existing table is used as is and not recreated."""
        failed = load_df_to_partitioned_table(
            bq_client,
            make_df([date(2026, 10, 1)]),
            TABLE,
            logger,
            create_if_missing=True,
        )

        assert failed == []
        bq_client.create_table.assert_not_called()
        bq_client.update_table.assert_not_called()
        assert bq_client.load_table_from_dataframe.call_count == 1

    @pytest.mark.parametrize("partition_field", [None, "event_date"])
    def test_existing_table_with_other_partition_is_rejected(
        self, bq_client: MagicMock, logger: MagicMock, partition_field: str | None
    ) -> None:
        """An existing table partitioned in another way is still a ValueError."""
        bq_client.get_table.return_value = make_table(partition_field)

        with pytest.raises(ValueError, match="must be partitioned by 'date'"):
            load_df_to_partitioned_table(
                bq_client,
                make_df([date(2026, 10, 1)]),
                TABLE,
                logger,
                create_if_missing=True,
            )

        bq_client.create_table.assert_not_called()
        bq_client.load_table_from_dataframe.assert_not_called()

    def test_table_created_concurrently_with_other_partition_is_rejected(
        self, missing_table_client: MagicMock, logger: MagicMock
    ) -> None:
        """If another run created the table differently, the load stops."""
        missing_table_client.create_table.side_effect = None
        missing_table_client.create_table.return_value = make_table("event_date")

        with pytest.raises(ValueError, match="must be partitioned by 'date'"):
            load_df_to_partitioned_table(
                missing_table_client,
                make_df([date(2026, 10, 1)]),
                TABLE,
                logger,
                create_if_missing=True,
            )

        missing_table_client.load_table_from_dataframe.assert_not_called()

    def test_empty_dataframe_does_not_create_the_table(
        self, missing_table_client: MagicMock, logger: MagicMock
    ) -> None:
        """Without rows there is nothing to load, so no table is created."""
        load_df_to_partitioned_table(
            missing_table_client, make_df([]), TABLE, logger, create_if_missing=True
        )

        missing_table_client.create_table.assert_not_called()

    def test_other_get_table_errors_are_raised(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A permission error reading the table is not mistaken for a missing one."""
        bq_client.get_table.side_effect = google_exceptions.Forbidden("denied")

        with pytest.raises(google_exceptions.Forbidden):
            load_df_to_partitioned_table(
                bq_client,
                make_df([date(2026, 10, 1)]),
                TABLE,
                logger,
                create_if_missing=True,
            )

        bq_client.create_table.assert_not_called()


class TestEnsurePartitionedTable:
    """Standalone helper ported from the older loader."""

    def test_creates_table_and_returns_true(
        self, bq_client: MagicMock, logger: MagicMock
    ) -> None:
        """A missing or matching table is ready; the definition has one column."""
        bq_client.create_table.side_effect = lambda table, exists_ok: table

        assert ensure_partitioned_table(bq_client, TABLE, logger) is True

        created = bq_client.create_table.call_args.args[0]
        assert [f.name for f in created.schema] == ["date"]
        assert created.time_partitioning.field == "date"
        logger.critical.assert_not_called()

    @pytest.mark.parametrize("partition_field", [None, "event_date"])
    def test_existing_table_with_other_partition_returns_false(
        self, bq_client: MagicMock, logger: MagicMock, partition_field: str | None
    ) -> None:
        """An existing table partitioned in another way is reported and alerted."""
        bq_client.create_table.return_value = make_table(partition_field)

        assert ensure_partitioned_table(bq_client, TABLE, logger) is False
        logger.critical.assert_called_once()
