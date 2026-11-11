"""Idempotent loads of DataFrames into date-partitioned BigQuery tables.

Each distinct date of the partition column is written with its own load job
(``WRITE_TRUNCATE`` on ``table$YYYYMMDD``), which swaps the partition
atomically: a failed load leaves the previous rows in place, so a date is
never left empty, and reprocessing a range does not duplicate rows.

The destination table must be partitioned by day on a ``DATE`` column. It can
be created on the fly with only that column; the first load adds the rest
with ``ALLOW_FIELD_ADDITION``, which BigQuery accepts together with
``WRITE_TRUNCATE`` when the destination is a partition decorator.

Errors are split in two groups. Transient ones (5xx and 429) are retried per
date and, if they persist, the date is reported back and the load moves on.
Anything else is a logical error that would repeat on every date, so it is
raised at once.
"""

from __future__ import annotations

import time
from datetime import date, datetime

import pandas as pd
from google.api_core import exceptions as google_exceptions
from google.cloud import bigquery

from d2b_data.workflow_logger import WorkflowLogger

# 5xx (backendError, internalError, 503) and 429 (rateLimitExceeded) go away
# on retry. Everything else (missing table, permissions, schema) would fail
# again on every date.
TRANSIENT_ERRORS = (
    google_exceptions.ServerError,
    google_exceptions.TooManyRequests,
)
DEFAULT_PARTITION_EXPIRATION_DAYS = 1096  # About three years.
MS_PER_DAY = 24 * 60 * 60 * 1000


def ensure_partitioned_table(
    bq_client: bigquery.Client,
    table_full_id: str,
    logger: WorkflowLogger,
    partition_column: str = "date",
    partition_expiration_days: int = DEFAULT_PARTITION_EXPIRATION_DAYS,
) -> bool:
    """Creates the date-partitioned table if it does not exist yet.

    The table is created with only the partition column; the remaining
    columns are added by the first load (``ALLOW_FIELD_ADDITION``). An
    existing table is left untouched.

    Args:
        bq_client: Authenticated BigQuery client.
        table_full_id: Table as 'project.dataset.table'.
        logger: Logger used for progress messages and critical alerts.
        partition_column: Name of the DATE column to partition by.
            Defaults to 'date'.
        partition_expiration_days: Days each partition is kept before
            BigQuery drops it. Defaults to 1096 (about three years).

    Returns:
        True if the table exists and is partitioned by the partition column,
        False otherwise.
    """
    table = bq_client.create_table(
        _new_partitioned_table(
            table_full_id, partition_column, partition_expiration_days
        ),
        exists_ok=True,
    )

    if (
        table.time_partitioning is None
        or table.time_partitioning.field != partition_column
    ):
        logger.critical(
            f"Table {table_full_id} exists but is not partitioned by "
            f"'{partition_column}'. Skipping load."
        )
        return False

    logger.info(f"ensure_partitioned_table | Table {table_full_id} ready")
    return True


def load_df_to_partitioned_table(
    bq_client: bigquery.Client,
    df: pd.DataFrame,
    table_full_id: str,
    logger: WorkflowLogger,
    partition_column: str = "date",
    max_retries: int = 3,
    retry_delay: int = 10,
    create_if_missing: bool = False,
    partition_expiration_days: int = DEFAULT_PARTITION_EXPIRATION_DAYS,
) -> list[date]:
    """Loads a DataFrame into a date-partitioned BigQuery table, one date at a time.

    For each distinct value of the partition column, replaces that partition
    with a single load job (WRITE_TRUNCATE on 'table$YYYYMMDD'). The swap is
    atomic: if the load fails the partition keeps its previous rows, so a
    failed date is never left empty. Reprocessing a date range is idempotent
    and dates absent from the DataFrame are not touched. New columns in the
    DataFrame are added to the table (ALLOW_FIELD_ADDITION); columns missing
    from the DataFrame are NULL in the replaced partitions.

    Before writing anything, checks that the partition column holds only
    `datetime.date` values and that the table exists and is partitioned by
    day on that column. With ``create_if_missing`` a missing table is first
    created with only the partition column.

    The load is retried on transient errors (5xx and 429) with a linear
    backoff. A transient error that exhausts the retries is logged as
    critical (which sends an alert) and the function moves on to the next
    date. Any other error is a logical one that would repeat on every date,
    so it is raised and the load stops.

    Args:
        bq_client: Authenticated BigQuery client.
        df: Data to load. The partition column must hold `datetime.date`
            values so it matches the DATE type of the table.
        table_full_id: Destination table as 'project.dataset.table'.
        logger: Logger used for progress messages and critical alerts.
        partition_column: Name of the DATE column the table is partitioned by.
            Defaults to 'date'.
        max_retries: Attempts per date before giving up on transient errors.
            Defaults to 3.
        retry_delay: Base seconds to wait between attempts; the wait is
            retry_delay * attempt. Defaults to 10.
        create_if_missing: If True and the table does not exist, create it
            partitioned by day on ``partition_column`` before loading.
            Defaults to False.
        partition_expiration_days: Partition expiration for a table created
            by ``create_if_missing``. Ignored for existing tables. Defaults
            to 1096 (about three years).

    Returns:
        The dates that could not be written because of transient errors. An
        empty list means every partition was loaded successfully.

    Raises:
        ValueError: If the partition column or the table fail validation,
            including an existing table partitioned in another way.
        Exception: Any non-transient error from BigQuery or from the
            DataFrame conversion, re-raised as is.
    """
    _validate_partition_column(df, partition_column)
    if df.empty:
        return []
    table = _get_or_create_table(
        bq_client,
        table_full_id,
        logger,
        partition_column,
        create_if_missing,
        partition_expiration_days,
    )
    _validate_table(table, table_full_id, partition_column)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        ],
        schema=_schema_for_dataframe(table, df),
    )

    failed_dates = []

    for unique_date in df[partition_column].unique():
        iter_df = df[df[partition_column] == unique_date].copy()

        logger.info(
            f"load_df_to_partitioned_table | Processing partition {unique_date}"
        )

        for attempt in range(1, max_retries + 1):
            try:
                # 'table$YYYYMMDD' makes WRITE_TRUNCATE replace only that day.
                partition_id = f"{table_full_id}${unique_date:%Y%m%d}"

                load_job = bq_client.load_table_from_dataframe(
                    iter_df,
                    partition_id,
                    job_config=job_config,
                )

                load_job.result()

                logger.info(
                    f"load_df_to_partitioned_table | Partition {unique_date} "
                    f"successfully updated in {table_full_id}"
                )

                break

            except TRANSIENT_ERRORS as e_transient:
                logger.warning(
                    f"RETRY {attempt}/{max_retries} - "
                    f"Transient error writing "
                    f"{table_full_id}: {str(e_transient)}"
                )

                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue

                logger.critical(
                    f"Failed to write {table_full_id} "
                    f"(partition {unique_date}): {str(e_transient)}"
                )
                failed_dates.append(unique_date)

            except Exception as e_logical:
                # A logical error would repeat on every date, so stop here and
                # let the caller decide how to alert.
                logger.error(
                    f"load_df_to_partitioned_table | Non-transient error writing "
                    f"{table_full_id} (partition {unique_date}), stopping load: "
                    f"{str(e_logical)}"
                )
                raise

    return failed_dates


def _new_partitioned_table(
    table_full_id: str, partition_column: str, partition_expiration_days: int
) -> bigquery.Table:
    """Builds a table definition with only the DATE partition column.

    Args:
        table_full_id: Table as 'project.dataset.table'.
        partition_column: Name of the DATE column to partition by.
        partition_expiration_days: Days each partition is kept.

    Returns:
        The table definition, not yet created in BigQuery.
    """
    table = bigquery.Table(
        table_full_id,
        schema=[bigquery.SchemaField(partition_column, "DATE")],
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field=partition_column,
        expiration_ms=partition_expiration_days * MS_PER_DAY,
    )
    return table


def _get_or_create_table(
    bq_client: bigquery.Client,
    table_full_id: str,
    logger: WorkflowLogger,
    partition_column: str,
    create_if_missing: bool,
    partition_expiration_days: int,
) -> bigquery.Table:
    """Fetches the destination table, creating it when allowed.

    Args:
        bq_client: Authenticated BigQuery client.
        table_full_id: Table as 'project.dataset.table'.
        logger: Logger used for progress messages.
        partition_column: Name of the DATE column to partition by.
        create_if_missing: Whether a missing table may be created.
        partition_expiration_days: Partition expiration for a new table.

    Returns:
        The existing or newly created table.

    Raises:
        ValueError: If the table does not exist and create_if_missing is
            False.
    """
    try:
        return bq_client.get_table(table_full_id)
    except google_exceptions.NotFound as e_not_found:
        if not create_if_missing:
            raise ValueError(
                f"Table {table_full_id} does not exist. Create it partitioned by "
                f"'{partition_column}' before loading."
            ) from e_not_found

    # exists_ok covers another run creating the table in the meantime; the
    # returned table is validated by the caller either way.
    table = bq_client.create_table(
        _new_partitioned_table(
            table_full_id, partition_column, partition_expiration_days
        ),
        exists_ok=True,
    )
    logger.info(
        f"load_df_to_partitioned_table | Created table {table_full_id} "
        f"partitioned by '{partition_column}'"
    )
    return table


def _validate_partition_column(df: pd.DataFrame, partition_column: str) -> None:
    """Checks that the partition column exists and holds only `datetime.date`.

    Args:
        df: Data to load.
        partition_column: Name of the partition column.

    Raises:
        ValueError: If the column is missing, has nulls, or holds values that
            are not `datetime.date` (e.g. strings, `datetime` or `Timestamp`).
    """
    if partition_column not in df.columns:
        raise ValueError(
            f"Partition column '{partition_column}' not found in DataFrame. "
            f"Columns: {list(df.columns)}"
        )

    values = df[partition_column]
    null_count = int(values.isna().sum())
    if null_count:
        raise ValueError(
            f"Partition column '{partition_column}' has {null_count} null values."
        )

    # datetime and pd.Timestamp subclass date, so they are excluded apart.
    is_plain_date = values.map(
        lambda v: isinstance(v, date) and not isinstance(v, datetime)
    )
    if not is_plain_date.all():
        bad_value = values[~is_plain_date].iloc[0]
        raise ValueError(
            f"Partition column '{partition_column}' must hold datetime.date "
            f"values; found {type(bad_value).__name__} ({bad_value!r}). "
            f"Convert it with pd.to_datetime(df['{partition_column}']).dt.date"
        )


def _validate_table(
    table: bigquery.Table, table_full_id: str, partition_column: str
) -> None:
    """Checks that the table is partitioned by day on the partition column.

    Args:
        table: Destination table.
        table_full_id: Table as 'project.dataset.table', for messages.
        partition_column: Name of the partition column.

    Raises:
        ValueError: If the table is not partitioned by `partition_column`
            or its partitions are not daily.
    """
    table_partition_field = (
        table.time_partitioning.field if table.time_partitioning else None
    )
    if table_partition_field != partition_column:
        raise ValueError(
            f"Table {table_full_id} must be partitioned by '{partition_column}'; "
            f"its partition column is {table_partition_field!r}."
        )

    # The $YYYYMMDD decorator only addresses daily partitions.
    partition_type = table.time_partitioning.type_
    if partition_type != bigquery.TimePartitioningType.DAY:
        raise ValueError(
            f"Table {table_full_id} must have daily partitions; "
            f"its partition type is {partition_type!r}."
        )


def _schema_for_dataframe(
    table: bigquery.Table, df: pd.DataFrame
) -> list[bigquery.SchemaField]:
    """Takes the table types for the DataFrame columns that already exist in it.

    With WRITE_TRUNCATE the client does not fetch the table schema by itself,
    so it is passed explicitly to keep the DataFrame types aligned with the
    table. Names are matched ignoring case, as BigQuery does, and keep the
    DataFrame spelling. Columns that are not in the table are left out so the
    client infers their type and ALLOW_FIELD_ADDITION adds them.

    Args:
        table: Destination table.
        df: Data to load.

    Returns:
        Schema fields for the DataFrame columns present in the table.
    """
    table_fields = {field.name.lower(): field for field in table.schema}
    schema = []
    for column in df.columns:
        field = table_fields.get(str(column).lower())
        if field:
            schema.append(
                bigquery.SchemaField(
                    column, field.field_type, mode=field.mode, fields=field.fields
                )
            )
    return schema
