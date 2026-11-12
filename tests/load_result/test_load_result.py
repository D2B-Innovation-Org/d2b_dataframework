from unittest.mock import MagicMock

import pytest

from d2b_data.load_guard import STATUS_ERROR, STATUS_OK
from d2b_data.load_result import (
    MAX_ALERT_ACCOUNTS,
    AccountResult,
    ErrorKind,
    ExtractionError,
    LoadReport,
    call_with_retries,
    classify_error,
)


class _HttpLikeError(Exception):
    """Imita requests.HTTPError: el código viene en response.status_code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = MagicMock(status_code=status_code)


class _GoogleApiClientLikeError(Exception):
    """Imita googleapiclient HttpError: el código viene en resp.status."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HttpError {status}")
        self.resp = MagicMock(status=status)


# --------------------------------------------------------------------- #
# classify_error
# --------------------------------------------------------------------- #
class TestClassifyError:
    """Tests for classify_error."""

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_status_codes(self, status: int) -> None:
        """401 and 403 are permission or credential errors."""
        assert classify_error(ExtractionError("x", status_code=status)) is ErrorKind.AUTH

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_status_codes(self, status: int) -> None:
        """429 and 5xx are transient and worth retrying."""
        error = ExtractionError("x", status_code=status)
        assert classify_error(error) is ErrorKind.TRANSIENT

    @pytest.mark.parametrize("status", [400, 404])
    def test_other_status_codes(self, status: int) -> None:
        """Other 4xx are neither auth nor transient."""
        assert classify_error(ExtractionError("x", status_code=status)) is ErrorKind.OTHER

    def test_requests_style_status(self) -> None:
        """The status code is read from response.status_code."""
        assert classify_error(_HttpLikeError(403)) is ErrorKind.AUTH

    def test_googleapiclient_style_status(self) -> None:
        """The status code is read from resp.status (the GA4 enovus case)."""
        error = _GoogleApiClientLikeError(
            403, "User does not have sufficient permissions for this property"
        )
        assert classify_error(error) is ErrorKind.AUTH

    def test_string_status_code(self) -> None:
        """A numeric string status code is accepted."""
        assert classify_error(ExtractionError("x", status_code="503")) is ErrorKind.TRANSIENT

    def test_non_http_code_attribute_is_ignored(self) -> None:
        """A 'code' attribute outside the HTTP range does not count as status."""
        error = ExtractionError("x")
        error.code = 40001
        assert classify_error(error) is ErrorKind.OTHER

    def test_permission_text_without_status(self) -> None:
        """Without status, PERMISSION_DENIED in the text means auth."""
        assert classify_error(RuntimeError("reason PERMISSION_DENIED")) is ErrorKind.AUTH

    def test_invalid_grant_text(self) -> None:
        """An expired OAuth refresh token is an auth error."""
        assert classify_error(Exception("invalid_grant: Token expired")) is ErrorKind.AUTH

    def test_timeout_exception_is_transient(self) -> None:
        """Timeouts are transient."""
        assert classify_error(TimeoutError("read timeout")) is ErrorKind.TRANSIENT

    def test_connection_error_is_transient(self) -> None:
        """Connection errors are transient."""
        assert classify_error(ConnectionError("reset")) is ErrorKind.TRANSIENT

    def test_rate_limit_text(self) -> None:
        """Rate limit text without status is transient."""
        assert classify_error(Exception("Rate limit exceeded")) is ErrorKind.TRANSIENT

    def test_unknown_error_is_other(self) -> None:
        """Anything unrecognized is OTHER."""
        assert classify_error(KeyError("date")) is ErrorKind.OTHER

    def test_empty_message_is_other(self) -> None:
        """An exception with no message and no status is OTHER."""
        assert classify_error(Exception()) is ErrorKind.OTHER


# --------------------------------------------------------------------- #
# call_with_retries
# --------------------------------------------------------------------- #
class TestCallWithRetries:
    """Tests for call_with_retries."""

    def test_success_first_try(self) -> None:
        """A successful call returns its value without sleeping."""
        sleep = MagicMock()
        assert call_with_retries(lambda: 42, sleep=sleep) == 42
        sleep.assert_not_called()

    def test_retries_transient_then_succeeds(self) -> None:
        """A transient error is retried and the later success is returned."""
        func = MagicMock(side_effect=[ExtractionError("x", status_code=503), "ok"])
        sleep = MagicMock()
        logger = MagicMock()
        assert call_with_retries(func, attempts=3, sleep=sleep, logger=logger) == "ok"
        assert func.call_count == 2
        assert sleep.call_count == 1
        logger.warning.assert_called_once()

    def test_transient_exhausts_attempts(self) -> None:
        """After the last attempt the transient error is raised."""
        error = ExtractionError("x", status_code=429)
        func = MagicMock(side_effect=error)
        sleep = MagicMock()
        with pytest.raises(ExtractionError):
            call_with_retries(func, attempts=3, sleep=sleep)
        assert func.call_count == 3
        assert sleep.call_count == 2

    def test_auth_error_not_retried(self) -> None:
        """A 403 is raised on the first attempt: retrying does not help."""
        func = MagicMock(side_effect=ExtractionError("x", status_code=403))
        sleep = MagicMock()
        with pytest.raises(ExtractionError):
            call_with_retries(func, attempts=5, sleep=sleep)
        assert func.call_count == 1
        sleep.assert_not_called()

    def test_other_error_not_retried(self) -> None:
        """An unrecognized error is raised without retrying."""
        func = MagicMock(side_effect=KeyError("x"))
        with pytest.raises(KeyError):
            call_with_retries(func, attempts=3, sleep=MagicMock())
        assert func.call_count == 1

    def test_delay_doubles(self) -> None:
        """Each wait is at least double the base of the previous one."""
        func = MagicMock(side_effect=ExtractionError("x", status_code=500))
        sleep = MagicMock()
        with pytest.raises(ExtractionError):
            call_with_retries(func, attempts=3, base_delay=1.0, sleep=sleep)
        first, second = (call.args[0] for call in sleep.call_args_list)
        assert 1.0 <= first < 2.0
        assert 2.0 <= second < 3.0

    def test_single_attempt(self) -> None:
        """attempts=1 means no retry even for transient errors."""
        func = MagicMock(side_effect=ExtractionError("x", status_code=503))
        sleep = MagicMock()
        with pytest.raises(ExtractionError):
            call_with_retries(func, attempts=1, sleep=sleep)
        sleep.assert_not_called()

    @pytest.mark.parametrize("attempts", [0, -1])
    def test_invalid_attempts(self, attempts: int) -> None:
        """attempts below 1 is invalid."""
        with pytest.raises(ValueError):
            call_with_retries(lambda: 1, attempts=attempts)


# --------------------------------------------------------------------- #
# AccountResult
# --------------------------------------------------------------------- #
class TestAccountResult:
    """Tests for AccountResult."""

    def test_clean_account(self) -> None:
        """An account without failures maps to status ok and no error."""
        result = AccountResult(account="1", rows=10, tables_ok=2)
        assert result.has_errors is False
        assert result.guard_status == STATUS_OK
        assert result.guard_error is None

    def test_fail_records_kind_and_message(self) -> None:
        """fail() stores the error text and returns its kind."""
        result = AccountResult(account="1")
        kind = result.fail(ExtractionError("no permission", status_code=403))
        assert kind is ErrorKind.AUTH
        assert result.error_kind is ErrorKind.AUTH
        assert "no permission" in result.error
        assert result.guard_status == STATUS_ERROR

    def test_fail_truncates_long_message(self) -> None:
        """Error text is capped at 1000 characters."""
        result = AccountResult(account="1")
        result.fail(RuntimeError("x" * 5000))
        assert len(result.error) == 1000

    def test_failed_tables_count_as_error(self) -> None:
        """A failed table alone marks the account as error."""
        result = AccountResult(account="1", tables_ok=3, tables_failed=["t_20261111"])
        assert result.has_errors is True
        assert result.guard_status == STATUS_ERROR
        assert "1 tablas fallaron: t_20261111" in result.guard_error

    def test_guard_error_combines_both(self) -> None:
        """Extraction error and failed tables both appear in guard_error."""
        result = AccountResult(account="1", tables_failed=["a", "b"])
        result.fail(RuntimeError("boom"))
        assert "boom" in result.guard_error
        assert "2 tablas fallaron" in result.guard_error

    def test_summary_with_error(self) -> None:
        """The alert line names the account and the error kind."""
        result = AccountResult(account="480007877")
        result.fail(ExtractionError("denied", status_code=403))
        assert result.summary().startswith("480007877: auth")

    def test_summary_with_failed_tables(self) -> None:
        """The alert line counts failed and OK tables."""
        result = AccountResult(account="1", tables_ok=4, tables_failed=["a"])
        assert result.summary() == "1: 1 tablas fallaron (a), 4 OK"


# --------------------------------------------------------------------- #
# LoadReport
# --------------------------------------------------------------------- #
def _report() -> tuple[LoadReport, MagicMock]:
    logger = MagicMock()
    return LoadReport("tiktok", "p.d.tiktok", logger), logger


class TestLoadReport:
    """Tests for LoadReport.finish."""

    def test_no_accounts_is_no_data(self) -> None:
        """An empty report is a successful run without data."""
        report, logger = _report()
        assert report.finish() == ("200 - sin datos", 200)
        logger.critical.assert_not_called()

    def test_all_ok(self) -> None:
        """All accounts loaded: 200 without alert."""
        report, logger = _report()
        account = report.account("1")
        account.rows, account.tables_ok = 10, 2
        assert report.finish() == ("200", 200)
        logger.critical.assert_not_called()

    def test_all_accounts_without_rows(self) -> None:
        """Accounts with zero rows and no errors: 200 'sin datos'."""
        report, logger = _report()
        report.account("1")
        report.account("2")
        assert report.finish() == ("200 - sin datos", 200)
        logger.critical.assert_not_called()

    def test_partial_load(self) -> None:
        """One account loaded and one failed: alert and 200."""
        report, logger = _report()
        ok = report.account("1")
        ok.rows, ok.tables_ok = 10, 2
        report.account("2").fail(ExtractionError("denied", status_code=403))
        message, status = report.finish()
        assert status == 200
        assert "carga parcial" in message
        logger.critical.assert_called_once()
        assert "2: auth" in logger.critical.call_args.args[0]

    def test_partial_within_one_account(self) -> None:
        """Some tables of the only account failed: alert and 200."""
        report, logger = _report()
        account = report.account("1")
        account.rows, account.tables_ok = 5, 1
        account.tables_failed.append("tiktok_1_20261111")
        assert report.finish()[1] == 200
        logger.critical.assert_called_once()

    def test_nothing_loaded_with_errors(self) -> None:
        """No table loaded and an error happened: alert and 500."""
        report, logger = _report()
        report.account("1").fail(ExtractionError("denied", status_code=403))
        message, status = report.finish()
        assert status == 500
        assert "Carga fallida" in message
        logger.critical.assert_called_once()

    def test_all_writes_failed(self) -> None:
        """Extraction worked but every write failed: 500."""
        report, _ = _report()
        report.account("1").tables_failed.extend(["a", "b"])
        assert report.finish()[1] == 500

    def test_account_is_reused(self) -> None:
        """account() returns the same object for the same id."""
        report, _ = _report()
        assert report.account(123) is report.account("123")
        assert len(report.results) == 1

    def test_alert_lists_at_most_max_accounts(self) -> None:
        """The alert caps the account list and says how many were left out."""
        report, logger = _report()
        for index in range(MAX_ALERT_ACCOUNTS + 3):
            report.account(str(index)).fail(RuntimeError("boom"))
        report.finish()
        text = logger.critical.call_args.args[0]
        assert "y 3 cuentas más" in text
