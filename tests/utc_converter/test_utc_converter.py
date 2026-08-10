from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from d2b_data.utc_converter import UTCConverter


ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"


def test_regions_map_contains_expected_countries():
    """The region map covers the countries the framework operates in."""
    assert set(UTCConverter.REGIONS) == {
        "chile", "brasil", "argentina", "peru", "colombia", "uruguay", "mexico",
    }
    assert UTCConverter.REGIONS["chile"] == "America/Santiago"


# --------------------------------------------------------------------- #
# get_now
# --------------------------------------------------------------------- #
def test_get_now_returns_iso_utc_string():
    """get_now returns a parseable UTC timestamp close to the real now."""
    result = UTCConverter.get_now()
    parsed = datetime.strptime(result, ISO_UTC).replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 60


def test_get_now_is_region_independent():
    """The same instant in UTC does not depend on the region asked for."""
    chile = datetime.strptime(UTCConverter.get_now("chile"), ISO_UTC)
    mexico = datetime.strptime(UTCConverter.get_now("mexico"), ISO_UTC)
    assert abs((chile - mexico).total_seconds()) < 5


def test_get_now_is_case_insensitive():
    """Region names are normalised to lowercase."""
    assert UTCConverter.get_now("CHILE").endswith("Z")


def test_get_now_falls_back_to_santiago_for_unknown_region():
    """An unknown region does not raise; it defaults to Santiago."""
    assert UTCConverter.get_now("narnia").endswith("Z")


# --------------------------------------------------------------------- #
# get_yesterday
# --------------------------------------------------------------------- #
def test_get_yesterday_returns_previous_local_day():
    """get_yesterday returns the day before in the region's own calendar."""
    result = UTCConverter.get_yesterday("chile")
    expected = datetime.now(ZoneInfo("America/Santiago")).date() - timedelta(days=1)
    assert result == expected.strftime("%Y-%m-%d")


def test_get_yesterday_has_no_time_component():
    """The output is a plain date, not a timestamp."""
    result = UTCConverter.get_yesterday()
    assert len(result) == 10
    assert "T" not in result


def test_get_yesterday_unknown_region_uses_santiago():
    """Unknown regions fall back instead of raising."""
    assert UTCConverter.get_yesterday("atlantis") == UTCConverter.get_yesterday("chile")


# --------------------------------------------------------------------- #
# convert — fechas simples
# --------------------------------------------------------------------- #
def test_convert_simple_date_from_chile():
    """Midnight in Santiago (UTC-3 in January) becomes 03:00 UTC."""
    assert UTCConverter.convert("2025-01-15", region="chile") == "2025-01-15T03:00:00Z"


def test_convert_simple_date_from_mexico():
    """Midnight in Mexico City (UTC-6 in January) becomes 06:00 UTC."""
    assert UTCConverter.convert("2025-01-15", region="mexico") == "2025-01-15T06:00:00Z"


def test_convert_respects_dst_offsets():
    """Santiago shifts between UTC-3 and UTC-4, and the conversion follows."""
    summer = UTCConverter.convert("2025-01-15", region="chile")   # UTC-3
    winter = UTCConverter.convert("2025-07-15", region="chile")   # UTC-4
    assert summer == "2025-01-15T03:00:00Z"
    assert winter == "2025-07-15T04:00:00Z"


# --------------------------------------------------------------------- #
# convert — modos start / end
# --------------------------------------------------------------------- #
def test_convert_mode_start_forces_local_midnight():
    """mode='start' pins the local time to 00:00:00."""
    assert UTCConverter.convert("2025-01-15", region="chile", mode="start") == "2025-01-15T03:00:00Z"


def test_convert_mode_end_forces_local_end_of_day():
    """mode='end' pins the local time to 23:59:59."""
    assert UTCConverter.convert("2025-01-15", region="chile", mode="end") == "2025-01-16T02:59:59Z"


def test_convert_mode_start_overrides_time_in_iso_input():
    """An explicit time in the input is discarded when a mode is given."""
    result = UTCConverter.convert("2025-01-15T14:35:15-03:00", region="chile", mode="start")
    assert result == "2025-01-15T03:00:00Z"


def test_convert_unknown_mode_is_ignored():
    """An unrecognised mode leaves the parsed time untouched."""
    assert UTCConverter.convert("2025-01-15", region="chile", mode="middle") == "2025-01-15T03:00:00Z"


# --------------------------------------------------------------------- #
# convert — formato ISO
# --------------------------------------------------------------------- #
def test_convert_iso_with_offset():
    """An ISO string with an offset is converted to UTC."""
    assert UTCConverter.convert("2026-02-02T14:35:15-03:00") == "2026-02-02T17:35:15Z"


def test_convert_iso_with_zero_offset_is_unchanged():
    """A timestamp already in UTC round-trips."""
    assert UTCConverter.convert("2026-02-02T14:35:15+00:00") == "2026-02-02T14:35:15Z"


# --------------------------------------------------------------------- #
# convert — errores
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_date", ["15-01-2025", "not-a-date", "2025-13-45"])
def test_convert_returns_error_message_on_bad_format(bad_date):
    """Malformed dates return a descriptive string instead of raising."""
    result = UTCConverter.convert(bad_date)
    assert result.startswith(f"Error de formato en fecha '{bad_date}'")


def test_convert_unknown_region_falls_back_to_santiago():
    """Unknown regions use Santiago rather than failing."""
    assert UTCConverter.convert("2025-01-15", region="narnia") == UTCConverter.convert("2025-01-15", region="chile")
