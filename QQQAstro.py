#!/usr/bin/env python3
"""
QQQ Astro Pipeline v 2.5 – High-Precision Ephemeris
===================================================
Adds to every 5-min QQQ bar:
  • Moon sign, nakshatra, house (Chalit-Taurus)
  • Solar-arc progressed Lagna (daily snapshot)
  • Lunar-return flag (± 0.274 ° ≈ 30 min)
  • Rahu/Ketu ± 2 ° Lagna-hit flag
  • Cusp-cross flags for Sun, Moon, Mercury, Venus,
    Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
    (Planetary longitudes calculated using Skyfield and JPL DE441 ephemeris)
  • Hora (planetary hour lord) for New York time
  • Transitory Lagna (Ascendant) longitude, sign, nakshatra, house (Chalit-Taurus)
  • Lagna house change flag (5-min interval)
  • Moon house change flag (5-min interval)
  • any_cusp_cross = OR of all planet flags

Run:
  python QQQAstro.py --csv /path/to/QQQ.csv  --out enriched.csv
If --csv is omitted, script searches $QQQ_DATA_DIR (fallback ./mnt/data).
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Union # Union for type hint

import pandas as pd
import pytz
from skyfield.api import load as skyfield_load
# Add to imports
from skyfield.api import Topos # For geographic location
from skyfield import almanac   # For sunrise/sunset calculations
from skyfield.framelib import ecliptic_frame

# IMPORTANT: If skyfield is not installed, run: pip install skyfield

# ── natal constants ───────────────────────────────────────────────
AYANAMSA          = 24.0
NATAL_SUN_LONG    = 325.722     # Aq 25°43′
NATAL_MOON_LONG   = 238.399     # Sc 28°23′
NATAL_LAGNA_LONG  = 38.6225     # Ta 08°37′
J2000             = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo",
         "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]

NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashirsha","Ardra",
    "Punarvasu","Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni",
    "Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta",
    "Shatabhisha","Purva Bhadrapada","Uttara Bhadrapada","Revati"
]

# ── Hora Calculation Constants ────────────────────────────────────
NY_LATITUDE = 40.7128    # Approximate latitude for New York City
NY_LONGITUDE = -74.0060  # Approximate longitude for New York City

# Day Lords: datetime.weekday() Monday is 0 and Sunday is 6
DAY_LORD_MAP = {
    0: "Moon",    # Monday
    1: "Mars",    # Tuesday
    2: "Mercury", # Wednesday
    3: "Jupiter", # Thursday
    4: "Venus",   # Friday
    5: "Saturn",  # Saturday
    6: "Sun"      # Sunday
}

# Sequence of Hora rulers (Chaldean order, applied cyclically starting with Day Lord)
HORA_RULER_SEQUENCE = ["Sun", "Venus", "Mercury", "Moon", "Saturn", "Jupiter", "Mars"]

# ── Skyfield Ephemeris Setup ──────────────────────────────────────
# This will download de441.bsp (approx 18MB for 1550-2650 range)
# to Skyfield's default cache directory on first run if not already present.
EPH = skyfield_load('de441.bsp')

SF_EARTH = EPH['earth']
SF_PLANET_MAPPING = {
    'sun': EPH['sun'],
    'moon': EPH['moon'],
    'mercury': EPH['mercury'], # Mercury planet itself
    'venus': EPH['venus'],   # Venus planet itself
    'mars': EPH['mars barycenter'], # Mars system barycenter
    'jupiter': EPH['jupiter barycenter'],
    'saturn': EPH['saturn barycenter'],
    'uranus': EPH['uranus barycenter'],
    'neptune': EPH['neptune barycenter'],
    'pluto': EPH['pluto barycenter'],
}
SF_TIMESCALER = skyfield_load.timescale()

def _get_skyfield_longitude(planet_key: str, dt_utc: datetime) -> float:
    """Helper to get apparent geocentric ecliptic longitude from Skyfield."""
    skyfield_body = SF_PLANET_MAPPING[planet_key]
    t = SF_TIMESCALER.from_datetime(dt_utc)
    astrometric = SF_EARTH.at(t).observe(skyfield_body)
    # Apparent geocentric ecliptic longitude of date
    _lat, lon, _dist = astrometric.apparent().ecliptic_latlon(epoch=t)
    return lon.degrees

# ── High-Precision Planetary Longitudes ───────────────────────────
def sun_lon(dt_utc: datetime) -> float:     return _get_skyfield_longitude('sun', dt_utc)
def moon_lon(dt_utc: datetime) -> float:    return _get_skyfield_longitude('moon', dt_utc)
def mercury_lon(dt_utc: datetime) -> float: return _get_skyfield_longitude('mercury', dt_utc)
def venus_lon(dt_utc: datetime) -> float:   return _get_skyfield_longitude('venus', dt_utc)
def mars_lon(dt_utc: datetime) -> float:    return _get_skyfield_longitude('mars', dt_utc)
def jupiter_lon(dt_utc: datetime) -> float: return _get_skyfield_longitude('jupiter', dt_utc)
def saturn_lon(dt_utc: datetime) -> float:  return _get_skyfield_longitude('saturn', dt_utc)
def uranus_lon(dt_utc: datetime) -> float:  return _get_skyfield_longitude('uranus', dt_utc)
def neptune_lon(dt_utc: datetime) -> float: return _get_skyfield_longitude('neptune', dt_utc)
def pluto_lon(dt_utc: datetime) -> float:   return _get_skyfield_longitude('pluto', dt_utc)

# Mean Rahu (North Node) - calculation remains the same
def rahu_lon(dt):    return (125.04452 - 0.0529538083 *
                             (dt - J2000).total_seconds()/86400) % 360

PLANET_FUNCS = {
    'sun': sun_lon, 'moon': moon_lon, 'mercury': mercury_lon,
    'venus': venus_lon, 'mars': mars_lon, 'jupiter': jupiter_lon,
    'saturn': saturn_lon, 'uranus': uranus_lon,
    'neptune': neptune_lon, 'pluto': pluto_lon,
}

# ── Lagna (Ascendant) Calculation ─────────────────────────────────
def calculate_lagna_longitude(dt_utc: datetime, observer_topos: Topos) -> Union[float, type(pd.NA)]:
    """
    Calculates the sidereal longitude of the Ascendant (Lagna) for a given
    UTC datetime and observer geographic location (Topos object).
    Uses global SF_TIMESCALER, SF_EARTH, ecliptic_frame, AYANAMSA.
    Returns pd.NA if input dt_utc is NA.
    """
    if pd.isna(dt_utc):
        return pd.NA

    t = SF_TIMESCALER.from_datetime(dt_utc)
    observer_at_time = (SF_EARTH + observer_topos).at(t)

    # Determine the ICRS coordinates of the point on the eastern horizon
    # (Altitude 0 degrees, Azimuth 90 degrees)
    eastern_horizon_point_icrs = observer_at_time.from_altaz(alt_degrees=0, az_degrees=90)

    # Convert these ICRS coordinates to ecliptic coordinates of date.
    # ecliptic_frame is J2000.0 mean ecliptic.
    # The epoch=t argument ensures the coordinates are for the true ecliptic and equinox of date.
    _ecl_lat, ecl_lon_tropical_of_date, _dist = eastern_horizon_point_icrs.ecliptic_latlon(epoch=t)
    
    tropical_lon_degrees = ecl_lon_tropical_of_date.degrees
    sidereal_lon_degrees = (tropical_lon_degrees - AYANAMSA) % 360
    return sidereal_lon_degrees

# ── zodiac helpers ────────────────────────────────────────────────
def to_sidereal(lon: float) -> Union[float, type(pd.NA)]:
    """Convert a tropical longitude to sidereal using AYANAMSA."""
    return (lon - AYANAMSA) % 360 if pd.notna(lon) else pd.NA

def sign_from_lon(lon: float) -> Union[str, type(pd.NA)]: return SIGNS[int(lon // 30)] if pd.notna(lon) else pd.NA
def nakshatra_from_lon(lon: float) -> Union[str, type(pd.NA)]:
    """Return nakshatra name for a tropical longitude."""
    if pd.notna(lon):
        idx = int(((lon - AYANAMSA) % 360) // (360 / 27))
        return NAKSHATRAS[idx]
    return pd.NA

# New helper for sidereal longitudes (no ayanamsha offset)
def nakshatra_from_sidereal_lon(lon: float) -> Union[str, type(pd.NA)]:
    """Return nakshatra name for a sidereal longitude."""
    if pd.notna(lon):
        idx = int((lon % 360) // (360 / 27))
        return NAKSHATRAS[idx]
    return pd.NA
def house_from_lon(lon: float) -> Union[int, type(pd.NA)]: return int(((lon - NATAL_LAGNA_LONG) % 360) // 30) + 1 if pd.notna(lon) else pd.NA
def ang_diff(a: float, b: float) -> Union[float, type(pd.NA)]: return abs((a-b+180)%360 - 180) if pd.notna(a) and pd.notna(b) else pd.NA

# House cusps (0 = Lagna, 30 = 2nd, …)
CUSPS = [(NATAL_LAGNA_LONG + 30*i) % 360 for i in range(12)]
CUSP_TOL = 0.5  # deg
def near_cusp(lon: float) -> bool: return any(ang_diff(lon,c) <= CUSP_TOL for c in CUSPS) if pd.notna(lon) else False

# ── Hora Calculation Helpers ──────────────────────────────────────

def get_sunrise_utc_for_date(target_date_obj: datetime.date,
                             ny_topos_obj: Topos,
                             ny_timezone_obj: pytz.BaseTzInfo,
                             sunrise_cache_dict: Dict[datetime.date, Union[datetime, None]]) -> Union[datetime, None]:
    """
    Calculates and caches the UTC sunrise time for a given date in New York.
    Uses global EPH and SF_TIMESCALER.
    """
    if target_date_obj in sunrise_cache_dict:
        return sunrise_cache_dict[target_date_obj]

    # Define the search window for sunrise on target_date_obj (NY time), converted to UTC for Skyfield
    dt_start_ny_naive = datetime.combine(target_date_obj, datetime.min.time())
    dt_start_ny_aware = ny_timezone_obj.localize(dt_start_ny_naive)
    
    # Search up to the start of the next NY day to ensure we capture the sunrise for target_date_obj
    dt_end_ny_naive = datetime.combine(target_date_obj + timedelta(days=1), datetime.min.time())
    dt_end_ny_aware = ny_timezone_obj.localize(dt_end_ny_naive)

    t0 = SF_TIMESCALER.from_datetime(dt_start_ny_aware.astimezone(timezone.utc))
    t1 = SF_TIMESCALER.from_datetime(dt_end_ny_aware.astimezone(timezone.utc))

    f = almanac.sunrise_sunset(EPH, ny_topos_obj)
    times_utc_skyfield, events = almanac.find_discrete(t0, t1, f)

    sunrise_dt_utc_val = None
    for t_sf, event_is_sunrise in zip(times_utc_skyfield, events):
        if event_is_sunrise:  # event == 1 means sunrise
            potential_sunrise_utc = t_sf.utc_datetime()
            # Ensure this sunrise, when viewed in NY time, falls on the target_date_obj
            potential_sunrise_ny = potential_sunrise_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)
            if potential_sunrise_ny.date() == target_date_obj:
                sunrise_dt_utc_val = potential_sunrise_utc
                break  # Found the correct sunrise for the target NY date

    if sunrise_dt_utc_val is None:
        print(f"⚠️  Warning: Could not determine sunrise for {target_date_obj} in New York.")
    
    sunrise_cache_dict[target_date_obj] = sunrise_dt_utc_val
    return sunrise_dt_utc_val

def get_sunset_utc_for_date(target_date_obj: datetime.date,
                            ny_topos_obj: Topos,
                            ny_timezone_obj: pytz.BaseTzInfo,
                            sunset_cache_dict: Dict[datetime.date, Union[datetime, None]]) -> Union[datetime, None]:
    """
    Calculates and caches the UTC sunset time for a given date in New York.
    Uses global EPH and SF_TIMESCALER.
    """
    if target_date_obj in sunset_cache_dict:
        return sunset_cache_dict[target_date_obj]

    dt_start_ny_naive = datetime.combine(target_date_obj, datetime.min.time())
    dt_start_ny_aware = ny_timezone_obj.localize(dt_start_ny_naive)
    
    dt_end_ny_naive = datetime.combine(target_date_obj + timedelta(days=1), datetime.min.time())
    dt_end_ny_aware = ny_timezone_obj.localize(dt_end_ny_naive)

    t0 = SF_TIMESCALER.from_datetime(dt_start_ny_aware.astimezone(timezone.utc))
    t1 = SF_TIMESCALER.from_datetime(dt_end_ny_aware.astimezone(timezone.utc))

    f = almanac.sunrise_sunset(EPH, ny_topos_obj) # f gives both sunrise (1) and sunset (0)
    times_utc_skyfield, events = almanac.find_discrete(t0, t1, f)

    sunset_dt_utc_val = None
    for t_sf, event_code in zip(times_utc_skyfield, events):
        if event_code == 0:  # event == 0 means sunset
            potential_sunset_utc = t_sf.utc_datetime()
            potential_sunset_ny = potential_sunset_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)
            if potential_sunset_ny.date() == target_date_obj:
                sunset_dt_utc_val = potential_sunset_utc
                break 
    sunset_cache_dict[target_date_obj] = sunset_dt_utc_val
    return sunset_dt_utc_val

def calculate_hora(timestamp_ny_aware: pd.Timestamp,
                   ny_topos_obj: Topos, # Not directly used if all needed sunrise/sunset times are pre-cached and found
                   ny_timezone_obj: pytz.BaseTzInfo,
                   sunrise_cache_utc: Dict[datetime.date, Union[datetime, None]],
                   sunset_cache_utc: Dict[datetime.date, Union[datetime, None]] # Added sunset_cache_utc
                  ) -> Union[str, None]:
    """
    Calculates the Hora lord for a given New York timestamp.
    Uses global DAY_LORD_MAP, HORA_RULER_SEQUENCE.
    This version calculates variable-length Horas based on actual day/night durations.
    """
    current_timestamp_date_ny = timestamp_ny_aware.date()

    # Determine the effective date for the Hora cycle
    # Note: The parameter was sunrise_cache_dict, renamed to sunrise_cache_utc for clarity
    sunrise_on_timestamp_date_utc = sunrise_cache_utc.get(current_timestamp_date_ny)

    effective_date_ny: Union[datetime.date, None] = None
    cycle_sunrise_ny: Union[datetime, None] = None # Sunrise that starts the current Hora cycle

    if sunrise_on_timestamp_date_utc:
        sunrise_on_timestamp_date_ny = sunrise_on_timestamp_date_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)
        if timestamp_ny_aware >= sunrise_on_timestamp_date_ny:
            effective_date_ny = current_timestamp_date_ny
            cycle_sunrise_ny = sunrise_on_timestamp_date_ny
        else:
            # Timestamp is before the sunrise of its own date, so it belongs to the previous day's cycle
            effective_date_ny = current_timestamp_date_ny - timedelta(days=1)
            sunrise_previous_day_utc = sunrise_cache_utc.get(effective_date_ny)
            if sunrise_previous_day_utc:
                cycle_sunrise_ny = sunrise_previous_day_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)
    elif current_timestamp_date_ny > min(sunrise_cache_utc.keys(), default=current_timestamp_date_ny): # Check if it's not the earliest date
        # Sunrise for current timestamp's date not found, try previous day if timestamp might fall there
        effective_date_ny = current_timestamp_date_ny - timedelta(days=1)
        sunrise_previous_day_utc = sunrise_cache_utc.get(effective_date_ny)
        if sunrise_previous_day_utc:
             cycle_sunrise_ny = sunrise_previous_day_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)
             # We must ensure timestamp_ny_aware is actually after this cycle_sunrise_ny
             if not (cycle_sunrise_ny and timestamp_ny_aware >= cycle_sunrise_ny):
                 cycle_sunrise_ny = None # Invalid scenario

    if not effective_date_ny or not cycle_sunrise_ny:
        print(f"⚠️ Warning: Could not determine effective sunrise/date for Hora calculation at {timestamp_ny_aware}.")
        return None

    # Get sunset of the effective day
    sunset_effective_day_utc = sunset_cache_utc.get(effective_date_ny) # Use the passed sunset_cache_utc
    if not sunset_effective_day_utc:
        print(f"⚠️ Warning: Sunset for effective date {effective_date_ny} not found for {timestamp_ny_aware}.")
        return None
    cycle_sunset_ny = sunset_effective_day_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)

    # Get sunrise of the day *after* the effective day (for night duration)
    date_after_effective_ny = effective_date_ny + timedelta(days=1)
    sunrise_next_day_utc = sunrise_cache_utc.get(date_after_effective_ny)
    if not sunrise_next_day_utc:
        print(f"⚠️ Warning: Sunrise for {date_after_effective_ny} (after effective) not found for {timestamp_ny_aware}.")
        return None
    cycle_next_sunrise_ny = sunrise_next_day_utc.replace(tzinfo=timezone.utc).astimezone(ny_timezone_obj)

    if not (cycle_sunrise_ny <= timestamp_ny_aware < cycle_next_sunrise_ny):
        print(f"⚠️ Timestamp {timestamp_ny_aware} is outside the calculated cycle [{cycle_sunrise_ny}, {cycle_next_sunrise_ny}).")
        return None

    day_duration_seconds = (cycle_sunset_ny - cycle_sunrise_ny).total_seconds()
    night_duration_seconds = (cycle_next_sunrise_ny - cycle_sunset_ny).total_seconds()

    if day_duration_seconds <= 0 and night_duration_seconds <= 0: # Should not happen in NY
        print(f"⚠️ Both day and night durations are zero or negative for {effective_date_ny}.")
        return None

    day_lord = DAY_LORD_MAP.get(effective_date_ny.weekday())
    if day_lord is None: return None # Should not happen
    try:
        start_lord_idx = HORA_RULER_SEQUENCE.index(day_lord)
    except ValueError: return None

    overall_hora_index = -1

    if timestamp_ny_aware < cycle_sunset_ny: # Day Hora
        if day_duration_seconds > 0:
            day_hora_len_sec = day_duration_seconds / 12.0
            time_since_sunrise = (timestamp_ny_aware - cycle_sunrise_ny).total_seconds()
            hora_idx_in_period = int(time_since_sunrise / day_hora_len_sec)
            overall_hora_index = min(hora_idx_in_period, 11) # Clamp to 0-11
        # If day_duration_seconds is 0, timestamp must be at sunrise=sunset, falls to night.
    else: # Night Hora
        if night_duration_seconds > 0:
            night_hora_len_sec = night_duration_seconds / 12.0
            time_since_sunset = (timestamp_ny_aware - cycle_sunset_ny).total_seconds()
            hora_idx_in_period = int(time_since_sunset / night_hora_len_sec)
            overall_hora_index = 12 + min(hora_idx_in_period, 11) # Clamp to 0-11 for night part

    if overall_hora_index == -1 or overall_hora_index >= 24:
        print(f"⚠️ Error calculating Hora index for {timestamp_ny_aware}. Index: {overall_hora_index}")
        return None

    final_lord_idx = (start_lord_idx + overall_hora_index) % len(HORA_RULER_SEQUENCE)
    return HORA_RULER_SEQUENCE[final_lord_idx]


# ── robust timestamp parser ───────────────────────────────────────
def parse_ts(series_input: pd.Series, pytz_ny: pytz.BaseTzInfo) -> pd.Series:
    series_str = series_input.astype(str).str.strip()

    # Initialize final result series (object dtype to hold tz-aware timestamps before final coercion)
    s_final_ny = pd.Series([pd.NaT] * len(series_str), index=series_str.index, dtype=object)

    # --- Attempt 1: Strict format, assumed to be NY naive time ---
    s_attempt1_naive = pd.to_datetime(series_str,
                       format='%m/%d/%y %H:%M', errors='coerce')
    
    mask_attempt1_success = s_attempt1_naive.notna()
    if mask_attempt1_success.any():
        # Localize these naive NY times to NY timezone
        localized_vals = s_attempt1_naive[mask_attempt1_success].dt.tz_localize(
            pytz_ny, nonexistent='shift_forward' # Use 'shift_forward' for DST "spring forward" non-existent times
        )
        s_final_ny.loc[mask_attempt1_success] = localized_vals

    # --- Attempt 2: For those that failed Attempt 1 (e.g. due to offsets or different format) ---
    mask_needs_inference = ~mask_attempt1_success
    if mask_needs_inference.any():
        strings_for_inference = series_str[mask_needs_inference]
        
        # Parse with inference. utc=True converts offset strings to UTC; naive strings remain naive.
        s_inferred = pd.to_datetime(
            strings_for_inference, 
            errors='coerce', 
            utc=True 
        )

        # Process successfully parsed inferred values (which are on the subset of rows)
        for idx, ts_val in s_inferred.items(): # Iterate over the subset defined by strings_for_inference.index
            if pd.isna(ts_val):
                # This is where parsing with utc=True failed for this specific string
                # that didn't match the initial format '%m/%d/%y %H:%M'.
                original_string_that_failed = series_str.loc[idx] # Get original string from series_str using the index
                # print(f"[DEBUG parse_ts] Inference with utc=True failed for original string: '{original_string_that_failed}'")
                continue # s_final_ny already has NaT for this idx if not set by format match

            if ts_val.tzinfo is not None: # It's timezone-aware (parsed as UTC)
                s_final_ny.loc[idx] = ts_val.tz_convert(pytz_ny)
            else: # It's naive (parsed from a string without an offset by inference)
                  # Assume these naive times are NY local time.
                s_final_ny.loc[idx] = pytz_ny.localize(ts_val, nonexistent='shift_forward')

    # Convert s_final_ny (object dtype) to a proper timezone-aware datetime Series
    s_final_ny = pd.to_datetime(s_final_ny, errors='coerce')
    
    # Ensure the final series has the target timezone, even if all values are NaT
    if s_final_ny.dt.tz is None and pytz_ny is not None:
        # If all were NaT or became NaT, tz_localize might fail if Series is empty or all NaT.
        # A robust way is to ensure the dtype is correct.
        try:
            s_final_ny = s_final_ny.dt.tz_localize(pytz_ny, nonexistent='shift_forward', ambiguous='NaT')
        except TypeError: # Handles cases like all NaT series
            if s_final_ny.isna().all():
                 s_final_ny = pd.Series(pd.NaT, index=s_final_ny.index).dt.tz_localize(pytz_ny)
            else:
                raise # Re-raise if it's another issue
    elif s_final_ny.dt.tz is not None and str(s_final_ny.dt.tz) != str(pytz_ny.zone):
        s_final_ny = s_final_ny.dt.tz_convert(pytz_ny)

    if s_final_ny.notna().sum() == 0 and not series_input.empty : # Check if series_input was not empty
        potential_timestamps = series_input.astype(str).str.strip().str.contains(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}')
        if potential_timestamps.any() and series_input[potential_timestamps].str.strip().any():
             # Only raise if there was something that looked like a timestamp
             # print("Warning: Timestamp-like data found but could not be definitively parsed into datetime objects after all attempts.")
             # Avoid raising error, let downstream handle NaTs.
             pass
            
    return s_final_ny

# ── pipeline ──────────────────────────────────────────────────────
def enrich(csv_path: Path, out_path: Path):
    print(f"ℹ️  Processing input CSV: {csv_path}")
    ny = pytz.timezone('America/New_York')
    df = pd.read_csv(csv_path)
    
    # Keep a copy of original timestamp strings for diagnostics
    original_timestamps_series = df['timestamp'].copy() 

    df['timestamp'] = parse_ts(df['timestamp'], ny) # Attempt to parse
    
    bad_rows_mask = df['timestamp'].isna()
    num_bad_rows = bad_rows_mask.sum()

    if num_bad_rows > 0:
        print(f"⚠︎  {num_bad_rows} timestamp rows will be dropped due to parsing errors.")
        # print("Original timestamp values from your input CSV that could not be parsed:")
        # print(original_timestamps_series[bad_rows_mask]) # Print the problematic original strings
        df = df.dropna(subset=['timestamp'])

    if df.empty: # Check if all rows were dropped
        print(f"🛑 Error: All {len(original_timestamps_series)} rows were dropped because their timestamps could not be parsed.")
        print("Please check the format of the 'timestamp' column in your input CSV:", csv_path)
        return

    df['utc'] = df['timestamp'].dt.tz_convert('UTC')

    # Hora Calculation Setup
    ny_topos = Topos(latitude_degrees=NY_LATITUDE, longitude_degrees=NY_LONGITUDE)
    sunrise_cache_utc: Dict[datetime.date, Union[datetime, None]] = {} 
    sunset_cache_utc: Dict[datetime.date, Union[datetime, None]] = {} # Cache for sunset times

    # Pre-populate sunrise and sunset cache for all relevant dates
    if not df.empty:
        # df['timestamp'] is already NY-localized and NaTs are dropped
        all_unique_dates_in_df_timestamps = df['timestamp'].dt.date.unique()
        
        required_dates_for_astro_events = set()
        for date_obj_val in all_unique_dates_in_df_timestamps:
            required_dates_for_astro_events.add(date_obj_val) # For current day's events
            required_dates_for_astro_events.add(date_obj_val - timedelta(days=1)) # For previous day's events
            required_dates_for_astro_events.add(date_obj_val + timedelta(days=1)) # For next day's events (e.g. next sunrise for night duration)

        print(f"ℹ️  Pre-calculating sunrises/sunsets for {len(required_dates_for_astro_events)} unique NY dates...")
        for date_to_calc in sorted(list(required_dates_for_astro_events)): 
            # Ensure get_sunrise/sunset functions handle missing events gracefully (e.g. polar night/day)
            # by returning None, which the cache will store.
            if get_sunrise_utc_for_date(date_to_calc, ny_topos, ny, sunrise_cache_utc) is None:
                 print(f"⚠️  Warning: Could not determine sunrise for {date_to_calc} in New York during pre-calculation.")
            if get_sunset_utc_for_date(date_to_calc, ny_topos, ny, sunset_cache_utc) is None:
                 print(f"⚠️  Warning: Could not determine sunset for {date_to_calc} in New York during pre-calculation.")
        print("✓ Sunrises/Sunsets pre-calculated.")

    # Pass both caches to the new calculate_hora function
    # Note: The calculate_hora function signature was updated in thought process to accept sunset_cache_utc
    # The lambda needs to pass ny_topos, ny_timezone, sunrise_cache, sunset_cache
    df['hora'] = df.apply(
        lambda row: calculate_hora(
            row['timestamp'], 
            ny_topos, # ny_topos_obj argument in calculate_hora
            ny,       # ny_timezone_obj argument in calculate_hora
            sunrise_cache_utc, 
            sunset_cache_utc # New argument for sunset cache
        ), 
        axis=1
    )

    # Transitory Lagna (Ascendant) calculations
    print("ℹ️  Calculating transitory Lagna, sign, nakshatra, and house...")
    df['lagna_long'] = df['utc'].apply(
        lambda dt_val: calculate_lagna_longitude(dt_val, ny_topos)
    )
    df['lagna_long'] = pd.to_numeric(df['lagna_long'], errors='coerce')

    # Longitude already sidereal – just lookup the sign directly
    df['lagna_sign'] = df['lagna_long'].apply(sign_from_lon)
    df['lagna_nakshatra'] = df['lagna_long'].apply(nakshatra_from_sidereal_lon)
    df['lagna_house'] = df['lagna_long'].apply(house_from_lon).astype(pd.Int64Dtype())

    # Moon sign / nakshatra / house
    df['moon_long'] = df['utc'].apply(moon_lon)
    df['moon_sign'] = df['moon_long'].apply(lambda L: sign_from_lon(to_sidereal(L)))
    df['nakshatra'] = df['moon_long'].apply(nakshatra_from_lon)
    df['moon_house'] = df['moon_long'].apply(
        lambda L: house_from_lon(to_sidereal(L))
    ).astype(pd.Int64Dtype())

    # Lagna and Moon House Change Flags (5-min interval)
    print("ℹ️  Calculating Lagna and Moon house change flags...")
    df['lagna_house_change_5m'] = (df['lagna_house'] != df['lagna_house'].shift(1)).fillna(False).astype(bool)
    df['moon_house_change_5m'] = (df['moon_house'] != df['moon_house'].shift(1)).fillna(False).astype(bool)
    
    # Ensure 'hora_lord' (already calculated) and 'mercury_cusp_cross' (will be calculated later) are present
    # The user request implies these are key outputs alongside the new ones.
    
    # Solar-arc progressed Lagna
    daily_arc = {}
    for d_utc_date_val in df['utc'].dt.date.unique(): # Iterate over unique UTC dates
        # Use a consistent time for daily arc calculation, e.g., noon UTC on that date
        dt_for_arc = datetime(d_utc_date_val.year, d_utc_date_val.month, d_utc_date_val.day, 12, 0, 0, tzinfo=timezone.utc)
        daily_arc[str(d_utc_date_val)] = (sun_lon(dt_for_arc) - NATAL_SUN_LONG) % 360
    df['solar_arc'] = df['utc'].dt.date.astype(str).map(daily_arc)
    df['prog_lagna_long'] = (NATAL_LAGNA_LONG + df['solar_arc']) % 360
    df['prog_lagna_house'] = df.apply(
        lambda r: house_from_lon(((r['moon_long'] - r['solar_arc']) - AYANAMSA) % 360),
        axis=1
    )

    # Lunar return flag
    df['lunar_return'] = df['moon_long'].apply(
        lambda L: ang_diff(L, NATAL_MOON_LONG) <= 0.274)

    # Node–Lagna hit
    df['rahu_long'] = df['utc'].apply(rahu_lon)
    df['ketu_long'] = (df['rahu_long'] + 180) % 360
    df['node_hits_lagna'] = df.apply(
        lambda r: ang_diff(r['rahu_long'],NATAL_LAGNA_LONG)<=2 or
                  ang_diff(r['ketu_long'],NATAL_LAGNA_LONG)<=2, axis=1)

    # Planet cusp-cross flags
    for name,func in PLANET_FUNCS.items():
        col_long = f"{name}_long"
        col_flag = f"{name}_cusp_cross"
        df[col_long] = df['utc'].apply(func)

        df[col_flag] = df[col_long].apply(lambda L: near_cusp(to_sidereal(L)))

    cusp_cols = [c for c in df.columns if c.endswith('_cusp_cross')]
    df['any_cusp_cross'] = df[cusp_cols].any(axis=1)

    df.to_csv(out_path, index=False)
    print(f"✓ Enriched file saved → {out_path}")

# ── CLI glue ──────────────────────────────────────────────────────
def locate_csv():
    root = Path(os.getenv('QQQ_DATA_DIR', './mnt/data'))
    all_csvs = list(root.glob('**/*.csv'))

    if not all_csvs:
        raise FileNotFoundError(f'No CSV files found in {root}')

    # Priority 1: Exact match for QQQ.csv (case-insensitive)
    exact_matches = [p for p in all_csvs if p.name.lower() == 'qqq.csv']
    if exact_matches:
        exact_matches.sort() # Ensure consistent pick if multiple (e.g. in subdirs)
        return exact_matches[0]

    # Priority 2: Files containing 'qqq' but not '_enriched'
    qqq_originals = [p for p in all_csvs if 'qqq' in p.name.lower() and '_enriched' not in p.name.lower()]
    if qqq_originals:
        qqq_originals.sort()
        return qqq_originals[0]

    # Priority 3: Fallback to any file containing 'qqq' if no "original" is found
    all_qqq_files = [p for p in all_csvs if 'qqq' in p.name.lower()]
    if all_qqq_files:
        all_qqq_files.sort()
        selected_file = all_qqq_files[0]
        if '_enriched' in selected_file.name.lower():
            print(f"⚠️  Warning: No primary QQQ data file found. Using potentially enriched file: {selected_file}")
        return selected_file

    raise FileNotFoundError(f'No QQQ CSV files found in {root} matching criteria.')

if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser(description='Enrich QQQ 5-min bars with astro data (all planets)')
    ap.add_argument('--csv', type=Path, help='Path to raw QQQ CSV')
    ap.add_argument('--out', type=Path, help='Output CSV (default *_enriched.csv)')
    args = ap.parse_args()

    try:
        csv_path = args.csv or locate_csv()
    except FileNotFoundError as e:
        sys.exit(str(e))

    out_path = args.out or csv_path.with_name(csv_path.stem+'_enriched.csv')
    enrich(csv_path, out_path)
