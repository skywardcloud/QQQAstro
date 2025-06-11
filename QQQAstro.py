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
  • any_cusp_cross = OR of all planet flags

Run:
  python QQQAstro.py --csv /path/to/QQQ.csv  --out enriched.csv
If --csv is omitted, script searches $QQQ_DATA_DIR (fallback ./mnt/data).
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import pytz
from skyfield.api import load as skyfield_load
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

# ── zodiac helpers ────────────────────────────────────────────────
def sign_from_lon(lon):       return SIGNS[int(lon // 30)]
def nakshatra_from_lon(lon):  return NAKSHATRAS[int(((lon-AYANAMSA)%360)//(360/27))]
def house_from_lon(lon):      return int(((lon - NATAL_LAGNA_LONG) % 360) // 30) + 1
def ang_diff(a,b):            return abs((a-b+180)%360 - 180)

# House cusps (0 = Lagna, 30 = 2nd, …)
CUSPS = [(NATAL_LAGNA_LONG + 30*i) % 360 for i in range(12)]
CUSP_TOL = 0.5  # deg

def near_cusp(lon): return any(ang_diff(lon,c) <= CUSP_TOL for c in CUSPS)

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
            pytz_ny, nonexistent='shift_forward'
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
                print(f"[DEBUG parse_ts] Inference with utc=True failed for original string: '{original_string_that_failed}'")
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
        s_final_ny = s_final_ny.dt.tz_localize(pytz_ny, nonexistent='shift_forward', ambiguous='NaT')
    elif s_final_ny.dt.tz is not None and str(s_final_ny.dt.tz) != str(pytz_ny.zone):
        s_final_ny = s_final_ny.dt.tz_convert(pytz_ny)

    if s_final_ny.notna().sum() == 0:
        potential_timestamps = series_input.astype(str).str.strip().str.contains(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}')
        if potential_timestamps.any() and series_input[potential_timestamps].str.strip().any():
             raise ValueError("Timestamp-like data found but could not be definitively parsed into datetime objects after all attempts.")
            
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
        print("Original timestamp values from your input CSV that could not be parsed:")
        print(original_timestamps_series[bad_rows_mask]) # Print the problematic original strings
        df = df.dropna(subset=['timestamp'])

    if df.empty: # Check if all rows were dropped
        print(f"🛑 Error: All {len(original_timestamps_series)} rows were dropped because their timestamps could not be parsed.")
        print("Please check the format of the 'timestamp' column in your input CSV:", csv_path)
        return

    df['utc'] = df['timestamp'].dt.tz_convert('UTC')

    # Moon sign / nakshatra / house
    df['moon_long'] = df['utc'].apply(moon_lon)
    df['moon_sign'] = df['moon_long'].apply(sign_from_lon)
    df['nakshatra'] = df['moon_long'].apply(nakshatra_from_lon)
    df['moon_house'] = df['moon_long'].apply(house_from_lon)

    # Solar-arc progressed Lagna
    daily_arc = {}
    for d in df['utc'].dt.date.unique():
        dt = datetime(d.year,d.month,d.day,tzinfo=timezone.utc)
        daily_arc[str(d)] = (sun_lon(dt) - NATAL_SUN_LONG) % 360
    df['solar_arc'] = df['utc'].dt.date.astype(str).map(daily_arc)
    df['prog_lagna_long'] = (NATAL_LAGNA_LONG + df['solar_arc']) % 360
    df['prog_lagna_house']= df.apply(
        lambda r: house_from_lon(r['moon_long']-r['solar_arc']), axis=1)

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
        df[col_flag] = df[col_long].apply(near_cusp)

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
