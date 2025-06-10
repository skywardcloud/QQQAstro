#!/usr/bin/env python3
"""
QQQ Astro Pipeline v 2.4 – all-planet cusp-cross flags
=====================================================

Adds to every 5-min QQQ bar:
  • Moon sign, nakshatra, house (Chalit-Taurus)
  • Solar-arc progressed Lagna (daily snapshot)
  • Lunar-return flag (± 0.274 ° ≈ 30 min)
  • Rahu/Ketu ± 2 ° Lagna-hit flag
  • Cusp-cross flags for Sun, Moon, Mercury, Venus,
    Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
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

# ── mean longitudes (1-degree accuracy) ───────────────────────────
def mean_longitude(base_deg: float, period_days: float, dt_utc: datetime) -> float:
    days = (dt_utc - J2000).total_seconds() / 86400
    return (base_deg + 360/period_days * days) % 360

def moon_lon(dt):    return mean_longitude(218.316, 27.321582, dt)
def sun_lon(dt):     return mean_longitude(280.460, 365.2422, dt)
def mercury_lon(dt): return mean_longitude(60.750, 87.969, dt)
def venus_lon(dt):   return mean_longitude(85.380, 224.701, dt)
def mars_lon(dt):    return mean_longitude(293.527, 686.98, dt)
def jupiter_lon(dt): return mean_longitude(238.049, 4332.59, dt)
def saturn_lon(dt):  return mean_longitude(266.564, 10759.22, dt)
def uranus_lon(dt):  return mean_longitude(314.055, 30685.4, dt)
def neptune_lon(dt): return mean_longitude(304.348, 60189.0, dt)
def pluto_lon(dt):   return mean_longitude(238.929, 90560.0, dt)
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
def parse_ts(series,pytz_ny):
    s = pd.to_datetime(series.astype(str).str.strip(),
                       format='%m/%d/%y %H:%M', errors='coerce')
    mask = s.isna()
    if mask.any():
        s.loc[mask] = pd.to_datetime(series[mask], errors='coerce',
                                     infer_datetime_format=True)
    if s.notna().sum()==0:
        raise ValueError("Could not parse ANY timestamps.")

    # Ensure 's' is truly datetimelike before using .dt accessor.
    # The previous steps should ideally result in a datetime64[ns] Series if parsing was successful.
    # However, if 's' ended up as an object Series with mixed types (e.g. some datetimes, some strings),
    # this explicit coercion will standardize it to datetime64[ns], turning unparseable items to NaT.
    s = pd.to_datetime(s, errors='coerce')

    # After final coercion, re-check if any valid datetimes remain.
    # This handles cases where initial s.notna().sum() > 0 was due to non-datetime objects
    # (e.g. strings) that were not successfully converted to datetimes.
    if s.notna().sum() == 0:
        raise ValueError("Timestamp-like data found but could not be definitively parsed into datetime objects after final coercion.")

    # Check if the Series is already timezone-aware
    if s.dt.tz is not None:
        # If already aware, convert to the target New York timezone
        return s.dt.tz_convert(pytz_ny)
    else:
        # If naive, localize to the New York timezone
        return s.dt.tz_localize(pytz_ny, nonexistent='shift_forward')

# ── pipeline ──────────────────────────────────────────────────────
def enrich(csv_path: Path, out_path: Path):
    ny = pytz.timezone('America/New_York')
    df = pd.read_csv(csv_path)

    df['timestamp'] = parse_ts(df['timestamp'], ny)
    bad = df['timestamp'].isna().sum()
    if bad:
        print(f"⚠︎  {bad} timestamp rows dropped.")
        df = df.dropna(subset=['timestamp'])
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
    root = Path(os.getenv('QQQ_DATA_DIR','./mnt/data'))
    cands=[p for p in root.glob('**/*.csv') if 'qqq' in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f'No QQQ CSV found in {root}')
    return cands[0]

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
