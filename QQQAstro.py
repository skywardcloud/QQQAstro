#!/usr/bin/env python3
"""QQQ Astro Pipeline v2
Adds:
  • Moon sign, nakshatra, NASDAQ-Lagna house (v1)
  • Solar‑arc progressed Lagna (daily)
  • Lunar‑return flag (±30 min)
  • Rahu/Ketu longitudes and ±2° Lagna‑contact flag

Run:

  python QQQAstro.py --csv /path/to/QQQ.csv --out enriched.csv
If --csv is omitted the script searches $QQQ_DATA_DIR (fallback: ./mnt/data).

"""

import math
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import pytz

# -------------------------- constants -------------------------------
AYANAMSA = 24.0  # fixed offset for crude sidereal
NATAL_SUN_LONG = 325.722   # Aquarius 25°43′22″
NATAL_MOON_LONG = 238.399  # Scorpio 28°23′55″
NATAL_LAGNA_LONG = 38.6225 # Taurus 8°37′21″

J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

SIGNS = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashirsha","Ardra",
    "Punarvasu","Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni",
    "Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta",
    "Shatabhisha","Purva Bhadrapada","Uttara Bhadrapada","Revati"
]

# ----------------------- astronomy helpers --------------------------
def mean_moon_longitude(dt_utc: datetime) -> float:
    days = (dt_utc - J2000).total_seconds()/86400.0
    return (218.316 + 13.176396*days) % 360

def mean_sun_longitude(dt_utc: datetime) -> float:
    days = (dt_utc - J2000).total_seconds()/86400.0
    return (280.460 + 0.9856474*days) % 360

def mean_rahu_longitude(dt_utc: datetime) -> float:
    days = (dt_utc - J2000).total_seconds()/86400.0
    return (125.04452 - 0.0529538083*days) % 360

# --------------------- zodiac helpers -------------------------------
def sign_from_longitude(lon: float) -> str:
    return SIGNS[int(lon//30)]

def nakshatra_from_longitude(lon: float) -> str:
    sid = (lon - AYANAMSA) % 360
    return NAKSHATRAS[int(sid // (360/27))]

def house_from_longitude(lon: float, lagna_long: float) -> int:
    return int(((lon - lagna_long) % 360)//30)+1

def ang_diff(a: float, b: float) -> float:
    return abs((a-b+180)%360 - 180)

# ----------------------- core pipeline ------------------------------
def enrich(csv_path: Path, out_path: Path):
    tz_ny = pytz.timezone("America/New_York")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None or df['timestamp'].dt.tz.iloc[0] is None:
        df['timestamp'] = df['timestamp'].apply(lambda x: tz_ny.localize(x))
    df['utc'] = df['timestamp'].dt.tz_convert('UTC')

    # Moon
    df['moon_long'] = df['utc'].apply(mean_moon_longitude)
    df['moon_sign'] = df['moon_long'].apply(sign_from_longitude)
    df['nakshatra'] = df['moon_long'].apply(nakshatra_from_longitude)
    df['moon_house'] = df['moon_long'].apply(
        lambda lon: house_from_longitude(lon, NATAL_LAGNA_LONG))

    # Solar‑arc progressed Lagna (daily snapshot)
    dates = df['utc'].dt.date.unique()
    sun_arc: Dict[str,float] = {}
    for d in dates:
        dt_mid = datetime(d.year,d.month,d.day,tzinfo=timezone.utc)
        arc = (mean_sun_longitude(dt_mid) - NATAL_SUN_LONG) % 360
        sun_arc[str(d)] = arc
    df['solar_arc'] = df['utc'].dt.date.astype(str).map(sun_arc)
    df['prog_lagna_long'] = (NATAL_LAGNA_LONG + df['solar_arc']) % 360
    df['prog_lagna_house'] = df.apply(
        lambda r: house_from_longitude(r['moon_long'], r['prog_lagna_long']), axis=1)

    # Lunar return ±30 min → ≈0.274°
    df['lunar_return'] = df['moon_long'].apply(
        lambda lon: ang_diff(lon, NATAL_MOON_LONG) <= 0.274)

    # Nodes
    df['rahu_long'] = df['utc'].apply(mean_rahu_longitude)
    df['ketu_long'] = (df['rahu_long'] + 180) % 360
    df['node_hits_lagna'] = df.apply(
        lambda r: (ang_diff(r['rahu_long'], NATAL_LAGNA_LONG) <= 2.0) or
                  (ang_diff(r['ketu_long'], NATAL_LAGNA_LONG) <= 2.0),
        axis=1)

    df.to_csv(out_path, index=False)
    print(f"✓ Enriched file saved → {out_path}")

# ------------------------- CLI glue ---------------------------------
def locate_default_csv() -> Path:
    env_root = os.getenv('QQQ_DATA_DIR')
    root = Path(env_root) if env_root else Path('./mnt/data')
    cands = [p for p in root.glob('**/*.csv') if 'qqq' in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f'No CSV containing "qqq" found in {root}.')
    return cands[0]

if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser(description='Enrich QQQ 5‑min bars with astro data')
    ap.add_argument('--csv', type=Path,
                    help='Path to raw QQQ CSV (default: search $QQQ_DATA_DIR)')
    ap.add_argument('--out', type=Path, help='Output path (default: same dir, *_enriched.csv)')
    args = ap.parse_args()

    try:
        csv_path = args.csv or locate_default_csv()
    except FileNotFoundError as e:
        sys.exit(str(e))

    out_path = args.out or csv_path.with_name(csv_path.stem + '_enriched.csv')
    enrich(csv_path, out_path)
