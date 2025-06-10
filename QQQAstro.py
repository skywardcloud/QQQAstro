#!/usr/bin/env python3
"""
QQQ Astro Pipeline v2.1 (robust timestamps)

Adds:
  • Moon sign, nakshatra, NASDAQ-Lagna house
  • Solar-arc progressed Lagna (daily snapshot)
  • Lunar-return flag (±30 min ≈ 0.274 °)
  • Rahu/Ketu longitudes and ±2 ° Lagna-contact flag

Run:
  python QQQAstro.py --csv /path/to/QQQ.csv --out enriched.csv
If --csv is omitted the script searches $QQQ_DATA_DIR (fallback: ./mnt/data).
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import pytz

# ────────────── natal constants ────────────────────────────────────
AYANAMSA          = 24.0                          # crude sidereal shift
NATAL_SUN_LONG    = 325.722                       # Aquarius 25°43′22″
NATAL_MOON_LONG   = 238.399                       # Scorpio 28°23′55″
NATAL_LAGNA_LONG  = 38.6225                       # Taurus  8°37′21″
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

# ────────────── ephemeris helpers (mean orbit ≈ 1 °) ───────────────
def mean_moon_longitude(dt_utc: datetime) -> float:
    d = (dt_utc - J2000).total_seconds() / 86400
    return (218.316 + 13.176396 * d) % 360

def mean_sun_longitude(dt_utc: datetime) -> float:
    d = (dt_utc - J2000).total_seconds() / 86400
    return (280.460 + 0.9856474 * d) % 360

def mean_rahu_longitude(dt_utc: datetime) -> float:
    d = (dt_utc - J2000).total_seconds() / 86400
    return (125.04452 - 0.0529538083 * d) % 360

# ────────────── zodiac/house helpers ───────────────────────────────
def sign_from_longitude(lon: float) -> str:
    return SIGNS[int(lon // 30)]

def nakshatra_from_longitude(lon: float) -> str:
    sid = (lon - AYANAMSA) % 360
    return NAKSHATRAS[int(sid // (360/27))]

def house_from_longitude(lon: float, lagna_long: float) -> int:
    return int(((lon - lagna_long) % 360) // 30) + 1

def ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference (deg) between a and b."""
    return abs((a - b + 180) % 360 - 180)

# ────────────── robust timestamp parser ────────────────────────────
def parse_timestamp(col: pd.Series, tz_ny) -> pd.Series:
    """
    Parse mixed-format timestamps. Unparseable strings → NaT.
    Returned Series is tz-aware NY-time.
    """
    dt = pd.to_datetime(col, errors="coerce", infer_datetime_format=True)
    if dt.dt.tz is None or dt.dt.tz.iloc[0] is None:
        dt = dt.apply(lambda x: tz_ny.localize(x) if pd.notna(x) else x)
    return dt

# ────────────── core pipeline ──────────────────────────────────────
def enrich(csv_path: Path, out_path: Path) -> None:
    tz_ny = pytz.timezone("America/New_York")
    df = pd.read_csv(csv_path)

    # 1️⃣ timestamps
    df["timestamp"] = parse_timestamp(df["timestamp"], tz_ny)
    bad_rows = df["timestamp"].isna().sum()
    if bad_rows:
        print(f"⚠︎ {bad_rows} rows had invalid timestamps and were removed.")
        df = df.dropna(subset=["timestamp"])
    df["utc"] = df["timestamp"].dt.tz_convert("UTC")

    # 2️⃣ Moon sign / nakshatra / house
    df["moon_long"]   = df["utc"].apply(mean_moon_longitude)
    df["moon_sign"]   = df["moon_long"].apply(sign_from_longitude)
    df["nakshatra"]   = df["moon_long"].apply(nakshatra_from_longitude)
    df["moon_house"]  = df["moon_long"].apply(
        lambda lon: house_from_longitude(lon, NATAL_LAGNA_LONG)
    )

    # 3️⃣ solar-arc progressed Lagna (daily)
    sun_arc: Dict[str, float] = {}
    for d in df["utc"].dt.date.unique():
        dt_mid = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        sun_arc[str(d)] = (mean_sun_longitude(dt_mid) - NATAL_SUN_LONG) % 360
    df["solar_arc"]        = df["utc"].dt.date.astype(str).map(sun_arc)
    df["prog_lagna_long"]  = (NATAL_LAGNA_LONG + df["solar_arc"]) % 360
    df["prog_lagna_house"] = df.apply(
        lambda r: house_from_longitude(r["moon_long"], r["prog_lagna_long"]),
        axis=1,
    )

    # 4️⃣ lunar-return flag (±30 min ≈ 0.274 °)
    df["lunar_return"] = df["moon_long"].apply(
        lambda lon: ang_diff(lon, NATAL_MOON_LONG) <= 0.274
    )

    # 5️⃣ Rahu / Ketu + Lagna-hit flag (±2 °)
    df["rahu_long"] = df["utc"].apply(mean_rahu_longitude)
    df["ketu_long"] = (df["rahu_long"] + 180) % 360
    df["node_hits_lagna"] = df.apply(
        lambda r: (
            ang_diff(r["rahu_long"], NATAL_LAGNA_LONG) <= 2.0
            or ang_diff(r["ketu_long"], NATAL_LAGNA_LONG) <= 2.0
        ),
        axis=1,
    )

    # save
    df.to_csv(out_path, index=False)
    print(f"✓ Enriched file saved → {out_path}")

# ────────────── CLI glue ───────────────────────────────────────────
def locate_default_csv() -> Path:
    root = Path(os.getenv("QQQ_DATA_DIR", "./mnt/data"))
    cands = [p for p in root.glob("**/*.csv") if "qqq" in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f'No CSV containing "qqq" found in {root}.')
    return cands[0]

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Enrich QQQ 5-min bars with astro data")
    parser.add_argument("--csv", type=Path,
                        help="Path to raw QQQ CSV (default: search $QQQ_DATA_DIR)")
    parser.add_argument("--out", type=Path,
                        help="Output path (default: same dir, *_enriched.csv)")
    args = parser.parse_args()

    try:
        csv_path = args.csv or locate_default_csv()
    except FileNotFoundError as e:
        sys.exit(str(e))

    out_path = args.out or csv_path.with_name(csv_path.stem + "_enriched.csv")
    enrich(csv_path, out_path)
