#!/usr/bin/env python3
"""QQQ Astro Pipeline v2.2 (robust timestamps + warnings)

Adds:
  • Moon sign, nakshatra, NASDAQ‑Lagna house
  • Solar‑arc progressed Lagna (daily snapshot)
  • Lunar‑return flag (±30 min ≈ 0.274 °)
  • Rahu/Ketu longitudes and ±2 ° Lagna‑contact flag

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

# ────────── natal constants ────────────────────────────────────────
AYANAMSA          = 24.0        # crude sidereal shift
NATAL_SUN_LONG    = 325.722     # Aquarius 25°43′22″
NATAL_MOON_LONG   = 238.399     # Scorpio 28°23′55″
NATAL_LAGNA_LONG  = 38.6225     # Taurus   8°37′21″
J2000             = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

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

# ────────── ephemeris helpers (mean orbits ~1° accuracy) ───────────
def mean_moon_longitude(dt_utc: datetime) -> float:
    d = (dt_utc - J2000).total_seconds() / 86400
    return (218.316 + 13.176396 * d) % 360

def mean_sun_longitude(dt_utc: datetime) -> float:
    d = (dt_utc - J2000).total_seconds() / 86400
    return (280.460 + 0.9856474 * d) % 360

def mean_rahu_longitude(dt_utc: datetime) -> float:
    d = (dt_utc - J2000).total_seconds() / 86400
    return (125.04452 - 0.0529538083 * d) % 360

# ────────── zodiac helpers ─────────────────────────────────────────
def sign_from_longitude(lon: float) -> str:
    return SIGNS[int(lon // 30)]

def nakshatra_from_longitude(lon: float) -> str:
    sid = (lon - AYANAMSA) % 360
    return NAKSHATRAS[int(sid // (360/27))]

def house_from_longitude(lon: float, lagna_long: float) -> int:
    return int(((lon - lagna_long) % 360) // 30) + 1

def ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular distance in degrees"""
    return abs((a - b + 180) % 360 - 180)

# ────────── robust timestamp parser ────────────────────────────────
def parse_timestamp(col: pd.Series, tz_ny) -> pd.Series:
    """Return tz‑aware NY datetimes; NaT for unparseable rows."""
    col_clean = col.astype(str).str.strip()

    # Pass 1: explicit M/D/YY H:M
    dt = pd.to_datetime(col_clean, format="%m/%d/%y %H:%M", errors="coerce")

    # Pass 2: ISO or other formats
    mask_nat = dt.isna()
    if mask_nat.any():
        # This step might introduce timezone-aware datetime objects if
        # the inferred format includes timezone information (e.g., 'Z' or offset).
        # The original 'dt' series contains naive datetimes or NaT.
        # Assigning potentially aware datetimes here can make 'dt' a mixed-type series.
        parsed_second_pass = pd.to_datetime(
            col_clean[mask_nat], errors="coerce", infer_datetime_format=True
        )
        dt.loc[mask_nat] = parsed_second_pass

    if dt.notna().sum() == 0:
        raise ValueError("Could not parse ANY timestamps – check raw data.")

    # 'dt' can now be a Series containing NaT, naive datetimes, or aware datetimes.
    # We need to convert all valid datetimes to tz_ny.

    # Initialize the result series with the target timezone and NaT values.
    # This ensures the correct dtype for the final Series.
    final_dt = pd.Series([pd.NaT]*len(dt), index=dt.index, dtype=f"datetime64[ns, {tz_ny.zone}]")
    
    not_na_mask = dt.notna()
    if not_na_mask.any():
        # Work with the subset of dt that has actual datetime objects
        active_dts = dt[not_na_mask]

        # Identify naive and aware timestamps within active_dts.
        # This list comprehension is generally efficient for attribute access.
        is_naive_list = [ts.tzinfo is None for ts in active_dts]
        naive_elements_mask = pd.Series(is_naive_list, index=active_dts.index)
        aware_elements_mask = ~naive_elements_mask

        # Process naive timestamps: localize to tz_ny
        if naive_elements_mask.any():
            series_to_localize = active_dts[naive_elements_mask]
            # series_to_localize contains naive pd.Timestamp objects (or similar datetime-like objects)
            localized_naive = series_to_localize.dt.tz_localize(tz_ny, ambiguous='NaT', nonexistent='shift_forward')
            final_dt.loc[series_to_localize.index] = localized_naive
            
        # Process aware timestamps: convert to tz_ny
        if aware_elements_mask.any():
            series_to_convert = active_dts[aware_elements_mask]
            # series_to_convert contains aware pd.Timestamp objects (or similar datetime-like objects)
            converted_aware = series_to_convert.dt.tz_convert(tz_ny)
            final_dt.loc[series_to_convert.index] = converted_aware
            
    return final_dt

# ────────── core pipeline ──────────────────────────────────────────
def enrich(csv_path: Path, out_path: Path) -> None:
    tz_ny = pytz.timezone("America/New_York")
    df = pd.read_csv(csv_path)

    # 1️⃣ Parse timestamps
    df["timestamp"] = parse_timestamp(df["timestamp"], tz_ny)
    bad = df["timestamp"].isna().sum()
    if bad:
        print(f"⚠︎  {bad} rows removed due to invalid timestamps.")
        df = df.dropna(subset=["timestamp"])
    df["utc"] = df["timestamp"].dt.tz_convert("UTC")

    # 2️⃣ Moon sign / nakshatra / house
    df["moon_long"]   = df["utc"].apply(mean_moon_longitude)
    df["moon_sign"]   = df["moon_long"].apply(sign_from_longitude)
    df["nakshatra"]   = df["moon_long"].apply(nakshatra_from_longitude)
    df["moon_house"]  = df["moon_long"].apply(
        lambda lon: house_from_longitude(lon, NATAL_LAGNA_LONG)
    )

    # 3️⃣ Solar‑arc progressed Lagna (daily)
    sun_arc: Dict[str, float] = {}
    for d in df["utc"].dt.date.unique():
        dt_mid = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        sun_arc[str(d)] = (mean_sun_longitude(dt_mid) - NATAL_SUN_LONG) % 360
    df["solar_arc"]        = df["utc"].dt.date.astype(str).map(sun_arc)
    df["prog_lagna_long"]  = (NATAL_LAGNA_LONG + df["solar_arc"]) % 360
    df["prog_lagna_house"] = df.apply(
        lambda r: house_from_longitude(r["moon_long"], r["prog_lagna_long"]),
        axis=1
    )

    # 4️⃣ Lunar return flag (±0.274° ≈ 30 min)
    df["lunar_return"] = df["moon_long"].apply(
        lambda lon: ang_diff(lon, NATAL_MOON_LONG) <= 0.274
    )

    # 5️⃣ Rahu / Ketu + Lagna contact (±2°)
    df["rahu_long"] = df["utc"].apply(mean_rahu_longitude)
    df["ketu_long"] = (df["rahu_long"] + 180) % 360
    df["node_hits_lagna"] = df.apply(
        lambda r: ang_diff(r["rahu_long"], NATAL_LAGNA_LONG) <= 2.0
               or ang_diff(r["ketu_long"], NATAL_LAGNA_LONG) <= 2.0,
        axis=1
    )

    df.to_csv(out_path, index=False)
    print(f"✓ Enriched file saved → {out_path}")

# ────────── CLI helpers ────────────────────────────────────────────
def locate_default_csv() -> Path:
    root = Path(os.getenv("QQQ_DATA_DIR", "./mnt/data"))
    cands = [p for p in root.glob("**/*.csv") if "qqq" in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f'No CSV containing "qqq" found in {root}.')
    return cands[0]

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Enrich QQQ 5‑min bars with astro data")
    ap.add_argument("--csv", type=Path,
                    help="Path to raw QQQ CSV (default: search $QQQ_DATA_DIR)")
    ap.add_argument("--out", type=Path,
                    help="Output path (default: same dir, *_enriched.csv)")
    args = ap.parse_args()

    try:
        csv_path = args.csv or locate_default_csv()
    except FileNotFoundError as e:
        sys.exit(str(e))

    out_path = args.out or csv_path.with_name(csv_path.stem + "_enriched.csv")
    enrich(csv_path, out_path)
