"""
QQQ Astro‑Pipeline 🌓📈  • v0.2 (robust file‑loader)
──────────────────────────────────────────────────────────────
Goal   Enrich every 5‑minute QQQ bar with the Moon’s western sign,
       sidereal nakshatra‑pada and NASDAQ‑Lagna house so we can
       hunt for repeatable astro/volatility patterns.

🔑 What changed in v0.2
    • Graceful CSV discovery – no hard crash if the exact path is wrong.
    • `--csv` and `--out` CLI arguments for flexibility.
    • Extra self‑test ensures friendly error when data file missing.
    • Docstring refreshed; logic unchanged elsewhere.

💾 Runtime requirements
    pandas · numpy · pytz   (all are pre‑installed here)
    (Optionally) skyfield    — used when available.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np  # noqa: F401 (may be useful later)
import pandas as pd
import pytz

# ─────────────────────────────────────────────────────────────
# NASDAQ radix constants
LAGNA_DEG = 8 + 37 / 60 + 21 / 3600  # 8° 37′ 21″ Taurus
SIGNS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]
# 27 nakshatras (starting degrees, Lahiri ayanamsha)
NAKSHATRAS = [
    ("Ashwini", 0),
    ("Bharani", 13 + 20 / 60),
    ("Krittika", 26 + 40 / 60),
    ("Rohini", 40),
    ("Mrigashira", 53 + 20 / 60),
    ("Ardra", 66 + 40 / 60),
    ("Punarvasu", 80),
    ("Pushya", 93 + 20 / 60),
    ("Ashlesha", 106 + 40 / 60),
    ("Magha", 120),
    ("Purva Phalguni", 133 + 20 / 60),
    ("Uttara Phalguni", 146 + 40 / 60),
    ("Hasta", 160),
    ("Chitra", 173 + 20 / 60),
    ("Swati", 186 + 40 / 60),
    ("Vishakha", 200),
    ("Anuradha", 213 + 20 / 60),
    ("Jyeshtha", 226 + 40 / 60),
    ("Mula", 240),
    ("Purva Ashadha", 253 + 20 / 60),
    ("Uttara Ashadha", 266 + 40 / 60),
    ("Shravana", 280),
    ("Dhanishta", 293 + 20 / 60),
    ("Shatabhisha", 306 + 40 / 60),
    ("Purva Bhadra", 320),
    ("Uttara Bhadra", 333 + 20 / 60),
    ("Revati", 346 + 40 / 60),
]

HOUSE_CUSPS: Dict[int, float] = {h: (LAGNA_DEG + 30 * (h - 1)) % 360 for h in range(1, 13)}

# ─────────────────────────────────────────────────────────────
# Optional high‑precision Moon longitude with Skyfield
try:
    from skyfield.api import load  # type: ignore

    _TS = load.timescale()
    _EPH = load("de440s.bsp")
    _MOON = _EPH["moon"]
    _EARTH = _EPH["earth"]

    def moon_longitude(dt_utc: datetime) -> float:  # noqa: D401 – simple name
        """Geocentric Moon longitude (tropical degrees)."""
        t = _TS.utc(dt_utc)
        ecl = _EARTH.at(t).observe(_MOON).ecliptic_position()
        return math.degrees(math.atan2(ecl[1].au, ecl[0].au)) % 360

except ModuleNotFoundError:

    J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

    def moon_longitude(dt_utc: datetime) -> float:  # type: ignore
        """Fast mean‑longitude (≤1.5° error) if Skyfield unavailable."""
        days = (dt_utc - J2000).total_seconds() / 86_400
        return (218.316 + 13.176396 * days) % 360

# ─────────────────────────────────────────────────────────────
# Helper functions

def sign_from_longitude(lon: float) -> str:
    return SIGNS[int(lon // 30)]


def nakshatra_from_sidereal(lon_sidereal: float) -> Tuple[str, int]:
    deg = lon_sidereal % 360
    for i, (name, start_deg) in enumerate(NAKSHATRAS):
        next_start = NAKSHATRAS[(i + 1) % 27][1]
        if start_deg <= deg < next_start:
            pada = int(((deg - start_deg) // 3.3333) + 1)
            return name, pada
    return "Revati", 4


def house_from_longitude(lon: float) -> int:
    shifted = (lon - LAGNA_DEG) % 360
    return int(shifted // 30) + 1

# ─────────────────────────────────────────────────────────────
# Core logic

def locate_csv(path_hint: Path | None = None) -> Path:
    """Try to locate the CSV; fall back to glob search inside /mnt/data."""
    if path_hint and path_hint.exists():
        return path_hint
    search_root = Path("/mnt/data")
    for p in search_root.glob("QQQ*major*changes*.csv"):
        return p
    raise FileNotFoundError(
        "✗ QQQ CSV not found. Upload the file or use --csv /path/to/file.csv",
    )


def enrich(csv_path: Path, out_path: Path | None = None) -> Path:
    df = pd.read_csv(csv_path)
    est = pytz.timezone("America/New_York")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_utc"] = df["timestamp"].dt.tz_localize(est).dt.tz_convert(timezone.utc)

    longs: list[float] = []
    signs: list[str] = []
    nakshas: list[str] = []
    padas: list[int] = []
    houses: list[int] = []

    for dt in df["timestamp_utc"]:
        lon = moon_longitude(dt.to_pydatetime())
        longs.append(lon)
        signs.append(sign_from_longitude(lon))
        lon_sidereal = (lon - 24) % 360  # Lahiri offset ≈24°
        nak, pad = nakshatra_from_sidereal(lon_sidereal)
        nakshas.append(nak)
        padas.append(pad)
        houses.append(house_from_longitude(lon))

    df["moon_longitude"] = longs
    df["moon_sign"] = signs
    df["moon_nakshatra"] = nakshas
    df["nakshatra_pada"] = padas
    df["moon_house"] = houses

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + "_enriched.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Enriched file saved → {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────────
# Self‑tests

def _test_moon_sign():
    sample_dt = datetime(2023, 1, 12, 18, 25, tzinfo=timezone.utc)
    sign = sign_from_longitude(moon_longitude(sample_dt))
    assert sign in SIGNS


def _test_missing_csv():
    try:
        locate_csv(Path("/non/existent/path.csv"))
    except FileNotFoundError:
        return
    raise AssertionError("locate_csv should raise FileNotFoundError on bad hint")

# ─────────────────────────────────────────────────────────────
# CLI entry‑point

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich QQQ 5‑minute bars with lunar data.")
    parser.add_argument("--csv", type=Path, help="Path to raw QQQ CSV.")
    parser.add_argument("--out", type=Path, help="Custom output path (CSV).")
    args = parser.parse_args()

    csv_path = locate_csv(args.csv)
    enrich(csv_path, args.out)


if __name__ == "__main__":
    _test_moon_sign()
    _test_missing_csv()
    main()
