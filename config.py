"""Central configuration for DRISHTI.

Credentials come from a gitignored .env file — never hardcode them.
Framing note (keep everywhere): this system is decision-support /
situational awareness. It is NOT a targeting tool.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = PROJECT_ROOT / "storage"
DB_PATH = STORAGE_DIR / "drishti.sqlite3"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from project root (gitignored).
load_dotenv(PROJECT_ROOT / ".env")

SPACETRACK_USER = os.environ.get("SPACETRACK_USER", "")
SPACETRACK_PASS = os.environ.get("SPACETRACK_PASS", "")

# --- Physical constants ------------------------------------------------
MU_EARTH_KM3_S2 = 398600.4418   # Earth GM, km^3/s^2
R_EARTH_KM = 6378.137           # equatorial radius, km
J2_EARTH = 1.08262668e-3        # Earth's second zonal harmonic (oblateness) —
                                 # drives nodal precession, the basis of SSO detection
SIDEREAL_DAY_S = 86164.0905     # Earth's sidereal rotation period, s (GEO/Tundra period)
GEO_ALTITUDE_KM = 35786.0       # geostationary altitude above equator, km
EARTH_ORBIT_RAD_PER_S = 1.99106e-7  # mean angular rate of Earth around Sun (SSO target
                                     # nodal precession rate; 2π / 365.2422 days)

# --- Data source endpoints ---------------------------------------------
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_SATCAT_URL = "https://celestrak.org/pub/satcat.csv"

# Celestrak group names we care about by default.
DEFAULT_CELESTRAK_GROUPS = ["active", "stations", "geo", "analyst"]
