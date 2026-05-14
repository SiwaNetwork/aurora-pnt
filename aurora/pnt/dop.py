"""
DOP (Dilution of Precision) calculations for PNT analysis.

Standard geometry matrix approach:
  H row = [cos(el)*sin(az), cos(el)*cos(az), sin(el), 1]
  Q = inv(H^T @ H)
  GDOP = sqrt(trace(Q))
  PDOP = sqrt(Q[0,0]+Q[1,1]+Q[2,2])
  HDOP = sqrt(Q[0,0]+Q[1,1])
  VDOP = sqrt(Q[2,2])
  TDOP = sqrt(Q[3,3])
"""

import math

import numpy as np

# DOP thresholds (ICAO/RTCA standards)
DOP_IDEAL = 1.0
DOP_EXCELLENT = 2.0
DOP_GOOD = 4.0
DOP_MODERATE = 6.0
DOP_POOR = 8.0


def compute_dop(az_rad: np.ndarray, el_rad: np.ndarray) -> dict | None:
    """
    Compute DOP values from arrays of visible satellite azimuth/elevation angles.

    Args:
        az_rad: Azimuth angles in radians (N,)
        el_rad: Elevation angles in radians (N,)

    Returns:
        Dict with gdop, pdop, hdop, vdop, tdop, n_sats.
        None if fewer than 4 satellites visible or geometry is degenerate.
    """
    n = len(az_rad)
    if n < 4:
        return None

    H = np.column_stack([
        np.cos(el_rad) * np.sin(az_rad),  # East
        np.cos(el_rad) * np.cos(az_rad),  # North
        np.sin(el_rad),                    # Up
        np.ones(n),                        # Clock bias
    ])

    HtH = H.T @ H
    try:
        Q = np.linalg.inv(HtH)
    except np.linalg.LinAlgError:
        return None

    # Guard against numerical noise producing tiny negative values on diagonal
    diag = np.maximum(np.diag(Q), 0.0)

    return {
        "gdop": math.sqrt(diag[0] + diag[1] + diag[2] + diag[3]),
        "pdop": math.sqrt(diag[0] + diag[1] + diag[2]),
        "hdop": math.sqrt(diag[0] + diag[1]),
        "vdop": math.sqrt(diag[2]),
        "tdop": math.sqrt(diag[3]),
        "n_sats": n,
    }


def compute_dop_multiconstellation(
    az_leo: np.ndarray,
    el_leo: np.ndarray,
    az_glo: np.ndarray,
    el_glo: np.ndarray,
) -> dict | None:
    """
    Multi-constellation DOP with separate clock bias column per system.

    H matrix layout:
      LEO row:     [cos(el)*sin(az), cos(el)*cos(az), sin(el), 1, 0]
      GLONASS row: [cos(el)*sin(az), cos(el)*cos(az), sin(el), 0, 1]

    Solves 5 unknowns: dx, dy, dz, dt_leo, dt_glonass (ISB between systems).
    Requires at least 1 satellite from each system AND total >= 5.

    Falls back to single-constellation 4-param DOP when only one system visible.
    """
    n_leo = len(az_leo)
    n_glo = len(az_glo)

    # Pure single-constellation fallbacks
    if n_leo == 0 and n_glo == 0:
        return None
    if n_leo == 0:
        return compute_dop(az_glo, el_glo)
    if n_glo == 0:
        return compute_dop(az_leo, el_leo)

    # Combined: 5 unknowns, need >= 5 total with >= 1 from each
    if n_leo + n_glo < 5:
        # Try single-constellation fallback
        if n_leo >= 4:
            return compute_dop(az_leo, el_leo)
        if n_glo >= 4:
            return compute_dop(az_glo, el_glo)
        return None

    rows_leo = np.column_stack([
        np.cos(el_leo) * np.sin(az_leo),
        np.cos(el_leo) * np.cos(az_leo),
        np.sin(el_leo),
        np.ones(n_leo),
        np.zeros(n_leo),
    ])
    rows_glo = np.column_stack([
        np.cos(el_glo) * np.sin(az_glo),
        np.cos(el_glo) * np.cos(az_glo),
        np.sin(el_glo),
        np.zeros(n_glo),
        np.ones(n_glo),
    ])
    H = np.vstack([rows_leo, rows_glo])  # (n_leo+n_glo, 5)

    HtH = H.T @ H
    try:
        Q = np.linalg.inv(HtH)
    except np.linalg.LinAlgError:
        return None

    diag = np.maximum(np.diag(Q), 0.0)

    return {
        "gdop": math.sqrt(diag[0] + diag[1] + diag[2] + diag[3] + diag[4]),
        "pdop": math.sqrt(diag[0] + diag[1] + diag[2]),
        "hdop": math.sqrt(diag[0] + diag[1]),
        "vdop": math.sqrt(diag[2]),
        "tdop": math.sqrt(diag[3]),   # LEO system TDOP
        "n_sats": n_leo + n_glo,
        "n_leo": n_leo,
        "n_glonass": n_glo,
        "multi_constellation": True,
    }


def dop_category(pdop: float) -> str:
    """Return qualitative PDOP category string."""
    if pdop <= DOP_IDEAL:
        return "ideal"
    if pdop <= DOP_EXCELLENT:
        return "excellent"
    if pdop <= DOP_GOOD:
        return "good"
    if pdop <= DOP_MODERATE:
        return "moderate"
    if pdop <= DOP_POOR:
        return "poor"
    return "very_poor"
