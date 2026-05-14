"""
Link budget physical models for LEO PNT signals.

Covers:
  - Free-Space Path Loss (FSPL)
  - Doppler shift on GSL and ISL
  - Received SNR / C/N0
  - Atmospheric loss (troposphere + ionosphere estimate)
"""

import math


_C = 299_792_458.0  # m/s
_K_DB = -228.6      # Boltzmann constant, dBW/K/Hz


# ── Frequency bands for PNT signals ─────────────────────────────────────────

BANDS = {
    "L1":  1575.42e6,   # GPS L1 / Galileo E1
    "L2":  1227.60e6,   # GPS L2
    "L5":  1176.45e6,   # GPS L5 / Galileo E5a
    "G1":  1602.00e6,   # GLONASS G1
    "G2":  1246.00e6,   # GLONASS G2
    "S":   2492.02e6,   # NavIC S-band
    "Ka":  26_500e6,    # Ka-band ISL
    "Ku":  14_000e6,    # Ku-band ISL/feeder
}


# ── Core formulas ─────────────────────────────────────────────────────────────

def fspl_db(distance_m: float, freq_hz: float) -> float:
    """Free-Space Path Loss in dB. FSPL = 20·log10(4π·d·f/c)"""
    return 20.0 * math.log10(4 * math.pi * distance_m * freq_hz / _C)


def received_power_dbw(
    eirp_dbw: float,
    fspl_db: float,
    rx_gain_dbi: float = 0.0,
    atm_loss_db: float = 0.0,
    pointing_loss_db: float = 0.0,
) -> float:
    """Pr = EIRP - FSPL + Gr - L_atm - L_point  (all in dB)"""
    return eirp_dbw - fspl_db + rx_gain_dbi - atm_loss_db - pointing_loss_db


def cn0_db(
    received_power_dbw: float,
    noise_temp_k: float = 290.0,
    noise_figure_db: float = 2.0,
) -> float:
    """
    Carrier-to-Noise-density ratio C/N0 (dB·Hz).
    C/N0 = Pr - k - T_sys
    where T_sys includes receiver noise figure.
    """
    t_sys = noise_temp_k * (10 ** (noise_figure_db / 10.0))
    t_sys_db = 10.0 * math.log10(t_sys)
    return received_power_dbw - _K_DB - t_sys_db


def snr_db(cn0_dbhz: float, bandwidth_hz: float) -> float:
    """SNR = C/N0 - 10·log10(BW)"""
    return cn0_dbhz - 10.0 * math.log10(bandwidth_hz)


def doppler_shift_hz(freq_hz: float, radial_velocity_m_s: float) -> float:
    """
    Doppler shift: Δf = f · v_r / c
    Positive v_r = source approaching receiver.
    """
    return freq_hz * radial_velocity_m_s / _C


def doppler_shift_ppm(freq_hz: float, radial_velocity_m_s: float) -> float:
    """Doppler shift in parts-per-million."""
    return 1e6 * radial_velocity_m_s / _C


def atmospheric_loss_db(elevation_deg: float) -> float:
    """
    Combined tropospheric + ionospheric loss estimate (dB).
    Uses a simplified mapping angle model (zenith ~0.5 dB, 10° elevation ~3 dB).
    """
    el_rad = math.radians(max(elevation_deg, 1.0))
    tropo = 0.5 / math.sin(el_rad)   # troposphere (dry + wet)
    iono  = 0.1 / math.sin(el_rad)   # ionosphere (L-band estimate)
    return tropo + iono


def max_leo_velocity_m_s(altitude_m: float) -> float:
    """Orbital velocity at given altitude (circular orbit)."""
    mu = 3.986004418e14  # Earth GM, m^3/s^2
    r = 6_371_000.0 + altitude_m
    return math.sqrt(mu / r)


# ── Link budget table for one configuration ──────────────────────────────────

def compute_link_budget(
    distance_m: float,
    elevation_deg: float,
    freq_hz: float,
    tx_power_dbw: float = 16.0,    # ~40 W EIRP typical LEO PNT sat
    tx_gain_dbi: float = 14.0,     # phased-array antenna
    rx_gain_dbi: float = 3.0,      # user receiver patch antenna
    noise_temp_k: float = 290.0,
    noise_figure_db: float = 2.0,
    bandwidth_hz: float = 10.23e6, # GPS C/A: 1.023 MHz primary, 10.23 P-code
    radial_velocity_m_s: float = 0.0,
) -> dict:
    """
    Compute full link budget. Returns dict with all intermediate values.
    """
    eirp = tx_power_dbw + tx_gain_dbi
    fsl = fspl_db(distance_m, freq_hz)
    atm = atmospheric_loss_db(elevation_deg)
    pr = received_power_dbw(eirp, fsl, rx_gain_dbi, atm)
    cn0 = cn0_db(pr, noise_temp_k, noise_figure_db)
    snr = snr_db(cn0, bandwidth_hz)
    dop_hz = doppler_shift_hz(freq_hz, radial_velocity_m_s)
    dop_ppm = doppler_shift_ppm(freq_hz, radial_velocity_m_s)

    return {
        "distance_km": distance_m / 1000.0,
        "elevation_deg": elevation_deg,
        "freq_mhz": freq_hz / 1e6,
        "eirp_dbw": round(eirp, 2),
        "fspl_db": round(fsl, 2),
        "atm_loss_db": round(atm, 2),
        "rx_power_dbw": round(pr, 2),
        "cn0_dbhz": round(cn0, 2),
        "snr_db": round(snr, 2),
        "doppler_hz": round(dop_hz, 1),
        "doppler_ppm": round(dop_ppm, 4),
        "link_margin_db": round(cn0 - 35.0, 2),  # 35 dBHz = min for GPS acquisition
    }
