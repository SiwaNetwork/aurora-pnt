"""
Two-way ranging and clock synchronization models for LEO PNT.

Reference:
  - IS-GPS-200: pseudorange error budget
  - ITU-R M.1787: satellite navigation measurement models
  - Parkinson & Spilker: GPS Theory and Applications
"""

import math


_C = 299_792_458.0          # m/s
_NS_PER_S = 1e9
_M_PER_NS = _C / _NS_PER_S  # ~0.3 m per nanosecond


# ── Pseudorange error budget ──────────────────────────────────────────────────

class RangingErrorBudget:
    """
    One-sigma ranging error contributors (meters).
    Default values are conservative estimates for a LEO PNT signal.
    """
    def __init__(
        self,
        clock_bias_m: float = 1.0,       # satellite clock (after sync): ~1 m = 3.3 ns
        ephemeris_m: float = 0.5,         # orbit determination: 0.5 m typical for LEO w/ISL
        iono_m: float = 2.0,              # ionospheric delay (single-freq): 2 m typical
        tropo_m: float = 0.5,             # tropospheric delay residual
        multipath_m: float = 0.3,         # multipath at receiver
        noise_m: float = 0.1,             # thermal noise (depends on C/N0 and bandwidth)
        relativistic_m: float = 0.01,     # relativistic correction residual
    ):
        self.clock_bias_m = clock_bias_m
        self.ephemeris_m = ephemeris_m
        self.iono_m = iono_m
        self.tropo_m = tropo_m
        self.multipath_m = multipath_m
        self.noise_m = noise_m
        self.relativistic_m = relativistic_m

    def uere(self) -> float:
        """User Equivalent Range Error (UERE) — combined one-sigma, meters."""
        terms = [
            self.clock_bias_m,
            self.ephemeris_m,
            self.iono_m,
            self.tropo_m,
            self.multipath_m,
            self.noise_m,
            self.relativistic_m,
        ]
        return math.sqrt(sum(t**2 for t in terms))

    def as_dict(self) -> dict:
        return {
            "clock_bias_m": self.clock_bias_m,
            "ephemeris_m": self.ephemeris_m,
            "iono_m": self.iono_m,
            "tropo_m": self.tropo_m,
            "multipath_m": self.multipath_m,
            "noise_m": self.noise_m,
            "relativistic_m": self.relativistic_m,
            "uere_m": round(self.uere(), 3),
        }


# ── Two-way ranging (TWR) ─────────────────────────────────────────────────────

def two_way_range(
    t_transmit_ns: float,
    t_receive_ns: float,
    t_transpond_ns: float,
    t_retransmit_ns: float,
) -> dict:
    """
    Two-Way Ranging measurement.

    Timeline:
      GS sends signal at t1 (t_transmit_ns)
      SAT receives at t2 (t_receive_ns)
      SAT retransmits at t3 (t_retransmit_ns)   [t3 = t2 + t_transpond_ns]
      GS receives at t4

    Range = c * (t4 - t1 - t_transpond) / 2

    Returns range in meters and clock offset in nanoseconds.
    """
    # Round-trip travel time (ns), excluding transponder delay
    t4 = t_receive_ns  # GS receive time (this is the measurement)
    rtt_ns = t4 - t_transmit_ns - t_transpond_ns
    range_m = _M_PER_NS * rtt_ns / 2.0

    # Clock offset estimate (GS clock relative to satellite)
    # Δt = (t2 + t3 - t1 - t4) / 2  in the symmetric TWR case
    clock_offset_ns = (t_receive_ns + t_retransmit_ns - t_transmit_ns - t4) / 2.0

    return {
        "range_m": range_m,
        "rtt_ns": rtt_ns,
        "clock_offset_ns": clock_offset_ns,
        "clock_offset_m": clock_offset_ns * _M_PER_NS,
    }


# ── Thermal noise on pseudorange ─────────────────────────────────────────────

def pseudorange_noise_m(cn0_dbhz: float, bandwidth_hz: float, coherent_ms: float = 20.0) -> float:
    """
    Thermal noise contribution to pseudorange (meters, 1-sigma).
    sigma_r = c / (2π · sqrt(2 · CN0 · T_coh)) / sqrt(BW)
    Simplified DLL noise model.
    """
    cn0_linear = 10 ** (cn0_dbhz / 10.0)
    t_coh = coherent_ms * 1e-3
    sigma_chips = 1.0 / (2.0 * math.pi * math.sqrt(2.0 * cn0_linear * t_coh))
    chip_width_m = _C / bandwidth_hz
    return sigma_chips * chip_width_m


# ── Clock synchronization via ISL ────────────────────────────────────────────

def isl_sync_error_m(
    isl_distance_km: float,
    oscillator_stability_ppb: float = 10.0,  # 10 ppb = typical TCXO in satellite
    sync_interval_s: float = 60.0,           # re-sync every 60 seconds
) -> float:
    """
    Estimated clock drift between ISL synchronization events.
    sigma_clock = oscillator_stability * sync_interval * c (in meters)
    """
    drift_s_per_s = oscillator_stability_ppb * 1e-9
    drift_m = drift_s_per_s * sync_interval_s * _C
    return drift_m


def propagation_delay_ns(distance_m: float) -> float:
    """One-way signal travel time in nanoseconds."""
    return distance_m / _C * _NS_PER_S


def relativistic_correction_m(altitude_m: float) -> float:
    """
    Combined gravitational + time-dilation relativistic correction (meters/s of orbit).
    Delta_f/f = (GM/c^2) * (1/R_earth - 1/R_sat) - v^2/(2c^2)
    """
    GM = 3.986004418e14  # m^3/s^2
    R_e = 6_371_000.0
    R_sat = R_e + altitude_m
    v_sat = math.sqrt(GM / R_sat)  # orbital velocity

    # Gravitational blueshift (sat is higher, runs faster)
    grav = GM / (_C ** 2) * (1.0 / R_e - 1.0 / R_sat)
    # Velocity-based redshift (sat is moving, runs slower)
    vel = v_sat ** 2 / (2 * _C ** 2)

    net_rate = grav - vel  # net fractional frequency offset
    return abs(net_rate) * _C  # m/s equivalent range rate


# ── Position error from UERE + DOP ───────────────────────────────────────────

def position_error_m(uere_m: float, pdop: float) -> float:
    """3D position error (1-sigma): sigma_pos = PDOP * UERE"""
    return pdop * uere_m


def position_error_horizontal_m(uere_m: float, hdop: float) -> float:
    """Horizontal position error (1-sigma): sigma_h = HDOP * UERE"""
    return hdop * uere_m


def time_error_ns(uere_m: float, tdop: float) -> float:
    """Timing error (1-sigma) in nanoseconds: sigma_t = TDOP * UERE / c"""
    return tdop * uere_m / _M_PER_NS
