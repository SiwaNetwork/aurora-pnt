"""
Общая конфигурация pytest для AURORA PNT.

Фикстуры для временных директорий вывода, эталонных констант, и базовых
утилит проверки графиков и CSV.
"""
import os
import sys
from pathlib import Path

import pytest

# Гарантируем UTF-8 stdout/stderr на Windows-консоли
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@pytest.fixture
def tmp_results(tmp_path):
    """Временный каталог для результатов конкретного теста."""
    out = tmp_path / "results"
    out.mkdir()
    return str(out)


@pytest.fixture
def project_root():
    """Корневой каталог проекта."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def aurora_constants():
    """Эталонные физические параметры AURORA для cross-проверок."""
    return {
        "n_sats": 300,
        "n_planes": 15,
        "n_per_plane": 20,
        "altitude_km": 1000.0,
        "inclination_deg": 75.0,
        "mu_earth": 3.986e14,        # м³/с²
        "R_earth_km": 6371.0,
        "semimajor_km": 7371.0,      # R_E + h
        "n_mean_motion": 9.96e-4,    # рад/с (для h=1000)
        "v_orbit_kmps": 7.35,
        "T_orbit_s": 6305.0,
        "freq_L1_Hz": 1575.42e6,
        "freq_L5_Hz": 1176.45e6,
        "freq_ISL_Hz": 26.5e9,
        "design_life_years": 7,
        "cn0_zenith_dbHz": 52.6,
        "doppler_max_Hz": 38600.0,
    }


def assert_png_exists(output_dir: str, fname: str, min_size_bytes: int = 1000):
    """Утверждает что PNG-файл создан и не пустой."""
    p = Path(output_dir) / fname
    assert p.exists(), f"PNG not created: {fname}"
    assert p.stat().st_size >= min_size_bytes, (
        f"PNG too small ({p.stat().st_size} B): {fname}"
    )


def assert_csv_valid(output_dir: str, fname: str, min_rows: int = 2):
    """Утверждает что CSV создан, имеет минимум min_rows строк, читается."""
    p = Path(output_dir) / fname
    assert p.exists(), f"CSV not created: {fname}"
    with open(p, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) >= min_rows, (
        f"CSV has only {len(lines)} non-empty lines, expected >= {min_rows}"
    )
