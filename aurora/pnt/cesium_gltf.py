"""
Cesium-сцена с реальными орбитами + glTF-модель КА АВРОРА (3D в браузере).

В отличие от точечных билбордов (`cesium_pnt`), здесь спутники несут полноценную
3D-glTF-модель (корпус + 2 солнечные панели), генерируемую программно с встроенным
base64-буфером (без внешних файлов). Орбиты — аналитический Walker 300/15 (i=75°,
1000 км), позиции в ECEF (FIXED). HTML-оболочка переиспользуется из `cesium_pnt`
(offline при наличии локального Cesium).

Выходы:
  aurora_sat_<label>.gltf   — модель КА (glTF 2.0, embedded buffer)
  cesium_gltf_<label>.html  — интерактивная сцена (открыть в браузере)
"""

import sys, os, math, json, base64, struct
from typing import Dict, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from aurora.pnt.constellation_anim import _build, _eci, _ecef, T_ORB, N_PLANE, N_PER, N_SAT
from aurora.pnt import cesium_pnt

_PLANE_COLORS = [[0, 230, 180, 255], [9, 132, 227, 255], [225, 112, 85, 255],
                 [253, 203, 110, 255], [108, 92, 231, 255]]


# ─────────────────────────────────────────────────────────────────────────────
#  Генерация glTF 2.0 модели КА (корпус + 2 панели), embedded base64 buffer
# ─────────────────────────────────────────────────────────────────────────────
def _box(cx, cy, cz, sx, sy, sz):
    """24 вершины (по 4 на грань) + нормали + 36 индексов."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    # грани: (+X,-X,+Y,-Y,+Z,-Z); для каждой — 4 вершины CCW и нормаль
    faces = [
        ([(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)], (1, 0, 0)),
        ([(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)], (-1, 0, 0)),
        ([(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)], (0, 1, 0)),
        ([(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)], (0, -1, 0)),
        ([(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], (0, 0, 1)),
        ([(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)], (0, 0, -1)),
    ]
    pos, nrm, idx = [], [], []
    for fi, (verts, n) in enumerate(faces):
        base = fi * 4
        for vx, vy, vz in verts:
            pos.append((cx + vx, cy + vy, cz + vz)); nrm.append(n)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return (np.array(pos, np.float32), np.array(nrm, np.float32),
            np.array(idx, np.uint16))


def _build_gltf_dict() -> dict:
    """Строит словарь glTF 2.0 модели КА (корпус + 2 панели + нав-антенна)."""
    parts = [
        (_box(0, 0, 0, 2.0, 1.2, 1.2),  [0.80, 0.63, 0.18, 1.0], 0.6, 0.45),   # корпус (золото MLI)
        (_box(0,  3.0, 0, 3.2, 4.2, 0.08), [0.09, 0.13, 0.32, 1.0], 0.1, 0.7), # панель +Y
        (_box(0, -3.0, 0, 3.2, 4.2, 0.08), [0.09, 0.13, 0.32, 1.0], 0.1, 0.7), # панель −Y
        (_box(1.05, 0, 0, 0.1, 0.7, 0.7), [0.85, 0.18, 0.12, 1.0], 0.2, 0.5),  # нав-антенна (надир)
    ]

    buf = bytearray()
    bufferViews, accessors, materials, prims = [], [], [], []
    FLOAT, USHORT = 5126, 5123
    ARRAY, ELEMENT = 34962, 34963

    def add_floats(arr, target):
        off = len(buf)
        b = arr.astype("<f4").tobytes()
        buf.extend(b)
        bufferViews.append({"buffer": 0, "byteOffset": off,
                            "byteLength": len(b), "target": target})
        return len(bufferViews) - 1

    def add_ushorts(arr):
        # выравнивание на 4 байта
        while len(buf) % 4 != 0:
            buf.append(0)
        off = len(buf)
        b = arr.astype("<u2").tobytes()
        buf.extend(b)
        bufferViews.append({"buffer": 0, "byteOffset": off,
                            "byteLength": len(b), "target": ELEMENT})
        return len(bufferViews) - 1

    for (pos, nrm, idx), color, metal, rough in parts:
        bv_p = add_floats(pos.reshape(-1), ARRAY)
        accessors.append({"bufferView": bv_p, "componentType": FLOAT, "count": len(pos),
                          "type": "VEC3",
                          "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()})
        a_pos = len(accessors) - 1
        bv_n = add_floats(nrm.reshape(-1), ARRAY)
        accessors.append({"bufferView": bv_n, "componentType": FLOAT, "count": len(nrm),
                          "type": "VEC3"})
        a_nrm = len(accessors) - 1
        bv_i = add_ushorts(idx)
        accessors.append({"bufferView": bv_i, "componentType": USHORT, "count": len(idx),
                          "type": "SCALAR"})
        a_idx = len(accessors) - 1
        materials.append({"pbrMetallicRoughness": {
            "baseColorFactor": color, "metallicFactor": metal, "roughnessFactor": rough}})
        prims.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nrm},
                      "indices": a_idx, "material": len(materials) - 1})

    b64 = base64.b64encode(bytes(buf)).decode("ascii")
    gltf = {
        "asset": {"version": "2.0", "generator": "AURORA cesium_gltf"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "AURORA_SAT"}],
        "meshes": [{"name": "AURORA", "primitives": prims}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": bufferViews,
        "buffers": [{"byteLength": len(buf),
                     "uri": "data:application/octet-stream;base64," + b64}],
    }
    # самопроверка структуры
    for bv in bufferViews:
        assert bv["byteOffset"] + bv["byteLength"] <= len(buf), "bufferView вне буфера"
    return gltf


def build_gltf_satellite(path: str) -> str:
    """Пишет glTF 2.0 модель КА в файл (для внешнего использования). Возвращает путь."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_build_gltf_dict(), f)
    return path


def gltf_data_uri() -> str:
    """glTF как data-URI — встраивается в CZML, грузится и под file:// (без CORS)."""
    s = json.dumps(_build_gltf_dict(), separators=(",", ":"))
    return "data:model/gltf+json;base64," + base64.b64encode(s.encode("utf-8")).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
#  CZML с аналитическими орбитами Walker + glTF-моделью
# ─────────────────────────────────────────────────────────────────────────────
def build_walker_czml(model_uri: str, label: str, n_orbits: float = 2.0,
                      step_s: float = 60.0, speed: int = 60,
                      model_every: int = 6) -> List[Dict]:
    import datetime
    raan, u0 = _build()
    dur = n_orbits * T_ORB
    times = np.arange(0, dur + step_s, step_s)
    epoch = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
    end = epoch + datetime.timedelta(seconds=float(times[-1]))
    epoch_iso = "2000-01-01T00:00:00Z"
    end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    # ВАЖНО: окно = реальному интервалу данных (иначе КА исчезают после выборок)
    avail = f"{epoch_iso}/{end_iso}"

    packets = [{
        "id": "document", "name": f"AURORA glTF — {label}", "version": "1.0",
        "clock": {"interval": avail, "currentTime": epoch_iso,
                  "multiplier": speed, "range": "LOOP_STOP",
                  "step": "SYSTEM_CLOCK_MULTIPLIER"},
    }]

    # позиции ECEF (м) по времени
    pos_cache = {si: [] for si in range(N_SAT)}
    for t in times:
        ecef = _ecef(_eci(raan, u0, t), t)  # м
        for si in range(N_SAT):
            pos_cache[si].extend([round(float(t), 1),
                                  float(ecef[si, 0]), float(ecef[si, 1]), float(ecef[si, 2])])

    for si in range(N_SAT):
        plane = si // N_PER
        col = _PLANE_COLORS[plane % len(_PLANE_COLORS)]
        pkt = {
            "id": f"sat_{si}", "name": f"AURORA Sat-{si+1:03d} | плоскость {plane+1}",
            "availability": avail,
            "position": {"interpolationAlgorithm": "LAGRANGE", "interpolationDegree": 5,
                         "referenceFrame": "FIXED", "epoch": epoch_iso,
                         "cartesian": pos_cache[si]},
            # ориентация модели по вектору скорости (иначе КА «смотрит» произвольно)
            "orientation": {"velocityReference": "#position"},
            "point": {"color": {"rgba": col}, "pixelSize": 6,
                      "outlineColor": {"rgba": [255, 255, 255, 160]}, "outlineWidth": 1},
        }
        # 3D-модель — на части КА (производительность), остальные — точки + путь
        if si % model_every == 0:
            pkt["model"] = {"gltf": model_uri, "scale": 1.0,
                            "minimumPixelSize": 48, "maximumScale": 250000,
                            "runAnimations": False}
        if plane in (0, 5, 10):
            pkt["path"] = {"show": [{"interval": avail, "boolean": True}],
                           "width": 1.2, "resolution": 120,
                           "leadTime": [{"number": 2200}], "trailTime": [{"number": 2200}],
                           "material": {"solidColor": {"color": {"rgba": col[:3] + [70]}}}}
        packets.append(pkt)
    return packets


def run_cesium_gltf(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    gltf_name = f"aurora_sat_{label}.gltf"
    gltf_path = os.path.join(output_dir, gltf_name)
    build_gltf_satellite(gltf_path)   # внешний файл — для http-режима/повторного использования

    # в CZML встраиваем модель как data-URI — грузится и под file:// (без CORS/XHR)
    czml = build_walker_czml(gltf_data_uri(), label)
    html_path = os.path.join(output_dir, f"cesium_gltf_{label}.html")
    cesium_pnt.write_cesium_html(czml, html_path, label)

    n_models = sum(1 for p in czml if "model" in p)
    return {"gltf": gltf_path, "html": html_path,
            "gltf_bytes": os.path.getsize(gltf_path),
            "n_sat": N_SAT, "n_models": n_models, "n_packets": len(czml)}


def print_cesium_gltf_summary(label: str, r: Dict) -> None:
    print(f"\n  Cesium + glTF -- {label}")
    print(f"    glTF-модель: {r['gltf']} ({r['gltf_bytes']} байт)")
    print(f"    Сцена: {r['n_sat']} КА, из них с 3D-моделью {r['n_models']}")
    print(f"    HTML (открыть в браузере): {r['html']}")
