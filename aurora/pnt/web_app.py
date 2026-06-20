"""
Генератор автономного веб-симулятора группировки АВРОРА (папка web/).

Полноценный одностраничный сайт на CesiumJS: 3D-глобус Земли, реальные
время-динамические орбиты Walker 300/15, glTF-модель КА, фирменный интерфейс
(панель миссии, переключение фаз развёртывания, легенда). Самодостаточен:
CZML и модель встроены, Cesium берётся локально (assets/cesium) или с CDN.

Сборка:  python -m aurora.pnt.cli web-app
Открыть: web/index.html в браузере.
"""

import sys, os, json, math
from typing import Dict
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from aurora.pnt.constellation_anim import (
    _build, T_ORB, N_PLANE, N_PER, N_SAT,
    GM, R_E, A, INCL, OMEGA_E, N_MEAN)
from aurora.pnt.cesium_gltf import gltf_data_uri
from aurora.pnt import cesium_pnt
from aurora.pnt.cesium_pnt import _gmst_rad   # GMST для TEME→ECEF

# Вековая прецессия восходящего узла от J2 (для круговой орбиты):
#   dΩ/dt = -1.5 · n · J2 · (R_E/a)² · cos i   — реальный дрейф плоскостей
_J2 = 1.08263e-3
RAAN_DOT = -1.5 * N_MEAN * _J2 * (R_E / A) ** 2 * math.cos(INCL)   # рад/с


def _prop_ecef(raan0, u0, t):
    """ECEF-позиции всех КА в момент t (круговая орбита + J2-прецессия RAAN + вращение Земли)."""
    u = u0 + N_MEAN * t
    raan = raan0 + RAAN_DOT * t          # ← J2: плоскости дрейфуют
    x0, y0 = A * np.cos(u), A * np.sin(u)
    x1 = x0; y1 = y0 * math.cos(INCL); z1 = y0 * math.sin(INCL)   # наклон вокруг X
    xe = x1 * np.cos(raan) - y1 * np.sin(raan)                    # поворот RAAN вокруг Z
    ye = x1 * np.sin(raan) + y1 * np.cos(raan)
    th = OMEGA_E * t                                             # ECEF: вращение Земли
    c, s = math.cos(-th), math.sin(-th)
    return np.stack([xe * c - ye * s, xe * s + ye * c, z1], axis=1)


def _nadir_quat(r, v, prev=None):
    """Кватернион [x,y,z,w] надирной ориентации: +X→надир, +Y→нормаль орбиты."""
    xb = -r / np.linalg.norm(r)                 # +X модели = надир (антенна на Землю)
    h = np.cross(r, v); yb = h / np.linalg.norm(h)   # +Y = нормаль орбиты (ось панелей)
    zb = np.cross(xb, yb)
    m00, m10, m20 = xb; m01, m11, m21 = yb; m02, m12, m22 = zb
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2; w = 0.25 * S
        x = (m21 - m12) / S; y = (m02 - m20) / S; z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / S
        x = 0.25 * S; y = (m01 + m10) / S; z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / S
        x = (m01 + m10) / S; y = 0.25 * S; z = (m12 + m21) / S
    else:
        S = math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / S
        x = (m02 + m20) / S; y = (m12 + m21) / S; z = 0.25 * S
    q = [x, y, z, w]
    if prev is not None and (q[0]*prev[0]+q[1]*prev[1]+q[2]*prev[2]+q[3]*prev[3]) < 0:
        q = [-c for c in q]                     # непрерывность знака для SLERP
    return q

# Канонические фазы (накопл. число КА) — §4.3
PHASES = [("Ф0", 3), ("Ф1", 12), ("Ф2", 90), ("Ф3", 180), ("Ф4", 300)]
PLANE_COLORS = [[0, 230, 180, 255], [9, 132, 227, 255], [225, 112, 85, 255],
                [253, 203, 110, 255], [108, 92, 231, 255]]


def _roundrobin_order():
    """Порядок КА «по кругу» плоскостям — чтобы первые N были распределены."""
    raan, u0 = _build()
    order = []
    for slot in range(N_PER):
        for p in range(N_PLANE):
            order.append(p * N_PER + slot)
    return order, raan, u0


def _satrecs_walker(epoch_dt):
    """300 объектов Satrec (Walker 300/15) через sgp4init — для реальной SGP4-пропагации."""
    from sgp4.api import Satrec, WGS72, jday
    jd, fr = jday(epoch_dt.year, epoch_dt.month, epoch_dt.day,
                  epoch_dt.hour, epoch_dt.minute, epoch_dt.second)
    epoch_days = (jd + fr) - 2433281.5          # дни от 1949-12-31 00:00 UT
    n_kozai = N_MEAN * 60.0                       # среднее движение, рад/мин
    sats = []
    for p in range(N_PLANE):
        raan = 2 * math.pi * p / N_PLANE
        for k in range(N_PER):
            mo = (2 * math.pi * k / N_PER + 2 * math.pi * p / N_SAT) % (2 * math.pi)
            s = Satrec()
            # whichconst, opsmode, satnum, epoch, bstar, ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo
            s.sgp4init(WGS72, 'i', p * N_PER + k + 1, epoch_days,
                       1e-4, 0.0, 0.0, 1e-4, 0.0, INCL, mo, n_kozai, raan)
            sats.append(s)
    return sats, jd, fr


def _build_czml_sgp4(model_uri: str, n_orbits=2.0, step_s=60.0, speed=60, model_every=6):
    """CZML на реальной SGP4-пропагации (J2/J3/J4 + торможение) + надирная ориентация."""
    import datetime
    from sgp4.api import SatrecArray
    order, _, _ = _roundrobin_order()
    modeled = {rank for rank in range(N_SAT) if rank % model_every == 0}
    epoch_dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    sats, jd0, fr0 = _satrecs_walker(epoch_dt)

    times = np.arange(0, n_orbits * T_ORB + step_s, step_s)
    jds = np.full(len(times), jd0, dtype=float)
    frs = fr0 + times / 86400.0
    e, r, _v = SatrecArray(sats).sgp4(jds, frs)   # r: (N_SAT, n_times, 3) TEME, км
    if int((e != 0).sum()) > 0:                    # сбой пропагации → фолбэк на аналитику (run_web_app)
        raise RuntimeError(f"SGP4: {int((e != 0).any(axis=1).sum())} КА с ошибкой пропагации")

    iso = epoch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (epoch_dt + datetime.timedelta(seconds=float(times[-1]))).strftime("%Y-%m-%dT%H:%M:%SZ")
    avail = f"{iso}/{end}"

    # TEME→ECEF: поворот на GMST вокруг Z, векторно по всем КА и эпохам (как _ecef_m, км→м)
    gmst = np.array([_gmst_rad(float(jds[ti]), float(frs[ti])) for ti in range(len(times))])
    cg, sg = np.cos(gmst)[None, :], np.sin(gmst)[None, :]
    xt, yt, zt = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    ecef = np.stack([(xt * cg + yt * sg) * 1000.0,
                     (-sg * xt + cg * yt) * 1000.0,
                     zt * 1000.0], axis=2)

    pos = {si: [] for si in range(N_SAT)}
    ori = {rank: [] for rank in modeled}; prev = {rank: None for rank in modeled}
    for ti, t in enumerate(times):
        for si in range(N_SAT):
            pos[si].extend([round(float(t), 1)] + [float(x) for x in ecef[si, ti]])
        for rank in modeled:
            si = order[rank]
            ti2 = min(ti + 1, len(times) - 1)
            v = ecef[si, ti2] - ecef[si, ti]
            if np.linalg.norm(v) < 1e-6:
                v = ecef[si, ti] - ecef[si, max(ti - 1, 0)]
            q = _nadir_quat(ecef[si, ti], v, prev[rank]); prev[rank] = q
            ori[rank].extend([round(float(t), 1)] + [float(c) for c in q])

    packets = [{"id": "document", "name": "AURORA (SGP4)", "version": "1.0",
                "clock": {"interval": avail, "currentTime": iso, "multiplier": speed,
                          "range": "LOOP_STOP", "step": "SYSTEM_CLOCK_MULTIPLIER"}}]
    for rank, si in enumerate(order):
        plane = si // N_PER
        col = PLANE_COLORS[plane % len(PLANE_COLORS)]
        pkt = {"id": f"sat_{rank}", "name": f"АВРОРА КА-{rank+1:03d} | плоскость {plane+1}",
               "availability": avail,
               "position": {"interpolationAlgorithm": "LAGRANGE", "interpolationDegree": 5,
                            "referenceFrame": "FIXED", "epoch": iso, "cartesian": pos[si]},
               "point": {"color": {"rgba": col}, "pixelSize": 7,
                         "outlineColor": {"rgba": [255, 255, 255, 170]}, "outlineWidth": 1}}
        if rank in modeled:
            pkt["model"] = {"gltf": model_uri, "scale": 1.0, "minimumPixelSize": 46,
                            "maximumScale": 250000, "runAnimations": False}
            pkt["orientation"] = {"interpolationAlgorithm": "LINEAR", "epoch": iso,
                                  "unitQuaternion": ori[rank]}
        packets.append(pkt)
    return packets


def _build_czml(model_uri: str, n_orbits=2.0, step_s=60.0, speed=60, model_every=6):
    import datetime
    order, raan, u0 = _roundrobin_order()
    modeled = {rank for rank in range(N_SAT) if rank % model_every == 0}
    times = np.arange(0, n_orbits * T_ORB + step_s, step_s)
    epoch = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
    end = (epoch + datetime.timedelta(seconds=float(times[-1]))).strftime("%Y-%m-%dT%H:%M:%SZ")
    avail = f"2000-01-01T00:00:00Z/{end}"

    pos = {si: [] for si in range(N_SAT)}
    ori = {rank: [] for rank in modeled}        # надирная ориентация только для КА с моделью
    prev = {rank: None for rank in modeled}
    for t in times:
        P = _prop_ecef(raan, u0, t)             # позиции (J2 + вращение Земли)
        V = _prop_ecef(raan, u0, t + 1.0) - P   # направление скорости (для ориентации)
        for si in range(N_SAT):
            pos[si].extend([round(float(t), 1), float(P[si, 0]), float(P[si, 1]), float(P[si, 2])])
        for rank in modeled:
            si = order[rank]
            q = _nadir_quat(P[si], V[si], prev[rank]); prev[rank] = q
            ori[rank].extend([round(float(t), 1)] + [float(c) for c in q])

    packets = [{"id": "document", "name": "AURORA", "version": "1.0",
                "clock": {"interval": avail, "currentTime": "2000-01-01T00:00:00Z",
                          "multiplier": speed, "range": "LOOP_STOP",
                          "step": "SYSTEM_CLOCK_MULTIPLIER"}}]
    for rank, si in enumerate(order):
        plane = si // N_PER
        col = PLANE_COLORS[plane % len(PLANE_COLORS)]
        pkt = {"id": f"sat_{rank}", "name": f"АВРОРА КА-{rank+1:03d} | плоскость {plane+1}",
               "availability": avail,
               "position": {"interpolationAlgorithm": "LAGRANGE", "interpolationDegree": 5,
                            "referenceFrame": "FIXED", "epoch": "2000-01-01T00:00:00Z",
                            "cartesian": pos[si]},
               "point": {"color": {"rgba": col}, "pixelSize": 7,
                         "outlineColor": {"rgba": [255, 255, 255, 170]}, "outlineWidth": 1}}
        if rank in modeled:
            pkt["model"] = {"gltf": model_uri, "scale": 1.0, "minimumPixelSize": 46,
                            "maximumScale": 250000, "runAnimations": False}
            pkt["orientation"] = {"interpolationAlgorithm": "LINEAR",
                                  "epoch": "2000-01-01T00:00:00Z", "unitQuaternion": ori[rank]}
        packets.append(pkt)
    return packets


_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>АВРОРА — симулятор группировки</title>
<script src="__CESIUM_JS__"></script>
<link href="__CESIUM_CSS__" rel="stylesheet">
<style>
  html,body,#globe{width:100%;height:100%;margin:0;padding:0;overflow:hidden;
    font-family:'Segoe UI',Roboto,sans-serif;background:#05080f;}
  .glass{position:absolute;background:rgba(10,18,32,0.72);backdrop-filter:blur(8px);
    border:1px solid rgba(95,168,255,0.25);border-radius:14px;color:#e8eef5;
    box-shadow:0 6px 24px rgba(0,0,0,0.45);z-index:50;}
  #hud{top:16px;left:16px;padding:16px 20px;max-width:300px;}
  #hud h1{margin:0 0 2px;font-size:24px;letter-spacing:2px;color:#00d6a4;}
  #hud .sub{font-size:12px;color:#9fb3c8;margin-bottom:12px;}
  #hud .row{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;
    border-bottom:1px solid rgba(255,255,255,0.06);}
  #hud .row b{color:#5fd0b4;}
  #hud .big{font-size:30px;color:#00ffc8;font-weight:700;margin-top:8px;}
  #phases{bottom:22px;left:50%;transform:translateX(-50%);padding:10px 14px;display:flex;gap:8px;align-items:center;}
  #phases span{font-size:12px;color:#9fb3c8;margin-right:4px;}
  .pbtn{background:rgba(95,168,255,0.12);border:1px solid rgba(95,168,255,0.35);
    color:#cfe3fb;border-radius:9px;padding:7px 13px;font-size:13px;cursor:pointer;transition:.15s;}
  .pbtn:hover{background:rgba(95,168,255,0.3);}
  .pbtn.active{background:#00d6a4;color:#05080f;border-color:#00d6a4;font-weight:700;}
  #legend{bottom:22px;right:16px;padding:12px 16px;font-size:12px;}
  #legend .li{display:flex;align-items:center;gap:8px;padding:2px 0;}
  #legend .dot{width:11px;height:11px;border-radius:50%;}
  #brand{position:absolute;top:18px;right:18px;font-size:12px;color:#6c8198;z-index:50;text-align:right;}
  .cesium-viewer-bottom{display:none;}
</style>
</head>
<body>
<div id="globe"></div>
<div id="hud" class="glass">
  <h1>АВРОРА</h1>
  <div class="sub">Низкоорбитальная PNT · ШИВА НЕТВОРК</div>
  <div class="row"><span>Орбита</span><b>1000 км</b></div>
  <div class="row"><span>Наклонение</span><b>75°</b></div>
  <div class="row"><span>Конфигурация</span><b>Walker 300/15</b></div>
  <div class="row"><span>Фаза</span><b id="phName">Ф4 — FOC</b></div>
  <div class="row"><span>КА на орбите</span><b id="satN">300</b></div>
  <div class="big" id="bigN">300 КА</div>
</div>
<div id="brand">Симулятор группировки<br>НИР «Сияние» · СИЯНИЕ-ТП-001</div>
<div id="phases" class="glass">
  <span>Фаза:</span>
  <button class="pbtn" data-n="3"   data-l="Ф0 — демо">Ф0 · 3</button>
  <button class="pbtn" data-n="12"  data-l="Ф1 — демо">Ф1 · 12</button>
  <button class="pbtn" data-n="90"  data-l="Ф2 — РФ 82%">Ф2 · 90</button>
  <button class="pbtn" data-n="180" data-l="Ф3 — РФ 100%">Ф3 · 180</button>
  <button class="pbtn active" data-n="300" data-l="Ф4 — FOC">Ф4 · 300</button>
</div>
<div id="legend" class="glass">
  <div style="color:#9fb3c8;margin-bottom:5px;">Орбитальные плоскости</div>
  <div class="li"><span class="dot" style="background:#00e6b4"></span>группа 1</div>
  <div class="li"><span class="dot" style="background:#0984e3"></span>группа 2</div>
  <div class="li"><span class="dot" style="background:#e17055"></span>группа 3</div>
</div>
<script>
var viewer = new Cesium.Viewer('globe', {
  animation:true, timeline:true, geocoder:false, homeButton:true,
  sceneModePicker:false, baseLayerPicker:false, navigationHelpButton:false,
  infoBox:true, selectionIndicator:true, fullscreenButton:true,
  imageryProvider:false, creditContainer:document.createElement('div')
});
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#05080f');
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#0d2040');
__EARTH_JS__

var ds = new Cesium.CzmlDataSource();
ds.load(__CZML__).then(function(){
  viewer.dataSources.add(ds);
  viewer.camera.setView({destination: Cesium.Cartesian3.fromDegrees(70.0, 50.0, 22000000)});
  setPhase(300, 'Ф4 — FOC');
});
viewer.clock.multiplier = 60; viewer.clock.shouldAnimate = true;

function setPhase(n, label){
  var ents = ds.entities.values;
  for (var i=0;i<ents.length;i++){
    var m = /sat_(\d+)/.exec(ents[i].id);
    if(m){ ents[i].show = (parseInt(m[1]) < n); }
  }
  document.getElementById('satN').textContent = n;
  document.getElementById('bigN').textContent = n + ' КА';
  if(label) document.getElementById('phName').textContent = label;
}
document.querySelectorAll('.pbtn').forEach(function(b){
  b.onclick=function(){
    document.querySelectorAll('.pbtn').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    setPhase(parseInt(b.dataset.n), b.dataset.l);
  };
});
</script>
</body>
</html>"""


def run_web_app(output_dir: str = "web", label: str = "phase4") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    uri = gltf_data_uri()
    try:
        czml = _build_czml_sgp4(uri); prop = "SGP4 (реальная, J2/J3/J4 + торможение)"
    except Exception as exc:
        czml = _build_czml(uri); prop = f"аналитич.+J2 (SGP4 недоступен: {exc})"
    czml_json = json.dumps(czml, separators=(",", ":"))

    # Пути к Cesium: локально (assets/cesium) относительно web/, иначе CDN
    if cesium_pnt.is_cesium_local():
        js = "../assets/cesium/Build/Cesium/Cesium.js"
        css = "../assets/cesium/Build/Cesium/Widgets/widgets.css"
        mode = "offline (локальный Cesium)"
    else:
        js = f"{cesium_pnt._CDN_BASE}/Cesium.js"
        css = f"{cesium_pnt._CDN_BASE}/Widgets/widgets.css"
        mode = "CDN"

    earth = cesium_pnt._earth_data_url()
    if earth:
        earth_js = ("Cesium.SingleTileImageryProvider.fromUrl('%s',{rectangle:"
                    "Cesium.Rectangle.fromDegrees(-180,-90,180,90)}).then(function(p){"
                    "viewer.imageryLayers.removeAll();viewer.imageryLayers.addImageryProvider(p);});" % earth)
    else:
        earth_js = "// нет встроенной текстуры Земли — однотонный глобус"

    html = (_HTML.replace("__CESIUM_JS__", js).replace("__CESIUM_CSS__", css)
            .replace("__EARTH_JS__", earth_js).replace("__CZML__", czml_json))
    path = os.path.join(output_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    # README
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Веб-симулятор группировки АВРОРА\n\n"
                "Откройте `index.html` в браузере. 3D-глобус, орбиты Walker 300/15, "
                "модель КА, переключение фаз развёртывания (Ф0…Ф4).\n\n"
                "Офлайн-режим требует локального Cesium — `aurora-pnt download-cesium` "
                "(каталог `assets/cesium/`). Иначе грузится с CDN (нужен интернет).\n\n"
                "Пересборка: `python -m aurora.pnt.cli web-app`.\n")
    return {"html": path, "mode": mode, "prop": prop, "size_kb": os.path.getsize(path) // 1024,
            "n_sat": N_SAT, "n_models": sum(1 for p in czml if "model" in p)}


def print_web_app_summary(label: str, r: Dict) -> None:
    print(f"\n  Web-симулятор АВРОРА -- {label}")
    print(f"    Пропагация: {r['prop']}")
    print(f"    Режим Cesium: {r['mode']}")
    print(f"    КА: {r['n_sat']} (с 3D-моделью {r['n_models']}); HTML {r['size_kb']} КБ")
    print(f"    Открыть: {r['html']}")
