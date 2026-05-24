# Changelog — AURORA PNT

All notable changes to AURORA PNT are documented here.

---

## [1.4.0] — 2026-05-24

### Прототипирование сигнала
- **`code_gen`** (`aurora-pnt code-gen`) — генератор кодов:
  - Legendre + Weil-перестановка для GPS L1C (n = 10 223)
  - Gold C/A на m-регистрах (n = 1023)
  - Extended Memory L5 (n = 10 230, прототип)
  - Численная проверка: АКФ Weil −31,2 дБ (vs Gold −20,8 дБ); XCorr подтверждены
- **`sdr_receiver`** (`aurora-pnt sdr-receiver`) — софтверный приёмник:
  - Захват: 2D-корреляция (FFT) Доплер × фаза кода; PSR 11,4 при бюджете
  - Слежение: Costas PLL (BW 1,4 Гц) + EML DLL (BW 0,5 Гц)
  - TTFF cold start 3,8 с при C/N₀ = 52,6 дБ-Гц (бюджет AURORA)
  - TTFF vs C/N₀ от 30 до 55 дБ-Гц (10 точек, 5 прогонов)

### Документация
- **PDF-версия** технического проекта (13,4 МБ) сгенерирована из DOCX
- README дополнен ссылками на технический проект, PDF, Cesium-демо
- §6.9 и §42.3 техпроекта — спецификации прототипов code_gen и SDR

---

## [1.3.0] — 2026-05-21

### Программные разделы технического проекта
- **§55 SRD** — 46 системных требований (MIS/FUN/SAT/GRN/EXT)
- **§56 CONOPS** — 8 режимов КА, state-машина, 6 сценариев
- **§57 Реестр рисков** — 25 рисков по 6 категориям, P×S матрица
- **§58 План работ** — 13 фаз, 7 вех 2026–2043 (M6: 300 КА к янв 2036)
- **§59 V&V план** — методы верификации, матрица для ключевых СТ
- **§60 Кибербезопасность** — 20 угроз STRIDE/PASTA, 15 контрмер (−70,7%)
- **§61 ПО** — бортовое ~140 kSLOC + наземное ~160 kSLOC
- **§62 Стандарты** — матрица соответствия 40+ стандартам

### Новые модули моделирования
- `risks` (P×S матрица), `schedule` (Gantt), `cybersec` (угрозы), `e2e` (PVT-симуляция)

### Аудит формул главного конструктора
- N_vis скорректирован: 36 → 11 (среднее) по всему документу
- §51 LCC пересчитан с российскими часами 5 млн ₽: 107 → 101,5 млрд ₽
- Стандарт частоты выделен отдельной строкой 1,5 млрд ₽

---

## [1.2.0] — 2026-05-19

### Фаза 2 расширения моделирования (10 новых модулей)
- `pvt_montecarlo` — Monte-Carlo PVT-бюджет
- `dop_temporal` — карты DOP и доступности
- `pod_filter` — точное определение орбиты
- `autonav_isl` — автономная навигация по ISL
- `araim` — ARAIM защитные уровни
- `integrity_budget` — целостность LPV-200/CAT-I
- `cost_model` — LCC модель
- `production_ait` — производство и AIT
- `launch_campaign` — кампания выведения
- `ground_segment` — наземный сегмент

### Аудит формул
- FSPL §10 правильное значение 156,4 дБ (не 147,4)
- Merit Factor Weil §6.6: 1703 → 6,1 (теоретический предел)
- σ_t часов §8.2: исправлен двойной множитель τ
- ΔV деорбита §22: 7,4 → 120 м/с (Хоманн)
- EPFD §43.4.2: правильная формула, −124 дБВт/м²

---

## [1.1.0] — 2026-05-17

### New analysis modules
- **User terminal link budget** — `aurora-pnt user-link-budget`
  - L1/L5 C/N₀ vs elevation (10°–90°) for Handheld / Survey / Maritime / Aviation terminals
  - Pseudorange thermal noise σ_PR per signal/terminal combination
  - Link budget waterfall chart (EIRP → FSPL → C/N₀ components)
  - Navigation margin: Handheld on L1 viable from 10° el (C/N₀ = 37.2 dB-Hz, margin +7.2 dB)
- **Eclipse / Earth shadow analysis** — `aurora-pnt eclipse`
  - Worst-case eclipse: 34.9 min (35% of orbit at 1000 km, beta=0°)
  - Eclipse fraction vs Sun beta angle: no eclipse above |beta| > 25.9°
  - Battery sizing: 43.6 Wh required, 54.5 Wh capacity (DOD=80%), 0.3 kg Li-ion
  - OCXO thermal effect: 8 ppb frequency shift (80 K temp swing), thermal oven required
  - Solar power: BOL 964 W, EOL 836 W (7-yr, 2%/yr GaAs degradation)
- **Navigation message structure** — `aurora-pnt nav-message`
  - Bit budget: ephemeris 527b + clock 94b + ionosphere 120b + OSNMA auth 524b + almanac (180 sat) 25920b
  - Frame: 300 bits/subframe, CRC-24Q, 1/2-rate FEC → 250 bps payload at 500 bps
  - TTFF: hot 2 s / warm 11 s / cold 149 s at 500 bps
  - ISL clock correction extension: 22 bits per satellite in clock field

### Bug fixes
- **CesiumJS HTML crash** — removed invalid `terrain: false` option from Viewer constructor
  (in Cesium 1.117, `defined(false) == true` causes `scene.setTerrain(false)` → TypeError)
- Regenerated all 5 phase HTML files (phase0–phase4) with fix applied

### Infrastructure
- Removed `assets/cesium/` from `.gitignore` — CesiumJS and Earth texture now tracked in repo
- `assets/earth_b64.txt`: 1024×512 Natural Earth II texture embedded as base64 (100 KB)

---

## [1.0.0] — 2026-05-14

### Project
- Renamed project to **AURORA PNT** (Advanced Universal Real-time Orbital Radio-navigation Architecture) by **Shiwa Network**
- Package renamed to `aurora`, CLI to `aurora-pnt`
- New repository: https://github.com/ShiwaNetwork/aurora-pnt

### New features
- **AURORA-T timing service protocol** — satellite-to-ground PTP/NTP grandmaster capability
  - `aurora-pnt timing-service`: accuracy budget by clock type, PTP class, protocol stack diagram
  - TESLA MAC authentication (Galileo OSNMA-compatible), 10 s frame interval
  - PTP Class 25 (<25 ns) achievable with Cs master clock
- **Mixed-clock architecture analysis** — `aurora-pnt clock-arch`
  - OCXO/Rb/Cs per satellite tier modeling
  - ISL chain sigma for each tier (Cs anchor -> Rb relay -> OCXO terminal)
  - Holdover time analysis at ISL disruption
  - Hardware budget (mass/power per constellation)
- **Combined LEO+GLONASS simulation** — `aurora-pnt combined`
  - 5-column H matrix with ISB (Inter-System Bias) as 5th unknown
  - Modes: combined (LEO+GLONASS), autonomous (LPT only), glonass (baseline)
  - PDOP p95: 5.04 (autonomous) -> 1.67 (combined)
- **Autonomous LPT time scale** — `aurora-pnt time-scale`
  - Sovereign time scale independent of GLONASS/GPS
  - sigma_ISL = sqrt(N_hops) x ppb x T_sync per clock type
  - UERE budget by mode: autonomous / combined / combined+SDCM
- **GLONASS constellation model** — `aurora/pnt/glonass.py`
  - Walker Delta 3x8, 64.8 deg, 19136 km, 5 deg elevation mask
- **RAIM integrity** — `aurora-pnt raim` (HPL/VPL analysis)
- **Resilience analysis** — `aurora-pnt resilience` (satellite failure sweep)
- **Clock analysis** — `aurora-pnt clock-analysis` (TCXO/Rb/Cs/Maser comparison)

### Simulation results (confirmed by numerical experiments)
- Phase 3 (180 sat, 12x15, 75 deg, 1000 km): 100% Russia coverage, PDOP<6 = 97%
- Phase 4 (300 sat, 15x20, 75 deg, 1000 km): 100% global, PDOP<6 = 97.3%
- Combined Phase 4: PDOP p95 = 1.67, CEP = 1.18 m
- Autonomous Phase 4: PDOP p95 = 5.04, CEP = 6.46 m
- Combined+SDCM: CEP = 0.67 m, Vertical 1-sigma = 1.14 m

### Configuration
- 13 YAML configs for all phases and parametric experiments
- 21 MCS ground stations finalized (Russia + allied countries)

---

## [0.1.1] — 2026 (initial simulation framework)

- LEO PNT simulation engine: Walker-Delta constellation, SGP4 propagation
- Coverage metrics: 4+ satellite visibility, PDOP, HDOP, VDOP
- Link budget: FSPL, Doppler shift, C/N0 for L1/L5 bands
- Network metrics: ISL/GSL topology, Hypatia path stability
- Ranging: UERE budget, position accuracy estimation
- CesiumJS interactive visualization
- Phase 0-4 parametric sweeps: altitude, inclination, satellite count
