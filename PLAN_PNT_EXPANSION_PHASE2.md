# План: Расширение моделирования AURORA PNT — Фаза 2

## Контекст

После Фазы 1 система имеет 56 модулей. Выявлены 4 крупных пробела для полного
технического проекта уровня системного предприятия. Цель Фазы 2 — 10 новых
модулей, симуляции, разделы техпроекта §47–§56, регенерация GOST DOCX.

Базовые параметры: Walker Delta 300/15/1, h=1000 км, i=75°, L1 BOC(1,1)+TMBOC,
L5 BPSK(10). Опора на уже исправленные бюджеты (§10 C/N₀=52,6 дБ-Гц, §40 ΔV).

---

## Блок A — Сквозная PVT-симуляция

### A1. `aurora/pnt/pvt_montecarlo.py` ★★★ КРИТИЧЕСКИЙ
**Считает:** End-to-end Monte-Carlo PVT. Связывает все бюджеты ошибок
(часы §8, эфемериды §37, ионосфера §33, тропосфера §34, многолучёвость,
шум приёмника, релятив. остаток §33) → CDF горизонт./верт. ошибки.
- UERE = RSS всех компонент по спутнику
- Position error = HDOP·σ_UERE (с реальной геометрией созвездия)
- N=10⁴ розыгрышей, разбивка вклада источников (tornado-диаграмма)
**Формулы:** σ_UERE² = Σσ_i²; CEP95 = 2,08·σ_H; ε_V = VDOP·σ_UERE
**Выход:** `results/pvt_montecarlo/`: pvt_error_cdf, uere_tornado,
pvt_vs_elevation_mask, pvt_global_box — 4 PNG + CSV
**CLI:** `aurora-pnt pvt-montecarlo -o results/pvt_montecarlo -l phase5`

### A2. `aurora/pnt/dop_temporal.py` ★★★ КРИТИЧЕСКИЙ
**Считает:** Временны́е карты DOP/доступности. Симуляция созвездия 24 ч,
сетка GDOP/PDOP/HDOP/VDOP по миру, % доступности vs угол маски (5/10/15°),
статистика навигационных «провалов» (outage) за сутки/год.
**Формулы:** DOP из (HᵀH)⁻¹, H — матрица направляющих косинусов;
availability = P(PDOP < порог & N_sat ≥ 4)
**Выход:** `results/dop_temporal/`: dop_world_map, availability_vs_mask,
dop_timeseries_msk, nsat_histogram — 4 PNG + CSV
**CLI:** `aurora-pnt dop-temporal -o results/dop_temporal -l phase5`

---

## Блок B — Точное определение орбиты (POD)

### B1. `aurora/pnt/pod_filter.py` ★★★ КРИТИЧЕСКИЙ
**Считает:** Batch-LSQ / EKF определение орбиты. Силовые модели:
J2–J6, лунно-солнечные, SRP (cannonball), атм. торможение на 1000 км.
Наблюдаемость от сети 21 станции, достижимая точность орбиты
(radial/along/cross), остатки валидации SLR.
**Формулы:** ẍ = −μr/r³ + a_J2..J6 + a_lunisol + a_SRP + a_drag;
ковариация P из (HᵀWH)⁻¹; SISRE = f(R, A, C, clk)
**Выход:** `results/pod/`: pod_force_budget, pod_accuracy_rac,
pod_slr_residuals, pod_observability — 4 PNG + CSV
**CLI:** `aurora-pnt pod -o results/pod -l phase5`

### B2. `aurora/pnt/autonav_isl.py` ★★ ВЫСОКИЙ
**Считает:** Автономная навигация по межспутниковым измерениям (Ka ISL).
Кросслинк-эфемериды без наземного сегмента, проблема вырождения ранга
(созвездие как жёсткое тело — неопределённость ориентации/вращения),
рост ошибки эфемерид со временем без наземного контакта (60/90/180 сут).
**Формулы:** Δρ_ij = |r_i − r_j| + cΔδt; rank-deficiency rotation;
ε_eph(t) ≈ ε₀ + k·t (drift без ground anchor)
**Выход:** `results/autonav/`: autonav_eph_growth, autonav_rank,
autonav_isl_geometry, autonav_vs_ground — 4 PNG + CSV
**CLI:** `aurora-pnt autonav -o results/autonav -l phase5`

---

## Блок C — Целостность и SBAS

### C1. `aurora/pnt/araim.py` ★★★ КРИТИЧЕСКИЙ
**Считает:** ARAIM. Мульти-гипотезное разделение решений (solution
separation), HPL/VPL, дерево распределения integrity risk, бюджет P_HMI.
**Формулы:** VPL = max_k(|Δ_k| + K_md·σ_ss,k); P_HMI ≤ Σ P_fault·P_md;
solution separation Δ_k между all-in-view и fault-excluded
**Выход:** `results/araim/`: araim_vpl_hpl, araim_risk_tree,
araim_isolation, araim_vpl_timeseries — 4 PNG + CSV
**CLI:** `aurora-pnt araim -o results/araim -l phase5`

### C2. `aurora/pnt/integrity_budget.py` ★★ ВЫСОКИЙ
**Считает:** Диаграмма Стэнфорда, защитные уровни vs alert limit,
continuity/availability для LPV-200 и CAT-I, параметры ISM
(Integrity Support Message: b_nom, σ_URA, P_sat, P_const).
**Формулы:** Stanford (PL vs PE vs AL); availability LPV-200 =
P(VPL<35 м & HPL<40 м); continuity risk budget
**Выход:** `results/integrity/`: stanford_diagram, pl_vs_al,
availability_lpv200_map, ism_parameters — 4 PNG + CSV
**CLI:** `aurora-pnt integrity -o results/integrity -l phase5`

---

## Блок D — Программно-экономический блок

### D1. `aurora/pnt/cost_model.py` ★★★ КРИТИЧЕСКИЙ
**Считает:** CAPEX/OPEX, стоимость КА (recurring/non-recurring),
запуск (rideshare vs выделенный), наземный сегмент, LCC 7–15 лет,
стоимость на пользователя. Кривая обучения для 300 КА.
**Формулы:** LCC = NRE + 300·T1·n^(log₂ b) + C_launch + C_ground +
OPEX·N_years; learning curve b≈0,9
**Выход:** `results/cost/`: cost_breakdown, learning_curve,
lcc_vs_years, cost_sensitivity — 4 PNG + CSV
**CLI:** `aurora-pnt cost -o results/cost -l phase5`

### D2. `aurora/pnt/production_ait.py` ★★ СРЕДНИЙ
**Считает:** Темп производства 300 КА, поток AIT, кривая обучения,
пропускная способность завода, график развёртывания.
**Формулы:** такт = T_total/300; throughput = N_lines/такт;
schedule с буферами
**Выход:** `results/production/`: ait_flow, production_rate,
factory_throughput, deployment_schedule — 4 PNG + CSV
**CLI:** `aurora-pnt production -o results/production -l phase5`

### D3. `aurora/pnt/launch_campaign.py` ★★ СРЕДНИЙ
**Считает:** Схема выведения, диспенсер/rideshare, число пусков,
ΔV довыведения, таймлайн заполнения Walker Delta 15 плоскостей.
**Формулы:** N_launch = ceil(300/N_per_LV); ΔV_raise (Хоманн
park→1000 км); RAAN-фазирование через дифф. J2-дрейф
**Выход:** `results/launch/`: launch_manifest, raan_phasing,
deployment_timeline, dv_raise_budget — 4 PNG + CSV
**CLI:** `aurora-pnt launch-campaign -o results/launch -l phase5`

### D4. `aurora/pnt/ground_segment.py` ★ СРЕДНИЙ
**Считает:** Архитектура MCS/TT&C, поток данных, задержки,
резервирование, кибербезопасность наземного сегмента.
**Формулы:** latency budget (TT&C↔MCS↔upload); доступность
MCS = 1−(1−R)ⁿ (резерв); пропускная TT&C для 300 КА
**Выход:** `results/ground_segment/`: gs_architecture,
gs_data_flow, gs_latency, gs_redundancy — 3 PNG + CSV
**CLI:** `aurora-pnt ground-segment -o results/ground_segment -l phase5`

---

## Фаза 2.1 — Регистрация в CLI
`aurora/pnt/cli.py`: 10 новых подкоманд (cmd_pvt_montecarlo … cmd_ground_segment).

## Фаза 2.2 — Запуск симуляций
Все 10 команд `aurora-pnt … -l phase5`.

## Фаза 2.3 — Техпроект
`AURORA_PNT_Technical_Project.md` — новые разделы перед Выводами:

| Раздел | Источник |
|---|---|
| §47 Сквозной бюджет PVT (Monte-Carlo) | pvt_montecarlo.py |
| §48 Временна́я доступность и DOP | dop_temporal.py |
| §49 Точное определение орбиты (POD) | pod_filter.py |
| §50 Автономная навигация (ISL AutoNav) | autonav_isl.py |
| §51 ARAIM и защитные уровни | araim.py |
| §52 Бюджет целостности (LPV-200/CAT-I) | integrity_budget.py |
| §53 Стоимостная модель (LCC) | cost_model.py |
| §54 Производство и AIT 300 КА | production_ait.py |
| §55 Кампания выведения | launch_campaign.py |
| §56 Наземный сегмент (MCS/TT&C) | ground_segment.py |

§45 Выводы / §46 Литература → §57 / §58.

## Фаза 2.4 — Регенерация DOCX
`python convert_gost.py` → обновлённый AURORA_PNT_GOST.docx.

---

## Верификация
- Каждый модуль запускается через CLI без ошибок
- В каждой `results/*/`: ≥3 PNG + 1 CSV
- MD: 58 разделов, ссылки на новые графики валидны
- Физ. правдоподобие:
  - PVT гориз. CEP95: 0,1–0,3 м (PPP-RTK), 1–3 м (одночастотн.)
  - Доступность PDOP<6: >99,5 % при маске 10°
  - POD radial: 2–5 см (с SLR), 10–20 см (только сеть)
  - Рост эфемерид AutoNav: ~1 м/сут без ground anchor
  - VPL LPV-200: <35 м при >99 % времени
  - LCC 300 КА: оценка с разбивкой NRE/recurring/launch/OPEX
```
```
