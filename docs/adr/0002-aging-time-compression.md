# ADR 0002: Annual TMY Time-Compression for 25-Year Arrhenius Degradation

## Status
Accepted

## Context

Full hourly simulation of Arrhenius-driven cell degradation over a 25-year project
life would require 219,600 time steps × n_modules × 72 cells per Monte Carlo draw.
The computational cost is prohibitive for routine yield-loss studies, and the input
data (TMY weather) does not support finer-than-annual inter-year resolution anyway.

The Arrhenius model couples degradation rate to instantaneous cell temperature:

    k(T) = k_ref · exp(−E_a / R · (1/T − 1/T_ref))

where E_a = 45 kJ/mol, R = 8.314 J/(mol·K), T_ref = 298.15 K (25°C), and
k_ref = 1 (dimensionless; rate is 1× at T_ref).

The degradation fraction after an exposure dose D is modelled as:

    f = exp(−k_deg · D)

where k_deg is the annual degradation rate [fraction per equivalent reference year]
and D is the cumulative Arrhenius dose in equivalent reference years.

## Decision

**Annual time-compression via a single representative TMY year.**

1. Simulate one 8760-hour TMY cycle through the 3-node thermal solver (Phase 2)
   to obtain per-module hourly cell temperatures T_c(h, m).

2. Compute the annual Arrhenius dose per module (in equivalent reference hours):

       D_annual(m) = Σ_{h=1}^{8760} k(T_c(h, m))   [dimensionless sum over 8760 h]

   This equals 8760 when all T_c ≡ T_ref, and is larger for hotter climates.
   For a Ghana plant (T_c ≈ 55°C daytime, 28°C night), D_annual ≈ 2.7 × 8760.

3. Normalize by dividing by N_HOURS_PER_YEAR = 8760 to get the annual dose factor:

       d_annual(m) = D_annual(m) / 8760   [equivalent reference years per calendar year]

4. At year y, the per-module degradation factor is:

       f(y, m) = exp(−k_deg · d_annual(m) · y)

5. Per-cell independence: cells within each module share the module-level T_c
   plus a small within-module thermal offset (σ ≈ 0.5 K, from position on the
   module laminate). The offset modifies the Arrhenius dose via a first-order
   linearisation:

       correction(m, c) = exp((E_a/R / T_c_mean(m)²) · δT(m, c))
       d_annual_cell(m, c) = d_annual(m) · correction(m, c)

6. Monotonicity is guaranteed by construction: since k(T) > 0 always,
   d_annual > 0, so f(y) is strictly decreasing for each cell. This satisfies
   the CLAUDE.md acceptance criterion directly without additional checks.

## Assumptions

1. **Annual stationarity**: the TMY year is representative of all 25 years.
   Inter-annual climate variability is ignored; its effect on Arrhenius dose
   is well within the Monte Carlo manufacturing-tolerance uncertainty band.

2. **Degradation does not feed back into temperature**: module efficiency loss
   slightly increases heat absorption, but the Δη over 25 years (≈15% relative)
   shifts T_c by less than 1°C — negligible compared to the weather-driven range.

3. **E_a = 45 kJ/mol applies uniformly**: this is the dominant EVA photo-oxidation
   pathway. Cell-level variation in E_a is outside scope for Phase 3 but can be
   introduced as a per-cell stochastic parameter in a future ADR.

## Compression scheme summary

| Symbol          | Value / Formula                                  | Units                  |
|-----------------|--------------------------------------------------|------------------------|
| E_a             | 45 000                                           | J mol⁻¹                |
| R               | 8.314                                            | J mol⁻¹ K⁻¹            |
| T_ref           | 298.15                                           | K                       |
| k(T)            | exp(−E_a/R · (1/T − 1/T_ref))                  | dimensionless           |
| D_annual(m)     | Σ_h k(T_c(h,m))                                 | equivalent ref. hours   |
| d_annual(m)     | D_annual(m) / 8760                              | equiv. ref. years/year  |
| f(y,m)          | exp(−k_deg · d_annual(m) · y)                   | fraction ∈ (0, 1]       |
| k_deg (default) | 0.005                                            | fraction per equiv. year |

## Consequences

**Positive**
- O(8760 × n_modules) computation once; O(n_years) projection — tractable.
- Spatial rack gradient preserved: corner modules (higher exposure, lower T_c)
  accumulate less Arrhenius dose than shaded center modules.
- Monotonic degradation guaranteed analytically, no post-hoc enforcement needed.
- Deterministic: given the same seed and T_c history, results are bit-for-bit
  reproducible across runs.

**Negative / limitations**
- Inter-annual weather variability is ignored. For stochastic annual dose, draw
  d_annual ~ N(d_ref, σ²) per year (future extension).
- Accelerated end-of-life aging (darkened EVA increases α_cell over time) is not
  modelled; this ADR treats optical properties as constant.
- PVsyst and IEC 61853-3 use different degradation parameterizations; bankability
  comparisons must note this model deviation explicitly.

## Validation target

Degradation trajectory must be monotonically non-increasing per cell across all
25 years (hard test). Year-1 combined mismatch + partial degradation loss should
fall within the 0.3–1.0% plausibility band (acceptance criterion from CLAUDE.md).
