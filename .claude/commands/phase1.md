Execute Phase 1 from CLAUDE.md (Continuous Perez Transposition Engine).
Order of work:
1. core/transposition/perez_discrete.py — standard Perez-Ineichen, validated
   against pvlib.irradiance.perez to within 1e-9 on synthetic + TMY inputs.
2. core/transposition/perez_continuous.py — mean-preserving cubic splines over
   clearness epsilon for all F coefficients; numerically verify C1 continuity.
3. core/transposition/reverse.py — POA→GHI reverse transposition via brentq.
4. validation/phase1_perez.py — annual POA comparison plots (discrete vs
   continuous vs pvlib) for 3 Ghana TMY datasets; save PNGs to validation/output/.
Stop and show me the plots + deviation table before starting Phase 2.
