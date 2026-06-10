"""
core.transposition — Perez irradiance transposition engines.

Public API
----------
transpose_discrete(...)   : Perez-Ineichen with discrete ε-bins (regression anchor)
transpose_continuous(...) : Perez-Ineichen with mean-preserving cubic splines
reverse_transpose(...)    : POA → GHI via Erbs decomposition + brentq
"""

from core.transposition.perez_continuous import (
    transpose as transpose_continuous,
)
from core.transposition.perez_discrete import (
    transpose as transpose_discrete,
)
from core.transposition.reverse import reverse_transpose

__all__ = [
    "transpose_discrete",
    "transpose_continuous",
    "reverse_transpose",
]
