"""lorenz63 — reproducible numerics for the Lorenz (1963) system.

Companion code for a course report arguing that the standard for trusting a
numerical result depends on the kind of system: conservative systems are
validated by invariants, chaotic ones only by statistics.
"""

from __future__ import annotations

from lorenz63.system import (
    RHO_HOMOCLINIC,
    LorenzParams,
    divergence,
    fixed_point_eigenvalues,
    fixed_points,
    hopf_rho,
    jacobian,
    rhs,
)

__version__ = "0.1.0"

__all__ = [
    "RHO_HOMOCLINIC",
    "LorenzParams",
    "__version__",
    "divergence",
    "fixed_point_eigenvalues",
    "fixed_points",
    "hopf_rho",
    "jacobian",
    "rhs",
]
