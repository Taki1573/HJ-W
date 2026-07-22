#!/usr/bin/env python3
"""Reproducible numerical experiments for the Wentzell road--field paper.

The program does four things:

1. directly solves the epsilon-scaled Fisher--KPP equation with the Wentzell
   boundary condition and compares its threshold fronts with the exact
   Hamilton--Jacobi (HJ) variational front;
2. computes directional propagation speeds and parameter-response curves;
3. computes symmetric two-road cone fronts using the rigorous folding law;
4. explores aligned and counter-flowing road/field drifts.

The default ``paper`` mode generates every figure, CSV table, JSON diagnostic,
and TeX table used by ``numerical_simulation_replacement.tex``.  All paths are
resolved relative to this file, so the script may be launched from any working
directory.

Numerical conventions
---------------------
The scaled parabolic problem is

    u_t = eps * Delta u - c * u_x + u(1-u)/eps,                  y > 0,
    a eps u_xx - b u_x + u_y = 0,                               y = 0.

The field drift is removed exactly by the Galilean coordinate z=x-c*t.  The
reaction is advanced by its exact logistic flow and diffusion by SSP-RK2 with
centered second differences.  After every stage, the Wentzell row is projected
by solving a strictly diagonally dominant tridiagonal M-matrix.  The boundary
tangential derivative is centered whenever the resulting row is an M-matrix
(as in all paper runs), with a sign-aware upwind fallback outside that regime.
Under the stated CFL restriction the paper scheme is positivity preserving;
assertions stop the run if positivity, the upper bound, boundary residual, or
artificial-boundary margin is violated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.lines import Line2D
from scipy.linalg import solve_banded
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import cKDTree


FIG = ROOT / "figures"
DATA = ROOT / "data"


@dataclass(frozen=True)
class Model:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float


@dataclass
class RDSolution:
    eps: float
    model: Model
    final_time: float
    grid: Grid
    u: np.ndarray
    dt: float
    nsteps: int
    boundary_residual: float
    road_tangential_scheme: str
    wall_seconds: float


def ensure_directories() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Exact Hamilton--Jacobi variational solution
# ---------------------------------------------------------------------------


def objective_derivative(
    s: np.ndarray | float,
    x: np.ndarray | float,
    y: np.ndarray | float,
    t: float,
    model: Model,
) -> np.ndarray:
    """Derivative with respect to s of the strictly convex objective."""
    s = np.asarray(s, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = -x + model.c * t + model.b * s
    d = t + model.a * s
    return (
        (2.0 * model.b * n * d - model.a * n * n) / (4.0 * d * d)
        + (y + s) / (2.0 * t)
    )


def phase_and_optimizer(
    x: np.ndarray | float,
    y: np.ndarray | float,
    t: float,
    model: Model,
    iterations: int = 58,
) -> tuple[np.ndarray, np.ndarray]:
    """Return phi*(x,y,t) and its unique optimizer s*.

    Since f_ss >= 1/(2t), if f_s(0)<0 then the unique root satisfies
    0 < s* <= -2t f_s(0).  Bisection on this certified bracket is therefore
    deterministic, vectorizable, and immune to local-minimum failures.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if model.a <= 0:
        raise ValueError("a must be positive")
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if np.any(y < -1.0e-14):
        raise ValueError("the half-plane phase requires y >= 0")
    y = np.maximum(y, 0.0)
    zero = np.zeros_like(x)
    f0 = objective_derivative(zero, x, y, t, model)
    active = f0 < 0.0
    lo = zero.copy()
    hi = np.where(active, -2.0 * t * f0, 0.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        go_right = active & (objective_derivative(mid, x, y, t, model) < 0.0)
        lo = np.where(go_right, mid, lo)
        hi = np.where(active & ~go_right, mid, hi)
    s = np.where(active, 0.5 * (lo + hi), 0.0)
    n = -x + model.c * t + model.b * s
    phi = n * n / (4.0 * (t + model.a * s)) + (y + s) ** 2 / (4.0 * t)
    return phi, s


def phase(
    x: np.ndarray | float,
    y: np.ndarray | float,
    t: float,
    model: Model,
) -> np.ndarray:
    return phase_and_optimizer(x, y, t, model)[0]


def directional_speed(theta: float, model: Model) -> float:
    """Outer radial speed r with phi*(r e_theta,1)=1.

    Scans in this program keep the origin inside the invaded Wulff set.  The
    routine nevertheless detects a violated assumption instead of returning a
    misleading root.
    """
    return float(directional_speed_array(np.asarray([theta]), model)[0])


def directional_speed_array(theta: np.ndarray, model: Model) -> np.ndarray:
    """Vectorized outer radial roots of phi*(r e_theta,1)=1."""
    theta = np.asarray(theta, float)
    shape = theta.shape
    flat = theta.ravel()
    ct, st = np.cos(flat), np.sin(flat)
    if float(phase(0.0, 0.0, 1.0, model)) > 1.0 + 1.0e-10:
        raise ValueError(
            "origin is outside the unit-time invaded set; radial speed is not "
            "single-valued for this parameter set"
        )
    lo = np.zeros_like(flat)
    hi = np.full_like(flat, 2.0)
    outside = phase(hi * ct, hi * st, 1.0, model) <= 1.0
    for _ in range(20):
        if not np.any(outside):
            break
        hi = np.where(outside, 2.0 * hi, hi)
        outside = phase(hi * ct, hi * st, 1.0, model) <= 1.0
    if np.any(outside):
        raise RuntimeError("failed to bracket one or more directional speeds")
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        inside = phase(mid * ct, mid * st, 1.0, model) <= 1.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return (0.5 * (lo + hi)).reshape(shape)


def field_radial_speed(theta: float | np.ndarray, c: float) -> np.ndarray:
    """Radial intercept of the translated field-only disk when it exists."""
    theta = np.asarray(theta, float)
    disc = 4.0 - c * c * np.sin(theta) ** 2
    out = c * np.cos(theta) + np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0.0) & (out >= 0.0), out, np.nan)


def road_endpoint_formula(model: Model) -> tuple[float, float]:
    """Return the right endpoint and the positive magnitude of the left endpoint."""
    def rad(s: float) -> float:
        return math.sqrt(max((1.0 + model.a * s) * (4.0 - s * s), 0.0))

    right = minimize_scalar(
        lambda s: -(model.b * s + rad(s)), bounds=(0.0, 2.0), method="bounded",
        options={"xatol": 1.0e-13},
    )
    left = minimize_scalar(
        lambda s: model.b * s - rad(s), bounds=(0.0, 2.0), method="bounded",
        options={"xatol": 1.0e-13},
    )
    x_plus = model.c - float(right.fun)
    x_minus = model.c + float(left.fun)
    return x_plus, -x_minus


def vertical_speed_formula(model: Model) -> float:
    """Independent scalar reduction for the x=0 radial intercept."""
    s_grid = np.linspace(0.0, 2.0, 20001)
    radicand = 4.0 - (model.c + model.b * s_grid) ** 2 / (1.0 + model.a * s_grid)
    values = np.where(radicand >= 0.0, -s_grid + np.sqrt(np.maximum(radicand, 0.0)), -np.inf)
    k = int(np.argmax(values))
    lo = s_grid[max(0, k - 2)]
    hi = s_grid[min(s_grid.size - 1, k + 2)]

    def negative_value(s: float) -> float:
        q = 4.0 - (model.c + model.b * s) ** 2 / (1.0 + model.a * s)
        if q < 0.0:
            return 1.0e6 + abs(q)
        return -max(-s + math.sqrt(q), 0.0)

    result = minimize_scalar(negative_value, bounds=(lo, hi), method="bounded")
    return max(-float(result.fun), float(max(values[k], 0.0)))


# ---------------------------------------------------------------------------
# Scaled reaction--diffusion solver
# ---------------------------------------------------------------------------


def make_grid(xlim: tuple[float, float], ymax: float, h: float) -> Grid:
    """Uniform grid anchored at x=0 and y=0; requested bounds are contained."""
    if not (xlim[0] < 0.0 < xlim[1] and ymax > 0.0 and h > 0.0):
        raise ValueError("invalid grid specification")
    nl = int(math.ceil(-xlim[0] / h))
    nr = int(math.ceil(xlim[1] / h))
    ny = int(math.ceil(ymax / h))
    x = np.arange(-nl, nr + 1, dtype=float) * h
    y = np.arange(0, ny + 1, dtype=float) * h
    return Grid(x=x, y=y, dx=h, dy=h)


class WentzellProjector:
    """Monotone projection of the algebraic Wentzell boundary row.

    A centered tangential derivative is used whenever its tridiagonal matrix
    remains an M-matrix.  Otherwise the code falls back to sign-aware upwinding.
    All production runs satisfy the centered-stencil condition.
    """

    def __init__(self, grid: Grid, eps: float, model: Model):
        n = grid.x.size - 2
        ax = model.a * eps / grid.dx**2
        centered_ok = ax + 1.0e-15 >= abs(model.b) / (2.0 * grid.dx)
        if centered_ok:
            lower = -ax - model.b / (2.0 * grid.dx)
            upper = -ax + model.b / (2.0 * grid.dx)
            diag = 2.0 * ax + 1.0 / grid.dy
            self.tangential_scheme = "centered"
        else:
            bplus = max(model.b, 0.0)
            bminus = max(-model.b, 0.0)
            lower = -ax - bplus / grid.dx
            upper = -ax - bminus / grid.dx
            diag = 2.0 * ax + abs(model.b) / grid.dx + 1.0 / grid.dy
            self.tangential_scheme = "upwind"
        ab = np.zeros((3, n), dtype=float)
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        self.ab = ab
        self.grid = grid
        self.eps = eps
        self.model = model

    def apply(self, u: np.ndarray) -> None:
        u[0, 1:-1] = solve_banded(
            (1, 1),
            self.ab,
            u[1, 1:-1] / self.grid.dy,
            overwrite_ab=False,
            overwrite_b=False,
            check_finite=False,
        )
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = 0.0

    def residual_inf(self, u: np.ndarray) -> float:
        dx, dy = self.grid.dx, self.grid.dy
        u0 = u[0]
        dxx = (u0[2:] - 2.0 * u0[1:-1] + u0[:-2]) / dx**2
        if self.tangential_scheme == "centered":
            dxu = (u0[2:] - u0[:-2]) / (2.0 * dx)
        elif self.model.b >= 0.0:
            dxu = (u0[1:-1] - u0[:-2]) / dx
        else:
            dxu = (u0[2:] - u0[1:-1]) / dx
        residual = self.model.a * self.eps * dxx - self.model.b * dxu
        residual += (u[1, 1:-1] - u0[1:-1]) / dy
        return float(np.max(np.abs(residual)))


def shrinking_seed(grid: Grid, eps: float) -> np.ndarray:
    """Compact C1-like seed U0(x/eps,y/eps) with O(eps) support.

    The fixed rescaled radius R0=4 supplies enough initial mass for all three
    diagnostic thresholds at the finite times used below.  Its physical radius
    is still 4*eps and therefore collapses to the HJ point source.
    """
    xx, yy = np.meshgrid(grid.x, grid.y)
    rho = np.hypot(xx, yy) / (4.0 * eps)
    u = np.zeros_like(rho)
    u[rho <= 0.5] = 1.0
    transition = (rho > 0.5) & (rho < 1.0)
    u[transition] = 0.5 * (
        1.0 + np.cos(2.0 * np.pi * (rho[transition] - 0.5))
    )
    return u


def logistic_flow(u: np.ndarray, tau: float, eps: float) -> np.ndarray:
    z = math.exp(-tau / eps)
    return u / (u + (1.0 - u) * z)


def transport_diffusion_rhs(
    u: np.ndarray, grid: Grid, eps: float, c: float
) -> np.ndarray:
    q = u[1:-1, 1:-1]
    lap = (u[1:-1, 2:] - 2.0 * q + u[1:-1, :-2]) / grid.dx**2
    lap += (u[2:, 1:-1] - 2.0 * q + u[:-2, 1:-1]) / grid.dy**2
    if c >= 0.0:
        ux = (q - u[1:-1, :-2]) / grid.dx
    else:
        ux = (u[1:-1, 2:] - q) / grid.dx
    rhs = np.zeros_like(u)
    rhs[1:-1, 1:-1] = eps * lap - c * ux
    return rhs


def solve_scaled_rd(
    eps: float,
    model: Model,
    final_time: float,
    h: float,
    xlim: tuple[float, float],
    ymax: float,
    cfl: float = 0.72,
    progress: bool = True,
) -> RDSolution:
    """Solve the epsilon-scaled Fisher--KPP/Wentzell problem."""
    tic = time.perf_counter()
    # Work in the exact Galilean coordinate z=x-c t.  This removes field
    # advection analytically instead of approximating it by an upwind operator;
    # the Wentzell law is unchanged because the road and field share the same
    # coordinate translation.  At the final time the grid is shifted back to x.
    computational_grid = make_grid(xlim, ymax, h)
    projector = WentzellProjector(computational_grid, eps, model)
    u = shrinking_seed(computational_grid, eps)
    projector.apply(u)
    dt_cfl = cfl / (
        2.0 * eps / computational_grid.dx**2
        + 2.0 * eps / computational_grid.dy**2
    )
    nsteps = int(math.ceil(final_time / dt_cfl))
    dt = final_time / nsteps
    for _ in range(nsteps):
        u[1:-1, 1:-1] = logistic_flow(u[1:-1, 1:-1], 0.5 * dt, eps)
        projector.apply(u)

        # SSP-RK2 for the transport--diffusion subflow.
        w = u + dt * transport_diffusion_rhs(u, computational_grid, eps, 0.0)
        projector.apply(w)
        z = w + dt * transport_diffusion_rhs(w, computational_grid, eps, 0.0)
        projector.apply(z)
        u = 0.5 * u + 0.5 * z
        projector.apply(u)

        u[1:-1, 1:-1] = logistic_flow(u[1:-1, 1:-1], 0.5 * dt, eps)
        projector.apply(u)
        if float(np.min(u)) < -2.0e-12 or float(np.max(u)) > 1.0 + 2.0e-12:
            raise RuntimeError(
                f"discrete maximum principle failed: [{u.min()}, {u.max()}]"
            )
    u = np.clip(u, 0.0, 1.0)
    projector.apply(u)
    residual = projector.residual_inf(u)
    if residual > 5.0e-10:
        raise RuntimeError(f"Wentzell residual too large: {residual:.3e}")
    elapsed = time.perf_counter() - tic
    grid = Grid(
        x=computational_grid.x + model.c * final_time,
        y=computational_grid.y,
        dx=computational_grid.dx,
        dy=computational_grid.dy,
    )
    if progress:
        print(
            f"  eps={eps:.4f}, h={h:.5f}, grid={u.shape[1]}x{u.shape[0]}, "
            f"steps={nsteps}, dt={dt:.3e}, residual={residual:.2e}, "
            f"road-Dx={projector.tangential_scheme}, wall={elapsed:.1f}s",
            flush=True,
        )
    return RDSolution(
        eps=eps,
        model=model,
        final_time=final_time,
        grid=grid,
        u=u,
        dt=dt,
        nsteps=nsteps,
        boundary_residual=residual,
        road_tangential_scheme=projector.tangential_scheme,
        wall_seconds=elapsed,
    )


def ray_domain_limit(theta: float, grid: Grid) -> float:
    ct, st = math.cos(theta), math.sin(theta)
    bounds: list[float] = []
    if ct > 1.0e-13:
        bounds.append(grid.x[-1] / ct)
    elif ct < -1.0e-13:
        bounds.append(grid.x[0] / ct)
    if st > 1.0e-13:
        bounds.append(grid.y[-1] / st)
    return 0.995 * min(bounds)


def radial_threshold_front(
    solution: RDSolution,
    theta: np.ndarray,
    threshold: float,
    samples_per_cell: float = 2.0,
) -> np.ndarray:
    """Extract the outer threshold crossing along each ray by bilinear sampling."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0,1)")
    grid, u = solution.grid, solution.u
    out = np.empty_like(theta, dtype=float)
    for k, angle in enumerate(theta):
        rmax = ray_domain_limit(float(angle), grid)
        nr = max(250, int(math.ceil(samples_per_cell * rmax / min(grid.dx, grid.dy))))
        r = np.linspace(0.0, rmax, nr)
        xq = r * math.cos(float(angle))
        yq = r * math.sin(float(angle))
        ix = (xq - grid.x[0]) / grid.dx
        iy = yq / grid.dy
        values = map_coordinates(u, np.vstack((iy, ix)), order=1, mode="nearest")
        above = values >= threshold
        if not above[0]:
            raise RuntimeError("origin is below threshold; seed/front extraction failed")
        crossings = np.flatnonzero(above[:-1] & ~above[1:])
        if crossings.size == 0:
            raise RuntimeError(
                f"threshold front did not close before artificial boundary at theta={angle}"
            )
        j = int(crossings[-1])
        v0, v1 = values[j], values[j + 1]
        frac = (v0 - threshold) / max(v0 - v1, np.finfo(float).eps)
        out[k] = r[j] + frac * (r[j + 1] - r[j])
    return out


def hausdorff_from_radial(
    theta: np.ndarray, r1: np.ndarray, r2: np.ndarray
) -> float:
    p1 = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))
    p2 = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))
    d12 = cKDTree(p2).query(p1, k=1)[0]
    d21 = cKDTree(p1).query(p2, k=1)[0]
    return float(max(np.max(d12), np.max(d21)))


def relative_symmetric_difference(solution: RDSolution, threshold: float) -> float:
    grid = solution.grid
    xx, yy = np.meshgrid(grid.x, grid.y)
    phi = phase(xx, yy, solution.final_time, solution.model)
    hj = phi <= solution.final_time
    rd = solution.u >= threshold
    return float(np.count_nonzero(hj ^ rd) / np.count_nonzero(hj))


def phase_band_l2(solution: RDSolution, band: float = 0.4) -> float:
    grid = solution.grid
    xx, yy = np.meshgrid(grid.x, grid.y)
    phi = phase(xx, yy, solution.final_time, solution.model)
    vlim = np.maximum(phi - solution.final_time, 0.0)
    veps = -solution.eps * np.log(np.clip(solution.u, 1.0e-300, 1.0))
    mask = vlim <= band
    return float(np.sqrt(np.mean((veps[mask] - vlim[mask]) ** 2)))


def artificial_boundary_margin(
    solution: RDSolution, theta: np.ndarray, r_hj: np.ndarray
) -> float:
    x = r_hj * np.cos(theta)
    y = r_hj * np.sin(theta)
    grid = solution.grid
    return float(
        min(
            np.min(x - grid.x[0]),
            np.min(grid.x[-1] - x),
            np.min(grid.y[-1] - y),
        )
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=220)
    plt.close(fig)


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Experiment 1: singular limit and solver verification
# ---------------------------------------------------------------------------


def run_singular_limit(mode: str) -> dict:
    print("[1/5] Singular-limit reaction--diffusion runs", flush=True)
    model = Model(a=2.0, b=1.0, c=0.5)
    final_time = 1.5
    if mode == "paper":
        eps_values = [0.20, 0.15, 0.12, 0.10]
        # Crucially, h/eps must tend to zero.  A fixed h/eps would retain
        # O(h/eps) boundary/upwind errors in the front layer and could converge
        # to a modified HJ law.  This production law gives h/eps=0.4*sqrt(eps).
        grid_size = lambda eps: 0.4 * eps**1.5
        angles = np.linspace(0.0, np.pi, 241)
        xlim, ymax = (-4.0, 8.0), 3.6
    else:
        eps_values = [0.20, 0.15]
        grid_size = lambda eps: 0.40 * eps
        angles = np.linspace(0.0, np.pi, 121)
        xlim, ymax = (-5.0, 8.5), 4.2

    r_hj = final_time * directional_speed_array(angles, model)
    threshold_values = (0.1, 0.5, 0.9)
    results: list[dict] = []
    fronts: dict[float, dict[float, np.ndarray]] = {}
    smallest_solution: RDSolution | None = None

    for eps in eps_values:
        sol = solve_scaled_rd(
            eps,
            model,
            final_time,
            h=grid_size(eps),
            xlim=xlim,
            ymax=ymax,
        )
        radial = {
            threshold: radial_threshold_front(sol, angles, threshold)
            for threshold in threshold_values
        }
        fronts[eps] = radial
        r50 = radial[0.5]
        err = r50 - r_hj
        margin = min(
            artificial_boundary_margin(sol, angles, r_hj),
            *(artificial_boundary_margin(sol, angles, radial[q]) for q in threshold_values),
        )
        if margin < 8.0 * max(sol.grid.dx, sol.grid.dy):
            raise RuntimeError(f"HJ front too close to artificial boundary: {margin}")
        row = {
            "epsilon": eps,
            "h": sol.grid.dx,
            "dt": sol.dt,
            "nx": sol.grid.x.size,
            "ny": sol.grid.y.size,
            "steps": sol.nsteps,
            "radial_Linf": float(np.max(np.abs(err))),
            "radial_L2": float(np.sqrt(np.trapezoid(err * err, angles) / np.pi)),
            "hausdorff": hausdorff_from_radial(angles, r50, r_hj),
            "symmetric_difference": relative_symmetric_difference(sol, 0.5),
            "phase_band_L2": phase_band_l2(sol),
            "threshold_width_mean": float(np.mean(radial[0.1] - radial[0.9])),
            "threshold_width_max": float(np.max(radial[0.1] - radial[0.9])),
            "boundary_residual": sol.boundary_residual,
            "road_tangential_scheme": sol.road_tangential_scheme,
            "boundary_margin": margin,
            "u_min": float(np.min(sol.u)),
            "u_max": float(np.max(sol.u)),
            "wall_seconds": sol.wall_seconds,
        }
        results.append(row)
        radial_rows = []
        for i, angle in enumerate(angles):
            radial_rows.append(
                {
                    "theta_rad": angle,
                    "r_HJ": r_hj[i],
                    "r_u_0p1": radial[0.1][i],
                    "r_u_0p5": radial[0.5][i],
                    "r_u_0p9": radial[0.9][i],
                }
            )
        write_csv(DATA / f"singular_front_eps_{eps:.3f}.csv", radial_rows)
        smallest_solution = sol

    write_csv(DATA / "singular_limit_metrics.csv", results)

    # Diagnostic observed log--log slopes: reported as empirical diagnostics,
    # not asserted as asymptotic convergence orders.
    eps_arr = np.asarray([row["epsilon"] for row in results])
    order = np.argsort(eps_arr)
    eps_sorted = eps_arr[order]
    haus_sorted = np.asarray([row["hausdorff"] for row in results])[order]
    set_sorted = np.asarray([row["symmetric_difference"] for row in results])[order]
    # Fit the three smallest epsilon values; this is an observed diagnostic,
    # not a theorem-level convergence-order claim.
    fit_slice = slice(0, min(3, len(eps_sorted)))
    slope_h = float(np.polyfit(np.log(eps_sorted[fit_slice]), np.log(haus_sorted[fit_slice]), 1)[0])
    slope_s = float(np.polyfit(np.log(eps_sorted[fit_slice]), np.log(set_sorted[fit_slice]), 1)[0])

    # Main convergence figure: four front overlays + two quantitative panels.
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.5))
    contour_axes = axes.flat[:4]
    for ax, eps in zip(contour_axes, eps_values):
        for threshold, color, style in (
            (0.1, "#4C78A8", "--"),
            (0.5, "#D62728", "-"),
            (0.9, "#59A14F", ":"),
        ):
            r = fronts[eps][threshold]
            ax.plot(r * np.cos(angles), r * np.sin(angles), color=color, ls=style)
        ax.plot(r_hj * np.cos(angles), r_hj * np.sin(angles), "k-", lw=2.0)
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        if mode == "paper":
            ax.set_title(rf"$\varepsilon={eps:.2f}$, $h=0.4\varepsilon^{{3/2}}$")
        else:
            ax.set_title(rf"$\varepsilon={eps:.2f}$ (quick grid)")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.grid(alpha=0.15)
    contour_axes[0].legend(
        handles=[
            Line2D([0], [0], color="k", lw=2, label="HJ front"),
            Line2D([0], [0], color="#4C78A8", ls="--", label=r"$u^\varepsilon=0.1$"),
            Line2D([0], [0], color="#D62728", label=r"$u^\varepsilon=0.5$"),
            Line2D([0], [0], color="#59A14F", ls=":", label=r"$u^\varepsilon=0.9$"),
        ],
        loc="upper left",
    )

    ax = axes[1, 1]
    ax.semilogy(eps_sorted, haus_sorted, "o-", label=r"$d_{\rm H}^{(241)}$")
    ax.semilogy(eps_sorted, set_sorted, "s-", label=r"$E_{\triangle}^h$")
    ax.semilogy(
        eps_sorted,
        haus_sorted[0] * eps_sorted / eps_sorted[0],
        color="0.5",
        ls="--",
        label="slope 1 guide",
    )
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("error")
    ax.set_title("Singular-limit error")
    ax.grid(which="both", alpha=0.2)
    ax.set_xticks(eps_sorted)
    ax.set_xticklabels([f"{q:.2f}" for q in eps_sorted])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend()

    ax = axes[1, 2]
    widths_mean = np.asarray([row["threshold_width_mean"] for row in results])[order]
    widths_max = np.asarray([row["threshold_width_max"] for row in results])[order]
    ax.semilogy(eps_sorted, widths_mean, "o-", label="mean width")
    ax.semilogy(eps_sorted, widths_max, "s-", label="maximum width")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$r_{0.1}-r_{0.9}$")
    ax.set_title("Finite-front thickness")
    ax.grid(which="both", alpha=0.2)
    ax.set_xticks(eps_sorted)
    ax.set_xticklabels([f"{q:.2f}" for q in eps_sorted])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend()
    fig.suptitle(
        rf"Direct Fisher--KPP/Wentzell fronts versus HJ front "
        rf"($a={model.a:g}$, $b={model.b:g}$, $c={model.c:g}$, $t={final_time:g}$)",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "fig_num_singular_limit")

    assert smallest_solution is not None
    sol = smallest_solution
    grid = sol.grid
    xx, yy = np.meshgrid(grid.x, grid.y)
    phi, sstar = phase_and_optimizer(xx, yy, final_time, model)
    vlim = np.maximum(phi - final_time, 0.0)
    veps = -sol.eps * np.log(np.clip(sol.u, 1.0e-12, 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    extent = [grid.x[0], grid.x[-1], 0.0, grid.y[-1]]
    im = axes[0].imshow(sol.u, origin="lower", extent=extent, aspect="equal", vmin=0, vmax=1, cmap="viridis")
    axes[0].contour(xx, yy, phi, levels=[final_time], colors="w", linewidths=1.8)
    axes[0].set_title(rf"$u^\varepsilon$ at $\varepsilon={sol.eps:g}$; HJ in white")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    cap = 0.6
    im = axes[1].imshow(
        np.minimum(veps, cap), origin="lower", extent=extent, aspect="equal",
        vmin=0, vmax=cap, cmap="magma",
    )
    axes[1].contour(xx, yy, vlim, levels=[1.0e-10, 0.2, 0.4], colors=["w", "0.8", "0.55"], linewidths=1.0)
    axes[1].set_title(r"$v^\varepsilon=-\varepsilon\log u^\varepsilon$ (clipped)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    masked_s = np.ma.masked_where(phi > final_time, sstar / final_time)
    im = axes[2].imshow(masked_s, origin="lower", extent=extent, aspect="equal", cmap="plasma")
    axes[2].contour(xx, yy, phi, levels=[final_time], colors="k", linewidths=1.5)
    axes[2].contour(xx, yy, sstar, levels=[1.0e-8], colors="w", linewidths=1.0)
    axes[2].set_title(r"Normalized optimal boundary local time $s^*/t$")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlim(-3.8, 8.0)
        ax.set_ylim(0.0, 3.8)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    save_figure(fig, "fig_num_phase_diagnostics")

    return {
        "model": asdict(model),
        "final_time": final_time,
        "eps_values": eps_values,
        "production_grid_law": "h=0.4*epsilon^(3/2)" if mode == "paper" else "quick",
        "results": results,
        "slope_hausdorff": slope_h,
        "slope_symmetric_difference": slope_s,
        "angles": angles,
        "r_hj": r_hj,
    }


def run_discretization_audit(mode: str) -> dict:
    print("[2/5] Grid, time-step, and domain audits", flush=True)
    model = Model(a=2.0, b=1.0, c=0.5)
    eps, final_time = 0.15, 1.5
    angles = np.linspace(0.0, np.pi, 181 if mode == "paper" else 91)
    production_ratio = 0.4 * math.sqrt(eps)
    ratios = [0.20, production_ratio, 0.12, 0.08] if mode == "paper" else [0.40, 0.25]
    fronts: list[np.ndarray] = []
    rows: list[dict] = []
    for ratio in ratios:
        sol = solve_scaled_rd(
            eps,
            model,
            final_time,
            h=ratio * eps,
            xlim=(-4.0, 8.0) if mode == "paper" else (-5.0, 8.5),
            ymax=3.6 if mode == "paper" else 4.2,
            cfl=0.72,
        )
        r = radial_threshold_front(sol, angles, 0.5)
        fronts.append(r)
        rows.append(
            {
                "test": "grid",
                "h": sol.grid.dx,
                "dt": sol.dt,
                "cfl": 0.72,
                "domain_xmin": sol.grid.x[0],
                "domain_xmax": sol.grid.x[-1],
                "domain_ymax": sol.grid.y[-1],
                "distance_to_finest": np.nan,
                "boundary_residual": sol.boundary_residual,
                "road_tangential_scheme": sol.road_tangential_scheme,
            }
        )
    reference = fronts[-1]
    for row, r in zip(rows, fronts):
        row["distance_to_finest"] = hausdorff_from_radial(angles, r, reference)

    # Time-step halving at the production grid; this isolates temporal error.
    time_fronts = []
    for cfl in ([0.72, 0.36] if mode == "paper" else [0.72]):
        sol = solve_scaled_rd(
            0.20,
            model,
            final_time,
            h=0.4 * 0.20**1.5 if mode == "paper" else 0.08,
            xlim=(-4.0, 8.0) if mode == "paper" else (-5.0, 8.5),
            ymax=3.6 if mode == "paper" else 4.2,
            cfl=cfl,
        )
        r = radial_threshold_front(sol, angles, 0.5)
        time_fronts.append(r)
        rows.append(
            {
                "test": "time",
                "h": sol.grid.dx,
                "dt": sol.dt,
                "cfl": cfl,
                "domain_xmin": sol.grid.x[0],
                "domain_xmax": sol.grid.x[-1],
                "domain_ymax": sol.grid.y[-1],
                "distance_to_finest": np.nan,
                "boundary_residual": sol.boundary_residual,
                "road_tangential_scheme": sol.road_tangential_scheme,
            }
        )
    if len(time_fronts) == 2:
        temporal_distance = hausdorff_from_radial(angles, time_fronts[0], time_fronts[1])
        rows[-2]["distance_to_finest"] = temporal_distance
        rows[-1]["distance_to_finest"] = 0.0
    else:
        temporal_distance = float("nan")

    # Domain-expansion audit, compared to the coupled-law production front.
    domain_sol = solve_scaled_rd(
        eps,
        model,
        final_time,
        h=0.4 * eps**1.5 if mode == "paper" else 0.40 * eps,
        xlim=(-5.0, 10.0) if mode == "paper" else (-6.25, 10.625),
        ymax=4.5 if mode == "paper" else 5.25,
        cfl=0.72,
    )
    domain_front = radial_threshold_front(domain_sol, angles, 0.5)
    standard_idx = int(np.argmin(np.abs(np.asarray(ratios) - production_ratio)))
    domain_distance = hausdorff_from_radial(angles, fronts[standard_idx], domain_front)
    rows.append(
        {
            "test": "domain",
            "h": domain_sol.grid.dx,
            "dt": domain_sol.dt,
            "cfl": 0.72,
            "domain_xmin": domain_sol.grid.x[0],
            "domain_xmax": domain_sol.grid.x[-1],
            "domain_ymax": domain_sol.grid.y[-1],
            "distance_to_finest": domain_distance,
            "boundary_residual": domain_sol.boundary_residual,
            "road_tangential_scheme": domain_sol.road_tangential_scheme,
        }
    )
    write_csv(DATA / "discretization_audit.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    hvals = np.asarray([row["h"] for row in rows if row["test"] == "grid"])
    distances = np.asarray([row["distance_to_finest"] for row in rows if row["test"] == "grid"])
    axes[0].semilogy(hvals[:-1], distances[:-1], "o-", color="#4C78A8")
    axes[0].set_xlabel(r"grid size $h$")
    axes[0].set_ylabel("distance to finest front")
    axes[0].set_title(r"Grid audit at $\varepsilon=0.15$")
    axes[0].grid(which="both", alpha=0.2)
    axes[0].set_xticks(hvals[:-1])
    axes[0].set_xticklabels([f"{q:.4f}" for q in hvals[:-1]], rotation=18)
    axes[0].tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    labels = ["time-step\nhalving", "domain\nexpansion"]
    values = [temporal_distance, domain_distance]
    axes[1].bar(labels, values, color=["#59A14F", "#F28E2B"])
    axes[1].set_ylabel("front Hausdorff difference")
    axes[1].set_title("Independent numerical perturbations")
    axes[1].grid(axis="y", alpha=0.2)
    save_figure(fig, "fig_num_discretization_audit")
    return {
        "rows": rows,
        "grid_distance_coarse": float(distances[0]),
        "grid_distance_middle": float(distances[-2]) if len(distances) > 2 else float(distances[0]),
        "grid_distance_production": float(distances[standard_idx]),
        "temporal_distance": temporal_distance,
        "domain_distance": domain_distance,
    }


# ---------------------------------------------------------------------------
# Experiment 2: directional speeds and parameter responses
# ---------------------------------------------------------------------------


def three_speeds(model: Model) -> tuple[float, float, float]:
    values = directional_speed_array(np.asarray([0.0, np.pi, np.pi / 2.0]), model)
    return float(values[0]), float(values[1]), float(values[2])


def scan_parameter(name: str, values: np.ndarray, base: Model) -> list[dict]:
    rows = []
    for value in values:
        params = asdict(base)
        params[name] = float(value)
        model = Model(**params)
        right, left, vertical = three_speeds(model)
        rows.append(
            {
                name: value,
                "C_right": right,
                "C_left": left,
                "C_vertical_radial": vertical,
                "field_right": model.c + 2.0,
                "field_left": 2.0 - model.c,
                "field_vertical_radial": math.sqrt(max(4.0 - model.c**2, 0.0)),
                "road_gain_right": right - (model.c + 2.0),
                "road_gain_left": left - (2.0 - model.c),
            }
        )
    return rows


def run_speed_analysis() -> dict:
    print("[3/5] Directional-speed parameter scans", flush=True)
    base = Model(a=2.0, b=1.0, c=0.5)
    scans = {
        "a": np.linspace(0.20, 6.0, 61),
        "b": np.linspace(-3.0, 3.0, 61),
        "c": np.linspace(-1.8, 1.8, 73),
    }
    all_rows: dict[str, list[dict]] = {}
    for name, values in scans.items():
        rows = scan_parameter(name, values, base)
        all_rows[name] = rows
        write_csv(DATA / f"speed_scan_{name}.csv", rows)

    colors = {"right": "#D62728", "left": "#4C78A8", "vertical": "#59A14F"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    for ax, name in zip(axes, ("a", "b", "c")):
        rows = all_rows[name]
        x = np.asarray([row[name] for row in rows])
        ax.plot(x, [row["C_right"] for row in rows], color=colors["right"], label=r"$C_+$")
        ax.plot(x, [row["C_left"] for row in rows], color=colors["left"], label=r"$C_-$")
        ax.plot(x, [row["C_vertical_radial"] for row in rows], color=colors["vertical"], label=r"$C_\perp^{\rm rad}$")
        ax.plot(x, [row["field_right"] for row in rows], color=colors["right"], ls="--", alpha=0.55)
        ax.plot(x, [row["field_left"] for row in rows], color=colors["left"], ls="--", alpha=0.55)
        ax.plot(x, [row["field_vertical_radial"] for row in rows], color=colors["vertical"], ls="--", alpha=0.55)
        ax.set_xlabel(rf"${name}$")
        ax.set_ylabel("radial speed")
        ax.set_title(rf"Vary ${name}$; other parameters fixed")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best")
    save_figure(fig, "fig_num_speed_parameters")

    theta = np.linspace(0.0, np.pi, 241)
    polar_models = [
        (Model(0.5, 1.0, 0.5), r"$a=0.5$"),
        (Model(2.0, 1.0, 0.5), r"$a=2$"),
        (Model(5.0, 1.0, 0.5), r"$a=5$"),
    ]
    fig = plt.figure(figsize=(7.2, 3.45))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    axx = fig.add_subplot(1, 2, 2)
    polar_rows = []
    for (model, label), color in zip(polar_models, ("#4C78A8", "#F28E2B", "#D62728")):
        speed = directional_speed_array(theta, model)
        axp.plot(theta, speed, color=color, label=label)
        axx.plot(speed * np.cos(theta), speed * np.sin(theta), color=color, label=label)
        for q, val in zip(theta, speed):
            polar_rows.append({"a": model.a, "theta_rad": q, "speed": val})
    axp.set_thetamin(0)
    axp.set_thetamax(180)
    axp.set_title("Directional radial speed")
    axp.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    axx.axhline(0, color="0.3", lw=0.8)
    axx.set_aspect("equal", adjustable="box")
    axx.set_xlabel(r"$x/t$")
    axx.set_ylabel(r"$y/t$")
    axx.set_title("Unit-time Wulff fronts")
    axx.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "fig_num_directional_speeds")
    write_csv(DATA / "directional_speed_polar.csv", polar_rows)

    right_formula, left_formula = road_endpoint_formula(base)
    right_root, left_root, vertical = three_speeds(base)
    return {
        "base": asdict(base),
        "base_right": right_root,
        "base_left": left_root,
        "base_vertical": vertical,
        "endpoint_formula_error": max(abs(right_formula - right_root), abs(left_formula - left_root)),
        "scans": all_rows,
    }


# ---------------------------------------------------------------------------
# Experiment 3: symmetric two-road cones
# ---------------------------------------------------------------------------


def critical_angle(model: Model) -> float:
    if abs(model.c) > 1.0e-14:
        raise ValueError("cone critical-angle law requires c=0")
    return float(
        brentq(
            lambda d: math.sin(d) - model.b * math.cos(d) - model.a * math.cos(d) ** 2,
            0.0,
            np.pi / 2.0,
        )
    )


def cone_speed(theta: np.ndarray, alpha: float, model: Model) -> np.ndarray:
    if abs(model.c) > 1.0e-14 or model.b < 0.0:
        raise ValueError("folded cone calculation requires c=0 and outward b>=0")
    if np.any(theta < -1.0e-13) or np.any(theta > 2.0 * alpha + 1.0e-13):
        raise ValueError("theta lies outside cone")
    delta = np.minimum(theta, 2.0 * alpha - theta)
    return directional_speed_array(delta, model)


def run_cone_analysis() -> dict:
    print("[4/5] Symmetric two-road cone fronts", flush=True)
    model = Model(a=2.0, b=1.0, c=0.0)
    alpha_degrees = [20.0, 45.0, 70.0]
    # Compute the one-road angular law once at high resolution; all cone plots
    # are exact foldings/interpolations of this single deterministic table.
    theta_master = np.linspace(0.0, np.pi, 12001)
    speed_master = directional_speed_array(theta_master, model)

    def c_of(delta: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(delta, float), theta_master, speed_master)

    rows: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
    for ax, degree in zip(axes.flat[:3], alpha_degrees):
        alpha = math.radians(degree)
        theta = np.linspace(0.0, 2.0 * alpha, 301)
        cross = c_of(np.minimum(theta, 2.0 * alpha - theta))
        single_reference = c_of(theta)
        bulk = np.full_like(theta, 2.0)
        x, y = cross * np.cos(theta), cross * np.sin(theta)
        ax.fill(np.r_[0.0, x, 0.0], np.r_[0.0, y, 0.0], color="#9ECAE1", alpha=0.55, label="two roads")
        ax.plot(x, y, color="#08519C", lw=2.0)
        ax.plot(single_reference * np.cos(theta), single_reference * np.sin(theta), "--", color="#D95F0E", label=r"one-road $\Gamma_0$ reference")
        ax.plot(bulk * np.cos(theta), bulk * np.sin(theta), ":", color="0.25", label="field-only")
        ray_length = 1.08 * float(np.max(cross))
        ax.plot([0, ray_length], [0, 0], color="0.25", lw=1.0)
        ax.plot(
            [0, ray_length * math.cos(2 * alpha)],
            [0, ray_length * math.sin(2 * alpha)],
            color="0.25",
            lw=1.0,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"half-opening $\alpha={degree:.0f}^\circ$")
        ax.set_xlabel(r"$x/t$")
        ax.set_ylabel(r"$y/t$")
        ax.grid(alpha=0.15)
        theta_area = np.linspace(0.0, 2.0 * alpha, 12001)
        cross_area_speed = c_of(np.minimum(theta_area, 2.0 * alpha - theta_area))
        area_cross = float(np.trapezoid(cross_area_speed * cross_area_speed, theta_area) / 2.0)
        area_bulk = 4.0 * alpha
        bisector_speed = float(c_of(alpha))
        rows.append(
            {
                "alpha_deg": degree,
                "critical_alpha_deg": math.degrees(critical_angle(model)),
                "bisector_speed": bisector_speed,
                "cross_area": area_cross,
                "field_wedge_area": area_bulk,
                "relative_area_gain": area_cross / area_bulk - 1.0,
            }
        )
    axes.flat[0].legend(loc="upper left")

    alpha_grid = np.linspace(math.radians(5.0), np.pi / 2.0, 120)
    bisector = c_of(alpha_grid)
    area_gain = []
    for alpha in alpha_grid:
        delta = np.linspace(0.0, alpha, 801)
        cdelta = c_of(delta)
        area_gain.append(float(np.trapezoid(cdelta * cdelta, delta) / (4.0 * alpha) - 1.0))
    ax = axes.flat[3]
    ax.plot(np.degrees(alpha_grid), bisector, color="#08519C", label=r"$C_\alpha^*(\alpha)$")
    ax.axvline(math.degrees(critical_angle(model)), color="#D62728", ls="--", label=r"$\alpha_{\rm c}$")
    ax.axhline(2.0, color="0.3", ls=":")
    ax.set_xlabel(r"half-opening $\alpha$ (degrees)")
    ax.set_ylabel("bisector speed")
    ax2 = ax.twinx()
    ax2.plot(np.degrees(alpha_grid), 100.0 * np.asarray(area_gain), color="#59A14F", alpha=0.8, label="area gain")
    ax2.set_ylabel("area gain over field wedge (%)", color="#3A7D44")
    ax.set_title("Angle law and geometric enhancement")
    ax.grid(alpha=0.2)
    lines = ax.get_lines()[:2] + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    save_figure(fig, "fig_num_cone_angles")
    write_csv(DATA / "cone_angle_summary.csv", rows)

    # Dynamics and topology at alpha=45 degrees.  The exact HJ dynamics are
    # homothetic; the topology panel visualizes which road is selected and the
    # amount of road time s*.
    alpha = math.radians(45.0)
    theta = np.linspace(0.0, 2.0 * alpha, 401)
    speed = c_of(np.minimum(theta, 2.0 * alpha - theta))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), constrained_layout=True)
    for t, color in zip((0.5, 1.0, 1.5), ("#9ECAE1", "#4292C6", "#08519C")):
        r = t * speed
        axes[0].plot(r * np.cos(theta), r * np.sin(theta), color=color, label=rf"$t={t:g}$")
    axes[0].plot([0, 5], [0, 0], color="0.25", lw=1)
    axes[0].plot([0, 0], [0, 5], color="0.25", lw=1)
    axes[0].set_aspect("equal", adjustable="box")
    dynamic_limit = 1.05 * 1.5 * float(np.max(speed))
    axes[0].set_xlim(-0.1, dynamic_limit)
    axes[0].set_ylim(-0.1, dynamic_limit)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_title(r"Homothetic cone dynamics, $\alpha=45^\circ$")
    axes[0].legend()
    axes[0].grid(alpha=0.15)

    rr = np.linspace(0.02, 1.04 * float(np.max(speed)), 300)
    tt = np.linspace(0.0, 2.0 * alpha, 240)
    RR, TT = np.meshgrid(rr, tt, indexing="xy")
    DD = np.minimum(TT, 2.0 * alpha - TT)
    XXloc, YYloc = RR * np.cos(DD), RR * np.sin(DD)
    phi_loc, s_loc = phase_and_optimizer(XXloc, YYloc, 1.0, model)
    X = RR * np.cos(TT)
    Y = RR * np.sin(TT)
    topo = np.ma.masked_where(phi_loc > 1.0, s_loc)
    pcm = axes[1].pcolormesh(X, Y, topo, shading="auto", cmap="plasma", vmin=0.0)
    axes[1].plot(speed * np.cos(theta), speed * np.sin(theta), color="k", lw=1.5)
    topology_limit = 1.04 * float(np.max(speed))
    axes[1].plot([0, topology_limit], [0, 0], color="w", lw=1.0)
    axes[1].plot([0, 0], [0, topology_limit], color="w", lw=1.0)
    axes[1].plot(
        [0, topology_limit / math.sqrt(2)],
        [0, topology_limit / math.sqrt(2)],
        color="w",
        ls="--",
        lw=1.0,
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel(r"$x/t$")
    axes[1].set_ylabel(r"$y/t$")
    axes[1].set_title(r"Normalized optimal boundary local time $s^*/t$")
    fig.colorbar(pcm, ax=axes[1], label=r"$s^*/t$")
    save_figure(fig, "fig_num_cone_dynamics")

    return {
        "model": asdict(model),
        "critical_alpha_deg": math.degrees(critical_angle(model)),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Experiment 4: aligned and opposite drifts
# ---------------------------------------------------------------------------


def run_opposite_drifts() -> dict:
    print("[5/5] Aligned and opposite drift configurations", flush=True)
    a = 2.0
    # Stronger |c| is used in the counter-flow panels to expose the vertical
    # radial-speed recovery caused by access to the oppositely directed road.
    cases = [(1.0, 1.50), (1.0, -1.50), (-1.0, 1.50), (-1.0, -1.50)]
    theta = np.linspace(0.0, np.pi, 361)
    rows: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), constrained_layout=True)
    for ax, (b, c) in zip(axes.flat, cases):
        model = Model(a=a, b=b, c=c)
        speed = directional_speed_array(theta, model)
        x, y = speed * np.cos(theta), speed * np.sin(theta)
        field = field_radial_speed(theta, c)
        _, sstar = phase_and_optimizer(x, y, 1.0, model)
        assisted = sstar > 1.0e-8
        ax.fill(np.r_[x, x[-1]], np.r_[y, 0.0], color="#DCEAF7", alpha=0.65)
        ax.plot(x, y, color="0.2", lw=1.4, label="road--field")
        ax.plot(field * np.cos(theta), field * np.sin(theta), "--", color="#7F7F7F", label="field-only")
        ax.scatter(x[assisted][::5], y[assisted][::5], s=7, color="#D62728", label=r"$s^*>0$")
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x/t$")
        ax.set_ylabel(r"$y/t$")
        relation = "aligned" if b * c > 0 else "opposed"
        ax.set_title(rf"$b={b:g}$, $c={c:g}$ ({relation})")
        ax.grid(alpha=0.15)
        right, left, vertical = three_speeds(model)
        xplus, left_formula = road_endpoint_formula(model)
        xminus = -left_formula
        asymmetry = (right - left) / (right + left)
        rows.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "flow_relation": relation,
                "C_right": right,
                "C_left": left,
                "C_vertical_radial": vertical,
                "x_plus": xplus,
                "x_minus": xminus,
                "asymmetry_index": asymmetry,
                "road_gain_right": right - (c + 2.0),
                "road_gain_left": left - (2.0 - c),
            }
        )
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(r"Drift reversal outside the standing assumption $b,c>0$ ($a=2$)", y=1.01)
    save_figure(fig, "fig_num_opposite_drifts")
    write_csv(DATA / "opposite_drift_summary.csv", rows)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Automated analytic and numerical tests; generated TeX tables
# ---------------------------------------------------------------------------


def run_analytic_tests() -> dict:
    rng = np.random.default_rng(20260716)
    generic_max = 0.0
    kkt_max = 0.0
    for _ in range(60):
        model = Model(
            a=float(rng.uniform(0.2, 5.0)),
            b=float(rng.uniform(-2.0, 2.0)),
            c=float(rng.uniform(-1.5, 1.5)),
        )
        x = float(rng.uniform(-4.0, 5.0))
        y = float(rng.uniform(0.0, 3.0))
        t = float(rng.uniform(0.3, 2.0))
        val, s = phase_and_optimizer(x, y, t, model)
        direct = minimize_scalar(
            lambda q: ((-x + model.b * q + model.c * t) ** 2) / (4.0 * (t + model.a * q))
            + (y + q) ** 2 / (4.0 * t),
            bounds=(0.0, max(10.0, float(s) + 3.0)),
            method="bounded",
            options={"xatol": 1.0e-13},
        )
        generic_max = max(generic_max, abs(float(val) - float(direct.fun)))
        derivative = float(objective_derivative(float(s), x, y, t, model))
        kkt = abs(derivative) if float(s) > 1.0e-9 else max(-derivative, 0.0)
        kkt_max = max(kkt_max, kkt)

    model = Model(2.0, 1.0, 0.7)
    x = rng.uniform(-3.0, 4.0, 100)
    y = rng.uniform(0.0, 2.5, 100)
    t = 1.3
    translation = float(
        np.max(np.abs(phase(x, y, t, model) - phase(x - model.c * t, y, t, Model(model.a, model.b, 0.0))))
    )
    reflection = float(
        np.max(
            np.abs(
                phase(x, y, t, Model(model.a, -model.b, model.c))
                - phase(2.0 * model.c * t - x, y, t, model)
            )
        )
    )
    scale = 1.7
    homogeneity = float(
        np.max(
            np.abs(
                phase(scale * x, scale * y, scale * t, model)
                - scale * phase(x, y, t, model)
            )
        )
    )
    cone_model = Model(2.0, 1.0, 0.0)
    alpha = math.radians(45.0)
    q = np.linspace(0.0, 2.0 * alpha, 101)
    cone_symmetry = float(np.max(np.abs(cone_speed(q, alpha, cone_model) - cone_speed(2.0 * alpha - q, alpha, cone_model))))
    a_values = [0.5, 1.0, 2.0, 4.0]
    right = [directional_speed(0.0, Model(a, 1.0, 0.5)) for a in a_values]
    monotone_violation = float(max(0.0, -np.min(np.diff(right))))
    endpoint_error = 0.0
    vertical_error = 0.0
    for test_model in (Model(0.5, -1.2, 0.3), Model(2.0, 1.0, 0.5), Model(4.0, 2.0, -0.7)):
        formula = road_endpoint_formula(test_model)
        roots = (directional_speed(0.0, test_model), directional_speed(np.pi, test_model))
        endpoint_error = max(endpoint_error, abs(formula[0] - roots[0]), abs(formula[1] - roots[1]))
        vertical_error = max(
            vertical_error,
            abs(vertical_speed_formula(test_model) - directional_speed(np.pi / 2.0, test_model)),
        )

    tests = {
        "generic_minimizer_max_abs_error": generic_max,
        "optimizer_KKT_max_residual": kkt_max,
        "translation_identity_max_abs_error": translation,
        "reflection_identity_max_abs_error": reflection,
        "homogeneity_max_abs_error": homogeneity,
        "cone_symmetry_max_abs_error": cone_symmetry,
        "a_monotonicity_violation": monotone_violation,
        "road_endpoint_formula_max_abs_error": endpoint_error,
        "vertical_formula_max_abs_error": vertical_error,
    }
    tolerances = {
        "generic_minimizer_max_abs_error": 2.0e-9,
        "optimizer_KKT_max_residual": 2.0e-10,
        "translation_identity_max_abs_error": 2.0e-12,
        "reflection_identity_max_abs_error": 2.0e-12,
        "homogeneity_max_abs_error": 2.0e-11,
        "cone_symmetry_max_abs_error": 2.0e-12,
        "a_monotonicity_violation": 2.0e-12,
        "road_endpoint_formula_max_abs_error": 2.0e-7,
        "vertical_formula_max_abs_error": 2.0e-7,
    }
    for key, value in tests.items():
        if value > tolerances[key]:
            raise AssertionError(f"analytic test {key} failed: {value} > {tolerances[key]}")
    return tests


def write_generated_tex(
    singular: dict,
    audit: dict,
    speeds: dict,
    cones: dict,
    opposite: dict,
) -> None:
    def tex_sci(value: float) -> str:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"

    results = singular["results"]
    singular_lines = [
        "% Auto-generated by road_field_numerics.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Quantitative singular-limit diagnostics at $t=1.5$ for $a=2$, $b=1$, $c=0.5$, with $h=0.4\varepsilon^{3/2}$. Here $d_{\rm H}^{(241)}$ is the bidirectional Hausdorff distance between the 241-ray samples of the $u^\varepsilon=0.5$ and HJ fronts, $E_2$ is the angular radial error, and $E_\triangle^h$ is the grid-quadrature approximation of the relative symmetric-difference area.}",
        r"\label{tab:num-singular-errors}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$\varepsilon$ & $h$ & $d_{\rm H}^{(241)}$ & $E_2$ & $E_\triangle^h$ & phase $L^2$ & boundary residual \\",
        r"\midrule",
    ]
    for row in results:
        singular_lines.append(
            f"{row['epsilon']:.2f} & {row['h']:.4f} & {row['hausdorff']:.4f} & "
            f"{row['radial_L2']:.4f} & {row['symmetric_difference']:.4f} & "
            f"{row['phase_band_L2']:.4f} & {row['boundary_residual']:.2e} \\\\"
        )
    singular_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    cone_rows = cones["rows"]
    cone_lines = [
        "% Auto-generated by road_field_numerics.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Symmetric two-road cone diagnostics for $a=2$, $b=1$, $c=0$. The opening is $2\alpha$ and the area gain is measured against the field-only sector of the same opening.}",
        r"\label{tab:num-cone-summary}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$\alpha$ & bisector speed & two-road area & relative area gain \\",
        r"\midrule",
    ]
    for row in cone_rows:
        cone_lines.append(
            rf"${row['alpha_deg']:.0f}^\circ$ & {row['bisector_speed']:.5f} & "
            + rf"{row['cross_area']:.5f} & {100.0 * row['relative_area_gain']:.2f}\% "
            + r"\\"
        )
    cone_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    opp_rows = opposite["rows"]
    opposite_lines = [
        "% Auto-generated by road_field_numerics.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Directional speeds in the aligned/counter-flow experiments ($a=2$). The positive quantities $C_+$ and $C_-$ are rightward and leftward radial speeds, and $\mathcal A=(C_+-C_-)/(C_++C_-)$ quantifies laboratory-frame asymmetry.}",
        r"\label{tab:num-opposite-drift}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$(b,c)$ & relation & $C_+$ & $C_-$ & $C_\perp^{\rm rad}$ & $\mathcal A$ \\",
        r"\midrule",
    ]
    for row in opp_rows:
        opposite_lines.append(
            rf"$({row['b']:.2f},{row['c']:.2f})$ & {row['flow_relation']} & "
            rf"{row['C_right']:.4f} & {row['C_left']:.4f} & "
            + rf"{row['C_vertical_radial']:.4f} & {row['asymmetry_index']:.4f} "
            + r"\\"
        )
    opposite_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    macro_lines = [
        "% Auto-generated by road_field_numerics.py; do not edit by hand.",
        rf"\newcommand{{\NumHausdorffSlope}}{{{singular['slope_hausdorff']:.3f}}}",
        rf"\newcommand{{\NumSetSlope}}{{{singular['slope_symmetric_difference']:.3f}}}",
        rf"\newcommand{{\NumGridCoarse}}{{{audit['grid_distance_coarse']:.4f}}}",
        rf"\newcommand{{\NumGridMiddle}}{{{audit['grid_distance_middle']:.4f}}}",
        rf"\newcommand{{\NumGridProduction}}{{{audit['grid_distance_production']:.4f}}}",
        rf"\newcommand{{\NumTimeAudit}}{{{tex_sci(audit['temporal_distance'])}}}",
        rf"\newcommand{{\NumDomainAudit}}{{{tex_sci(audit['domain_distance'])}}}",
        rf"\newcommand{{\NumConeCriticalDeg}}{{{cones['critical_alpha_deg']:.3f}}}",
        rf"\newcommand{{\NumBaseRight}}{{{speeds['base_right']:.4f}}}",
        rf"\newcommand{{\NumBaseLeft}}{{{speeds['base_left']:.4f}}}",
        rf"\newcommand{{\NumBaseVertical}}{{{speeds['base_vertical']:.4f}}}",
    ]
    (ROOT / "table_singular_limit.tex").write_text("\n".join(singular_lines) + "\n", encoding="utf-8")
    (ROOT / "table_cone_summary.tex").write_text("\n".join(cone_lines) + "\n", encoding="utf-8")
    (ROOT / "table_opposite_drift.tex").write_text("\n".join(opposite_lines) + "\n", encoding="utf-8")
    (ROOT / "numerical_generated_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    global FIG, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("paper", "quick"),
        default="paper",
        help="paper: all production runs; quick: lower-cost smoke/reproduction run",
    )
    args = parser.parse_args(argv)
    if args.mode == "quick":
        # A smoke test must never overwrite the production figures, tables, or
        # CSV files consumed by the manuscript.
        FIG = ROOT / "quick_output" / "figures"
        DATA = ROOT / "quick_output" / "data"
    ensure_directories()
    configure_plotting()
    start = time.perf_counter()
    tests = run_analytic_tests()
    singular = run_singular_limit(args.mode)
    audit = run_discretization_audit(args.mode)
    speeds = run_speed_analysis()
    cones = run_cone_analysis()
    opposite = run_opposite_drifts()
    if args.mode == "paper":
        write_generated_tex(singular, audit, speeds, cones, opposite)

    metadata = {
        "mode": args.mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "analytic_tests": tests,
        "singular_limit": {
            key: value
            for key, value in singular.items()
            if key not in {"angles", "r_hj", "results"}
        },
        "singular_limit_results": singular["results"],
        "discretization_audit": audit,
        "speed_summary": {key: value for key, value in speeds.items() if key != "scans"},
        "cone_summary": cones,
        "opposite_drift_summary": opposite,
        "total_wall_seconds": time.perf_counter() - start,
    }
    (DATA / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Completed {args.mode} run in {metadata['total_wall_seconds']:.1f}s. "
        f"Figures: {FIG}; data: {DATA}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
