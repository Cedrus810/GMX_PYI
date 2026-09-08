"""In-memory analysis backend.

The gmx analysis tools re-read the trajectory from disk on every call and
run single-threaded.  These do the same arithmetic on frames that are
already in memory, so N analyses cost one read instead of N.

Not everything belongs here: ``gmx dssp`` is both faster and more
informative than the mdtraj equivalent, so secondary structure stays on gmx.

Definitions follow GROMACS, not whatever the library defaults to: RMSD and
the superposition are mass-weighted, Rg is mass-weighted about the centre of
mass.  Frames must already be whole (see ``Simulation.frames``, which runs
one ``trjconv -pbc mol`` pass first) -- nothing here fixes periodicity.
"""

from __future__ import annotations

import numpy as np


def masses(traj, indices):
    out = np.array([traj.topology.atom(int(i)).element.mass for i in indices],
                   dtype=float)
    out[out <= 0] = 1.0
    return out


def _fit(coords, reference, weights):
    """Mass-weighted Kabsch superposition of one frame onto ``reference``."""
    w = weights[:, None] / weights.sum()
    x = coords - (w * coords).sum(0)
    y = reference - (w * reference).sum(0)
    u, _, vt = np.linalg.svd((w * x).T @ y)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return x @ rotation, y


def rmsd(traj, indices, reference=None, mass_weighted=True):
    """Mass-weighted RMSD after least-squares fit, exactly as ``gmx rms``."""
    indices = np.asarray(indices, dtype=int)
    xyz = traj.xyz[:, indices]
    ref = xyz[0] if reference is None else np.asarray(reference)[indices]
    w = masses(traj, indices) if mass_weighted else np.ones(len(indices))
    weights = w / w.sum()
    out = np.empty(len(xyz))
    for i, frame in enumerate(xyz):
        fitted, centred_ref = _fit(frame, ref, w)
        out[i] = np.sqrt((weights * ((fitted - centred_ref) ** 2).sum(1)).sum())
    return traj.time, out


def rmsf(traj, indices, mass_weighted=True):
    """Per-atom fluctuation about the mean structure, after fitting."""
    indices = np.asarray(indices, dtype=int)
    xyz = traj.xyz[:, indices]
    w = masses(traj, indices) if mass_weighted else np.ones(len(indices))
    fitted = np.empty_like(xyz)
    ref = xyz[0]
    for i, frame in enumerate(xyz):
        fitted[i] = _fit(frame, ref, w)[0]
    average = fitted.mean(axis=0)
    fluctuation = np.sqrt((((fitted - average) ** 2).sum(axis=2)).mean(axis=0))
    return np.arange(1, len(indices) + 1), fluctuation


def radius_of_gyration(traj, indices):
    indices = np.asarray(indices, dtype=int)
    xyz = traj.xyz[:, indices]
    w = masses(traj, indices)
    total = w.sum()
    centre = (xyz * w[None, :, None]).sum(axis=1) / total
    delta = xyz - centre[:, None, :]
    squared = (w[None, :, None] * delta ** 2).sum(axis=(1, 2)) / total
    return traj.time, np.sqrt(squared)


def rdf(traj, reference, selection, bin_width=0.002, rmax=None):
    """g(r) between two index sets, minimum-image aware."""
    import mdtraj

    reference = np.asarray(reference, dtype=int)
    selection = np.asarray(selection, dtype=int)
    if rmax is None:
        rmax = float(traj.unitcell_lengths.min()) / 2 if traj.unitcell_lengths is not None else 2.0
    pairs = np.stack(np.meshgrid(reference, selection, indexing="ij"), -1).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    r, g = mdtraj.compute_rdf(traj, pairs, r_range=(0.0, rmax),
                              bin_width=bin_width)
    return r, g


def sasa(traj, indices, probe=0.14, per_residue=False):
    import mdtraj

    sliced = traj.atom_slice(np.asarray(indices, dtype=int))
    mode = "residue" if per_residue else "atom"
    areas = mdtraj.shrake_rupley(sliced, probe_radius=probe, mode=mode)
    if per_residue:
        return np.arange(1, areas.shape[1] + 1), areas.mean(axis=0)
    return traj.time, areas.sum(axis=1)


def msd(traj, indices, lag_fraction=0.5):
    """Mean squared displacement, averaged over time origins."""
    indices = np.asarray(indices, dtype=int)
    xyz = traj.xyz[:, indices]
    n_frames = len(xyz)
    max_lag = max(1, int(n_frames * lag_fraction))
    out = np.zeros(max_lag)
    for lag in range(1, max_lag):
        delta = xyz[lag:] - xyz[:-lag]
        out[lag] = (delta ** 2).sum(axis=2).mean()
    dt = float(traj.time[1] - traj.time[0]) if n_frames > 1 else 1.0
    times = np.arange(max_lag) * dt
    return times, out


def diffusion_constant(times, values, fit_from=0.1, fit_to=0.9):
    """Einstein fit of the MSD, in 1e-5 cm^2/s (GROMACS' unit)."""
    n = len(times)
    lo, hi = int(n * fit_from), max(int(n * fit_to), int(n * fit_from) + 2)
    if hi - lo < 2:
        return None
    slope = np.polyfit(times[lo:hi], values[lo:hi], 1)[0]   # nm^2/ps
    return slope / 6.0 * 100.0                              # -> 1e-5 cm^2/s
