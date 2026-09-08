"""Did the simulation actually go well?

``sim.check()`` runs the checks everybody does by hand after a run --
temperature against the thermostat setting, density and energy drift, LINCS
complaints in the log, whether the box was big enough -- and returns a
verdict per check instead of a wall of plots.
"""

from __future__ import annotations

import re
from pathlib import Path

from .mdp import MDP

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"


class Check:
    def __init__(self, name, status, message, value=None):
        self.name = name
        self.status = status
        self.message = message
        self.value = value

    @property
    def passed(self):
        return self.status in (OK, SKIP)

    def __repr__(self):
        mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "--  "}[self.status]
        return f"[{mark}] {self.name}: {self.message}"


class CheckReport:
    def __init__(self, checks):
        self.checks = list(checks)

    @property
    def ok(self):
        return all(c.passed for c in self.checks)

    @property
    def problems(self):
        return [c for c in self.checks if not c.passed]

    def __iter__(self):
        return iter(self.checks)

    def __len__(self):
        return len(self.checks)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.checks[key]
        return next(c for c in self.checks if c.name == key)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([{"check": c.name, "status": c.status,
                              "value": c.value, "message": c.message}
                             for c in self.checks])

    def __str__(self):
        return "\n".join(repr(c) for c in self.checks)

    def __repr__(self):
        bad = len(self.problems)
        return (f"<CheckReport {len(self.checks) - bad}/{len(self.checks)} ok"
                + (f", {bad} to look at" if bad else "") + ">")


def _slope(series):
    """Least-squares slope, per unit of x."""
    n = len(series)
    if n < 3:
        return 0.0
    mean_x = sum(series.x) / n
    mean_y = sum(series.y) / n
    denominator = sum((x - mean_x) ** 2 for x in series.x)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y)
               for x, y in zip(series.x, series.y)) / denominator


def _run_mdp(sim):
    """The mdp actually used, if it is still on disk."""
    for candidate in (sim.workdir / f"{sim.name}_mdout.mdp",
                      sim.workdir / f"{sim.name}.mdp"):
        if Path(candidate).exists():
            try:
                return MDP.read(candidate)
            except Exception:
                pass
    return None


def _n_atoms(sim):
    for source in (lambda: sim.trajectory.n_atoms,
                   lambda: sim.checkpoint.n_atoms):
        try:
            value = source()
            if value:
                return value
        except Exception:
            pass
    return None


def check(sim, drift_tolerance=0.05, temperature_tolerance=5.0,
          drift_per_atom=0.5, box_check=True):
    """Run the standard post-run checks.  Cheap: energies and the log only
    (plus one periodic-image pass if ``box_check``)."""
    checks = []
    log, mdp = sim.log, _run_mdp(sim)

    # --- did it finish -------------------------------------------------
    if not log.exists:
        checks.append(Check("finished", FAIL, f"no log at {log.path.name}"))
        return CheckReport(checks)
    if log.minimisation:
        checks.append(Check(
            "converged", OK if log.converged else WARN,
            f"Fmax = {log.max_force:.4g}, Epot = {log.potential_energy:.6g}"
            + ("" if log.converged else " (hit the step limit)"),
            log.max_force))
        return CheckReport(checks)
    checks.append(Check("finished", OK if log.finished else FAIL,
                        "mdrun finished normally" if log.finished
                        else "mdrun did not reach the end"))

    # --- constraint trouble --------------------------------------------
    lincs = len(re.findall(r"LINCS WARNING|Constraint error", log.text))
    checks.append(Check(
        "constraints", OK if lincs == 0 else (WARN if lincs < 10 else FAIL),
        "no LINCS warnings" if lincs == 0 else f"{lincs} LINCS warning(s) in the log",
        lincs))

    try:
        energy = sim.energy
        terms = energy.terms
    except Exception as exc:
        checks.append(Check("energy", FAIL, f"cannot read the edr: {exc}"))
        return CheckReport(checks)

    # --- thermostat ----------------------------------------------------
    if "Temperature" in terms:
        temperature = energy["Temperature"]
        target = mdp.number("ref-t") if mdp else None
        if target is None and mdp is not None:
            first = str(mdp.get("ref-t", "")).split()
            target = float(first[0]) if first else None
        offset = abs(temperature.mean - target) if target else None
        status = OK if offset is None or offset <= temperature_tolerance else WARN
        checks.append(Check(
            "temperature", status,
            f"{temperature.mean:.1f} +/- {temperature.std:.1f} K"
            + (f" (target {target:g} K)" if target else " (no ref-t found)"),
            temperature.mean))

    # --- drifts ---------------------------------------------------------
    for term, label in (("Density", "density"), ("Volume", "volume")):
        if term not in terms:
            continue
        series = energy[term]
        span = series.x[-1] - series.x[0] if len(series) > 1 else 0
        change = _slope(series) * span
        relative = abs(change) / abs(series.mean) if series.mean else 0
        checks.append(Check(
            f"{label} drift", OK if relative <= drift_tolerance else WARN,
            f"{relative * 100:.2f}% over {span:g} ps "
            f"(mean {series.mean:.4g})", relative))

    # --- energy conservation --------------------------------------------
    conserved = next((t for t in ("Conserved En.", "Total Energy") if t in terms),
                     None)
    if conserved:
        series = energy[conserved]
        span = series.x[-1] - series.x[0] if len(series) > 1 else 0
        n_atoms = _n_atoms(sim)
        temperature = energy["Temperature"].mean if "Temperature" in terms else 300.0
        if span < 20:
            checks.append(Check(
                "energy drift", SKIP,
                f"only {span:g} ps: too short to mean anything"))
        elif n_atoms:
            # the usual acceptance criterion: kT per ns per atom
            per_ns = _slope(series) * 1000.0
            per_atom = per_ns / n_atoms / (0.0083144626 * temperature)
            checks.append(Check(
                "energy drift", OK if abs(per_atom) <= drift_per_atom else WARN,
                f"{per_atom:+.3g} kT/ns/atom ({per_ns:+.4g} kJ/mol per ns)",
                per_atom))

    # --- was the box big enough -----------------------------------------
    if box_check and sim.trajectory.exists:
        try:
            distance = sim.analysis.periodic_image_distance()
            cutoff = mdp.number("rvdw", 1.0) if mdp else 1.0
            margin = distance.min - 2 * cutoff
            checks.append(Check(
                "periodic images", OK if margin > 0 else WARN,
                f"closest image {distance.min:.3f} nm, twice the cut-off is "
                f"{2 * cutoff:g} nm", distance.min))
        except Exception as exc:
            checks.append(Check("periodic images", SKIP, str(exc)[:120]))

    return CheckReport(checks)
