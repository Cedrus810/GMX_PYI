"""MDP files as Python objects.

Keys are stored GROMACS-style (``ref-t``), but accepted and exposed in
Python style (``ref_t``) too, so ``mdp.ref_t`` and ``mdp["ref-t"]`` are the
same thing.  Values may be :class:`~gmxpy.units.Quantity` -- they are
written out in canonical GROMACS units.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import MdpError
from .units import Quantity

# Not exhaustive -- unknown keys are reported as warnings, never rejected,
# because grompp remains the authority on what is valid.
KNOWN_KEYS = set("""
integrator tinit dt nsteps init-step simulation-part comm-mode nstcomm comm-grps
emtol emstep niter fcstep nstcgsteep nbfgscorr
nstxout nstvout nstfout nstlog nstcalcenergy nstenergy nstxout-compressed
compressed-x-precision compressed-x-grps energygrps
cutoff-scheme nstlist pbc periodic-molecules verlet-buffer-tolerance rlist
coulombtype coulomb-modifier rcoulomb-switch rcoulomb epsilon-r epsilon-rf
vdwtype vdw-modifier rvdw-switch rvdw dispcorr table-extension
fourierspacing fourier-nx fourier-ny fourier-nz pme-order ewald-rtol
ewald-rtol-lj lj-pme-comb-rule ewald-geometry epsilon-surface
implicit-solvent
tcoupl nsttcouple tc-grps tau-t ref-t
pcoupl pcoupltype nstpcouple tau-p compressibility ref-p refcoord-scaling
gen-vel gen-temp gen-seed
constraints constraint-algorithm continuation shake-tol lincs-order
lincs-iter lincs-warnangle morse
energygrp-excl
freezegrps freezedim cos-acceleration deform
free-energy expanded init-lambda init-lambda-state delta-lambda nstdhdl
fep-lambdas mass-lambdas coul-lambdas vdw-lambdas bonded-lambdas
restraint-lambdas temperature-lambdas calc-lambda-neighbors
sc-alpha sc-power sc-r-power sc-sigma sc-coul couple-moltype couple-lambda0
couple-lambda1 couple-intramol dhdl-derivatives dhdl-print-energy
separate-dhdl-file dh-hist-size dh-hist-spacing
pull pull-ngroups pull-ncoords
awh awh-nbias
disre disre-weighting disre-mixed disre-fc disre-tau nstdisreout
orire orire-fc orire-tau orire-fitgrp nstorireout
qmmm qmmm-grps
simulated-tempering nstexpanded lmc-stats lmc-move lmc-seed
userint1 userint2 userint3 userint4 userreal1 userreal2 userreal3 userreal4
define include
nsteps-per-checkpoint
""".split())

_ENUMS = {
    "integrator": {"md", "md-vv", "md-vv-avek", "sd", "bd", "steep", "cg",
                   "l-bfgs", "nm", "tpi", "tpic", "mimic"},
    "tcoupl": {"no", "berendsen", "nose-hoover", "v-rescale", "andersen",
               "andersen-massive"},
    "pcoupl": {"no", "berendsen", "parrinello-rahman", "c-rescale", "mttk"},
    "constraints": {"none", "h-bonds", "all-bonds", "h-angles", "all-angles"},
    "cutoff-scheme": {"verlet", "group"},
    "pbc": {"xyz", "no", "xy"},
    "coulombtype": {"cut-off", "ewald", "pme", "p3m-ad", "reaction-field",
                    "user", "pme-switch", "pme-user", "pme-user-switchnb"},
    "vdwtype": {"cut-off", "pme", "shift", "switch", "user"},
    "dispcorr": {"no", "enerpres", "ener", "allenerpres", "allener"},
}


def _key(name):
    return name.strip().lower().replace("_", "-")


def _fmt(value):
    if isinstance(value, Quantity):
        return f"{value.canonical:g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return " ".join(_fmt(v) for v in value)
    return str(value)


class MDP:
    """An ``.mdp`` parameter set."""

    def __init__(self, _base=None, **kwargs):
        self._data = {}
        if _base:
            self.update(**{k: v for k, v in dict(_base).items()})
        self.update(**kwargs)

    # -- construction --------------------------------------------------
    @classmethod
    def read(cls, path):
        mdp = cls()
        for line in Path(path).read_text().splitlines():
            line = line.split(";")[0].strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            mdp._data[_key(key)] = value.strip()
        return mdp

    @classmethod
    def preset(cls, kind, **overrides):
        """Sane starting points: ``em``, ``nvt``, ``npt``, ``md``."""
        common = dict(
            cutoff_scheme="Verlet", nstlist=20, coulombtype="PME",
            rcoulomb=1.0, rvdw=1.0, pbc="xyz", dispcorr="EnerPres",
        )
        if kind == "em":
            base = dict(integrator="steep", emtol=1000.0, emstep=0.01,
                        nsteps=50000, nstenergy=100, nstlog=100, **common)
        elif kind in ("nvt", "npt", "md"):
            base = dict(
                integrator="md", dt=0.002, nsteps=500000,
                nstxout_compressed=5000, nstenergy=1000, nstlog=1000,
                continuation="no" if kind == "nvt" else "yes",
                constraints="h-bonds", constraint_algorithm="lincs",
                tcoupl="V-rescale", tc_grps="System", tau_t=0.1, ref_t=300,
                gen_vel="yes" if kind == "nvt" else "no",
                **common)
            if kind == "nvt":
                base.update(gen_temp=300, gen_seed=-1)
            if kind in ("npt", "md"):
                base.update(pcoupl="C-rescale", pcoupltype="isotropic",
                            tau_p=2.0, ref_p=1.0, compressibility=4.5e-5)
        else:
            raise MdpError(f"unknown preset {kind!r} (em, nvt, npt, md)")
        base.update(overrides)
        return cls(**base)

    # -- mapping interface ---------------------------------------------
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value
        return self

    def copy(self, **overrides):
        return MDP(self._data).update(**overrides)

    def __setitem__(self, key, value):
        self._data[_key(key)] = _fmt(value)

    def __getitem__(self, key):
        return self._data[_key(key)]

    def __contains__(self, key):
        return _key(key) in self._data

    def __delitem__(self, key):
        del self._data[_key(key)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        return self._data.get(_key(key), default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def to_dict(self):
        return dict(self._data)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            raw = self._data[_key(name)]
        except KeyError:
            pass
        else:
            # attribute access returns numbers as numbers; use mdp["key"] for
            # the raw string
            try:
                return int(raw)
            except ValueError:
                pass
            try:
                return float(raw)
            except ValueError:
                return raw
        if True:
            raise AttributeError(f"{name!r} is not set in this MDP")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    # -- typed accessors -----------------------------------------------
    def number(self, key, default=None):
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @property
    def simulation_time(self):
        """nsteps * dt, in ps (None for a minimisation)."""
        dt, nsteps = self.number("dt"), self.number("nsteps")
        if dt is None or nsteps is None:
            return None
        return Quantity(dt * nsteps, "ps")

    # -- validation / output -------------------------------------------
    def validate(self, strict=False):
        """Cheap pre-flight check.  grompp stays the real authority."""
        problems = []
        for key, value in self._data.items():
            if key not in KNOWN_KEYS and not key.startswith(("pull-", "awh-", "user")):
                problems.append(f"unknown parameter {key!r}")
            allowed = _ENUMS.get(key)
            if allowed and value.lower() not in allowed:
                problems.append(f"{key} = {value!r} (expected one of "
                                f"{', '.join(sorted(allowed))})")
        for key in ("dt", "nsteps", "nstlist", "rvdw", "rcoulomb"):
            v = self.number(key)
            if v is not None and v < 0:
                problems.append(f"{key} must be >= 0, got {v}")
        if self.get("tcoupl", "no").lower() != "no":
            groups = str(self.get("tc-grps", "")).split()
            for key in ("tau-t", "ref-t"):
                n = len(str(self.get(key, "")).split())
                if groups and n != len(groups):
                    problems.append(
                        f"{key} has {n} value(s) but tc-grps has {len(groups)}")
        if self.get("pcoupl", "no").lower() != "no" and "ref-p" not in self:
            problems.append("pcoupl is on but ref-p is not set")
        if strict and problems:
            raise MdpError("invalid MDP:\n  " + "\n  ".join(problems))
        return problems

    def as_text(self):
        width = max((len(k) for k in self._data), default=0)
        return "".join(f"{k:<{width}} = {v}\n" for k, v in self._data.items())

    def write(self, path):
        path = Path(path)
        path.write_text(self.as_text())
        return path

    def __repr__(self):
        head = ", ".join(f"{k}={v}" for k, v in list(self._data.items())[:4])
        return f"MDP({head}{', ...' if len(self._data) > 4 else ''})"
