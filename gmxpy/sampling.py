"""Enhanced sampling: pull coordinates, umbrella windows + WHAM, temperature
REMD, and the PLUMED hook.

    pull = Pull(sim.select.protein, sim.select.ligand, k=1000)   # umbrella
    sim = Simulation(structure=..., topology=..., mdp=mdp, pull=pull, name="md")

``Simulation(pull=...)`` turns the two selections into index groups, adds the
``pull-*`` mdp keys and passes the index to grompp; mdrun writes
``<name>_pullx.xvg`` / ``<name>_pullf.xvg`` next to the other output.

Umbrella sampling: pick a starting frame per window from any trajectory that
already spans the reaction coordinate, run the windows (locally, or as one
PBS array job via ``project.submit()``), then WHAM them:

    umb = Umbrella(pull, values=np.arange(0.4, 1.21, 0.05), k=1000,
                   structure=..., topology=..., mdp=MDP.preset("md"),
                   workdir="umb")
    umb.from_trajectory(traj)          # frame nearest each value
    umb.prepare(); ...run...
    pmf = wham(umb)
    pmf.plot()

Temperature REMD builds one directory per replica and runs them together
with ``mdrun -multidir``; ``demux()`` reads the exchange history back out of
the logs.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import command as cmd
from .command import abspath
from .data import Series, read_xvg
from .errors import AnalysisError, GmxpyError
from .mdp import MDP
from .units import value_in

_GEOMETRIES = ("distance", "direction", "position", "cylinder")
_PULL_TYPES = ("umbrella", "constraint", "constant-force", "flat-bottom",
               "external-potential")


class Pull:
    """One pull coordinate between two selections.

    ``k`` is the force constant in GROMACS canonical units (kJ/mol/nm^2) --
    there is no composite unit in :mod:`gmxpy.units`, so a bare float it is.
    ``init`` is the reference distance, ``rate`` the pulling speed (nm/ps).
    """

    def __init__(self, a, b, geometry="distance", type="umbrella", k=None,
                 rate=None, init=None, start=None, vec=None,
                 nstxout=50, nstfout=50):
        if geometry not in _GEOMETRIES:
            raise GmxpyError(f"unknown pull geometry {geometry!r} "
                             f"(one of {', '.join(_GEOMETRIES)})")
        if type not in _PULL_TYPES:
            raise GmxpyError(f"unknown pull type {type!r} "
                             f"(one of {', '.join(_PULL_TYPES)})")
        self.a, self.b = a, b
        self.geometry = geometry
        self.type = type
        self.k = k                          # kJ/mol/nm^2 (canonical)
        self.rate = rate                    # nm/ps (canonical)
        self.init = value_in(init, "nm") if init is not None else None
        self.start = start
        self.vec = vec
        self.nstxout, self.nstfout = nstxout, nstfout

    # -- mdp --------------------------------------------------------------
    def mdp_keys(self):
        """The ``pull-*`` block, as a dict MDP.update() accepts."""
        keys = {
            "pull": "yes",
            "pull-ngroups": 2,
            "pull-ncoords": 1,
            "pull-group1-name": "gmxpy_pull0",
            "pull-group2-name": "gmxpy_pull1",
            "pull-coord1-groups": "1 2",
            "pull-coord1-type": self.type,
            "pull-coord1-geometry": self.geometry,
            "pull-nstxout": self.nstxout,
            "pull-nstfout": self.nstfout,
        }
        if self.k is not None:
            keys["pull-coord1-k"] = self.k
        if self.init is not None:
            keys["pull-coord1-init"] = self.init
        if self.start is not None:
            keys["pull-coord1-start"] = "yes" if self.start else "no"
        if self.rate is not None:
            keys["pull-coord1-rate"] = self.rate
        if self.vec is not None:
            keys["pull-coord1-vec"] = " ".join(f"{float(v):g}" for v in self.vec)
        # grompp refuses a pull group larger than half the box without an
        # explicit reference atom; write_ndx() fills these in per group
        if getattr(self, "_pbcatom", None):
            for n, (label, value) in enumerate(self._pbcatom.items(), start=1):
                keys[f"pull-group{n}-pbcatom"] = value
            # a group that spans half the box also wants this (grompp says so)
            keys["pull-pbc-ref-prev-step-com"] = "yes"
        return keys

    def copy(self, **changes):
        """A clone with some fields changed (used to vary ``init`` per window)."""
        options = dict(a=self.a, b=self.b, geometry=self.geometry,
                       type=self.type, k=self.k, rate=self.rate,
                       init=self.init, start=self.start, vec=self.vec,
                       nstxout=self.nstxout, nstfout=self.nstfout)
        options.update(changes)
        return Pull(**options)

    # -- index groups -----------------------------------------------------
    def _selection(self, side, context):
        """Bind one side (Selection / node / raw string) to a context."""
        from .selection import Selection, _node

        if isinstance(side, Selection):
            return Selection(side.node, context or side.context, side.label)
        return Selection(_node(side), context)

    def write_ndx(self, path, user_index=None, context=None):
        """The two pull groups (plus the user's groups) as one index file.

        pull-groupX-name only ever accepts index group names, so selections
        are evaluated once and written as ``gmxpy_pull0`` / ``gmxpy_pull1``.
        Also picks each group's pbcatom (nearest-to-centre atom), which
        grompp demands for groups larger than half the box -- call this
        before mdp_keys() for it to land in the mdp.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = Path(user_index).read_text() if user_index else ""
        self._pbcatom = {}
        gro = context._gro() if context is not None else None
        for n, (label, side) in enumerate((("gmxpy_pull0", self.a),
                                           ("gmxpy_pull1", self.b)), start=1):
            selection = self._selection(side, context)
            part = path.with_name(f"{path.stem}_{label}.part")
            selection.write_ndx(part, name=label)
            text += part.read_text()
            part.unlink(missing_ok=True)
            if gro is not None:
                self._pbcatom[f"gmxpy_pull{n}"] = _pbcatom(
                    selection.to_indices(), gro)
        path.write_text(text)
        return path

    def __repr__(self):
        return (f"<Pull {self.geometry} {self.type}"
                + (f" k={self.k}" if self.k is not None else "") + ">")


class Umbrella:
    """A set of umbrella windows along one pull coordinate."""

    def __init__(self, pull, values, k=None, structure=None, topology=None,
                 mdp=None, workdir="umbrella", env=None, name="md"):
        self.pull = pull.copy(k=k) if k is not None else pull
        self.values = [value_in(v, "nm") for v in values]
        if not self.values:
            raise GmxpyError("umbrella sampling needs at least one window")
        self.structure = abspath(structure)
        self.topology = abspath(topology)
        self.mdp = mdp if isinstance(mdp, MDP) else (
            MDP.read(mdp) if mdp else MDP.preset("md"))
        self.workdir = abspath(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.name = name
        self.windows = []

    # -- windows ------------------------------------------------------------
    def build(self, structures=None):
        """Create one Simulation per window value.

        ``structures``: one starting .gro per value (e.g. from
        :meth:`from_trajectory`); without it every window starts from
        ``structure=`` and the umbrella pulls it into place.
        """
        from .simulation import Simulation

        structures = structures or [self.structure] * len(self.values)
        if len(structures) != len(self.values):
            raise GmxpyError(f"{len(self.values)} window values but "
                             f"{len(structures)} starting structures")
        self.windows = []
        for i, (value, gro) in enumerate(zip(self.values, structures)):
            directory = self.workdir / f"win_{i:02d}"
            pull = self.pull.copy(init=value)
            self.windows.append(Simulation(
                structure=gro, topology=self.topology, mdp=self.mdp.copy(),
                name=self.name, workdir=directory, env=self.env, pull=pull))
        return self.windows

    def coordinate_series(self, trajectory):
        """The pull coordinate (distance geometry) over a trajectory."""
        import numpy as np
        from . import fast
        from .selection import SelectionContext

        if self.pull.geometry != "distance":
            raise GmxpyError("coordinate_series() only does geometry='distance'")
        context = SelectionContext(trajectory.tpr or trajectory.structure,
                                   env=self.env or trajectory.env,
                                   workdir=trajectory.path.parent)
        traj = trajectory.load(engine="mdtraj")
        ia = np.asarray(self.pull._selection(self.pull.a, context).to_indices())
        ib = np.asarray(self.pull._selection(self.pull.b, context).to_indices())
        delta = _min_image(traj, _com(traj, ia), _com(traj, ib))
        return traj.time, np.linalg.norm(delta, axis=1)

    def from_trajectory(self, trajectory):
        """Pick the frame nearest each window value and build the windows."""
        times, coordinate = self.coordinate_series(trajectory)
        used, structures = set(), []
        for value in self.values:
            best = min((abs(float(c) - value), i)
                       for i, c in enumerate(coordinate) if i not in used)[1]
            used.add(best)
            gro = self.workdir / f"win_{len(structures):02d}_start.gro"
            trajectory.frame(time=float(times[best]), output=gro)
            structures.append(gro)
        return self.build(structures)

    # -- running ------------------------------------------------------------
    def prepare(self, maxwarn=1, **kwargs):
        return [w.prepare(maxwarn=maxwarn, **kwargs) for w in self.windows]

    def project(self):
        """The windows as a :class:`~gmxpy.project.Project` (submit/wham)."""
        from .project import Project
        return Project(self.workdir)


def _com(traj, indices):
    """Mass-weighted centre of mass per frame, in nm."""
    import numpy as np
    from . import fast

    w = fast.masses(traj, indices)
    return (traj.xyz[:, indices] * w[None, :, None]).sum(axis=1) / w.sum()


def _pbcatom(indices, gro_path):
    """The group-internal (1-based) index of the atom nearest the centre.

    GROMACS demands a reference atom once a pull group spans more than half
    the box; the centre-nearest atom is what it recommends.
    """
    lines = Path(gro_path).read_text(errors="replace").splitlines()
    pts = [(float(lines[2 + i][20:28]), float(lines[2 + i][28:36]),
            float(lines[2 + i][36:44])) for i in indices]
    centre = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
    distances = [sum((p[k] - centre[k]) ** 2 for k in range(3)) for p in pts]
    return distances.index(min(distances)) + 1


def _min_image(traj, a, b):
    """b - a reduced into the primary image of a triclinic box (the best of
    the 27 neighbouring images -- exact for the shapes editconf produces)."""
    import numpy as np

    delta = b - a
    box = traj.unitcell_vectors          # (n_frames, 3, 3), rows are vectors
    if box is None:
        return delta
    best = delta.copy()
    best_norm = np.linalg.norm(best, axis=1, keepdims=True)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                shift = (box * np.array([i, j, k])[None, :, None]).sum(axis=1)
                shifted = delta + shift
                norm = np.linalg.norm(shifted, axis=1, keepdims=True)
                closer = norm < best_norm
                best = np.where(closer, shifted, best)
                best_norm = np.where(closer, norm, best_norm)
    return best


# ---------------------------------------------------------------------
# WHAM
# ---------------------------------------------------------------------

def _wham_inputs(source):
    """(tprs, pullfs, workdir) from lists of files, an Umbrella or a Project."""
    from .project import Project

    if isinstance(source, Umbrella):
        if not source.windows:
            raise GmxpyError("build() the umbrella windows first")
        tprs = [w.tpr for w in source.windows]
        pullfs = [w.workdir / f"{w.name}_pullf.xvg" for w in source.windows]
        return tprs, pullfs, source.workdir
    if isinstance(source, Project):
        tprs = [sim.tpr for sim in source.runs.values()]
        pullfs = [sim.workdir / f"{sim.name}_pullf.xvg"
                  for sim in source.runs.values()]
        return tprs, pullfs, source.workdir
    files = [Path(abspath(f)) for f in source]
    if not files:
        raise GmxpyError("wham() got no window files")
    return files, None, files[0].parent


def wham(source, pullfs=None, temperature=300.0, bins=200, begin=None,
         end=None, bootstrap=0, env=None, workdir=None, **extra):
    """Umbrella WHAM: ``gmx wham`` -> the PMF as a Series (kJ/mol).

        pmf = wham(umbrella)                  # an Umbrella or a Project
        pmf = wham(sorted(glob("win_*/md.tpr")),
                   pullfs=sorted(glob("win_*/md_pullf.xvg")))

    ``bootstrap`` is the number of bootstrap samples for the error estimate;
    the result then carries ``.bootstrap`` and ``.error``.
    """
    tprs, found_pullfs, base = _wham_inputs(source)
    if pullfs is None:
        pullfs = found_pullfs
    if pullfs is None:
        # default naming: the pullf sits next to each tpr's run
        pullfs = [Path(t).parent / f"{Path(t).stem}_pullf.xvg" for t in tprs]
    pullfs = [Path(abspath(f)) for f in pullfs]
    missing = [f.name for f in pullfs if not f.exists()]
    if missing:
        raise AnalysisError("missing pull output(s): " + ", ".join(missing))

    scratch = Path(workdir or base) / ".gmxpy"
    scratch.mkdir(parents=True, exist_ok=True)
    tpr_list = scratch / "wham_tpr.dat"
    pullf_list = scratch / "wham_pullf.dat"
    tpr_list.write_text("".join(f"{Path(t).resolve()}\n" for t in tprs))
    pullf_list.write_text("".join(f"{f.resolve()}\n" for f in pullfs))

    out, hist = scratch / "wham_pmf.xvg", scratch / "wham_hist.xvg"
    kwargs = dict(tprs=str(tpr_list), pullfs=str(pullf_list),
                  output=str(out), histogram=str(hist),
                  temperature=value_in(temperature, "K"), bins=bins)
    bsres = None
    if begin is not None:
        kwargs["b"] = value_in(begin, "ps")
    if end is not None:
        kwargs["e"] = value_in(end, "ps")
    if bootstrap:
        bsres = scratch / "wham_bsres.xvg"
        kwargs.update(n_bootstrap=bootstrap, bsres=str(bsres))
    cmd.Wham(env=env, **kwargs, **extra).run()

    columns, _ = read_xvg(out)
    xlabel = "Distance (nm)"        # distance is the geometry we build
    series = Series(columns[0], columns[1], "PMF", xlabel, "PMF (kJ/mol)", out)
    hist_columns, _ = read_xvg(hist)
    series.histogram = Series(hist_columns[0], hist_columns[1], "Histogram",
                              xlabel, "Count", hist)
    if bsres is not None and bsres.exists():
        bs, _ = read_xvg(bsres)
        series.bootstrap = Series(bs[0], bs[1], "PMF (bootstrap mean)",
                                  xlabel, "PMF (kJ/mol)", bsres)
        series.error = list(bs[2]) if len(bs) > 2 else None
    return series


# ---------------------------------------------------------------------
# temperature REMD
# ---------------------------------------------------------------------

def temperature_ladder(low, high, n):
    """Geometrically spaced temperatures -- the usual REMD ladder."""
    low, high = value_in(low, "K"), value_in(high, "K")
    if n < 2 or high <= low:
        raise GmxpyError("need n >= 2 and high > low")
    ratio = (high / low) ** (1.0 / (n - 1))
    return [low * ratio ** i for i in range(n)]


class TemperatureREMD:
    """One directory per temperature, run together by ``mdrun -multidir``."""

    def __init__(self, mdp, temperatures, structure, topology, workdir="remd",
                 env=None, name="md"):
        self.temperatures = [value_in(t, "K") for t in temperatures]
        if len(self.temperatures) < 2:
            raise GmxpyError("REMD needs at least two temperatures")
        self.structure = abspath(structure)
        self.topology = abspath(topology)
        self.workdir = abspath(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.name = name
        self.mdp = mdp if isinstance(mdp, MDP) else (
            MDP.read(mdp) if mdp else MDP.preset("md"))
        from .simulation import Simulation

        self.sims = []
        for i, temperature in enumerate(self.temperatures):
            stage = self.mdp.copy(ref_t=temperature, gen_temp=temperature)
            if str(stage.get("gen-vel", "no")).lower() == "yes":
                stage.update(gen_seed=int(stage.number("gen-seed") or i + 1))
            self.sims.append(Simulation(
                structure=self.structure, topology=self.topology, mdp=stage,
                name=name, workdir=self.workdir / f"r{i}", env=self.env))

    def prepare(self, maxwarn=1, **kwargs):
        return [sim.prepare(maxwarn=maxwarn, **kwargs) for sim in self.sims]

    # -- the mdrun that runs every replica at once -------------------------
    def _argv(self, replex=100, **kwargs):
        """``mdrun -multidir`` argv, via an external-MPI build when needed.

        A thread-MPI build refuses -multidir ("Multi-simulations are only
        supported ... with a proper external MPI library"), but such
        installs usually ship gmx_mpi next to gmx -- use it via mpirun.
        """
        import shutil

        env = self.env
        if env is None:
            from .environment import Environment
            env = self.env = Environment.detect()
        options = dict(tpr=f"{self.name}.tpr",
                       multidir=[str(sim.workdir) for sim in self.sims],
                       replex=replex,
                       **{k: v for k, v in kwargs.items() if v is not None})
        mpirun = shutil.which("mpirun")     # multidir wants a rank per replica
        if not mpirun:
            raise GmxpyError("REMD needs mdrun -multidir under mpirun, but "
                             "there is no mpirun on $PATH")
        if env.has("mpi"):
            exe = env.executable
        else:
            exe = str(Path(env.executable).parent / "gmx_mpi")
            if not Path(exe).exists():
                raise GmxpyError(
                    f"{env.executable} is a thread-MPI build, which cannot "
                    "run -multidir; and no gmx_mpi sits next to it")
        prefix = [mpirun, "-np", str(len(self.sims))]
        runner = cmd.Mdrun(env=env, **options)
        return prefix + [exe, "-quiet", "mdrun"] + runner.build_command()[3:]

    def command(self, replex=100, **kwargs):
        """The single mdrun line that runs every replica."""
        import shlex
        return " ".join(shlex.quote(a) for a in self._argv(replex, **kwargs))

    def run(self, replex=100, check=True, echo=False, **kwargs):
        import subprocess

        from .errors import classify

        if not all(sim.tpr.exists() for sim in self.sims):
            self.prepare()
        argv = self._argv(replex, **kwargs)
        if echo:
            print(" ".join(argv))
        proc = subprocess.run(argv, capture_output=True, text=True,
                              cwd=str(self.workdir), env=self.env.env)
        if proc.returncode != 0:
            raise classify(proc.stderr + proc.stdout, argv,
                           proc.returncode, "mdrun")
        if check:
            unfinished = [sim.name for sim in self.sims
                          if not sim.log.exists or not sim.log.finished]
            if unfinished:
                raise GmxpyError("REMD mdrun exited 0 but the log(s) of "
                                 + ", ".join(unfinished) + " did not finish")
        return proc

    def demux(self):
        """The exchange history, parsed from the replicas' logs."""
        return demux(*[sim.paths["log"] for sim in self.sims],
                     temperatures=self.temperatures)

    def __repr__(self):
        return (f"<TemperatureREMD {len(self.temperatures)} replicas "
                f"{self.temperatures[0]:g}..{self.temperatures[-1]:g} K "
                f"in {self.workdir}>")


class ReplTrace:
    """Replica-exchange history: which temperature each directory ran."""

    def __init__(self, times, temperatures, exchange_fractions=None,
                 exchange_counts=None):
        self.times = list(times)
        self.temperatures = temperatures       # per directory, aligned to times
        self.n_replicas = len(temperatures[0]) if temperatures else 0
        self.exchange_fractions = exchange_fractions
        self.exchange_counts = exchange_counts

    def series(self, replica):
        """One directory's temperature over time."""
        return Series(self.times, [row[replica] for row in self.temperatures],
                      f"r{replica}", "Time (ps)", "Temperature (K)")

    def to_dataframe(self):
        import pandas as pd
        data = {"time": self.times}
        for r in range(self.n_replicas):
            data[f"r{r}"] = [row[r] for row in self.temperatures]
        return pd.DataFrame(data)

    def __repr__(self):
        extra = ""
        if self.exchange_fractions:
            extra = f", lowest exchange {min(self.exchange_fractions) * 100:.1f}%"
        return (f"<ReplTrace {len(self.times)} exchanges over "
                f"{self.n_replicas} replicas{extra}>")


def demux(*logs, temperatures=None):
    """Parse the exchange history out of REMD mdrun logs.

    What GROMACS' demux.pl writes as replica_temp.xvg, without the perl.
    Understands both log layouts: the old ``Repl t=...: 0x 1X`` lines and
    the ``Replica exchange at step ...`` blocks GROMACS has written since
    2024 (``Repl ex  0 x  1`` marks an accepted swap).  ``temperatures``
    (the ladder, in order) turns index traces into kelvin.
    """
    times, rows = [], []
    state = None                     # ladder index per directory
    counts = None
    fractions = None
    pending = None                   # time of the block being read

    def apply_new(tokens):
        """``Repl ex`` tokens: replica labels with 'x' between swapped pairs."""
        nonlocal state, counts
        label, accepted = None, set()
        for token in tokens:
            if token == "x":
                if label is not None:
                    accepted.add(label)
                continue
            try:
                label = int(token)
            except ValueError:
                label = None
        if state is None:
            n = max(accepted) + 2 if accepted else 2
            state = list(range(n))
            counts = [0] * (n - 1)
        for pair in sorted(a for a in accepted if a + 1 < len(state)):
            state[pair], state[pair + 1] = state[pair + 1], state[pair]
            counts[pair] += 1

    for log in logs:
        path = Path(log)
        text = path.read_text(errors="replace") if path.exists() else ""
        lines = text.splitlines()
        for index, line in enumerate(lines):
            match = re.match(
                r"^Replica exchange at step \d+ time ([\d.eE+-]+)", line)
            if match:
                pending = float(match.group(1))
                continue
            match = re.match(r"^Repl ex\s+(.*)$", line)
            if match and pending is not None:
                apply_new(match.group(1).split())
                times.append(pending)
                rows.append(list(state))
                pending = None
                continue
            match = re.match(r"^Repl t=([\d.eE+-]+)\s+step=\d+:\s*(.+)$", line)
            if match:                          # the pre-2024 layout
                tokens = re.findall(r"(\d+)\s*([xX*])", match.group(2))
                if not tokens:
                    continue
                if state is None:
                    state = list(range(len(tokens) + 1))
                    counts = [0] * len(tokens)
                for pair, status in tokens:
                    if status in "X*":         # accepted: the configs swap
                        pair = int(pair)
                        state[pair], state[pair + 1] = state[pair + 1], state[pair]
                        counts[pair] += 1
                times.append(float(match.group(1)))
                rows.append(list(state))
            if "average number of exchanges:" in line \
                    or re.match(r"^Repl\s+av\. x:", line):
                # old format carries the values on the same line; the new
                # one puts them on the lines after the header
                same_line = re.match(r"^Repl\s+av\. x:\s*(.+)$", line)
                candidates = [same_line.group(1)] if same_line else []
                candidates += [later.strip().removeprefix("Repl")
                               for later in lines[index + 1:index + 5]]
                for candidate in candidates:
                    values = candidate.split()
                    if values and all(_looks_numeric(v) for v in values) \
                            and not all(v.isdigit() for v in values):
                        fractions = [float(v) for v in values]
                        break

    if state is None:
        raise AnalysisError(
            f"no exchange records in {', '.join(map(str, logs))}")
    ladder = [float(t) for t in temperatures] if temperatures \
        else list(range(len(state)))
    temperatures_rows = [[ladder[s] for s in row] for row in rows]
    return ReplTrace(times, temperatures_rows, fractions, counts)


def _looks_numeric(token):
    try:
        float(token)
    except ValueError:
        return False
    return True
