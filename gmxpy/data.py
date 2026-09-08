"""Data layer: one object per GROMACS output file.

The user asks for ``sim.energy["Potential"]`` and gets numbers.  XVG files
are an internal detail -- produced in a scratch directory, parsed, and never
shown.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from .errors import AnalysisError, GromacsError
from .units import Quantity


# ---------------------------------------------------------------------
# xvg (internal only)
# ---------------------------------------------------------------------

_XMGRACE_GREEK = {"a": "alpha", "b": "beta", "g": "gamma", "d": "delta",
                  "p": "pi", "D": "Delta", "w": "omega"}


def _clean_legend(text):
    """Strip xmgrace markup: ``\\xa\\f{}-Helices`` -> ``alpha-Helices``."""
    out = re.sub(r"\\x(.)\\f\{\}", lambda m: _XMGRACE_GREEK.get(m.group(1), m.group(1)),
                 text)
    return re.sub(r"\\[sSN]|\\f\{[^}]*\}", "", out).strip()


def read_xvg(path):
    """Return ``(columns, legends)``.  Columns are lists of floats."""
    columns, legends, title, axes = [], [], "", {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("@"):
            m = re.match(r'@\s*s\d+\s+legend\s+"(.*)"', line)
            if m:
                legends.append(_clean_legend(m.group(1)))
            m = re.match(r'@\s*(xaxis|yaxis)\s+label\s+"(.*)"', line)
            if m:
                axes[m.group(1)] = m.group(2)
            m = re.match(r'@\s*title\s+"(.*)"', line)
            if m:
                title = m.group(1)
            continue
        if line.startswith(("#", "&")):
            continue
        values = line.split()
        if not columns:
            columns = [[] for _ in values]
        if len(values) != len(columns):
            continue
        for col, value in zip(columns, values):
            try:
                col.append(float(value))
            except ValueError:
                col.append(float("nan"))
    return columns, {"legends": legends, "title": title, **axes}


# ---------------------------------------------------------------------
# result series
# ---------------------------------------------------------------------

class Series:
    """An (x, y) data series with a name and units.

    Behaves like a light pandas Series: ``.mean``, ``.plot()``,
    ``.to_dataframe()``, iteration, slicing by x value.
    """

    def __init__(self, x, y, name="", xlabel="Time (ps)", ylabel="", source=None):
        self.x = list(x)
        self.y = list(y)
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel or name
        self.source = source

    # aliases that read naturally for time series
    @property
    def time(self):
        return self.x

    @property
    def values(self):
        return self.y

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        return iter(zip(self.x, self.y))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Series(self.x[item], self.y[item], self.name,
                          self.xlabel, self.ylabel, self.source)
        return self.y[item]

    def __array__(self, dtype=None, copy=None):
        import numpy as np
        return np.asarray(self.y, dtype=dtype)

    # -- statistics ----------------------------------------------------
    @property
    def mean(self):
        return sum(self.y) / len(self.y) if self.y else float("nan")

    @property
    def std(self):
        if len(self.y) < 2:
            return float("nan")
        m = self.mean
        return (sum((v - m) ** 2 for v in self.y) / (len(self.y) - 1)) ** 0.5

    @property
    def min(self):
        return min(self.y)

    @property
    def max(self):
        return max(self.y)

    def describe(self):
        return {"n": len(self), "mean": self.mean, "std": self.std,
                "min": self.min, "max": self.max}

    # -- transforms ----------------------------------------------------
    def since(self, start):
        """Drop everything before x >= start (e.g. equilibration)."""
        keep = [i for i, xv in enumerate(self.x) if xv >= float(start)]
        return Series([self.x[i] for i in keep], [self.y[i] for i in keep],
                      self.name, self.xlabel, self.ylabel, self.source)

    def rolling(self, window):
        """Centred running mean.  Cheap smoothing for plots."""
        if window <= 1 or window > len(self.y):
            return self
        half = window // 2
        out = []
        for i in range(len(self.y)):
            lo, hi = max(0, i - half), min(len(self.y), i + half + 1)
            chunk = self.y[lo:hi]
            out.append(sum(chunk) / len(chunk))
        return Series(self.x, out, f"{self.name} (rolling {window})",
                      self.xlabel, self.ylabel, self.source)

    def equilibration_time(self, tolerance=1.0, candidates=20):
        """Where the series stops trending, as an x value.

        Scans candidate start points and takes the earliest one whose
        remaining mean already sits within ``tolerance`` standard errors of
        the second-half mean.
        """
        # ponytail: 20-point scan against the second-half mean; swap in
        # pymbar.timeseries.detect_equilibration if this ever misjudges
        n = len(self.y)
        if n < 10:
            return self.x[0]
        tail = self.y[n // 2:]
        reference = sum(tail) / len(tail)
        spread = (sum((v - reference) ** 2 for v in tail) / max(len(tail) - 1, 1)) ** 0.5
        error = spread / max(len(tail), 1) ** 0.5 or 1e-12
        for k in range(candidates):
            start = int(n * k / (2 * candidates))       # scan the first half
            chunk = self.y[start:]
            if abs(sum(chunk) / len(chunk) - reference) <= tolerance * error:
                return self.x[start]
        return self.x[n // 2]

    def equilibrated(self, **kwargs):
        """The series with its burn-in dropped."""
        return self.since(self.equilibration_time(**kwargs))

    def block_average(self, n_blocks=10):
        """Block averages -- a quick convergence check."""
        size = max(1, len(self.y) // n_blocks)
        xs, ys = [], []
        for start in range(0, len(self.y) - size + 1, size):
            chunk = self.y[start:start + size]
            xs.append(self.x[start + len(chunk) // 2])
            ys.append(sum(chunk) / len(chunk))
        return Series(xs, ys, f"{self.name} (block average)",
                      self.xlabel, self.ylabel, self.source)

    # -- output --------------------------------------------------------
    def plot(self, **kwargs):
        from .plotting import plot
        return plot(self, **kwargs)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({self.xlabel: self.x, self.name or "value": self.y})

    def to_series(self):
        import pandas as pd
        return pd.Series(self.y, index=self.x, name=self.name or "value")

    def to_csv(self, path):
        Path(path).write_text(
            f"{self.xlabel},{self.name or 'value'}\n"
            + "".join(f"{a},{b}\n" for a, b in zip(self.x, self.y)))
        return Path(path)

    def __repr__(self):
        return (f"<Series {self.name!r} n={len(self)} "
                f"mean={self.mean:.4g} std={self.std:.3g}>")


# ---------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------

_EDR_MAGIC = -55555


def read_edr_terms(path):
    """Term names from an .edr header (XDR).  No GROMACS call needed."""
    with open(path, "rb") as fh:
        data = fh.read(65536)
    pos = 0

    def i32():
        nonlocal pos
        value = struct.unpack_from(">i", data, pos)[0]
        pos += 4
        return value

    def text():
        nonlocal pos
        n = i32()
        if n < 0 or n > 4096:
            raise AnalysisError(f"{path}: not a readable .edr header")
        raw = data[pos:pos + n]
        pos += (n + 3) // 4 * 4
        return raw.decode("ascii", "replace").rstrip("\x00").strip()

    magic = i32()
    if magic == _EDR_MAGIC:
        version = i32()
        n_terms = i32()
    else:                      # pre-2004 layout: magic is the term count
        version, n_terms = 1, magic
    terms = []
    for _ in range(n_terms):
        name = text()
        unit = text() if version >= 5 else ""
        terms.append((name, unit))
    return terms


class Energy:
    """``.edr`` contents, addressed by term name."""

    def __init__(self, path, env=None, workdir=None, tpr=None):
        from .command import abspath
        self.path = abspath(path)
        self.env = env
        self.tpr = tpr
        self.workdir = abspath(workdir or self.path.parent)
        self._terms = None
        self._units = {}
        self._cache = {}
        self._frame = None       # pandas DataFrame from panedr, if available

    # -- terms ---------------------------------------------------------
    @property
    def terms(self):
        if self._terms is None:
            if not self.path.exists():
                raise AnalysisError(f"no energy file: {self.path}")
            pairs = read_edr_terms(self.path)
            self._terms = [name for name, _ in pairs]
            self._units = {name: unit for name, unit in pairs}
        return list(self._terms)

    def unit(self, term):
        self.terms
        return self._units.get(self._resolve(term), "")

    def __contains__(self, term):
        try:
            self._resolve(term)
            return True
        except KeyError:
            return False

    def _resolve(self, term):
        """Accept exact, case-insensitive, dashed or partial term names."""
        terms = self.terms
        if term in terms:
            return term
        wanted = str(term).lower().replace("-", " ").replace(".", "")
        norm = {t.lower().replace("-", " ").replace(".", ""): t for t in terms}
        if wanted in norm:
            return norm[wanted]
        hits = [t for key, t in norm.items() if wanted in key]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise KeyError(f"{term!r} is ambiguous: {', '.join(hits)}")
        raise KeyError(f"no energy term {term!r}; available: {', '.join(terms)}")

    # -- data ----------------------------------------------------------
    def __getitem__(self, term):
        name = self._resolve(term)
        if name not in self._cache:
            self._cache[name] = self._extract([name])[0]
        return self._cache[name]

    def _panedr(self):
        if self._frame is None:
            try:
                import panedr
            except ImportError:
                return None
            self._frame = panedr.edr_to_df(str(self.path))
        return self._frame

    def _extract(self, names):
        frame = self._panedr()
        if frame is not None:
            time = list(frame["Time"])
            return [Series(time, list(frame[n]), n, "Time (ps)",
                           f"{n} ({self.unit(n)})", self.path) for n in names]
        return self._extract_via_gmx(names)

    def _extract_via_gmx(self, names):
        """Fallback: drive ``gmx energy`` and answer its menu ourselves."""
        from .command import Energy as EnergyCmd

        scratch = self.workdir / ".gmxpy"
        scratch.mkdir(parents=True, exist_ok=True)
        out = scratch / "energy.xvg"
        numbers = [str(self.terms.index(n) + 1) for n in names]
        cmd = EnergyCmd(env=self.env, edr=str(self.path), output=str(out),
                        xvg="none")
        proc = cmd.run(stdin="\n".join(numbers) + "\n\n", check=False)
        if proc.returncode != 0 or not out.exists():
            raise AnalysisError(
                f"could not read {names} from {self.path.name}:\n"
                + proc.stderr[-500:])
        columns, _ = read_xvg(out)
        out.unlink(missing_ok=True)
        if len(columns) < len(names) + 1:
            raise AnalysisError(f"gmx energy returned {len(columns)} columns "
                                f"for {len(names)} term(s)")
        return [Series(columns[0], columns[i + 1], n, "Time (ps)",
                       f"{n} ({self.unit(n)})", self.path)
                for i, n in enumerate(names)]

    def average(self, term):
        return self[term].mean

    def to_dataframe(self, terms=None):
        import pandas as pd
        frame = self._panedr()
        if frame is not None:
            return frame if terms is None else frame[["Time"] + [self._resolve(t) for t in terms]]
        names = [self._resolve(t) for t in (terms or self.terms)]
        series = self._extract(names)
        data = {"Time": series[0].x}
        data.update({s.name: s.y for s in series})
        return pd.DataFrame(data)

    def plot(self, *terms, **kwargs):
        from .plotting import plot
        return plot([self[t] for t in terms], **kwargs)

    def __repr__(self):
        try:
            n = len(self.terms)
        except Exception:
            return f"<Energy {self.path.name} (unreadable)>"
        return f"<Energy {self.path.name}: {n} terms>"


# ---------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------

class Trajectory:
    """``.xtc``/``.trr`` handle.  Frame loading is delegated to mdtraj/MDA."""

    def __init__(self, path, tpr=None, structure=None, env=None):
        from .command import abspath
        self.path = abspath(path)
        self.tpr = tpr
        self.structure = structure
        self.env = env
        self._info = None

    @property
    def exists(self):
        return self.path.exists()

    def _check(self):
        if self._info is None:
            from .command import Check
            proc = Check(env=self.env, trajectory=str(self.path)).run(check=False)
            text = proc.stderr + proc.stdout
            get = lambda pat, cast=float: (
                cast(re.search(pat, text, re.M).group(1))
                if re.search(pat, text, re.M) else None)
            self._info = {
                "n_frames": get(r"^Coords\s+(\d+)", int),
                "dt": get(r"^Coords\s+\d+\s+([\d.eE+-]+)"),
                "n_atoms": get(r"^#\s*Atoms\s+(\d+)", int),
                "last_time": get(r"Last frame\s+\d+\s+time\s+([\d.eE+-]+)"),
                "text": text,
            }
        return self._info

    @property
    def n_frames(self):
        return self._check()["n_frames"]

    @property
    def n_atoms(self):
        return self._check()["n_atoms"]

    @property
    def dt(self):
        value = self._check()["dt"]
        return Quantity(value, "ps") if value is not None else None

    @property
    def length(self):
        value = self._check()["last_time"]
        return Quantity(value, "ps") if value is not None else None

    @property
    def time(self):
        dt, n = self._check()["dt"], self.n_frames
        if dt is None or n is None:
            return []
        return [i * dt for i in range(n)]

    def _coordinate_topology(self):
        """A .gro the engines can always read, converting the tpr if needed."""
        if self.structure and Path(self.structure).suffix.lower() in (".gro", ".pdb"):
            return str(self.structure)
        if self.tpr:
            scratch = self.path.parent / ".gmxpy"
            scratch.mkdir(parents=True, exist_ok=True)
            cached = scratch / (Path(self.tpr).stem + "_top.gro")
            if not cached.exists():
                from .command import Editconf
                Editconf(env=self.env, structure=str(self.tpr),
                         output=str(cached)).run()
            return str(cached)
        return str(self.structure or "")

    def _topologies_for(self, engine, top=None):
        """Topology candidates, best first.

        mdtraj cannot read .tpr at all.  MDAnalysis can, but only up to the
        tpx version its release knows -- so keep the .gro as a fallback.
        """
        if top:
            return [str(top)]
        if engine == "mdanalysis" and self.tpr:
            return [str(self.tpr), self._coordinate_topology()]
        return [self._coordinate_topology()]

    def load(self, engine=None, top=None, **kwargs):
        """Load frames.  ``engine`` is ``mdtraj`` or ``mdanalysis``."""
        engines = [engine] if engine else ["mdtraj", "mdanalysis"]
        problems = []
        for name in engines:
            try:
                if name in ("mdtraj", "mdanalysis"):
                    return self._load_with(name, self._topologies_for(name, top),
                                           problems, **kwargs)
                raise AnalysisError(f"unknown engine {name!r}")
            except ImportError as exc:
                problems.append(f"{name}: {exc}")
        raise AnalysisError(
            "no trajectory engine available (" + "; ".join(problems)
            + ").  pip install mdtraj  or  pip install MDAnalysis")

    # -- processing (trjconv) ------------------------------------------
    def _ndx_for(self, selections, tag):
        """Write a multi-group ndx; group numbers are the list order."""
        from .selection import Selection, SelectionContext, Selector

        scratch = self.path.parent / ".gmxpy"
        scratch.mkdir(parents=True, exist_ok=True)
        context = None
        text = ""
        for i, (name, selection) in enumerate(selections):
            if not isinstance(selection, Selection):
                if context is None:
                    context = SelectionContext(self.tpr or self.structure,
                                               env=self.env,
                                               workdir=self.path.parent)
                selector = Selector(context)
                selection = (selector.expr(selection) if isinstance(selection, str)
                             else selector.all)
            part = scratch / f"_{tag}_{i}.ndx"
            text += selection.write_ndx(part, name=name).read_text()
            part.unlink(missing_ok=True)
        path = scratch / f"{tag}.ndx"
        path.write_text(text)
        return path

    def process(self, output=None, pbc=None, center=None, fit=None,
                fit_group=None, group=None, unit_cell="compact", skip=None,
                begin=None, end=None, dt=None, **extra):
        """trjconv without the menu.  Returns the new :class:`Trajectory`.

            traj = sim.trajectory.process(pbc="mol", center=sim.select.protein,
                                          fit="rot+trans")

        GROMACS refuses to treat PBC and do a rotational fit in one pass
        ("do the PBC condition treatment first and then run trjconv in a
        second step"), so when both are asked for this runs the two passes
        itself.
        """
        from . import command as cmd

        output = Path(output) if output else self.path.with_name(
            self.path.stem + "_processed" + self.path.suffix)
        if not (self.tpr or self.structure):
            raise AnalysisError("processing needs a tpr or structure")
        out_group = group if group is not None else "all"
        needs_pbc = pbc is not None or center is not None
        needs_fit = fit not in (None, "none")

        source = str(self.path)
        scratch = self.path.parent / ".gmxpy"
        scratch.mkdir(parents=True, exist_ok=True)
        common = {k: v for k, v in
                  dict(skip=skip, begin=begin, end=end, dt=dt).items()
                  if v is not None}
        common.update(extra)

        if needs_pbc:
            target = str(scratch / ("_pbc" + output.suffix)) if needs_fit else str(output)
            # when a fit pass follows, keep every atom: the second pass'
            # index numbers still refer to the original system
            groups = [("center", center if center is not None else out_group),
                      ("output", "all" if needs_fit else out_group)]
            ndx = self._ndx_for(groups, "trjconv_pbc")
            options = dict(pbc=pbc or "mol", unit_cell=unit_cell)
            stdin = "1\n"
            if center is not None:
                options["center"] = True
                stdin = "0\n1\n"
            cmd.Trjconv(env=self.env, tpr=str(self.tpr or self.structure),
                        trajectory=source, index=str(ndx), output=target,
                        **options, **common).run(stdin=stdin)
            source, common = target, {}     # frame selection already applied

        if needs_fit:
            groups = [("fit", fit_group if fit_group is not None else out_group),
                      ("output", out_group)]
            ndx = self._ndx_for(groups, "trjconv_fit")
            cmd.Trjconv(env=self.env, tpr=str(self.tpr or self.structure),
                        trajectory=source, index=str(ndx), output=str(output),
                        fit=fit, **common).run(stdin="0\n1\n")

        if not needs_pbc and not needs_fit:
            ndx = self._ndx_for([("output", out_group)], "trjconv_out")
            cmd.Trjconv(env=self.env, tpr=str(self.tpr or self.structure),
                        trajectory=source, index=str(ndx), output=str(output),
                        **common).run(stdin="0\n")

        return Trajectory(output, tpr=self.tpr, structure=self.structure,
                          env=self.env)

    def frame(self, time=0, output=None, group=None):
        """Extract a single frame (``.gro``/``.pdb``) at ``time`` ps."""
        from . import command as cmd

        output = Path(output or self.path.with_name(
            f"{self.path.stem}_{float(time):g}ps.pdb"))
        ndx = self._ndx_for([("output", group if group is not None else "all")],
                            "trjconv_frame")
        cmd.Trjconv(env=self.env, tpr=str(self.tpr or self.structure),
                    trajectory=str(self.path), index=str(ndx),
                    output=str(output), dump=float(time)).run(stdin="0\n")
        return output

    def _load_with(self, engine, tops, problems, **kwargs):
        for top in tops:
            try:
                if engine == "mdtraj":
                    import mdtraj
                    return mdtraj.load(str(self.path), top=top, **kwargs)
                import MDAnalysis
                return MDAnalysis.Universe(top, str(self.path), **kwargs)
            except ImportError:
                raise
            except Exception as exc:                     # unreadable topology
                problems.append(f"{engine}+{Path(top).name}: {exc}")
        raise AnalysisError(f"{engine} could not read {self.path.name}: "
                            + "; ".join(problems))

    def __repr__(self):
        if not self.exists:
            return f"<Trajectory {self.path.name} (missing)>"
        return f"<Trajectory {self.path.name}: {self.n_frames} frames>"


# ---------------------------------------------------------------------
# log / checkpoint
# ---------------------------------------------------------------------

class LogFile:
    """mdrun ``.log``: did it finish, how fast, and what went wrong."""

    def __init__(self, path):
        from .command import abspath
        self.path = abspath(path)

    @property
    def exists(self):
        return self.path.exists()

    @property
    def text(self):
        return self.path.read_text(errors="replace") if self.exists else ""

    @property
    def tail(self):
        return self.text[-4000:]

    @property
    def finished(self):
        return "Finished mdrun on rank" in self.text

    @property
    def performance(self):
        """ns/day, or None."""
        m = re.search(r"Performance:\s+([\d.]+)", self.text)
        return float(m.group(1)) if m else None

    @property
    def wall_time(self):
        """mdrun wall-clock seconds."""
        m = re.search(r"^\s*Time:\s+[\d.]+\s+([\d.]+)", self.text, re.M)
        return float(m.group(1)) if m else None

    @property
    def steps(self):
        matches = re.findall(r"^\s+Step\s+Time\s*$\n\s+(\d+)", self.text, re.M)
        return int(matches[-1]) if matches else None

    @property
    def version(self):
        m = re.search(r"GROMACS version:\s*(.+)", self.text)
        return m.group(1).strip() if m else ""

    # -- energy minimisation -------------------------------------------
    @property
    def minimisation(self):
        return bool(re.search(r"^(Steepest Descents|Polak-Ribiere|L-BFGS)",
                              self.text, re.M))

    @property
    def converged(self):
        """True/False for a minimisation, None for anything else."""
        if not self.minimisation:
            return None
        return "did not converge" not in self.text

    @property
    def potential_energy(self):
        m = re.search(r"Potential Energy\s*=\s*([-\d.eE+]+)", self.text)
        return float(m.group(1)) if m else None

    @property
    def max_force(self):
        m = re.search(r"Maximum force\s*=\s*([-\d.eE+]+)", self.text)
        return float(m.group(1)) if m else None

    @property
    def warnings(self):
        return [line.strip() for line in self.text.splitlines()
                if "WARNING" in line]

    @property
    def notes(self):
        return re.findall(r"^NOTE:.*", self.text, re.M)

    def __repr__(self):
        if not self.exists:
            return f"<LogFile {self.path.name} (missing)>"
        return (f"<LogFile {self.path.name} finished={self.finished} "
                f"performance={self.performance} ns/day>")


class Checkpoint:
    """``.cpt`` handle -- enough to decide whether a restart is sane."""

    def __init__(self, path, env=None):
        from .command import abspath
        self.path = abspath(path)
        self.env = env
        self._info = None

    @property
    def exists(self):
        return self.path.exists()

    def _dump(self):
        if self._info is None:
            from .command import Dump
            proc = Dump(env=self.env, checkpoint=str(self.path)).run(check=False)
            text = proc.stdout + proc.stderr
            get = lambda pat: (re.search(pat, text, re.M) or [None, None])[1]
            self._info = {
                "step": get(r"^step\s*=\s*(\d+)"),
                "time": get(r"^t\s*=\s*([\d.eE+-]+)"),
                "n_atoms": get(r"^#atoms\s*=\s*(\d+)"),
                "output_file": get(r"^output filename\s*=\s*(\S+)"),
                "text": text,
            }
        return self._info

    @property
    def step(self):
        value = self._dump()["step"]
        return int(value) if value is not None else None

    @property
    def time(self):
        value = self._dump()["time"]
        return Quantity(float(value), "ps") if value is not None else None

    @property
    def n_atoms(self):
        value = self._dump()["n_atoms"]
        return int(value) if value is not None else None

    def __repr__(self):
        if not self.exists:
            return f"<Checkpoint {self.path.name} (missing)>"
        return f"<Checkpoint {self.path.name} step={self.step}>"
