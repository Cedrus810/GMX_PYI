"""Many trajectories at once: replicas, lambda windows, continuation chunks.

    project = Project("runs")            # finds every <name>.tpr underneath
    project.summary()                    # did they all finish?
    project.energy["Potential"].plot()   # one line per run
    project.analysis.rmsd(sel).plot(band=True)   # mean +/- sd across replicas
    project.free_energy()                # BAR over the lambda windows

A :class:`Selection` handed to the project is re-bound to each run, so one
selection object works across every trajectory.
"""

from __future__ import annotations

import re
from pathlib import Path

from .command import abspath
from .data import Series
from .environment import Environment
from .errors import AnalysisError, GmxpyError
from .selection import Selection, Selector
from .simulation import Simulation


def _natural(name):
    """Sort lambda_2 before lambda_10."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", str(name))]


class SeriesGroup:
    """The same observable measured on several runs."""

    def __init__(self, items, name=""):
        self.items = dict(items)          # run name -> Series
        self.name = name or next(iter(self.items.values())).name

    # -- mapping -------------------------------------------------------
    @property
    def names(self):
        return list(self.items)

    def __getitem__(self, key):
        return self.items[key]

    def __iter__(self):
        return iter(self.items.values())

    def __len__(self):
        return len(self.items)

    # -- statistics ----------------------------------------------------
    @property
    def means(self):
        """Per-run mean."""
        return {name: series.mean for name, series in self.items.items()}

    def _aligned(self):
        series = list(self.items.values())
        if not series:
            raise AnalysisError("empty group")
        n = min(len(s) for s in series)
        if n == 0:
            raise AnalysisError("a run produced no data")
        return series[0].x[:n], [s.y[:n] for s in series]

    def mean_series(self):
        """Mean across runs, frame by frame."""
        x, ys = self._aligned()
        values = [sum(column) / len(column) for column in zip(*ys)]
        first = next(iter(self.items.values()))
        return Series(x, values, f"{self.name} (mean of {len(ys)})",
                      first.xlabel, first.ylabel)

    def std_series(self):
        x, ys = self._aligned()
        out = []
        for column in zip(*ys):
            if len(column) < 2:
                out.append(0.0)
                continue
            m = sum(column) / len(column)
            out.append((sum((v - m) ** 2 for v in column) / (len(column) - 1)) ** 0.5)
        first = next(iter(self.items.values()))
        return Series(x, out, f"{self.name} (sd)", first.xlabel, first.ylabel)

    # -- output --------------------------------------------------------
    def plot(self, band=False, ax=None, **kwargs):
        """Overlay every run, or the mean with a +/- sd band."""
        from .plotting import Plot, _pyplot, plot as plot_series

        if not band:
            handle = None
            for name, series in self.items.items():
                handle = plot_series(series, ax=handle.ax if handle else ax,
                                     label=name, title=self.name, **kwargs)
            handle.ax.legend()
            return handle

        plt = _pyplot()
        mean, sd = self.mean_series(), self.std_series()
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 3.6)))
        else:
            fig = ax.figure
        lower = [m - s for m, s in zip(mean.y, sd.y)]
        upper = [m + s for m, s in zip(mean.y, sd.y)]
        ax.fill_between(mean.x, lower, upper, alpha=0.25, label="+/- sd")
        ax.plot(mean.x, mean.y, label=f"mean of {len(self)}", **kwargs)
        ax.set_xlabel(mean.xlabel)
        ax.set_ylabel(mean.ylabel)
        ax.set_title(self.name)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return Plot(fig, ax)

    def to_dataframe(self, wide=False):
        import pandas as pd
        if wide:
            x, ys = self._aligned()
            data = {"x": x}
            data.update({name: y for name, y in zip(self.names, ys)})
            return pd.DataFrame(data)
        frames = []
        for name, series in self.items.items():
            frame = series.to_dataframe()
            frame.insert(0, "run", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def __repr__(self):
        return (f"<SeriesGroup {self.name!r} over {len(self)} runs: "
                + ", ".join(f"{k}={v:.4g}" for k, v in list(self.means.items())[:4])
                + (", ..." if len(self) > 4 else "") + ">")


class ProjectEnergy:
    """``project.energy["Potential"]`` -> one Series per run."""

    def __init__(self, project):
        self.project = project

    @property
    def terms(self):
        return next(iter(self.project.runs.values())).energy.terms

    def __contains__(self, term):
        return term in next(iter(self.project.runs.values())).energy

    def __getitem__(self, term):
        return SeriesGroup({name: sim.energy[term]
                            for name, sim in self.project.runs.items()}, term)

    def to_dataframe(self, terms=None):
        import pandas as pd
        frames = []
        for name, sim in self.project.runs.items():
            frame = sim.energy.to_dataframe(terms)
            frame.insert(0, "run", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def __repr__(self):
        return f"<ProjectEnergy over {len(self.project)} runs>"


class ProjectAnalysis:
    """Any :class:`~gmxpy.analysis.Analysis` method, fanned out over runs."""

    def __init__(self, project):
        self.project = project

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def fanned(*args, **kwargs):
            results = {}
            for run, sim in self.project.runs.items():
                method = getattr(sim.analysis, name, None)
                if method is None:
                    raise AttributeError(f"Analysis has no {name!r}")
                bound = [_rebind(a, sim) for a in args]
                bound_kwargs = {k: _rebind(v, sim) for k, v in kwargs.items()}
                results[run] = method(*bound, **bound_kwargs)
            if results and isinstance(next(iter(results.values())), Series):
                return SeriesGroup(results)      # named after the observable
            return results

        fanned.__name__ = name
        return fanned

    def __repr__(self):
        return f"<ProjectAnalysis over {len(self.project)} runs>"


def _rebind(value, sim):
    """A selection built against one run works against all of them."""
    if isinstance(value, Selection):
        return Selection(value.node, sim.context, value.label)
    return value


class Project:
    """A set of related simulations."""

    def __init__(self, workdir=".", runs=None, name=None, env=None,
                 pattern="*"):
        self.workdir = abspath(workdir)
        self.name = name or self.workdir.name
        self.env = env or Environment.detect()
        self.runs = dict(runs) if runs else self._discover(pattern)
        if not self.runs:
            raise GmxpyError(f"no runs found under {self.workdir} "
                             "(a run is a directory containing <name>.tpr)")

    # -- discovery -----------------------------------------------------
    def _discover(self, pattern):
        found = {}
        for tpr in sorted(self.workdir.glob(f"{pattern}/*.tpr"),
                          key=lambda p: _natural(p)):
            directory = tpr.parent
            label = directory.name if directory != self.workdir else tpr.stem
            if len(list(directory.glob("*.tpr"))) > 1:
                label = f"{directory.name}/{tpr.stem}"
            found[label] = Simulation.open(tpr.stem, workdir=directory,
                                           env=self.env)
        for tpr in sorted(self.workdir.glob("*.tpr"), key=lambda p: _natural(p)):
            found.setdefault(tpr.stem,
                             Simulation.open(tpr.stem, workdir=self.workdir,
                                             env=self.env))
        return dict(sorted(found.items(), key=lambda kv: _natural(kv[0])))

    def add(self, name, simulation):
        self.runs[name] = simulation
        return self

    # -- mapping -------------------------------------------------------
    @property
    def names(self):
        return list(self.runs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.runs[self.names[key]]
        return self.runs[key]

    def __iter__(self):
        return iter(self.runs.values())

    def __len__(self):
        return len(self.runs)

    @property
    def first(self):
        return next(iter(self.runs.values()))

    # -- data ----------------------------------------------------------
    @property
    def energy(self):
        return ProjectEnergy(self)

    @property
    def analysis(self):
        return ProjectAnalysis(self)

    @property
    def select(self):
        """Selector bound to the first run; selections re-bind per run."""
        return Selector(self.first.context)

    def summary(self, as_frame=False):
        rows = []
        for name, sim in self.runs.items():
            row = {"run": name}
            row.update(sim.result.summary())
            rows.append(row)
        if as_frame:
            import pandas as pd
            return pd.DataFrame(rows)
        return rows

    @property
    def finished(self):
        return all(sim.result.finished for sim in self)

    # -- operations ----------------------------------------------------
    def concatenate(self, output=None, runs=None, mode="continuation", **kwargs):
        """trjcat the trajectories into one.

        ``mode="continuation"`` (default) is for chunks of one run: frames
        with duplicate times are dropped.  ``mode="append"`` keeps every
        frame, which is what you want when stacking independent replicas --
        the default would silently throw two thirds of them away.
        """
        from . import command as cmd
        from .data import Trajectory

        selected = [self.runs[name] for name in (runs or self.names)]
        files = [str(sim.trajectory.path) for sim in selected
                 if sim.trajectory.exists]
        if len(files) < 2:
            raise AnalysisError("need at least two trajectories to concatenate")
        if mode not in ("continuation", "append"):
            raise AnalysisError("mode must be 'continuation' or 'append'")
        target = abspath(output or self.workdir / f"{self.name}_all.xtc")
        cmd.Trjcat(env=self.env, trajectory=files, output=str(target),
                   cat=(mode == "append"), **kwargs).run(cwd=self.workdir)
        return Trajectory(target, tpr=selected[0].tpr, env=self.env)

    @property
    def dhdl_files(self):
        """``<name>.xvg`` written by mdrun for a free-energy run, in run order."""
        out = []
        for name, sim in self.runs.items():
            path = sim.workdir / f"{sim.name}.xvg"
            if path.exists():
                out.append(path)
        return out

    def free_energy(self, temperature=None, begin=None, end=None):
        """BAR over the lambda windows, in run order."""
        from .analysis import bar

        files = self.dhdl_files
        if len(files) < 2:
            raise AnalysisError(
                "no dhdl output found; a free-energy run writes <deffnm>.xvg")
        return bar(files, temperature=temperature, begin=begin, end=end,
                   env=self.env, workdir=self.workdir)

    def report(self, path=None, **kwargs):
        from .report import project_report
        return project_report(self, path, **kwargs)

    def __repr__(self):
        done = sum(1 for sim in self if sim.result.finished)
        return (f"<Project {self.name!r}: {len(self)} runs, {done} finished "
                f"[{', '.join(self.names[:5])}"
                f"{', ...' if len(self) > 5 else ''}]>")
