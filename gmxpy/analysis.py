"""Analysis.

Every method takes selections and returns a :class:`~gmxpy.data.Series`.
GROMACS does the geometry (it already knows the topology, PBC and the tpr);
the XVG it emits is written to a scratch directory, parsed, and deleted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import command as cmd
from .command import abspath
from .data import Series, read_xvg
from .errors import AnalysisError, SelectionError
from .selection import Selection
from .units import value_in

# bump when a method changes what it asks gmx for, so old cached results in
# .gmxpy/ are not silently reused
_CACHE_VERSION = 2


class Analysis:
    """``sim.analysis`` -- bound to one simulation's tpr + trajectory."""

    def __init__(self, owner, cache=True, engine="auto"):
        self.sim = owner
        self.cache = cache
        self.engine = engine        # "auto" | "mdtraj" | "gmx"
        self._memo = {}             # in-memory results (the fast backend
                                    # writes no file to cache)

    # -- plumbing ------------------------------------------------------
    @property
    def env(self):
        return self.sim.env

    def _scratch(self):
        path = Path(self.sim.workdir) / ".gmxpy"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _inputs(self, trajectory=None):
        tpr = self.sim.tpr or self.sim.structure
        traj = trajectory or self.sim.trajectory.path
        if not tpr or not Path(tpr).exists():
            raise AnalysisError("no tpr/structure available; run sim.prepare() first")
        if not Path(traj).exists():
            raise AnalysisError(f"no trajectory at {traj}; run sim.run() first")
        return str(tpr), str(traj)

    # -- backend choice ------------------------------------------------
    def _use_fast(self):
        """In-memory backend if it is available and not overridden.

        gmx re-reads the trajectory for every tool it runs; mdtraj reads it
        once.  Both give GROMACS' definitions -- see gmxpy/fast.py.
        """
        if self.engine == "gmx":
            return False
        try:
            import mdtraj  # noqa: F401
        except ImportError:
            if self.engine == "mdtraj":
                raise AnalysisError("engine='mdtraj' needs mdtraj installed")
            return False
        return True

    def _remember(self, key, produce):
        """Memoise an in-memory result for this Analysis instance."""
        if not self.cache:
            return produce()
        if key not in self._memo:
            self._memo[key] = produce()
        return self._memo[key]

    def _sel(self, selection):
        """Accept a Selection, a raw string, or None (= whole system)."""
        if selection is None:
            return self.sim.select.all
        if isinstance(selection, Selection):
            return selection
        if isinstance(selection, str):
            return self.sim.select.expr(selection)
        raise SelectionError(f"cannot use {selection!r} as a selection")

    def _ndx(self, selection, tag):
        path = self._scratch() / f"{tag}.ndx"
        return selection.write_ndx(path, name=tag)

    def _multi_ndx(self, pairs, tag):
        """One .ndx holding several groups; group numbers are the list order.

        Legacy tools ask for groups by number -- we write the file, so we
        already know the answer and the user never sees the menu.
        """
        scratch = self._scratch()
        text = ""
        for i, (name, selection) in enumerate(pairs):
            part = scratch / f"_{tag}_{i}.ndx"
            text += self._sel(selection).write_ndx(part, name=name).read_text()
            part.unlink(missing_ok=True)
        path = scratch / f"{tag}.ndx"
        path.write_text(text)
        return path

    def _xvg(self, tag, *key):
        """Deterministic scratch path: same question -> same file."""
        if not key:
            return self._scratch() / f"{tag}.xvg"
        parts = [str(_CACHE_VERSION)] + [str(k) for k in key]
        digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
        return self._scratch() / f"{tag}_{digest}.xvg"

    def _input_mtime(self):
        newest = 0.0
        for path in (self.sim.tpr, self.sim.structure,
                     self.sim.trajectory.path):
            if path and Path(path).exists():
                newest = max(newest, Path(path).stat().st_mtime)
        return newest

    def _stale(self, path, extra_outputs=()):
        """Must this result be recomputed?"""
        outputs = [Path(path)] + [Path(p) for p in extra_outputs]
        if not self.cache or not all(p.exists() for p in outputs):
            return True
        return min(p.stat().st_mtime for p in outputs) < self._input_mtime()

    def _produce(self, path, command, stdin=None, extra_outputs=()):
        """Run ``command`` unless ``path`` already holds a fresh answer.

        Post-processing is interactive: re-running a notebook cell should not
        re-read the trajectory again.  The cache key is in the filename, and
        anything older than the tpr/trajectory is discarded.
        """
        if self._stale(path, extra_outputs):
            command.run(stdin=stdin)
        return path

    @staticmethod
    def _series_set(path, xlabel=None, ylabel="", source=None, names=None):
        """Every y column of an xvg, named from its legends."""
        columns, meta = read_xvg(path)
        legends = names or meta.get("legends") or []
        out = []
        for i, column in enumerate(columns[1:]):
            name = legends[i] if i < len(legends) else f"column {i + 1}"
            out.append(Series(columns[0], column, name,
                              xlabel or meta.get("xaxis", "Time (ps)"),
                              ylabel or name, source))
        return out

    @staticmethod
    def _series(path, ycol=1, **kwargs):
        columns, meta = read_xvg(path)
        if len(columns) <= ycol:
            raise AnalysisError(f"{Path(path).name}: expected column {ycol}, "
                                f"got {len(columns)}")
        return Series(columns[0], columns[ycol], **kwargs)

    # -- trajectory analyses -------------------------------------------
    def rmsd(self, selection=None, fit=None, trajectory=None, tu="ps"):
        """RMSD after least-squares fit.  ``fit`` defaults to ``selection``."""
        if trajectory is None and fit is None and self._use_fast():
            from . import fast
            target = self._sel(selection)

            def compute():
                x, y = fast.rmsd(self.sim.frames, target.to_indices())
                return Series(list(x), list(y), "RMSD", f"Time ({tu})",
                              "RMSD (nm)", self.sim.trajectory.path)
            return self._remember(("rmsd", target.to_gromacs(), tu), compute)
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection)
        fit_sel = self._sel(fit) if fit is not None else target
        out = self._xvg("rmsd", tpr, traj, target.to_gromacs(),
                        fit_sel.to_gromacs(), tu)
        if fit_sel.to_gromacs() == target.to_gromacs():
            ndx = self._ndx(target, "rmsd")
            stdin = "0\n0\n"
        else:
            ndx = self._scratch() / "rmsd.ndx"
            text = (fit_sel.write_ndx(self._scratch() / "_fit.ndx", "fit").read_text()
                    + target.write_ndx(self._scratch() / "_tgt.ndx", "target").read_text())
            ndx.write_text(text)
            stdin = "0\n1\n"
        self._produce(out, cmd.Rms(env=self.env, tpr=tpr, trajectory=traj,
                                   index=str(ndx), output=str(out),
                                   xvg="none", tu=tu), stdin=stdin)
        return self._series(out, name="RMSD", xlabel=f"Time ({tu})",
                            ylabel="RMSD (nm)", source=traj)

    def rmsf(self, selection=None, per_residue=False, trajectory=None):
        """Per-atom (or per-residue) fluctuation."""
        if trajectory is None and not per_residue and self._use_fast():
            from . import fast
            target = self._sel(selection)
            def compute():
                x, y = fast.rmsf(self.sim.frames, target.to_indices())
                return Series(list(x), list(y), "RMSF", "Atom", "RMSF (nm)",
                              self.sim.trajectory.path)
            return self._remember(("rmsf", target.to_gromacs()), compute)
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection)
        ndx = self._ndx(target, "rmsf")
        out = self._xvg("rmsf", tpr, traj, target.to_gromacs(), per_residue)
        self._produce(out, cmd.Rmsf(env=self.env, tpr=tpr, trajectory=traj,
                                    index=str(ndx), output=str(out),
                                    xvg="none", res=per_residue), stdin="0\n")
        return self._series(out, name="RMSF",
                            xlabel="Residue" if per_residue else "Atom",
                            ylabel="RMSF (nm)", source=traj)

    def radius_of_gyration(self, selection=None, trajectory=None, tu="ps"):
        if trajectory is None and self._use_fast():
            from . import fast
            target = self._sel(selection)
            def compute():
                x, y = fast.radius_of_gyration(self.sim.frames,
                                               target.to_indices())
                return Series(list(x), list(y), "Radius of gyration",
                              f"Time ({tu})", "Rg (nm)",
                              self.sim.trajectory.path)
            return self._remember(("rg", target.to_gromacs(), tu), compute)
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection)
        out = self._xvg("gyrate", tpr, traj, target.to_gromacs(), tu)
        self._produce(out, cmd.Gyrate(env=self.env, tpr=tpr, trajectory=traj,
                                      output=str(out),
                                      selection=target.to_gromacs(),
                                      xvg="none", tu=tu))
        return self._series(out, name="Radius of gyration",
                            xlabel=f"Time ({tu})", ylabel="Rg (nm)", source=traj)

    gyration = radius_of_gyration

    def distance(self, a, b=None, trajectory=None, tu="ps"):
        """Distance between two selections (centre of mass by default)."""
        tpr, traj = self._inputs(trajectory)
        if b is None:
            expression = self._sel(a).to_gromacs()
        else:
            first, second = self._sel(a), self._sel(b)
            expression = (f"{_as_position(first)} plus {_as_position(second)}")
        out = self._xvg("distance", tpr, traj, expression, tu)
        self._produce(out, cmd.Distance(env=self.env, tpr=tpr, trajectory=traj,
                                        output=str(out), selection=expression,
                                        xvg="none", tu=tu))
        return self._series(out, name="Distance", xlabel=f"Time ({tu})",
                            ylabel="Distance (nm)", source=traj)

    def rdf(self, reference, selection, bin_width=0.002, cutoff=None,
            trajectory=None):
        """Radial distribution function g(r)."""
        if trajectory is None and self._use_fast():
            from . import fast
            ref, sel = self._sel(reference), self._sel(selection)

            def compute():
                x, y = fast.rdf(self.sim.frames, ref.to_indices(),
                                sel.to_indices(),
                                bin_width=value_in(bin_width, "nm"),
                                rmax=value_in(cutoff, "nm") if cutoff else None)
                return Series(list(x), list(y), "RDF", "r (nm)", "g(r)",
                              self.sim.trajectory.path)
            return self._remember(("rdf", ref.to_gromacs(), sel.to_gromacs(),
                                   bin_width, cutoff), compute)
        tpr, traj = self._inputs(trajectory)
        kwargs = {}
        if cutoff is not None:
            kwargs["rmax"] = value_in(cutoff, "nm")
        ref, sel = self._sel(reference), self._sel(selection)
        out = self._xvg("rdf", tpr, traj, ref.to_gromacs(), sel.to_gromacs(),
                        bin_width, cutoff)
        self._produce(out, cmd.Rdf(env=self.env, tpr=tpr, trajectory=traj,
                                   output=str(out),
                                   reference=ref.to_gromacs(),
                                   selection=sel.to_gromacs(),
                                   bin=value_in(bin_width, "nm"),
                                   xvg="none", **kwargs))
        return self._series(out, name="RDF", xlabel="r (nm)", ylabel="g(r)",
                            source=traj)

    def msd(self, selection=None, trajectory=None):
        """Mean squared displacement (and the fitted diffusion constant)."""
        # only on request: the two backends fit D over different windows and
        # disagree by ~15%, and D is a number people quote
        if trajectory is None and self.engine == "mdtraj" and self._use_fast():
            from . import fast
            target = self._sel(selection)
            x, y = fast.msd(self.sim.frames, target.to_indices())
            series = Series(list(x), list(y), "MSD", "Time (ps)",
                            "MSD (nm^2)", self.sim.trajectory.path)
            series.diffusion_constant = fast.diffusion_constant(x, y)
            series.diffusion_error = None
            return series
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection)
        out = self._xvg("msd", tpr, traj, target.to_gromacs())
        # no -xvg none here: gmx puts the fitted D in the series legend
        self._produce(out, cmd.Msd(env=self.env, tpr=tpr, trajectory=traj,
                                   output=str(out),
                                   selection=target.to_gromacs()))
        columns, meta = read_xvg(out)
        series = Series(columns[0], columns[1], "MSD", "Time (ps)",
                        "MSD (nm^2)", traj)
        legend = meta["legends"][0] if meta["legends"] else ""
        series.diffusion_constant = _grep_float(       # 1e-5 cm^2/s
            legend, r"=\s*([-\d.eE+]+)")
        series.diffusion_error = _grep_float(legend, r"\+/-\s*([-\d.eE+]+)")
        return series

    def hbonds(self, donors=None, acceptors=None, cutoff=0.35, angle=30,
               trajectory=None):
        """Hydrogen-bond count over time."""
        tpr, traj = self._inputs(trajectory)
        scratch = self._scratch()
        ref = self._sel(donors)
        target = self._sel(acceptors) if acceptors is not None else ref
        out = self._xvg("hbnum", tpr, traj, ref.to_gromacs(),
                        target.to_gromacs(), cutoff, angle)
        self._produce(out, cmd.Hbond(env=self.env, tpr=tpr, trajectory=traj,
                                     output=str(out),
                                     o=str(scratch / "hbond.ndx"),
                                     r=ref.to_gromacs(), t=target.to_gromacs(),
                                     hbr=value_in(cutoff, "nm"), hba=angle,
                                     xvg="none"))
        return self._series(out, name="H-bonds", xlabel="Time (ps)",
                            ylabel="Number of H-bonds", source=traj)


    # -- surface / contacts --------------------------------------------
    def sasa(self, selection=None, per_residue=False, probe=0.14,
             trajectory=None, tu="ps"):
        """Solvent-accessible surface area (total, or per residue)."""
        if trajectory is None and self._use_fast():
            from . import fast
            target = self._sel(selection if selection is not None
                               else self.sim.select.protein)
            def compute():
                x, y = fast.sasa(self.sim.frames, target.to_indices(),
                                 probe=value_in(probe, "nm"),
                                 per_residue=per_residue)
                return Series(list(x), list(y),
                              "SASA per residue" if per_residue else "SASA",
                              "Residue" if per_residue else f"Time ({tu})",
                              "Area (nm^2)", self.sim.trajectory.path)
            return self._remember(("sasa", target.to_gromacs(), probe,
                                   per_residue, tu), compute)
        tpr, traj = self._inputs(trajectory)
        surface = self._sel(selection if selection is not None
                            else self.sim.select.protein)
        key = (tpr, traj, surface.to_gromacs(), probe, per_residue, tu)
        out = self._xvg("sasa", *key)
        residue_out = self._xvg("sasa_res", *key) if per_residue else None
        self._produce(out, cmd.Sasa(
            env=self.env, tpr=tpr, trajectory=traj, output=str(out),
            surface=surface.to_gromacs(), probe=probe,
            per_residue=str(residue_out) if residue_out else None,
            xvg="none", tu=tu),
            extra_outputs=[residue_out] if residue_out else ())
        if per_residue:
            return self._series(residue_out, name="SASA per residue",
                                xlabel="Residue", ylabel="Area (nm^2)",
                                source=traj)
        return self._series(out, name="SASA", xlabel=f"Time ({tu})",
                            ylabel="Area (nm^2)", source=traj)

    def mindist(self, a, b=None, cutoff=0.6, contacts=False, trajectory=None,
                tu="ps"):
        """Minimum distance between two groups (and the contact count)."""
        tpr, traj = self._inputs(trajectory)
        if b is None:
            return self.periodic_image_distance(a, trajectory=trajectory, tu=tu)
        ndx = self._multi_ndx([("a", a), ("b", b)], "mindist")
        key = (tpr, traj, self._sel(a).to_gromacs(), self._sel(b).to_gromacs(),
               cutoff, tu)
        out, ncont = self._xvg("mindist", *key), self._xvg("ncontacts", *key)
        self._produce(out, cmd.Mindist(
            env=self.env, tpr=tpr, trajectory=traj, index=str(ndx),
            output=str(out), contacts=str(ncont), d=value_in(cutoff, "nm"),
            xvg="none", tu=tu), stdin="0\n1\n", extra_outputs=[ncont])
        if contacts:
            return self._series(ncont, name="Contacts", xlabel=f"Time ({tu})",
                                ylabel=f"Contacts within {value_in(cutoff, 'nm')} nm",
                                source=traj)
        return self._series(out, name="Minimum distance",
                            xlabel=f"Time ({tu})", ylabel="Distance (nm)",
                            source=traj)

    def periodic_image_distance(self, selection=None, trajectory=None, tu="ps"):
        """Smallest distance to a periodic image -- the standard box sanity check.

        If this drops near the cut-off, the box was too small.
        """
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection if selection is not None
                           else self.sim.select.protein)
        ndx = self._ndx(target, "pi")
        out = self._xvg("mindist_pi", tpr, traj, target.to_gromacs(), tu)
        self._produce(out, cmd.Mindist(
            env=self.env, tpr=tpr, trajectory=traj, index=str(ndx),
            output=str(out), pi=True, xvg="none", tu=tu), stdin="0\n")
        return self._series(out, name="Periodic image distance",
                            xlabel=f"Time ({tu})", ylabel="Distance (nm)",
                            source=traj)

    # -- structure ------------------------------------------------------
    def secondary_structure(self, selection=None, trajectory=None,
                            simplified=True):
        """DSSP: number of residues in each secondary-structure class."""
        # no fast path: mdtraj's DSSP is slower here and only reports three
        # classes, where gmx reports ten
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection if selection is not None
                           else self.sim.select.protein)
        counts = self._xvg("dssp_num", tpr, traj, target.to_gromacs())
        self._produce(counts, cmd.Dssp(     # no -xvg none: keep the legends
            env=self.env, tpr=tpr, trajectory=traj,
            output=str(self._scratch() / "dssp.dat"), counts=str(counts),
            selection=target.to_gromacs()))
        return self._series_set(counts, xlabel="Time (ps)",
                                ylabel="Residues", source=traj)

    dssp = secondary_structure

    def angle(self, selection, kind="angle", trajectory=None, tu="ps"):
        """Angle over time.  ``kind``: angle (3N atoms) or dihedral (4N)."""
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection)
        out = self._xvg("angle", tpr, traj, target.to_gromacs(), kind, tu)
        self._produce(out, cmd.Gangle(
            env=self.env, tpr=tpr, trajectory=traj, output=str(out),
            g1=kind, group1=target.to_gromacs(), xvg="none", tu=tu))
        return self._series(out, name=kind.capitalize(),
                            xlabel=f"Time ({tu})", ylabel="Angle (degrees)",
                            source=traj)

    def dihedral(self, selection, **kwargs):
        return self.angle(selection, kind="dihedral", **kwargs)

    def density(self, selection=None, axis="z", slices=50, kind="mass",
                trajectory=None):
        """Density profile along one box axis (membranes, interfaces)."""
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection)
        ndx = self._ndx(target, "density")
        out = self._xvg("density", tpr, traj, target.to_gromacs(), axis,
                        slices, kind)
        self._produce(out, cmd.Density(
            env=self.env, tpr=tpr, trajectory=traj, index=str(ndx),
            output=str(out), axis=axis.upper(), slices=slices, dens=kind,
            xvg="none"), stdin="0\n")
        return self._series(out, name="Density", xlabel=f"{axis} (nm)",
                            ylabel="Density (kg/m^3)", source=traj)

    # -- ensemble -------------------------------------------------------
    def cluster(self, selection=None, cutoff=0.1, method="gromos",
                fit=None, trajectory=None):
        """Cluster the trajectory.  Returns sizes plus the parsed summary."""
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection if selection is not None
                           else self.sim.select.protein)
        fit_sel = self._sel(fit) if fit is not None else target
        ndx = self._multi_ndx([("fit", fit_sel), ("output", target)], "cluster")
        scratch = self._scratch()
        sizes = self._xvg("clust_size", tpr, traj, target.to_gromacs(),
                          fit_sel.to_gromacs(), method, cutoff)
        log = scratch / "cluster.log"
        self._produce(sizes, cmd.Cluster(
            env=self.env, tpr=tpr, trajectory=traj, index=str(ndx),
            sizes=str(sizes), log=str(log),
            matrix=str(scratch / "rmsd-clust.xpm"),
            raw_matrix=str(scratch / "rmsd-raw.xpm"),
            ids=str(scratch / "clust-id.xvg"),
            structures=str(scratch / "clusters.pdb"),
            method=method, cutoff=value_in(cutoff, "nm"),
            xvg="none"), stdin="0\n1\n", extra_outputs=[log])
        series = self._series(sizes, name="Cluster size", xlabel="Cluster",
                              ylabel="Frames", source=traj)
        series.n_clusters = len(series)
        series.log = log
        series.structures = scratch / "clusters.pdb"
        return series

    def pca(self, selection=None, n_components=2, fit=None, trajectory=None):
        """Essential dynamics: covariance eigenvalues + the first projections."""
        tpr, traj = self._inputs(trajectory)
        target = self._sel(selection if selection is not None
                           else self.sim.select.protein & self.sim.select.name("CA"))
        fit_sel = self._sel(fit) if fit is not None else target
        ndx = self._multi_ndx([("fit", fit_sel), ("analysis", target)], "pca")
        scratch = self._scratch()
        key = (tpr, traj, target.to_gromacs(), fit_sel.to_gromacs())
        eigenvalues = self._xvg("eigenval", *key)
        vectors = scratch / f"eigenvec_{eigenvalues.stem.split('_')[-1]}.trr"
        self._produce(eigenvalues, cmd.Covar(
            env=self.env, tpr=tpr, trajectory=traj, index=str(ndx),
            eigenvalues=str(eigenvalues), eigenvectors=str(vectors),
            average=str(scratch / "average.pdb"),
            log=str(scratch / "covar.log"),
            xvg="none"), stdin="0\n1\n", extra_outputs=[vectors])
        values = self._series(eigenvalues, name="Eigenvalue",
                              xlabel="Component", ylabel="Eigenvalue (nm^2)",
                              source=traj)
        components = []
        for i in range(1, n_components + 1):
            projection = self._xvg(f"proj{i}", *key)
            self._produce(projection, cmd.Anaeig(
                env=self.env, tpr=tpr, trajectory=traj, index=str(ndx),
                eigenvectors=str(vectors), eigenvalues=str(eigenvalues),
                projection=str(projection), first=i, last=i,
                xvg="none"), stdin="0\n1\n")   # fit group, eigenvector group
            components.append(self._series(projection, name=f"PC{i}",
                                           xlabel="Time (ps)",
                                           ylabel=f"PC{i} (nm)", source=traj))
        values.components = components
        total = sum(values.y) or 1.0
        values.explained = [v / total for v in values.y[:n_components]]
        return values

    # -- convenience ---------------------------------------------------
    def convergence(self, series, n_blocks=10):
        """Block averages of any series -- has it settled?"""
        return series.block_average(n_blocks)

    def __repr__(self):
        methods = [name for name in dir(self)
                   if not name.startswith("_") and callable(getattr(self, name))]
        return "<Analysis " + "/".join(methods) + ">"


def _as_position(selection):
    """Position expression for gmx distance: keep explicit com/cog as-is."""
    expression = selection.to_gromacs()
    if expression.startswith(("com of", "cog of")):
        return expression
    return f"com of ({expression})"


def _grep_float(text, pattern):
    import re
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------
# free energy
# ---------------------------------------------------------------------

class BarResult:
    """What ``gmx bar`` computed: the total and the per-pair breakdown."""

    def __init__(self, delta_g, error, pairs, unit="kJ/mol", output=""):
        self.delta_g = delta_g
        self.error = error
        self.pairs = pairs            # [(lambda_a, lambda_b, dG, err), ...]
        self.unit = unit
        self.output = output

    @property
    def profile(self):
        """Cumulative dG along the lambda path, as a Series."""
        total, xs, ys = 0.0, [0], [0.0]
        for i, (_, _, value, _) in enumerate(self.pairs, start=1):
            total += value
            xs.append(i)
            ys.append(total)
        return Series(xs, ys, "Cumulative dG", "lambda point",
                      f"dG ({self.unit})")

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.pairs,
                            columns=["lambda_a", "lambda_b", "dG", "error"])

    def __repr__(self):
        return (f"<BarResult dG = {self.delta_g:.3f} +/- {self.error:.3f} "
                f"{self.unit} over {len(self.pairs)} pair(s)>")


def bar(files, temperature=None, begin=None, end=None, env=None, workdir=None):
    """BAR free-energy difference from dhdl.xvg (or .edr) files.

        dg = bar(sorted(glob("lambda_*/md.xvg")))
        print(dg.delta_g, dg.error)

    Files must be given in lambda order.
    """
    import re

    files = [str(abspath(f)) for f in files]
    if len(files) < 2:
        raise AnalysisError("BAR needs at least two lambda points")
    scratch = Path(workdir or Path(files[0]).parent) / ".gmxpy"
    scratch.mkdir(parents=True, exist_ok=True)
    key = "edr" if files[0].endswith(".edr") else "dhdl"
    proc = cmd.Bar(env=env, output=str(scratch / "bar.xvg"),
                   integral=str(scratch / "barint.xvg"),
                   temperature=temperature, begin=begin, end=end,
                   **{key: files}).run()
    text = proc.stdout + proc.stderr

    pairs = []
    for a, b, value, error in re.findall(
            r"^point\s+(\S+)\s*-\s*(\S+),\s*DG\s+([-\d.eE+]+)\s*\+/-\s*([\d.eE+]+)",
            text, re.M):
        pairs.append((a, b, float(value), float(error)))
    total = re.search(
        r"^total\s+\S+\s*-\s*\S+,\s*DG\s+([-\d.eE+]+)\s*\+/-\s*([\d.eE+]+)",
        text, re.M)
    if not total:
        raise AnalysisError("could not parse gmx bar output:\n" + text[-800:])
    unit = "kJ/mol" if "kJ/mol" in text else "kT"
    return BarResult(float(total.group(1)), float(total.group(2)), pairs,
                     unit, text)


def landscape(x, y, temperature=300.0, bins=50, max_energy=None):
    """Free-energy surface from two observables: ``-kT ln P(x, y)``.

        fes = landscape(rmsd, rg)
        fes.plot()

    Pure NumPy -- no gmx sham, no XPM.
    """
    import numpy as np

    xs = np.asarray(x.y if hasattr(x, "y") else x, dtype=float)
    ys = np.asarray(y.y if hasattr(y, "y") else y, dtype=float)
    counts, x_edges, y_edges = np.histogram2d(xs, ys, bins=bins)
    probability = counts / counts.sum()
    kT = 0.0083144626 * float(temperature)          # kJ/mol/K
    with np.errstate(divide="ignore"):
        energy = -kT * np.log(probability)
    energy -= np.nanmin(energy[np.isfinite(energy)])
    if max_energy is not None:
        energy[energy > max_energy] = np.nan
    energy[~np.isfinite(energy)] = np.nan
    return Landscape(energy.T, x_edges, y_edges,
                     getattr(x, "ylabel", "x"), getattr(y, "ylabel", "y"), kT)


class Landscape:
    """A 2D free-energy surface in kJ/mol."""

    def __init__(self, energy, x_edges, y_edges, xlabel, ylabel, kT):
        self.energy = energy
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.kT = kT

    @property
    def minimum(self):
        """(x, y) of the global minimum."""
        import numpy as np
        j, i = np.unravel_index(np.nanargmin(self.energy), self.energy.shape)
        centre = lambda edges, k: (edges[k] + edges[k + 1]) / 2
        return centre(self.x_edges, i), centre(self.y_edges, j)

    def plot(self, levels=20, cmap="viridis", figsize=(6, 4.6), **kwargs):
        from .plotting import Plot, _pyplot
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=figsize)
        mesh = ax.contourf(self.x_edges[:-1], self.y_edges[:-1], self.energy,
                           levels=levels, cmap=cmap, **kwargs)
        fig.colorbar(mesh, ax=ax, label="Free energy (kJ/mol)")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        fig.tight_layout()
        return Plot(fig, ax)

    def __repr__(self):
        import numpy as np
        return (f"<Landscape {self.energy.shape} max "
                f"{np.nanmax(self.energy):.1f} kJ/mol, minimum at "
                f"{tuple(round(v, 3) for v in self.minimum)}>")
