"""The Simulation object -- the thing this whole package exists to provide.

One simulation = one ``-deffnm``.  Inputs are objects, outputs are objects,
and the fragmented ``.tpr/.xtc/.edr/.log/.cpt/.gro`` set is addressed
through one handle.
"""

from __future__ import annotations

from pathlib import Path

from . import command as cmd
from .command import abspath
from .analysis import Analysis
from .data import Checkpoint, Energy, LogFile, Trajectory
from .environment import Environment
from .errors import CheckpointError, GmxpyError, MdrunError
from .mdp import MDP
from .selection import SelectionContext, Selector
from .topology import Structure, Topology
from .units import Quantity


class Result:
    """What came out of ``sim.run()`` -- status plus every output file."""

    def __init__(self, sim, returncode=None):
        self.sim = sim
        self.return_code = returncode

    # -- status --------------------------------------------------------
    @property
    def finished(self):
        return self.sim.log.finished

    @property
    def steps(self):
        return self.sim.log.steps

    @property
    def performance(self):
        """ns/day."""
        return self.sim.log.performance

    @property
    def wall_time(self):
        """seconds"""
        return self.sim.log.wall_time

    @property
    def minimisation(self):
        return self.sim.log.minimisation

    @property
    def converged(self):
        return self.sim.log.converged

    @property
    def potential_energy(self):
        return self.sim.log.potential_energy

    @property
    def max_force(self):
        return self.sim.log.max_force

    @property
    def simulation_time(self):
        if self.minimisation:
            return None            # "time" in a minimisation is the step count
        length = self.sim.trajectory.length if self.sim.trajectory.exists else None
        if length is not None:
            return length
        checkpoint = self.sim.checkpoint
        return checkpoint.time if checkpoint.exists else None

    time = simulation_time

    # -- outputs -------------------------------------------------------
    @property
    def structure(self):
        return self.sim.paths["structure"]

    @property
    def trajectory(self):
        return self.sim.trajectory

    @property
    def energy(self):
        return self.sim.energy

    @property
    def checkpoint(self):
        return self.sim.checkpoint

    @property
    def log(self):
        return self.sim.log

    @property
    def tpr(self):
        return self.sim.tpr

    def summary(self):
        time = self.simulation_time
        out = {
            "name": self.sim.name,
            "finished": self.finished,
            "steps": self.steps,
            "return_code": self.return_code,
        }
        if self.minimisation:
            out.update(converged=self.converged,
                       potential_energy=self.potential_energy,
                       max_force=self.max_force)
        else:
            out.update(simulation_time_ps=float(time) if time is not None else None,
                       performance_ns_per_day=self.performance,
                       wall_time_s=self.wall_time)
        return out

    def __repr__(self):
        state = "finished" if self.finished else "incomplete"
        if self.minimisation:
            extra = (f", Epot {self.potential_energy:.6g}"
                     f", Fmax {self.max_force:.4g}"
                     if self.potential_energy is not None else "")
        else:
            extra = f", {self.performance} ns/day" if self.performance else ""
        return f"<Result {self.sim.name!r} {state}{extra}>"


class Simulation:
    """A GROMACS simulation as a Python object."""

    def __init__(self, structure=None, topology=None, mdp=None, name="sim",
                 workdir=".", tpr=None, index=None, env=None):
        self.name = name
        self.workdir = abspath(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = env or Environment.detect()
        self.mdp = mdp if not isinstance(mdp, (str, Path)) else abspath(mdp)
        self.index = abspath(index)
        self._structure = abspath(structure)
        self._topology = abspath(topology)
        self._tpr = abspath(tpr)
        self._selector = None
        self._context = None
        self._frames = None
        self._analysis = None
        self.result = Result(self)

    # -- construction --------------------------------------------------
    @classmethod
    def open(cls, name, workdir=".", env=None):
        """Attach to an existing ``-deffnm`` run on disk."""
        workdir = abspath(workdir)
        tpr = workdir / f"{name}.tpr"
        if not tpr.exists():
            raise GmxpyError(f"no {tpr} -- nothing to open")
        return cls(name=name, workdir=workdir, tpr=tpr, env=env)

    # -- paths ---------------------------------------------------------
    @property
    def paths(self):
        base = self.workdir / self.name
        return {
            "tpr": self.tpr,
            "trajectory": base.with_suffix(".xtc"),
            "full_trajectory": base.with_suffix(".trr"),
            "energy": base.with_suffix(".edr"),
            "log": base.with_suffix(".log"),
            "checkpoint": base.with_suffix(".cpt"),
            "structure": base.with_suffix(".gro"),
            "mdp": base.with_suffix(".mdp"),
        }

    @property
    def tpr(self):
        return self._tpr or (self.workdir / f"{self.name}.tpr")

    @property
    def structure(self):
        """Input structure (or the final frame once the run is done)."""
        return self._structure

    @property
    def topology(self):
        return Topology(self._topology) if self._topology else None

    # -- preparation ---------------------------------------------------
    def prepare(self, maxwarn=0, restraint=None, checkpoint=None, force=False,
                **kwargs):
        """grompp: mdp + structure + topology -> tpr."""
        tpr = self.tpr
        if tpr.exists() and not force and self.mdp is None:
            return tpr
        if not (self._structure and self._topology):
            raise GmxpyError("prepare() needs structure= and topology=")
        mdp_path = self.paths["mdp"]
        if isinstance(self.mdp, MDP):
            problems = self.mdp.validate()
            if problems:
                print("MDP warnings:\n  " + "\n  ".join(problems))
            self.mdp.write(mdp_path)
        elif self.mdp is not None:
            mdp_path = abspath(self.mdp)
        else:
            raise GmxpyError("prepare() needs mdp=")

        cmd.Grompp(
            env=self.env, mdp=str(mdp_path), structure=str(self._structure),
            topology=str(self._topology), output=str(tpr),
            mdout=str(self.workdir / f"{self.name}_mdout.mdp"),
            index=str(self.index) if self.index else None,
            restraint=str(abspath(restraint)) if restraint else None,
            checkpoint=str(abspath(checkpoint)) if checkpoint else None,
            maxwarn=maxwarn or None, **kwargs
        ).run(cwd=self.workdir)
        self._tpr = tpr
        self._context = None            # selections now resolve against the tpr
        return tpr

    # -- running -------------------------------------------------------
    def _mdrun(self, resume=False, append=None, **kwargs):
        options = dict(deffnm=str(self.workdir / self.name))
        if resume:
            options["checkpoint"] = str(self.paths["checkpoint"])
            if append is not None:
                options["append"] = append
        elif append is not None:
            options["append"] = append
        options.update({k: v for k, v in kwargs.items() if v is not None})
        return cmd.Mdrun(env=self.env, **options)

    def command(self, **kwargs):
        """The mdrun command line, for a job script or a sanity check."""
        return self._mdrun(**kwargs).command_line()

    def run(self, ntomp=None, ntmpi=None, nb=None, pme=None, bonded=None,
            update=None, gpu_id=None, nsteps=None, maxh=None, resume=False,
            append=None, check=True, echo=False, **extra):
        """Run mdrun.  Returns :class:`Result`."""
        if not self.tpr.exists():
            self.prepare()
        if resume:
            self._check_restart()
        if ntomp is not None and ntmpi is None and self.env.has("gpu"):
            # a GPU build refuses -ntomp without -ntmpi; one rank is what
            # "give me N threads on this node" means anyway
            ntmpi = 1
        runner = self._mdrun(
            resume=resume, append=append, ntomp=ntomp, ntmpi=ntmpi, nb=nb,
            pme=pme, bonded=bonded, update=update, gpu_id=gpu_id,
            nsteps=nsteps, maxh=maxh, **extra)
        proc = runner.run(cwd=self.workdir, check=check, echo=echo)
        self.result = Result(self, proc.returncode)
        if check and not self.result.finished:
            raise MdrunError(
                "mdrun exited 0 but the log does not say it finished",
                command=runner.build_command(), returncode=proc.returncode,
                output=self.log.tail)
        return self.result

    def resume(self, **kwargs):
        """Continue from ``<name>.cpt``."""
        return self.run(resume=True, **kwargs)

    def _check_restart(self):
        """Restarts silently doing the wrong thing is the classic footgun."""
        checkpoint = self.checkpoint
        if not checkpoint.exists:
            raise CheckpointError(f"no checkpoint at {checkpoint.path}")
        if not self.tpr.exists():
            raise CheckpointError(f"no tpr at {self.tpr} to restart against")
        n_cpt = checkpoint.n_atoms
        n_tpr = self._tpr_atoms()
        if n_cpt and n_tpr and n_cpt != n_tpr:
            raise CheckpointError(
                f"checkpoint has {n_cpt} atoms but {self.tpr.name} has {n_tpr} "
                "-- they are not the same system")
        return True

    def _tpr_atoms(self):
        try:
            proc = cmd.Dump(env=self.env, tpr=str(self.tpr)).run(check=False)
        except Exception:
            return None
        import re
        m = re.search(r"natoms\s*=\s*(\d+)", proc.stdout)
        return int(m.group(1)) if m else None

    def extend(self, until=None, extend=None, nsteps=None, output=None):
        """convert-tpr: lengthen a finished run, then ``resume()``."""
        target = abspath(output or self.tpr)
        cmd.ConvertTpr(env=self.env, tpr=str(self.tpr), output=str(target),
                       until=until, extend=extend, nsteps=nsteps
                       ).run(cwd=self.workdir)
        self._tpr = target
        return target

    # -- data ----------------------------------------------------------
    @property
    def energy(self):
        return Energy(self.paths["energy"], env=self.env,
                      workdir=self.workdir, tpr=self.tpr)

    @property
    def trajectory(self):
        traj = self.paths["trajectory"]
        if not traj.exists() and self.paths["full_trajectory"].exists():
            traj = self.paths["full_trajectory"]
        return Trajectory(traj, tpr=self.tpr if self.tpr.exists() else None,
                          structure=self._structure, env=self.env)

    @property
    def frames(self):
        """The trajectory in memory, whole molecules, loaded once.

        One ``trjconv -pbc mol`` pass (cached on disk) plus one mdtraj load;
        every in-memory analysis after that is free of trajectory I/O.
        """
        if self._frames is None:
            traj = self.trajectory
            if not traj.exists:
                raise GmxpyError(f"no trajectory at {traj.path}")
            scratch = self.workdir / ".gmxpy"
            scratch.mkdir(parents=True, exist_ok=True)
            whole = scratch / f"{self.name}_whole.xtc"
            if (not whole.exists()
                    or whole.stat().st_mtime < traj.path.stat().st_mtime):
                traj.process(output=whole, pbc="mol")
            self._frames = Trajectory(whole, tpr=self.tpr,
                                      structure=self._structure,
                                      env=self.env).load(engine="mdtraj")
        return self._frames

    def unload(self):
        """Drop the in-memory trajectory."""
        self._frames = None
        return self

    @property
    def parts(self):
        """Trajectory chunks left by ``-noappend`` restarts, in order.

        ``prod.part0001.xtc``, ``prod.part0002.xtc``, ... plus ``prod.xtc``
        if it exists.  Empty when the run appended normally.
        """
        chunks = sorted(self.workdir.glob(f"{self.name}.part*.xtc"))
        base = self.paths["trajectory"]
        return (chunks + [base]) if (chunks and base.exists()) else chunks

    def merge_parts(self, output=None):
        """trjcat the ``-noappend`` chunks back into one trajectory."""
        parts = self.parts
        if len(parts) < 2:
            raise GmxpyError(
                f"{self.name} has {len(parts)} trajectory chunk(s); nothing to merge")
        target = abspath(output or self.workdir / f"{self.name}_merged.xtc")
        cmd.Trjcat(env=self.env, trajectory=[str(p) for p in parts],
                   output=str(target)).run(cwd=self.workdir)
        return Trajectory(target, tpr=self.tpr, structure=self._structure,
                          env=self.env)

    @property
    def log(self):
        return LogFile(self.paths["log"])

    @property
    def checkpoint(self):
        return Checkpoint(self.paths["checkpoint"], env=self.env)

    # -- selection / analysis ------------------------------------------
    @property
    def context(self):
        if self._context is None:
            reference = self.tpr if self.tpr.exists() else self._structure
            if reference is None:
                raise GmxpyError("no structure or tpr to select against")
            self._context = SelectionContext(reference, env=self.env,
                                             workdir=self.workdir)
        return self._context

    @property
    def select(self):
        if self._selector is None or self._selector.context is not self._context:
            self._selector = Selector(self.context)
        return self._selector

    @property
    def analysis(self):
        # one instance per simulation: caches, and holds the engine choice
        if self._analysis is None:
            self._analysis = Analysis(self)
        return self._analysis

    @property
    def command_runner(self):
        """Low-level escape hatch: ``sim.command_runner('trjconv', ...)``."""
        def run(subcommand, **kwargs):
            return cmd.gmx(subcommand, env=self.env, cwd=self.workdir, **kwargs)
        return run

    def check(self, **kwargs):
        """Post-run sanity checks: thermostat, drift, constraints, box size."""
        from .quality import check
        return check(self, **kwargs)

    # -- reporting -----------------------------------------------------
    def report(self, path=None, **kwargs):
        from .report import html_report
        return html_report(self, path, **kwargs)

    def __repr__(self):
        state = "prepared" if self.tpr.exists() else "not prepared"
        if self.log.exists:
            state = "finished" if self.log.finished else "running/incomplete"
        return f"<Simulation {self.name!r} [{state}] in {self.workdir}>"
