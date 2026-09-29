"""The equilibration chain: em -> nvt -> npt -> production, one call.

    system = (System.from_pdb("protein.pdb", workdir="prep")
              .pdb2gmx().box(padding=1.0 * u.nanometer)
              .solvate().add_ions())
    eq = system.equilibrate(workdir="prep/eq", nvt=100 * u.ps, npt=1 * u.ns)
    eq.ok                                   # every stage finished
    prod = eq.production(MDP.preset("md", nsteps=5_000_000))
    prod.run(ntomp=8)                       # or prod.submit(...)

Each stage is an ordinary :class:`~gmxpy.simulation.Simulation` under
``workdir``; the chain hands the output structure of one stage to the next
and passes the checkpoint to grompp (``-t``) so velocities survive.  Stages
whose log already says finished are skipped, so an interrupted equilibration
is resumed by running the same call again.
"""

from __future__ import annotations

from pathlib import Path

from .command import abspath
from .environment import Environment
from .errors import GmxpyError
from .mdp import MDP
from .simulation import Simulation
from .units import value_in


def _steps(duration, dt):
    """A duration (Quantity, or a bare number of ps) as a step count."""
    ps = value_in(duration, "ps")
    if ps <= 0:
        raise GmxpyError("stage duration must be positive")
    return max(1, round(ps / dt))


class Protocol:
    """What the equilibration stages are, without touching a system.

        Protocol(nvt=200 * u.ps, npt=1 * u.ns)               # durations
        Protocol(em=False)                                   # already minimised
        Protocol(em={"nsteps": 5000}, nvt={"gen_seed": 7})   # mdp overrides

    ``posre``: ``"auto"`` (default) restrains nvt/npt when the topology has
    a ``POSRES`` block (pdb2gmx writes one), and quietly does not when it
    does not; ``True`` demands the block and errors out without it;
    ``False`` never restrains.
    """

    def __init__(self, em=True, nvt=100.0, npt=500.0, dt=0.002,
                 temperature=300.0, pressure=1.0, tau_t=0.1,
                 posre="auto", maxwarn=1,
                 em_overrides=None, nvt_overrides=None, npt_overrides=None):
        self.dt = value_in(dt, "ps")
        self.temperature = value_in(temperature, "K")
        self.pressure = value_in(pressure, "bar")
        self.tau_t = tau_t
        self.posre = posre
        self.maxwarn = maxwarn
        self.overrides = {"em": em_overrides or {}, "nvt": nvt_overrides or {},
                          "npt": npt_overrides or {}}
        # dict-form stage settings are overrides on top of the default
        # duration; a number replaces it
        self._durations = {"nvt": 100.0, "npt": 500.0}
        self._stages = []
        if em:
            # for em a bare number is a step count (steep descent has no dt)
            if isinstance(em, dict):
                self.overrides["em"].update(em)
            elif em is not True:
                self.overrides["em"]["nsteps"] = int(em)
            self._stages.append("em")
        for key, setting in (("nvt", nvt), ("npt", npt)):
            if setting is False:
                continue
            if isinstance(setting, dict):
                self.overrides[key].update(setting)
            else:
                self._durations[key] = setting
            self._stages.append(key)

    # -- per-stage mdp ---------------------------------------------------
    def stage_mdp(self, name):
        """The mdp for one stage.

        A stage's duration is ``self._durations[name]`` (ps) unless its
        overrides carry ``nsteps=`` or ``duration=`` directly.
        """
        overrides = dict(self.overrides[name])
        duration = overrides.pop("duration", None) or self._durations.get(name)
        nsteps = overrides.pop("nsteps", None)

        if name == "em":
            return MDP.preset("em", nsteps=int(nsteps or 50000), **overrides)

        base = dict(dt=self.dt, ref_t=self.temperature, tau_t=self.tau_t,
                    gen_temp=self.temperature)
        if name == "npt":
            base.update(ref_p=self.pressure)
        if nsteps is None:
            if duration is None:
                raise GmxpyError(
                    f"the {name} stage needs a duration ({name}=100 * u.ps) "
                    "or nsteps= in its overrides")
            nsteps = _steps(duration, self.dt)
        return MDP.preset(name, nsteps=int(nsteps), **base, **overrides)

    @property
    def stage_names(self):
        return list(self._stages)

    def __repr__(self):
        return f"<Protocol {' -> '.join(self.stage_names)}>"


class Equilibration:
    """A chain run (or being run): the stages plus where they ended up."""

    def __init__(self, structure, topology, workdir=".", env=None,
                 protocol=None, index=None, **protocol_kwargs):
        self.env = env or Environment.detect()
        self.protocol = protocol or Protocol(**protocol_kwargs)
        self.structure = abspath(structure)
        self.topology = abspath(topology)
        self.workdir = abspath(workdir)
        self.index = abspath(index)
        self.stages = {}
        self.notes = []

    # -- posre ------------------------------------------------------------
    def _posre_block(self):
        """``define`` value for restrained stages, or ``None`` to go free."""
        text = Path(self.topology).read_text(errors="replace")
        if self.protocol.posre is False:
            return None
        if "#ifdef POSRES" in text:
            return "-DPOSRES"
        if self.protocol.posre is True:
            raise GmxpyError(
                f"{Path(self.topology).name} has no '#ifdef POSRES' block, so "
                "position restraints cannot be switched on; build with "
                "pdb2gmx (it writes posre.itp) or pass posre=False")
        self.notes.append("no POSRES block in the topology: equilibrating "
                          "without position restraints")
        return None

    # -- running ----------------------------------------------------------
    def run(self, run_kwargs=None, force=False):
        """Run every stage in order.  Finished stages are skipped."""
        define = self._posre_block()
        structure, checkpoint = self.structure, None
        for name in self.protocol.stage_names:
            mdp = self.protocol.stage_mdp(name)
            if define and name in ("nvt", "npt"):
                mdp.update(define=define)
            sim = Simulation(structure=structure, topology=self.topology,
                             mdp=mdp, name=name, workdir=self.workdir,
                             env=self.env, index=self.index)
            self.stages[name] = sim
            if sim.log.exists and sim.log.finished and not force:
                structure = sim.paths["structure"]
                checkpoint = sim.paths["checkpoint"]
                continue
            # em writes no checkpoint (no dynamics) -- only hand one over
            # when the previous stage actually produced it
            sim.prepare(maxwarn=self.protocol.maxwarn,
                        restraint=structure if define else None,
                        checkpoint=checkpoint
                        if checkpoint and Path(checkpoint).exists() else None)
            sim.run(**(run_kwargs or {}))
            structure = sim.paths["structure"]
            checkpoint = sim.paths["checkpoint"]
        return self

    # -- state ------------------------------------------------------------
    @property
    def last(self):
        """The final stage's Simulation."""
        if not self.stages:
            raise GmxpyError("nothing has run yet")
        return self.stages[self.protocol.stage_names[-1]]

    @property
    def ok(self):
        return bool(self.stages) and all(
            sim.log.exists and sim.log.finished for sim in self.stages.values())

    def check(self):
        """``sim.check()`` per stage, as a dict."""
        return {name: sim.check() for name, sim in self.stages.items()}

    # -- handover ----------------------------------------------------------
    def production(self, mdp=None, name="md", posre=False, workdir=None,
                   prepare=True):
        """A Simulation wired to the end of the chain (structure + checkpoint)."""
        if not self.stages:
            raise GmxpyError("run() first")
        mdp = mdp if mdp is not None else MDP.preset("md")
        if not isinstance(mdp, MDP):
            mdp = MDP.read(mdp)
        define = self._posre_block() if posre else None
        if posre and define is None:
            raise GmxpyError("posre=True needs a POSRES block in the topology")
        if define:
            mdp.update(define=define)
        head = self.last
        sim = Simulation(structure=head.paths["structure"],
                         topology=self.topology, mdp=mdp, name=name,
                         workdir=workdir or self.workdir, env=self.env,
                         index=self.index)
        if prepare:
            checkpoint = head.paths["checkpoint"]
            sim.prepare(maxwarn=self.protocol.maxwarn,
                        restraint=head.paths["structure"] if posre else None,
                        checkpoint=checkpoint if checkpoint.exists() else None)
        return sim

    def __repr__(self):
        stages = ", ".join(f"{name}: {'done' if sim.log.finished else 'pending'}"
                           for name, sim in self.stages.items()) or "not run"
        return (f"<Equilibration {' -> '.join(self.protocol.stage_names)} "
                f"| {stages}>")
