"""Preparation: from a PDB to something ``grompp`` will accept.

    system = System.from_pdb("protein.pdb", workdir="prep")
    system.pdb2gmx(forcefield="amber99sb-ildn", water="tip3p")
    system.box(shape="dodecahedron", padding=1.0 * u.nanometer)
    system.solvate()
    system.add_ions(concentration=0.15)
    sim = system.simulation(mdp=MDP.preset("em"), name="em")

Each step returns ``self``, so the whole preparation is one chain.  The
interactive prompts (force field, water model, the group genion replaces)
are answered here.
"""

from __future__ import annotations

from pathlib import Path

from . import command as cmd
from .command import abspath
from .environment import Environment
from .errors import GmxpyError
from .mdp import MDP
from .topology import Structure, Topology
from .units import value_in

SOLVENTS = {"tip3p": "spc216.gro", "spc": "spc216.gro", "spce": "spc216.gro",
            "tip4p": "tip4p.gro", "tip4pew": "tip4p.gro", "tip5p": "tip5p.gro"}


class System:
    """A molecular system on its way to being simulatable."""

    def __init__(self, structure, topology=None, workdir=".", env=None,
                 water="tip3p"):
        self.env = env or Environment.detect()
        self.workdir = abspath(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.structure = abspath(structure)
        self.topology = abspath(topology)
        self.water = water
        self.history = []

    @classmethod
    def from_pdb(cls, path, workdir=".", env=None):
        return cls(structure=path, workdir=workdir, env=env)

    @classmethod
    def from_gro(cls, path, topology, workdir=".", env=None):
        """Already parameterised -- skip straight to box/solvate/ions."""
        return cls(structure=path, topology=topology, workdir=workdir, env=env)

    # -- steps ---------------------------------------------------------
    def rename_residues(self, mapping, output=None):
        """Rename residues in the PDB (``{"NMA": "NME", "HIE": "HIS"}``).

        Force fields and PDB files disagree about cap and protonation-state
        names constantly; pdb2gmx just refuses.  This is the fix.
        """
        out = abspath(output or self.workdir / (self.structure.stem + "_renamed.pdb"))
        lines = []
        for line in self.structure.read_text(errors="replace").splitlines(True):
            if line.startswith(("ATOM", "HETATM")):
                name = line[17:20].strip()
                if name in mapping:
                    line = line[:17] + f"{mapping[name]:>3}" + line[20:]
            lines.append(line)
        out.write_text("".join(lines))
        self.structure = out
        self.history.append(f"rename_residues({mapping})")
        return self

    def pdb2gmx(self, forcefield="amber99sb-ildn", water="tip3p",
                ignore_hydrogens=True, output=None, topology=None, **kwargs):
        """Assign a force field.  Ligands and non-standard residues will fail
        here -- parameterise those separately and use ``from_gro``."""
        out = abspath(output or self.workdir / "processed.gro")
        top = abspath(topology or self.workdir / "topol.top")
        cmd.Pdb2gmx(env=self.env, structure=str(self.structure), output=str(out),
                    topology=str(top), forcefield=forcefield, water=water,
                    i=str(self.workdir / "posre.itp"), ignh=ignore_hydrogens,
                    **kwargs).run(cwd=self.workdir)
        self.structure, self.topology, self.water = out, top, water
        self.history.append(f"pdb2gmx({forcefield}, {water})")
        return self

    def box(self, shape="dodecahedron", padding=1.0, center=True, size=None,
            output=None, **kwargs):
        """Put the solute in a box (``padding`` nm of space around it)."""
        out = abspath(output or self.workdir / "boxed.gro")
        options = dict(shape=shape, center=center)
        if size is not None:
            options["box"] = [value_in(v, "nm") for v in
                              (size if isinstance(size, (list, tuple)) else [size] * 3)]
        else:
            options["padding"] = value_in(padding, "nm")
        cmd.Editconf(env=self.env, structure=str(self.structure), output=str(out),
                     **options, **kwargs).run(cwd=self.workdir)
        self.structure = out
        self.history.append(f"box({shape}, {padding})")
        return self

    def solvate(self, model=None, output=None, **kwargs):
        """Fill the box with water."""
        self._need_topology("solvate")
        solvent = model or SOLVENTS.get(self.water, "spc216.gro")
        out = abspath(output or self.workdir / "solvated.gro")
        cmd.Solvate(env=self.env, structure=str(self.structure), solvent=solvent,
                    output=str(out), topology=str(self.topology),
                    **kwargs).run(cwd=self.workdir)
        self.structure = out
        self.history.append(f"solvate({solvent})")
        return self

    def add_ions(self, concentration=0.15, neutral=True, positive="NA",
                 negative="CL", solvent_group="SOL", output=None, **kwargs):
        """Neutralise and salt the box (genion, via a throwaway tpr)."""
        self._need_topology("add_ions")
        tpr = self.workdir / "ions.tpr"
        MDP.preset("em", nsteps=1).write(self.workdir / "ions.mdp")
        cmd.Grompp(env=self.env, mdp=str(self.workdir / "ions.mdp"),
                   structure=str(self.structure), topology=str(self.topology),
                   output=str(tpr), mdout=str(self.workdir / "ions_mdout.mdp"),
                   maxwarn=2).run(cwd=self.workdir)
        out = abspath(output or self.workdir / "ionised.gro")
        cmd.Genion(env=self.env, tpr=str(tpr), output=str(out),
                   topology=str(self.topology), pname=positive, nname=negative,
                   neutral=neutral, concentration=concentration, **kwargs
                   ).run(stdin=f"{solvent_group}\n", cwd=self.workdir)
        self.structure = out
        self.history.append(f"add_ions({concentration} M)")
        return self

    def minimise(self, mdp=None, name="em", **run_kwargs):
        """Energy-minimise in place; the result becomes the new structure."""
        sim = self.simulation(mdp or MDP.preset("em"), name=name)
        sim.prepare()
        sim.run(**run_kwargs)
        self.structure = sim.paths["structure"]
        self.history.append(f"minimise({name})")
        return self

    minimize = minimise

    # -- handover ------------------------------------------------------
    def simulation(self, mdp, name="sim", workdir=None, **kwargs):
        """Hand the prepared system to a :class:`~gmxpy.simulation.Simulation`."""
        from .simulation import Simulation

        self._need_topology("simulation")
        return Simulation(structure=self.structure, topology=self.topology,
                          mdp=mdp, name=name, env=self.env,
                          workdir=workdir or self.workdir, **kwargs)

    # -- inspection ----------------------------------------------------
    @property
    def n_atoms(self):
        return Structure(self.structure).n_atoms

    @property
    def box_vectors(self):
        return Structure(self.structure).box

    @property
    def molecules(self):
        return Topology(self.topology).molecules if self.topology else []

    def _need_topology(self, step):
        if not self.topology:
            raise GmxpyError(
                f"{step}() needs a topology: run pdb2gmx() first, or build the "
                "system with System.from_gro(structure, topology)")

    def __repr__(self):
        return (f"<System {self.structure.name} ({self.n_atoms} atoms)"
                + (f" after {' -> '.join(self.history)}" if self.history else "")
                + ">")
