"""Command layer: build and run ``gmx`` subcommands.

One generic runner plus a flag map per subcommand.  Unknown keyword
arguments are passed straight through (``ntomp=8`` -> ``-ntomp 8``), so the
full GROMACS option surface stays reachable without wrapping every flag.

Interactive numbered menus are answered here via ``stdin`` -- callers deal
in names and selections, never in group numbers.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .environment import Environment
from .errors import classify
from .units import Quantity


def abspath(path):
    """Absolute, ~-expanded, lexically normalised -- but symlinks kept.

    Every path handed to gmx must be absolute: commands run with cwd set to
    the simulation's workdir, and on NFS the same tree has different relative
    positions on different machines.  Symlinks are deliberately not resolved,
    so automounted paths stay the ones the user typed.
    """
    if path is None:
        return None
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _fmt(value):
    if isinstance(value, Quantity):
        return f"{value.canonical:g}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class GromacsCommand:
    """Base class: ``arguments`` in, argv out, subprocess run."""

    name = ""          # gmx subcommand
    flags = {}         # python kwarg -> gmx flag (only for renames)

    def __init__(self, env=None, _cwd=None, _stdin=None, **arguments):
        self.env = env or Environment.detect()
        self.arguments = {k: v for k, v in arguments.items() if v is not None}
        self.cwd = _cwd
        self.stdin = _stdin
        self.result = None

    # -- building ------------------------------------------------------
    def build_command(self):
        argv = [self.env.executable, "-quiet", self.name]
        for key, value in self.arguments.items():
            flag = self.flags.get(key, "-" + key.replace("_", "-"))
            if value is True:
                argv.append(flag)
            elif value is False:
                argv.append("-no" + flag.lstrip("-"))
            elif isinstance(value, (list, tuple)):
                argv.append(flag)
                argv += [_fmt(v) for v in value]
            else:
                argv += [flag, _fmt(value)]
        return argv

    def command_line(self):
        return " ".join(shlex.quote(a) for a in self.build_command())

    # -- running -------------------------------------------------------
    def run(self, stdin=None, cwd=None, check=True, timeout=None, echo=False):
        argv = self.build_command()
        stdin = stdin if stdin is not None else self.stdin
        if echo:
            print(self.command_line())
        proc = subprocess.run(
            argv, input=stdin, capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd or self.cwd or Path.cwd()), env=self.env.env)
        self.result = proc
        if check and proc.returncode != 0:
            raise classify(proc.stderr + proc.stdout, argv, proc.returncode, self.name)
        return proc

    def __repr__(self):
        return f"<{type(self).__name__} {self.command_line()}>"


def gmx(subcommand, env=None, stdin=None, cwd=None, check=True, **arguments):
    """Escape hatch: run any gmx subcommand.  ``sim.command.gmx(...)``."""
    cmd = GromacsCommand(env=env, **arguments)
    cmd.name = subcommand
    return cmd.run(stdin=stdin, cwd=cwd, check=check)


# ---------------------------------------------------------------------
# preparation / simulation commands
# ---------------------------------------------------------------------

class Pdb2gmx(GromacsCommand):
    name = "pdb2gmx"
    flags = {"structure": "-f", "output": "-o", "topology": "-p", "index": "-n",
             "forcefield": "-ff", "position_restraints": "-i"}


class Editconf(GromacsCommand):
    name = "editconf"
    flags = {"structure": "-f", "output": "-o", "index": "-n",
             "shape": "-bt", "padding": "-d", "center": "-c"}


class Solvate(GromacsCommand):
    name = "solvate"
    flags = {"structure": "-cp", "solvent": "-cs", "output": "-o",
             "topology": "-p"}


class Genion(GromacsCommand):
    name = "genion"
    flags = {"tpr": "-s", "output": "-o", "topology": "-p", "index": "-n",
             "concentration": "-conc"}


class Grompp(GromacsCommand):
    name = "grompp"
    flags = {"mdp": "-f", "structure": "-c", "topology": "-p", "output": "-o",
             "index": "-n", "restraint": "-r", "checkpoint": "-t",
             "mdout": "-po"}


class Mdrun(GromacsCommand):
    name = "mdrun"
    flags = {"tpr": "-s", "checkpoint": "-cpi", "output": "-o", "log": "-g",
             "energy": "-e", "structure": "-c"}


class ConvertTpr(GromacsCommand):
    name = "convert-tpr"
    flags = {"tpr": "-s", "output": "-o", "index": "-n"}


class MakeNdx(GromacsCommand):
    name = "make_ndx"
    flags = {"structure": "-f", "output": "-o", "index": "-n"}


class Select(GromacsCommand):
    name = "select"
    flags = {"structure": "-s", "trajectory": "-f", "index": "-n",
             "ndx_out": "-on", "indices_out": "-oi", "sizes_out": "-os"}


# ---------------------------------------------------------------------
# data / analysis commands (used internally; users see result objects)
# ---------------------------------------------------------------------

class Energy(GromacsCommand):
    name = "energy"
    flags = {"edr": "-f", "output": "-o", "tpr": "-s"}


class Rms(GromacsCommand):
    name = "rms"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o"}


class Rmsf(GromacsCommand):
    name = "rmsf"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o"}


class Gyrate(GromacsCommand):
    name = "gyrate"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "selection": "-sel"}


class Rdf(GromacsCommand):
    name = "rdf"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "reference": "-ref", "selection": "-sel"}


class Distance(GromacsCommand):
    name = "distance"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n",
             "output": "-oav", "all": "-oall", "selection": "-select"}


class Msd(GromacsCommand):
    name = "msd"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "selection": "-sel"}


class Hbond(GromacsCommand):
    name = "hbond"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-num"}


class Sasa(GromacsCommand):
    name = "sasa"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "surface": "-surface", "per_residue": "-or", "per_atom": "-oa",
             "volume": "-tv"}


class Mindist(GromacsCommand):
    name = "mindist"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-od",
             "contacts": "-on", "per_residue": "-or"}


class Density(GromacsCommand):
    name = "density"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "axis": "-d", "slices": "-sl"}


class Dssp(GromacsCommand):
    name = "dssp"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "counts": "-num", "selection": "-sel"}


class Gangle(GromacsCommand):
    name = "gangle"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-oav",
             "all": "-oall", "histogram": "-oh"}


class Cluster(GromacsCommand):
    name = "cluster"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "log": "-g",
             "sizes": "-sz", "ids": "-clid", "structures": "-cl",
             "matrix": "-o", "raw_matrix": "-om"}


class Covar(GromacsCommand):
    name = "covar"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n",
             "eigenvalues": "-o", "eigenvectors": "-v", "average": "-av",
             "log": "-l"}


class Anaeig(GromacsCommand):
    name = "anaeig"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n",
             "eigenvectors": "-v", "eigenvalues": "-eig", "projection": "-proj",
             "projection_2d": "-2d", "rmsf": "-rmsf", "extremes": "-extr"}


class Bar(GromacsCommand):
    name = "bar"
    flags = {"dhdl": "-f", "edr": "-g", "output": "-o", "integral": "-oi",
             "histogram": "-oh", "temperature": "-temp", "begin": "-b",
             "end": "-e"}


class Trjconv(GromacsCommand):
    name = "trjconv"
    flags = {"tpr": "-s", "trajectory": "-f", "index": "-n", "output": "-o",
             "skip": "-skip", "begin": "-b", "end": "-e", "unit_cell": "-ur"}


class Trjcat(GromacsCommand):
    name = "trjcat"
    flags = {"trajectory": "-f", "output": "-o", "index": "-n"}


class Check(GromacsCommand):
    name = "check"
    flags = {"trajectory": "-f", "tpr": "-s"}


class Dump(GromacsCommand):
    name = "dump"
    flags = {"tpr": "-s", "checkpoint": "-cp", "edr": "-e", "trajectory": "-f"}
