"""Topology and structure handles.

Deliberately shallow: GROMACS' preprocessor stays authoritative.  We read
what is cheap and unambiguous (includes, defines, the ``[ molecules ]``
table, atom counts, box) and hand everything else to ``grompp``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import TopologyError
from .units import Quantity

_INCLUDE = re.compile(r'^\s*#include\s+["<]([^">]+)[">]', re.M)
_DEFINE = re.compile(r"^\s*#define\s+(\S+)\s*(.*)$", re.M)
_SECTION = re.compile(r"^\s*\[\s*(\S+)\s*\]", re.M)


class Topology:
    """A ``.top`` file.  Text-level view only."""

    def __init__(self, path):
        from .command import abspath
        self.path = abspath(path)
        if not self.path.exists():
            raise TopologyError(f"topology not found: {self.path}")
        self._text = None

    @property
    def text(self):
        if self._text is None:
            self._text = self.path.read_text(errors="replace")
        return self._text

    @property
    def includes(self):
        return _INCLUDE.findall(self.text)

    @property
    def defines(self):
        return {name: value.strip() for name, value in _DEFINE.findall(self.text)}

    @property
    def sections(self):
        return _SECTION.findall(self.text)

    def _section_body(self, name):
        out, inside = [], False
        for line in self.text.splitlines():
            stripped = line.split(";")[0].strip()
            header = _SECTION.match(line)
            if header:
                inside = header.group(1).lower() == name.lower()
                continue
            if inside and stripped and not stripped.startswith("#"):
                out.append(stripped)
        return out

    @property
    def molecules(self):
        """``[(name, count), ...]`` from ``[ molecules ]``, in file order."""
        out = []
        for line in self._section_body("molecules"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                out.append((parts[0], int(parts[1])))
        return out

    @property
    def molecule_names(self):
        return [name for name, _ in self.molecules]

    @property
    def moleculetypes(self):
        """Names defined by ``[ moleculetype ]`` blocks in this file."""
        out, expect = [], False
        for line in self.text.splitlines():
            stripped = line.split(";")[0].strip()
            header = _SECTION.match(line)
            if header:
                expect = header.group(1).lower() == "moleculetype"
                continue
            if expect and stripped and not stripped.startswith("#"):
                out.append(stripped.split()[0])
                expect = False
        return out

    @property
    def n_atoms_declared(self):
        """Sum over ``[ molecules ]`` -- None if a moleculetype is external."""
        sizes = self._moleculetype_sizes()
        total = 0
        for name, count in self.molecules:
            if name not in sizes:
                return None
            total += sizes[name] * count
        return total

    def _moleculetype_sizes(self):
        sizes, current, count, in_atoms = {}, None, 0, False
        for line in self.text.splitlines():
            stripped = line.split(";")[0].strip()
            header = _SECTION.match(line)
            if header:
                section = header.group(1).lower()
                if current and in_atoms:
                    sizes[current] = count
                if section == "moleculetype":
                    current, count, in_atoms = "", 0, False
                elif section == "atoms" and current is not None:
                    in_atoms, count = True, 0
                else:
                    in_atoms = False
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if current == "" and not in_atoms:
                current = stripped.split()[0]
            elif in_atoms:
                count += 1
        if current and in_atoms:
            sizes[current] = count
        return sizes

    def compile(self, structure, mdp, output=None, env=None, **kwargs):
        """Run grompp -- the only complete topology semantics there is."""
        from .command import Grompp
        from .mdp import MDP

        from .command import abspath
        output = abspath(output or self.path.with_suffix(".tpr"))
        mdp_path = mdp
        if isinstance(mdp, MDP):
            mdp_path = output.with_name(output.stem + "_grompp.mdp")
            mdp.write(mdp_path)
        Grompp(env=env, mdp=mdp_path, structure=str(structure),
               topology=str(self.path), output=str(output),
               mdout=str(output.with_name(output.stem + "_mdout.mdp")),
               **kwargs).run()
        return output

    def __repr__(self):
        return f"Topology({self.path.name!r}, molecules={self.molecules})"


class Structure:
    """A coordinate file (.gro / .pdb).  Header-level view only."""

    def __init__(self, path):
        from .command import abspath
        self.path = abspath(path)
        if not self.path.exists():
            raise TopologyError(f"structure not found: {self.path}")

    @property
    def format(self):
        return self.path.suffix.lower().lstrip(".")

    @property
    def n_atoms(self):
        if self.format == "gro":
            with self.path.open() as fh:
                fh.readline()
                return int(fh.readline().strip())
        count = 0
        with self.path.open(errors="replace") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    count += 1
                elif line.startswith("ENDMDL"):
                    break
        return count

    @property
    def box(self):
        """Box vector lengths in nm, or None."""
        if self.format == "gro":
            last = self.path.read_bytes()[-200:].decode(errors="replace").strip().splitlines()[-1]
            try:
                return tuple(Quantity(float(v), "nm") for v in last.split()[:3])
            except ValueError:
                return None
        with self.path.open(errors="replace") as fh:
            for line in fh:
                if line.startswith("CRYST1"):
                    return tuple(Quantity(float(line[6 + 9 * i:15 + 9 * i]) / 10, "nm")
                                 for i in range(3))
        return None

    @property
    def title(self):
        if self.format == "gro":
            return self.path.read_text(errors="replace").splitlines()[0].strip()
        return self.path.stem

    def __repr__(self):
        return f"Structure({self.path.name!r}, n_atoms={self.n_atoms})"
