"""Selection AST.

Users compose Python objects; the AST compiles to a GROMACS selection
string (or an MDAnalysis one), and evaluates to atom indices via
``gmx select``.  No selection strings and no index-group numbers in user
code.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

from .errors import SelectionError
from .units import Quantity, value_in

Atom = namedtuple("Atom", "index resid resname name")
Residue = namedtuple("Residue", "resid resname indices")


# ---------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------

class Node:
    def to_gromacs(self):
        raise NotImplementedError

    def to_mdanalysis(self):
        raise SelectionError(
            f"{type(self).__name__} has no MDAnalysis equivalent")

    def __repr__(self):
        return f"{type(self).__name__}({self.to_gromacs()!r})"


class Raw(Node):
    """Escape hatch: a literal GROMACS selection string."""

    def __init__(self, text):
        self.text = text

    def to_gromacs(self):
        return self.text

    def to_mdanalysis(self):
        return self.text


class All(Node):
    def to_gromacs(self):
        return "all"

    def to_mdanalysis(self):
        return "all"


class Nothing(Node):
    def to_gromacs(self):
        return "none"

    def to_mdanalysis(self):
        return "not all"


class Group(Node):
    """A default GROMACS index group, e.g. ``Protein``, ``Water``."""

    _MDA = {"protein": "protein", "water": "resname SOL HOH TIP3 WAT",
            "backbone": "backbone", "c-alpha": "name CA",
            "non-protein": "not protein", "non-water": "not (resname SOL HOH WAT)",
            "sidechain": "not backbone", "ion": "resname NA CL K MG CA ZN NA+ CL-",
            "system": "all"}

    def __init__(self, name):
        self.name = name

    def to_gromacs(self):
        return f'group "{self.name}"'

    def to_mdanalysis(self):
        try:
            return self._MDA[self.name.lower()]
        except KeyError:
            raise SelectionError(
                f"index group {self.name!r} has no MDAnalysis equivalent") from None


class _StringKeyword(Node):
    keyword = ""
    mda_keyword = ""

    def __init__(self, *values):
        if not values:
            raise SelectionError(f"{self.keyword} needs at least one value")
        self.values = [str(v) for v in values]

    def _quoted(self):
        return " ".join(f'"{v}"' if any(c in v for c in "*? ") else v
                        for v in self.values)

    def to_gromacs(self):
        return f"{self.keyword} {self._quoted()}"

    def to_mdanalysis(self):
        return f"{self.mda_keyword} {' '.join(self.values)}"


class Name(_StringKeyword):
    keyword = "name"
    mda_keyword = "name"


class Resname(_StringKeyword):
    keyword = "resname"
    mda_keyword = "resname"


class Chain(_StringKeyword):
    keyword = "chain"
    mda_keyword = "segid"


class Atomtype(_StringKeyword):
    keyword = "type"
    mda_keyword = "type"


class _IntKeyword(Node):
    keyword = ""
    mda_keyword = ""

    def __init__(self, *values):
        if not values:
            raise SelectionError(f"{self.keyword} needs at least one value")
        if len(values) == 2 and all(isinstance(v, int) for v in values) \
                and values[0] < values[1]:
            self.spec = f"{values[0]} to {values[1]}"
            self.mda_spec = f"{values[0]}:{values[1]}"
        else:
            flat = " ".join(str(int(v)) for v in values)
            self.spec = flat
            self.mda_spec = flat

    def to_gromacs(self):
        return f"{self.keyword} {self.spec}"

    def to_mdanalysis(self):
        return f"{self.mda_keyword} {self.mda_spec}"


class Resid(_IntKeyword):
    keyword = "resid"
    mda_keyword = "resid"


class Resindex(_IntKeyword):
    keyword = "resindex"
    mda_keyword = "resindex"


class Atomnr(_IntKeyword):
    keyword = "atomnr"
    mda_keyword = "bynum"


class Within(Node):
    """Atoms within ``distance`` of ``target`` (distance in nm)."""

    def __init__(self, distance, target):
        self.distance = value_in(distance, "nm")
        self.target = target

    def to_gromacs(self):
        return f"within {self.distance:g} of ({self.target.to_gromacs()})"

    def to_mdanalysis(self):
        return f"around {self.distance * 10:g} ({self.target.to_mdanalysis()})"


class SameResidueAs(Node):
    def __init__(self, inner):
        self.inner = inner

    def to_gromacs(self):
        return f"same residue as ({self.inner.to_gromacs()})"

    def to_mdanalysis(self):
        return f"same residue as ({self.inner.to_mdanalysis()})"


class _Binary(Node):
    op = ""

    def __init__(self, left, right):
        self.left, self.right = left, right

    def to_gromacs(self):
        return f"({self.left.to_gromacs()}) {self.op} ({self.right.to_gromacs()})"

    def to_mdanalysis(self):
        return f"({self.left.to_mdanalysis()}) {self.op} ({self.right.to_mdanalysis()})"


class And(_Binary):
    op = "and"


class Or(_Binary):
    op = "or"


class Not(Node):
    def __init__(self, inner):
        self.inner = inner

    def to_gromacs(self):
        return f"not ({self.inner.to_gromacs()})"

    def to_mdanalysis(self):
        return f"not ({self.inner.to_mdanalysis()})"


class Position(Node):
    """A position expression (``com of ...``), for distance/angle tools."""

    def __init__(self, inner, kind="com", pbc=True):
        self.inner, self.kind, self.pbc = inner, kind, pbc

    def to_gromacs(self):
        return f"{self.kind} of ({self.inner.to_gromacs()}){' pbc' if self.pbc else ''}"


# ---------------------------------------------------------------------
# user-facing selection
# ---------------------------------------------------------------------

class Selection:
    """A selection expression bound (optionally) to a structure."""

    def __init__(self, node, context=None, label=None):
        self.node = node
        self.context = context
        self.label = label

    # -- composition ---------------------------------------------------
    def _wrap(self, node, label=None):
        return Selection(node, self.context, label)

    def __and__(self, other):
        return self._wrap(And(self.node, _node(other)))

    def __or__(self, other):
        return self._wrap(Or(self.node, _node(other)))

    def __invert__(self):
        return self._wrap(Not(self.node))

    def __sub__(self, other):
        return self._wrap(And(self.node, Not(_node(other))))

    def within(self, distance, target):
        return self._wrap(And(self.node, Within(distance, _node(target))))

    def by_residue(self):
        return self._wrap(SameResidueAs(self.node))

    # chained filters: sim.select.protein.residue(100).atom("CA")
    def name(self, *names):
        return self._wrap(And(self.node, Name(*names)))

    atom = name

    def resname(self, *names):
        return self._wrap(And(self.node, Resname(*names)))

    def resid(self, *values):
        return self._wrap(And(self.node, Resid(*values)))

    residue = resid

    def chain(self, *values):
        return self._wrap(And(self.node, Chain(*values)))

    @property
    def center_of_mass(self):
        return Selection(Position(self.node, "com"), self.context, self.label)

    @property
    def center_of_geometry(self):
        return Selection(Position(self.node, "cog"), self.context, self.label)

    # -- compilation ---------------------------------------------------
    def to_gromacs(self):
        return self.node.to_gromacs()

    def to_mdanalysis(self):
        return self.node.to_mdanalysis()

    def __str__(self):
        return self.to_gromacs()

    def __repr__(self):
        return f"<Selection {self.to_gromacs()!r}>"

    # -- evaluation ----------------------------------------------------
    def _ctx(self, context=None):
        ctx = context or self.context
        if ctx is None:
            raise SelectionError(
                "this selection is not bound to a structure; use "
                "sim.select.* or pass context=SelectionContext(...)")
        return ctx

    def to_indices(self, context=None):
        """0-based atom indices."""
        return self._ctx(context).indices(self.to_gromacs())

    indices = property(to_indices)

    def evaluate(self, context=None):
        return self.to_indices(context)

    @property
    def n_atoms(self):
        return len(self.to_indices())

    def __len__(self):
        return self.n_atoms

    @property
    def atoms(self):
        ctx = self._ctx()
        table = ctx.atom_table()
        return [table[i] for i in self.to_indices()]

    @property
    def residues(self):
        seen, out = {}, []
        for atom in self.atoms:
            key = (atom.resid, atom.resname)
            if key not in seen:
                seen[key] = Residue(atom.resid, atom.resname, [])
                out.append(seen[key])
            seen[key].indices.append(atom.index)
        return out

    def write_ndx(self, path, name=None):
        """Write a single-group index file (1-based, GROMACS convention)."""
        indices = self.to_indices()
        if not indices:
            raise SelectionError(f"selection matched no atoms: {self.to_gromacs()}")
        group = name or self.label or "gmxpy_selection"
        lines = [f"[ {group} ]"]
        row = []
        for i in indices:
            row.append(f"{i + 1:>6}")
            if len(row) == 15:
                lines.append(" ".join(row))
                row = []
        if row:
            lines.append(" ".join(row))
        path = Path(path)
        path.write_text("\n".join(lines) + "\n")
        return path


def _node(obj):
    if isinstance(obj, Selection):
        return obj.node
    if isinstance(obj, Node):
        return obj
    if isinstance(obj, str):
        return Raw(obj)
    raise SelectionError(f"cannot use {obj!r} as a selection")


# ---------------------------------------------------------------------
# evaluation context
# ---------------------------------------------------------------------

class SelectionContext:
    """Binds selections to a structure/tpr so they can be evaluated."""

    def __init__(self, structure, env=None, workdir=None):
        from .command import abspath
        self.structure = abspath(structure)
        self.env = env
        self.workdir = abspath(workdir or self.structure.parent)
        self._cache = {}
        self._atoms = None

    def _env(self):
        if self.env is None:
            from .environment import Environment
            self.env = Environment.detect()
        return self.env

    def indices(self, expression):
        if expression in self._cache:
            return self._cache[expression]
        from .command import Select

        scratch = self.workdir / ".gmxpy"
        scratch.mkdir(parents=True, exist_ok=True)
        ndx = scratch / "select.ndx"
        proc = Select(env=self._env(), structure=str(self.structure),
                      select=expression, ndx_out=str(ndx)).run(check=False)
        if proc.returncode != 0:
            raise SelectionError(
                f"invalid selection {expression!r}\n"
                + (_last_error(proc.stderr) or proc.stderr[-400:]))
        result = _read_ndx_first_group(ndx)
        self._cache[expression] = result
        return result

    def atom_table(self):
        """Per-atom (index, resid, resname, name), 0-based index."""
        if self._atoms is None:
            self._atoms = _read_atoms(self._gro())
        return self._atoms

    def _gro(self):
        if self.structure.suffix.lower() in (".gro", ".pdb"):
            return self.structure
        scratch = self.workdir / ".gmxpy"
        scratch.mkdir(parents=True, exist_ok=True)
        cached = scratch / (self.structure.stem + "_atoms.gro")
        if not cached.exists():
            from .command import Editconf
            Editconf(env=self._env(), structure=str(self.structure),
                     output=str(cached)).run()
        return cached


def _last_error(text):
    for marker in ("Invalid selection", "Fatal error", "syntax error"):
        idx = (text or "").find(marker)
        if idx >= 0:
            return text[idx:idx + 400].strip()
    return ""


def _read_ndx_first_group(path):
    out, started = [], False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            if started:
                break
            started = True
            continue
        if started:
            out += [int(v) - 1 for v in line.split()]
    return out


def _read_atoms(path):
    path = Path(path)
    atoms = []
    if path.suffix.lower() == ".gro":
        lines = path.read_text(errors="replace").splitlines()
        n = int(lines[1])
        for i, line in enumerate(lines[2:2 + n]):
            atoms.append(Atom(i, int(line[0:5]), line[5:10].strip(),
                              line[10:15].strip()))
    else:
        i = 0
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(("ATOM", "HETATM")):
                atoms.append(Atom(i, int(line[22:26]), line[17:20].strip(),
                                  line[12:16].strip()))
                i += 1
            elif line.startswith("ENDMDL"):
                break
    return atoms


# ---------------------------------------------------------------------
# selector factory -- this is ``sim.select``
# ---------------------------------------------------------------------

class Selector:
    """Namespace of ready-made selections, bound to one structure."""

    def __init__(self, context=None):
        self.context = context

    def _sel(self, node, label=None):
        return Selection(node, self.context, label)

    # default GROMACS index groups
    @property
    def all(self):
        return self._sel(All(), "System")

    system = all

    @property
    def none(self):
        return self._sel(Nothing(), "None")

    @property
    def protein(self):
        return self._sel(Group("Protein"), "Protein")

    @property
    def backbone(self):
        return self._sel(Group("Backbone"), "Backbone")

    @property
    def c_alpha(self):
        return self._sel(Name("CA") , "C-alpha")

    @property
    def sidechain(self):
        return self._sel(Group("SideChain"), "SideChain")

    @property
    def water(self):
        return self._sel(Group("Water"), "Water")

    solvent = water

    @property
    def ions(self):
        return self._sel(Group("Ion"), "Ion")

    @property
    def non_water(self):
        return self._sel(Group("non-Water"), "non-Water")

    @property
    def ligand(self):
        """Everything that is not protein, water or ion.

        GROMACS' default ``Other`` group; for a standard protein-ligand box
        that is exactly the ligand.
        """
        return self._sel(Group("Other"), "Other")

    @property
    def heavy_atoms(self):
        return self._sel(Not(Name("H*")), "heavy")

    @property
    def protein_heavy(self):
        return self._sel(And(Group("Protein"), Not(Name("H*"))), "Protein-H")

    # parameterised selectors
    def name(self, *names):
        return self._sel(Name(*names))

    def resname(self, *names):
        return self._sel(Resname(*names))

    def resid(self, *values):
        return self._sel(Resid(*values))

    residue = resid

    def resindex(self, *values):
        return self._sel(Resindex(*values))

    def chain(self, *values):
        return self._sel(Chain(*values))

    def atomnr(self, *values):
        return self._sel(Atomnr(*values))

    def group(self, name):
        return self._sel(Group(name), name)

    def expr(self, text):
        """Raw GROMACS selection text, when you really want it."""
        return self._sel(Raw(text))

    def molecule(self, name):
        return self._sel(Resname(name), name)

    # high-level semantics
    def binding_site(self, ligand=None, cutoff=0.5, by_residue=True):
        lig = self.ligand if ligand is None else (
            self.resname(ligand) if isinstance(ligand, str) else ligand)
        site = self.protein.within(cutoff, lig)
        return site.by_residue() if by_residue else site

    def __repr__(self):
        return "<Selector protein/water/ions/ligand/backbone/... >"
