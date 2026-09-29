"""gmxpy checks.

    python tests/test_gmxpy.py                 # unit tests, no GROMACS needed
    python tests/test_gmxpy.py --integration   # + a real grompp/mdrun/analysis run
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gmxpy import (MDP, Selector, Series, Topology, errors, landscape,
                   units as u)
from gmxpy.data import read_xvg
from gmxpy.errors import classify
from gmxpy.selection import _read_atoms, _read_ndx_first_group

TEST_SYSTEM = Path(os.environ.get("GMXPY_TEST_SYSTEM",
                              "~/abfe-benchmark/parameters/brd4/ligand1")).expanduser()


def needs_gmx(func):
    """Skip an integration test without a test system or a reachable gmx.

    Works both under pytest (shows as a skip, not a failure) and under
    ``python tests/test_gmxpy.py``.
    """
    def wrapper(*args, **kwargs):
        if not TEST_SYSTEM.exists():
            print("  (skipped: no test system)")
            return
        try:
            from gmxpy.environment import Environment
            Environment.detect()
        except Exception:
            print("  (skipped: no gmx -- module load gromacs/... first)")
            return
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


def test_units():
    assert (5 * u.angstrom).canonical == 0.5
    assert (0.5 * u.nanometer).value_in("angstrom") == 5
    assert float(2 * u.femtosecond) == 0.002
    assert (300 * u.kelvin).to("K").magnitude == 300
    assert (1 * u.nanosecond) > (100 * u.picosecond)
    assert ((1 * u.nanometer) + (5 * u.angstrom)).canonical == 1.5
    try:
        (5 * u.angstrom) + (300 * u.kelvin)
    except u.UnitError:
        pass
    else:
        raise AssertionError("mixing dimensions must fail")


def test_mdp():
    mdp = MDP(integrator="md", dt=2 * u.femtosecond, nsteps=1000,
              ref_t=300 * u.kelvin)
    assert mdp["dt"] == "0.002" and mdp.nsteps == 1000
    assert mdp.simulation_time.value_in("ps") == 2.0
    assert mdp.validate() == []

    bad = MDP(integrator="verlet", dt=-1, nonsense="x", tcoupl="v-rescale",
              tc_grps="Protein Water", ref_t=300, tau_t=0.1)
    problems = " ".join(bad.validate())
    assert "integrator" in problems and "nonsense" in problems
    assert "dt must be" in problems and "ref-t has 1" in problems
    try:
        bad.validate(strict=True)
    except errors.MdpError:
        pass
    else:
        raise AssertionError("strict validate must raise")

    with tempfile.TemporaryDirectory() as tmp:
        path = MDP.preset("npt", nsteps=42).write(Path(tmp) / "a.mdp")
        assert MDP.read(path).nsteps == 42
        assert MDP.read(path).pcoupl.lower() == "c-rescale"


def test_selection_compiles():
    s = Selector()
    assert s.protein.to_gromacs() == 'group "Protein"'
    assert s.protein.within(5 * u.angstrom, s.resname("LIG")).to_gromacs() == \
        '(group "Protein") and (within 0.5 of (resname LIG))'
    assert s.binding_site("LIG", 5 * u.angstrom).to_gromacs().startswith(
        "same residue as")
    assert (s.chain("A") | s.chain("B")).to_mdanalysis() == "(segid A) or (segid B)"
    assert (s.protein & ~s.resname("GLY")).to_gromacs() == \
        '(group "Protein") and (not (resname GLY))'
    assert s.resid(10, 30).to_gromacs() == "resid 10 to 30"
    assert s.residue(100).atom("CA").center_of_mass.to_gromacs() == \
        "com of ((resid 100) and (name CA)) pbc"
    # unbound selections cannot be evaluated
    try:
        s.protein.n_atoms
    except errors.SelectionError:
        pass
    else:
        raise AssertionError("unbound selection must refuse to evaluate")


def test_error_classification():
    grompp_out = """
-------------------------------------------------------
Program:     gmx grompp, version 2026.3

Fatal error:
No such moleculetype LIG

For more information and tips for troubleshooting, please check the GROMACS
website at http://www.gromacs.org/Documentation/Errors
-------------------------------------------------------
"""
    err = classify(grompp_out, ["gmx", "grompp"], 1, "grompp")
    assert isinstance(err, errors.TopologyError)
    assert err.message == "No such moleculetype LIG"

    lincs = ("Step 42, time 0.084 (ps)\nLINCS WARNING\n"
             "relative constraint deviation after LINCS:\natoms 12 13 original")
    err = classify(lincs, None, 1, "mdrun")
    assert isinstance(err, errors.LincsError) and err.step == 42
    assert err.atoms == [12, 13]

    err = classify("something went wrong", None, 1, "mdrun")
    assert isinstance(err, errors.MdrunError)

    # grompp puts the real reason in an ERROR block; the Fatal error only counts
    grompp_error = """
ERROR 1 [file md.mdp]:
  The largest distance between excluded atoms is 1.181 nm, which is larger
  than the cut-off distance.

-------------------------------------------------------
Fatal error:
There was 1 error in input file(s)
-------------------------------------------------------
"""
    message = classify(grompp_error, None, 1, "grompp").message
    assert "largest distance" in message and "1 error in input" not in message


def test_equilibration():
    """Burn-in detection: a ramp then a plateau."""
    ramp = [10.0 - 9.0 * i / 20 for i in range(20)]     # settles at ~1.0
    plateau = [1.0 + (0.05 if i % 2 else -0.05) for i in range(180)]
    series = Series(list(range(200)), ramp + plateau, "test")
    start = series.equilibration_time()
    assert 0 < start < 60
    trimmed = series.equilibrated()
    assert abs(trimmed.mean - 1.0) < 0.2 and trimmed.std < series.std


def test_series():
    series = Series(list(range(10)), [float(i) for i in range(10)], "x")
    assert len(series) == 10 and series.mean == 4.5
    assert series.since(5).x[0] == 5
    assert len(series.rolling(3)) == 10
    assert len(series.block_average(5)) == 5
    assert series[2] == 2.0 and len(series[2:5]) == 3


def test_xvg():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.xvg"
        path.write_text('# comment\n@ title "t"\n@ s0 legend "RMSD"\n'
                        "0 1.0\n1 2.0\n2 3.0\n")
        columns, meta = read_xvg(path)
        assert columns == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
        assert meta["legends"] == ["RMSD"]


def test_ndx_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        indices = list(range(0, 40))
        path = Path(tmp) / "a.ndx"
        # write_ndx needs evaluation, so build the file the same way by hand
        lines = ["[ test ]"]
        for start in range(0, len(indices), 15):
            lines.append(" ".join(f"{i + 1:>6}" for i in indices[start:start + 15]))
        path.write_text("\n".join(lines) + "\n")
        assert _read_ndx_first_group(path) == indices


def test_topology_and_structure():
    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    top = Topology(TEST_SYSTEM / "complex.top")
    assert top.molecule_names == ["system1", "LIG", "HOH", "NA", "CL"]
    assert dict(top.molecules)["HOH"] == 12478
    assert top.n_atoms_declared == 39689
    atoms = _read_atoms(TEST_SYSTEM / "complex.gro")
    assert len(atoms) == 39689
    assert atoms[0].resname == "ACE" and atoms[0].resid == 1


# ---------------------------------------------------------------------
# new features: hpc / protocol / sampling (unit, no GROMACS)
# ---------------------------------------------------------------------

def test_job_scripts():
    from gmxpy.hpc import JobSpec, render, render_array, write

    spec = JobSpec(command="gmx mdrun -deffnm prod", workdir="/runs/prod",
                   name="prod", ncpus=8, ngpus=1, walltime="12:00:00",
                   modules=("gromacs/2026.3",), env={"GMXPY_GMX": "/x/gmx"})
    pbs = render(spec, "pbs")
    assert "#PBS -N prod" in pbs
    assert "#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1" in pbs
    assert "#PBS -l walltime=12:00:00" in pbs
    assert "module load gromacs/2026.3" in pbs
    assert "cd /runs/prod" in pbs and "gmx mdrun -deffnm prod" in pbs

    torque = render(spec, "torque")
    assert "#PBS -l nodes=1:ppn=8:gpus=1" in torque

    slurm = render(spec, "slurm")
    assert "--gres=gpu:1" in slurm and "--cpus-per-task=8" in slurm

    try:
        render(spec, "lsf")
    except errors.HpcError:
        pass
    else:
        raise AssertionError("unknown scheduler must raise")

    other = JobSpec(command="gmx mdrun -deffnm md", workdir="/runs/win_01")
    array = render_array([spec, other], "pbs", name="umb")
    assert "#PBS -J 0-1" in array
    assert "/runs/prod" in array and "/runs/win_01" in array
    assert "${PBS_ARRAY_INDEX:-0}" in array and "DIRS=(" in array

    with tempfile.TemporaryDirectory() as tmp:
        path = write(spec, Path(tmp) / "job.sh", "pbs")
        assert Path(path).read_text() == pbs


def test_pull_mdp():
    from gmxpy import Pull

    pull = Pull("Protein", "resname LIG", k=1000, init=0.5)
    keys = pull.mdp_keys()
    assert keys["pull"] == "yes"
    assert keys["pull-ngroups"] == 2 and keys["pull-ncoords"] == 1
    assert keys["pull-group1-name"] == "gmxpy_pull0"
    assert keys["pull-coord1-groups"] == "1 2"
    assert keys["pull-coord1-type"] == "umbrella"
    assert keys["pull-coord1-geometry"] == "distance"
    assert keys["pull-coord1-k"] == 1000
    assert keys["pull-coord1-init"] == 0.5

    window = pull.copy(init=0.7)
    assert window.mdp_keys()["pull-coord1-init"] == 0.7
    assert pull.mdp_keys()["pull-coord1-init"] == 0.5      # original untouched

    quantity = Pull("a", "b", init=5 * u.angstrom)
    assert quantity.mdp_keys()["pull-coord1-init"] == 0.5

    vec = Pull("a", "b", geometry="direction", vec=(0, 0, 1)).mdp_keys()
    assert vec["pull-coord1-vec"] == "0 0 1"

    for bad in (dict(geometry="torsion"), dict(type="spring")):
        try:
            Pull("a", "b", **bad)
        except errors.GmxpyError:
            pass
        else:
            raise AssertionError("bad pull options must raise")

    # merged into an mdp exactly the way Simulation(pull=) does it
    mdp = MDP.preset("md").update(**keys)
    assert mdp["pull-coord1-k"] == "1000"


def test_demux_parser():
    from gmxpy import demux

    log = """
Repl t=1.0 step=500: 0x 1x
Repl t=2.0 step=1000: 0X 1x
Repl t=3.0 step=1500: 0x 1X
Repl count[ 0]=        1
Repl count[ 1]=        1
Repl  av. x:          0.333    0.333
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "md.log"
        path.write_text(log)
        trace = demux(path)
        assert trace.times == [1.0, 2.0, 3.0]
        # directory temperatures: r0/r1 swap at t=2 (pair 0), then at t=3 the
        # config now in r1 swaps with r2... only 2 pairs, 3 dirs
        assert trace.temperatures[0] == [0, 1, 2]
        assert trace.temperatures[1] == [1, 0, 2]
        assert trace.temperatures[2] == [1, 2, 0]
        assert trace.exchange_counts == [1, 1]
        assert trace.exchange_fractions == [0.333, 0.333]
        try:
            frame = trace.to_dataframe()
        except ImportError:                        # pandas is optional
            pass
        else:
            assert frame.shape == (3, 4) and list(frame["r1"]) == [1, 0, 2]

        hot = demux(path, temperatures=[300, 320, 340])
        assert hot.series(0).y == [300, 320, 320]
        assert hot.series(2).name == "r2"

        empty = Path(tmp) / "empty.log"
        empty.write_text("nothing here")
        try:
            demux(empty)
        except errors.AnalysisError:
            pass
        else:
            raise AssertionError("a log without exchanges must raise")

    # the layout GROMACS has written since 2024
    new_format = """
Replica exchange at step 50 time 0.10000
Repl 0 <-> 1  dE_term =  2.556e-01 (kT)
Repl ex  0    1
Repl pr   .77
Replica exchange at step 100 time 0.20000
Repl ex  0 x  1
Repl pr   1.0
Replica exchange at step 150 time 0.30000
Repl ex  0    1
Repl pr   .65
Repl  average number of exchanges:
Repl     0    1
Repl      .33
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "md.log"
        path.write_text(new_format)
        trace = demux(path, temperatures=[300, 320])
        assert trace.times == [0.1, 0.2, 0.3]
        # no swap, swap, no swap: directory 0 ends up running 320 K
        assert trace.temperatures == [[300, 320], [320, 300], [320, 300]]
        assert trace.exchange_counts == [1]
        assert trace.exchange_fractions == [0.33]
        assert trace.series(0).y == [300, 320, 320]


def test_protocol_stage_mdp():
    from gmxpy import Protocol
    from gmxpy.protocol import _steps

    assert _steps(100 * u.ps, 0.002) == 50000
    assert _steps(0.5, 0.002) == 250
    try:
        _steps(-1, 0.002)
    except errors.GmxpyError:
        pass
    else:
        raise AssertionError("a negative duration must raise")

    protocol = Protocol(nvt=100 * u.ps, npt=1 * u.ns, dt=2 * u.femtosecond,
                        temperature=310)
    assert protocol.stage_names == ["em", "nvt", "npt"]
    nvt = protocol.stage_mdp("nvt")
    assert nvt.nsteps == 50000 and nvt.ref_t == 310
    assert nvt.gen_vel == "yes"                    # velocities start here
    npt = protocol.stage_mdp("npt")
    assert npt.nsteps == 500000 and npt.ref_t == 310 and npt.gen_vel == "no"
    assert npt.pcoupl.lower() == "c-rescale"
    assert protocol.stage_mdp("em").nsteps == 50000

    # bare em number is a step count; dict overrides merge, keeping defaults
    assert Protocol(em=137).stage_mdp("em").nsteps == 137
    custom = Protocol(em=False, nvt={"gen_seed": 7})
    assert custom.stage_names == ["nvt", "npt"]
    seeded = custom.stage_mdp("nvt")
    assert seeded.gen_seed == 7 and seeded.nsteps == 50000   # 100 ps default
    assert Protocol(nvt={"nsteps": 50}).stage_mdp("nvt").nsteps == 50
    assert Protocol(nvt={"duration": 0.06}).stage_mdp("nvt").nsteps == 30
    # a stage with neither duration nor nsteps cannot be built
    stripped = Protocol(nvt=100.0)
    stripped._durations.pop("nvt")
    try:
        stripped.stage_mdp("nvt")
    except errors.GmxpyError as exc:
        assert "duration" in str(exc)
    else:
        raise AssertionError("nvt without nsteps/duration must raise")


def test_protocol_posre_detection():
    from gmxpy.protocol import Equilibration

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "s.gro").write_text("t\n0\n 0.0 0.0 0.0\n")
        restrained = tmp / "with_posre.top"
        restrained.write_text('#include "forcefield.itp"\n'
                              '#ifdef POSRES\n#include "posre.itp"\n#endif\n')
        free = tmp / "plain.top"
        free.write_text('#include "forcefield.itp"\n')

        eq = Equilibration(structure=tmp / "s.gro", topology=restrained,
                           workdir=tmp, env=object())
        assert eq._posre_block() == "-DPOSRES"

        eq = Equilibration(structure=tmp / "s.gro", topology=free,
                           workdir=tmp, env=object(), posre="auto")
        assert eq._posre_block() is None and eq.notes

        try:
            Equilibration(structure=tmp / "s.gro", topology=free, workdir=tmp,
                          env=object(), posre=True)._posre_block()
        except errors.GmxpyError:
            pass
        else:
            raise AssertionError("posre=True without a POSRES block must raise")


def test_temperature_ladder():
    from gmxpy import temperature_ladder

    ladder = temperature_ladder(300, 344.2, 3)
    assert len(ladder) == 3 and abs(ladder[0] - 300) < 1e-9
    assert abs(ladder[1] ** 2 - 300 * 344.2) < 1.0       # geometric middle
    assert ladder == sorted(ladder)
    try:
        temperature_ladder(300, 300, 3)
    except errors.GmxpyError:
        pass
    else:
        raise AssertionError("a flat ladder must raise")


def test_wham_inputs():
    from gmxpy.sampling import _wham_inputs
    from gmxpy import Umbrella, Pull

    with tempfile.TemporaryDirectory() as tmp:
        tprs = [Path(tmp) / f"w{i}.tpr" for i in range(3)]
        found, pullfs, base = _wham_inputs(tprs)
        assert found == tprs and pullfs is None and base == tprs[0].parent

        umb = Umbrella(Pull("a", "b"), [0.4, 0.5], workdir=tmp, env=object())
        try:
            _wham_inputs(umb)
        except errors.GmxpyError:
            pass
        else:
            raise AssertionError("unbuilt windows must raise")


# ---------------------------------------------------------------------
# integration -- needs a working gmx and the test system
# ---------------------------------------------------------------------

@needs_gmx
def test_integration():
    from gmxpy import Simulation

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(
            structure=TEST_SYSTEM / "complex.gro",
            topology=TEST_SYSTEM / "complex.top",
            mdp=MDP.preset("md", nsteps=100, nstenergy=10,
                           nstxout_compressed=10, nstlog=50, gen_vel="yes",
                           gen_temp=300, gen_seed=1, continuation="no"),
            name="test", workdir=tmp)
        sim.prepare(maxwarn=1)
        assert sim.tpr.exists()

        result = sim.run(ntomp=8, ntmpi=1)
        assert result.finished and result.steps == 100
        assert result.performance > 0

        # selections resolve against the tpr
        assert sim.select.protein.n_atoms == 2130
        assert sim.select.ligand.n_atoms == 56
        site = sim.select.binding_site(cutoff=5 * u.angstrom)
        assert 5 < len(site.residues) < 40
        assert "TRP41" in [f"{r.resname}{r.resid}" for r in site.residues]

        # energy: both backends must agree
        potential = sim.energy["Potential"]
        assert len(potential) == 11 and potential.mean < 0
        by_gmx = sim.energy._extract_via_gmx(["Potential"])[0]
        assert abs(by_gmx.mean - potential.mean) < 1e-3
        assert sim.energy.unit("Temperature") == "K"

        # partial and case-insensitive term names
        assert sim.energy["temperature"].mean > 0

        # analysis
        assert len(sim.analysis.rmsd(sim.select.protein)) == 11
        assert sim.analysis.radius_of_gyration(sim.select.protein).mean > 1.0
        assert sim.analysis.distance(sim.select.ligand, site).mean < 1.0
        assert len(sim.analysis.rmsf(sim.select.protein, per_residue=True)) > 100

        # trajectory
        assert sim.trajectory.n_frames == 11
        assert sim.trajectory.n_atoms == 39689

        # restart
        sim.extend(extend=0.2)
        assert sim.resume(ntomp=8).steps == 200

        # a bad selection is a SelectionError, not a CalledProcessError
        try:
            sim.select.expr("not a selection").n_atoms
        except errors.SelectionError:
            pass
        else:
            raise AssertionError("bad selection must raise SelectionError")

        # the rest of the common analysis set
        assert 20 < sim.analysis.sasa(sim.select.protein).mean < 200
        assert len(sim.analysis.sasa(sim.select.protein, per_residue=True)) > 100
        assert sim.analysis.mindist(sim.select.protein, sim.select.ligand).mean < 0.6
        assert sim.analysis.mindist(sim.select.protein, sim.select.ligand,
                                    contacts=True).mean > 0
        assert sim.analysis.periodic_image_distance(sim.select.protein).mean > 1.0
        assert sim.analysis.density(sim.select.water, axis="z").mean > 100

        helices = {s.name: s.mean for s in
                   sim.analysis.secondary_structure(sim.select.protein)}
        assert helices["alpha-Helices"] > 40      # BRD4 BD1 is a helix bundle

        pca = sim.analysis.pca(sim.select.protein & sim.select.name("CA"),
                               n_components=2)
        assert len(pca.components) == 2
        assert len(pca.components[0]) == sim.trajectory.n_frames
        assert 0 < pca.explained[0] < 1

        clusters = sim.analysis.cluster(sim.select.protein & sim.select.name("CA"))
        assert clusters.n_clusters >= 1 and clusters.structures.exists()

        # trjconv: pbc + fit needs two passes, and must renumber correctly
        processed = sim.trajectory.process(
            output=Path(tmp) / "fit.xtc", pbc="mol",
            center=sim.select.protein, fit="rot+trans",
            group=sim.select.non_water, skip=2)
        assert processed.n_atoms == sim.select.non_water.n_atoms
        assert processed.n_frames < sim.trajectory.n_frames
        frame = sim.trajectory.frame(time=0, output=Path(tmp) / "f.pdb",
                                     group=sim.select.protein)
        assert frame.exists()

        # free-energy surface from two observables (pure numpy)
        surface = landscape(sim.analysis.rmsd(sim.select.protein),
                            sim.analysis.radius_of_gyration(sim.select.protein),
                            bins=8)
        assert surface.energy.shape == (8, 8) and len(surface.minimum) == 2

        # report
        report = sim.report()
        assert report.exists() and report.stat().st_size > 10000
        assert "not available" not in report.read_text()


@needs_gmx
def test_preparation():
    """PDB -> force field -> box -> water -> ions -> minimised."""
    from gmxpy import System

    pdb = TEST_SYSTEM.parent / "protein.pdb"
    if not pdb.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        capless = Path(tmp) / "capless.pdb"      # this PDB has Amber ACE/NMA caps
        capless.write_text("".join(
            line for line in pdb.read_text().splitlines(True)
            if not (line.startswith(("ATOM", "HETATM"))
                    and line[17:20].strip() in ("ACE", "NMA"))))

        system = System.from_pdb(capless, workdir=tmp)
        system.pdb2gmx(forcefield="amber99sb-ildn", water="tip3p")
        assert system.n_atoms > 2000
        dry = system.n_atoms

        system.box(shape="dodecahedron", padding=1.0 * u.nanometer)
        assert system.n_atoms == dry and system.box_vectors[0].magnitude > 5

        system.solvate()
        assert system.n_atoms > dry * 5

        system.add_ions(concentration=0.15)
        molecules = dict(system.molecules)
        assert molecules["NA"] > 0 and molecules["CL"] > 0

        sim = system.simulation(MDP.preset("em", nsteps=100), name="em")
        sim.prepare()
        result = sim.run(ntomp=8)
        assert result.finished and result.minimisation
        assert result.potential_energy < 0 and result.max_force > 0
        assert result.simulation_time is None      # steps are not picoseconds


@needs_gmx
def test_project():
    """Several trajectories at once: fan-out, mean +/- sd, concatenation."""
    from gmxpy import Project, Simulation

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(2):
            mdp = MDP.preset("md", nsteps=200, nstenergy=20,
                             nstxout_compressed=20, nstlog=200, gen_vel="yes",
                             gen_temp=300, gen_seed=rep + 1, continuation="no")
            sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                             topology=TEST_SYSTEM / "complex.top", mdp=mdp,
                             name="md", workdir=Path(tmp) / f"rep{rep}")
            sim.prepare(maxwarn=1)
            sim.run(ntomp=8)

        project = Project(tmp)
        assert project.names == ["rep0", "rep1"] and project.finished
        assert len(project.summary()) == 2

        temperature = project.energy["Temperature"]
        assert len(temperature) == 2 and all(v > 250 for v in temperature.means.values())
        assert len(temperature.mean_series()) == len(temperature["rep0"])
        assert all(v >= 0 for v in temperature.std_series().y)

        # one selection object, re-bound to every run
        rmsd = project.analysis.rmsd(project.select.protein)
        assert rmsd.name == "RMSD" and len(rmsd) == 2
        assert rmsd.to_dataframe().shape[0] == sum(len(s) for s in rmsd)
        assert rmsd.to_dataframe(wide=True).shape[1] == 3

        frames = project["rep0"].trajectory.n_frames
        assert project.concatenate(mode="append").n_frames == 2 * frames
        assert project.concatenate(output=Path(tmp) / "c.xtc").n_frames == frames

        report = project.report()
        assert report.exists() and "not available" not in report.read_text()


@needs_gmx
def test_analysis_cache():
    """A repeated question must not re-read the trajectory."""
    from gmxpy import Simulation
    from gmxpy.analysis import Analysis

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                         topology=TEST_SYSTEM / "complex.top",
                         mdp=MDP.preset("md", nsteps=100, nstxout_compressed=10,
                                        nstenergy=50, nstlog=100, gen_vel="yes",
                                        gen_temp=300, gen_seed=1,
                                        continuation="no"),
                         name="c", workdir=tmp)
        sim.prepare(maxwarn=1)
        sim.run(ntomp=8)

        # the disk cache is the gmx backend's; the fast one memoises in RAM
        analysis = Analysis(sim, engine="gmx")
        first = analysis.rmsd(sim.select.protein)
        cached = next(Path(tmp).glob(".gmxpy/rmsd_*.xvg"))
        stamp = cached.stat().st_mtime_ns

        again = analysis.rmsd(sim.select.protein)
        assert cached.stat().st_mtime_ns == stamp      # not regenerated
        assert again.y == first.y

        # a different question is a different file
        analysis.rmsd(sim.select.protein & sim.select.name("CA"))
        assert len(list(Path(tmp).glob(".gmxpy/rmsd_*.xvg"))) == 2

        # cache=False always recomputes
        fresh = Analysis(sim, cache=False, engine="gmx")
        fresh.rmsd(sim.select.protein)
        assert cached.stat().st_mtime_ns > stamp


@needs_gmx
def test_checks_and_cli():
    """sim.check() and the command line, on a run that is deliberately young."""
    from gmxpy import Simulation
    from gmxpy.__main__ import main
    from gmxpy.quality import FAIL, SKIP, WARN

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                         topology=TEST_SYSTEM / "complex.top",
                         mdp=MDP.preset("md", nsteps=200, nstenergy=20,
                                        nstxout_compressed=20, nstlog=200,
                                        gen_vel="yes", gen_temp=300,
                                        gen_seed=1, continuation="no"),
                         name="q", workdir=tmp)
        sim.prepare(maxwarn=1)
        sim.run(ntomp=8)

        report = sim.check()
        assert report["finished"].status != FAIL
        assert report["constraints"].value == 0
        # gen_vel without minimising first: potential relaxation heats it up,
        # and the check is expected to say so
        assert report["temperature"].value > 200
        assert report["temperature"].status == WARN
        # 0.4 ps is far too short to judge energy conservation, and it says so
        assert report["energy drift"].status == SKIP
        assert "periodic images" in [c.name for c in report]
        assert isinstance(str(report), str) and len(report) > 3

        assert main(["info", tmp]) == 0
        assert main(["energy", tmp, "Temperature"]) == 0
        assert main(["report", tmp, "-o", str(Path(tmp) / "r.html")]) == 0
        assert (Path(tmp) / "r.html").exists()


@needs_gmx
def test_analysis_engines():
    """The in-memory backend must agree with gmx on what it replaces."""
    from gmxpy import Simulation
    from gmxpy.analysis import Analysis

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    try:
        import mdtraj  # noqa: F401
    except ImportError:
        print("  (skipped: no mdtraj)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                         topology=TEST_SYSTEM / "complex.top",
                         mdp=MDP.preset("md", nsteps=500, nstenergy=50,
                                        nstxout_compressed=50, nstlog=500,
                                        gen_vel="yes", gen_temp=300,
                                        gen_seed=1, continuation="no"),
                         name="e", workdir=tmp)
        sim.prepare(maxwarn=1)
        sim.run(ntomp=8)

        slow = Analysis(sim, cache=False, engine="gmx")
        fast = Analysis(sim, cache=False, engine="mdtraj")
        protein = sim.select.protein

        def agree(a, b, tolerance):
            values_a, values_b = a.y, b.y
            assert len(values_a) == len(values_b)
            scale = max(abs(v) for v in values_a) or 1.0
            worst = max(abs(x - y) for x, y in zip(values_a, values_b))
            assert worst / scale < tolerance, f"{worst / scale:.4f} >= {tolerance}"

        agree(slow.rmsd(protein), fast.rmsd(protein), 0.01)
        agree(slow.rmsf(protein), fast.rmsf(protein), 0.01)
        agree(slow.radius_of_gyration(protein),
              fast.radius_of_gyration(protein), 0.01)
        # different surface algorithms; same answer to a few percent
        agree(slow.sasa(protein), fast.sasa(protein), 0.05)

        # auto picks the fast path, and one load serves every analysis
        auto = Analysis(sim, cache=False)
        assert auto._use_fast()
        assert sim.frames.n_frames == sim.trajectory.n_frames
        # msd stays on gmx even on auto: the fitted D differs between them
        assert auto.msd(sim.select.water).diffusion_error is not None


@needs_gmx
def test_free_energy():
    """A short decoupling, then BAR."""
    from gmxpy import Simulation, bar

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        dhdl = []
        for state in range(2):
            mdp = MDP.preset(
                "md", nsteps=500, nstenergy=100, nstxout_compressed=0,
                nstlog=500, gen_vel="yes", gen_temp=300, gen_seed=1,
                continuation="no", free_energy="yes", init_lambda_state=state,
                coul_lambdas="0.0 1.0", vdw_lambdas="0.0 0.0",
                couple_moltype="LIG", couple_lambda0="vdw-q",
                couple_lambda1="none", couple_intramol="yes",
                sc_alpha=0.5, sc_power=1, sc_sigma=0.3, nstdhdl=10,
                calc_lambda_neighbors=-1)
            sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                             topology=TEST_SYSTEM / "complex.top", mdp=mdp,
                             name=f"l{state}", workdir=Path(tmp) / f"l{state}")
            sim.prepare(maxwarn=2)
            sim.run(ntomp=8)
            dhdl.append(sim.workdir / f"l{state}.xvg")
        result = bar(dhdl, workdir=tmp)
        assert len(result.pairs) == 1
        assert result.error >= 0 and result.unit == "kJ/mol"
        assert len(result.profile) == 2


@needs_gmx
def test_equilibrate():
    """The em -> nvt -> npt chain, and the handover to production."""
    from gmxpy import Equilibration, MDP
    from gmxpy.simulation import Simulation

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        chain = Equilibration(
            structure=TEST_SYSTEM / "complex.gro",
            topology=TEST_SYSTEM / "complex.top",
            workdir=tmp,
            protocol=None,
            em=100, nvt=0.06 * u.picosecond, npt=0.06 * u.picosecond,
            dt=0.002)
        chain.run(run_kwargs=dict(ntomp=8))
        assert chain.ok and list(chain.stages) == ["em", "nvt", "npt"]
        # amber-style topology: no POSRES block, so it says so and runs free
        assert chain.notes and "POSRES" in chain.notes[0]

        # nvt produced velocities; npt ran from its checkpoint
        assert chain.stages["nvt"].result.finished
        assert chain.stages["npt"].checkpoint.exists

        checks = chain.check()
        assert set(checks) == {"em", "nvt", "npt"}

        prod = chain.production(
            MDP.preset("md", nsteps=50, nstenergy=10, nstlog=50,
                       nstxout_compressed=50), name="prod")
        assert (Path(tmp) / "prod.tpr").exists()
        prod.run(ntomp=8)
        assert prod.result.finished

        try:            # this topology cannot do restrained production
            chain.production(name="prod2", posre=True)
        except errors.GmxpyError:
            pass
        else:
            raise AssertionError("posre=True without a POSRES block must raise")

        # a finished chain is skipped on rerun
        assert chain.run() and all(sim.log.finished for sim in chain.stages.values())

        # the standalone path also works
        direct = Equilibration(structure=TEST_SYSTEM / "complex.gro",
                               topology=TEST_SYSTEM / "complex.top",
                               workdir=Path(tmp) / "direct2", em=50,
                               nvt={"nsteps": 20}, npt={"nsteps": 20})
        direct.run(run_kwargs=dict(ntomp=8))
        assert direct.ok


@needs_gmx
def test_pull_and_wham():
    """Pull code, umbrella windows, and a (tiny) WHAM roundtrip."""
    from gmxpy import MDP, Pull, Simulation, Umbrella, wham
    from gmxpy.selection import SelectionContext, Selector

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        context = SelectionContext(TEST_SYSTEM / "complex.gro", workdir=tmp)
        sel = Selector(context)
        pull = Pull(sel.group("Protein"), sel.resname("LIG"),
                    type="umbrella", k=1000)

        mdp = MDP.preset("md", nsteps=100, nstenergy=10, nstlog=100,
                         nstxout_compressed=0, gen_vel="yes", gen_temp=300,
                         gen_seed=1, continuation="no")
        sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                         topology=TEST_SYSTEM / "complex.top", mdp=mdp,
                         pull=pull, name="steer", workdir=tmp / "steer")
        sim.prepare(maxwarn=1)
        assert "pull-group1-name" in (tmp / "steer" / "steer.mdp").read_text()
        sim.run(ntomp=8)
        assert (tmp / "steer" / "steer_pullf.xvg").exists()
        assert (tmp / "steer" / "steer_pullx.xvg").exists()

        # windows: same start, different reference distances
        base = MDP.preset("md", nsteps=100, nstenergy=10, nstlog=100,
                          nstxout_compressed=0, gen_vel="yes", gen_temp=300,
                          gen_seed=1, continuation="no")
        umb = Umbrella(pull, values=[0.40, 0.48, 0.56], structure=TEST_SYSTEM
                       / "complex.gro", topology=TEST_SYSTEM / "complex.top",
                       mdp=base, workdir=tmp / "umb")
        windows = umb.build()
        assert len(windows) == 3
        assert (tmp / "umb" / "win_01").exists()
        assert umb.windows[1].pull.mdp_keys()["pull-coord1-init"] == 0.48
        umb.prepare(maxwarn=1)
        for window in windows:
            window.run(ntomp=8)
            assert window.result.finished
            assert (window.workdir / "md_pullf.xvg").exists()

        pmf = wham(umb, temperature=300, begin=0.0)
        assert len(pmf) > 10 and pmf.name == "PMF"
        assert len(pmf.histogram) == len(pmf)
        pmf.plot().save(tmp / "pmf.png")

        # as a project: array submission script
        project = umb.project()
        script = project.submit(scheduler="pbs", array=True, ncpus=8,
                                modules=("gromacs/2026.3",))
        text = script.read_text()
        assert "#PBS -J 0-2" in text and "win_02" in text


@needs_gmx
def test_remd():
    """Two-temperature REMD via mdrun -multidir, plus demux."""
    from gmxpy import MDP, Simulation, TemperatureREMD, demux

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mdp = MDP.preset("md", nsteps=100, nstenergy=10, nstlog=10,
                         nstxout_compressed=0, gen_vel="yes",
                         gen_seed=1, continuation="no")
        remd = TemperatureREMD(mdp, [300, 320],
                               structure=TEST_SYSTEM / "complex.gro",
                               topology=TEST_SYSTEM / "complex.top",
                               workdir=tmp)
        assert remd.temperatures == [300.0, 320.0]
        line = remd.command(replex=25)
        assert "mdrun" in line and "-multidir" in line and "-replex 25" in line
        remd.prepare(maxwarn=1)
        remd.run(replex=25, ntomp=4)
        assert all(sim.result.finished for sim in remd.sims)

        trace = demux(*[sim.paths["log"] for sim in remd.sims],
                      temperatures=remd.temperatures)
        assert trace.n_replicas == 2 and len(trace.times) >= 2
        assert all(row[0] in (300.0, 320.0) for row in trace.temperatures)


@needs_gmx
def test_submit_script():
    """Job scripts from a real simulation (no actual submission)."""
    from gmxpy import MDP, Simulation

    if not TEST_SYSTEM.exists():
        print("  (skipped: no test system)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(structure=TEST_SYSTEM / "complex.gro",
                         topology=TEST_SYSTEM / "complex.top",
                         mdp=MDP.preset("md", nsteps=100), name="prod",
                         workdir=tmp)
        script = sim.submit(scheduler="pbs", ncpus=8, ngpus=1,
                            walltime="4:00:00",
                            modules=("gromacs/2026.3",),
                            run_kwargs=dict(nb="gpu"))
        text = Path(script).read_text()
        assert "#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1" in text
        assert "-deffnm" in text and "-nb gpu" in text
        assert "module load gromacs/2026.3" in text

        torque = sim.submit(scheduler="torque", ncpus=4, output=Path(tmp) / "t.sh")
        assert "#PBS -l nodes=1:ppn=4" in Path(torque).read_text()


def test_tui():
    """The TUI, headless: render helpers plus a bootstrap over a fake run.

    Needs textual (pip install "gmxpy[tui]"); skips without it.  No GROMACS:
    a fake Environment stands in for the real detection, and the run has a
    log but no edr, so the energy tab exercises its error path.
    """
    try:
        import textual
    except ImportError:
        print("  (skipped: no textual)")
        return

    from gmxpy.tui import render

    # -- render helpers, no app needed ------------------------------------
    x = [i * 0.5 for i in range(40)]
    y = [300.0 + 5.0 * ((i * 2654435761) % 97) / 97 for i in range(40)]
    series = Series(x, y, name="Temperature", xlabel="Time (ps)",
                    ylabel="Temperature (K)")
    stats = render.stats_markup(series)
    assert "Temperature" in stats and "n=40" in stats
    assert render.plot_png(object()) is None          # not plottable
    other = Series(x, [v - 20.0 for v in y], name="Potential",
                   xlabel="Time (ps)", ylabel="Potential (kJ/mol)")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  (partly skipped: no matplotlib)")
    else:
        png = render.plot_png(series)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert render.plot_png([series, other])[:8] == b"\x89PNG\r\n\x1a\n"
        try:
            from textual_image.widget import Image  # noqa: F401
        except ImportError:
            pass
        else:
            assert render.image_widget(png) is not None

    # -- the app, headless --------------------------------------------------
    import asyncio

    from gmxpy.environment import Environment

    fake_env = Environment(executable="/bin/true", version="test",
                           source="test-env")
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        runs.mkdir()
        (runs / "demo.tpr").write_bytes(b"")
        (runs / "demo.log").write_text("Finished mdrun on rank 0.\n")

        async def scenario():
            from gmxpy.tui.app import GmxpyTUI
            app = GmxpyTUI(runs, env=fake_env)
            async with app.run_test(size=(120, 40)) as pilot:
                for _ in range(100):
                    if app._opened:
                        break
                    await asyncio.sleep(0.05)
                assert app._opened
                assert app.sim is not None and app.sim.name == "demo"
                await pilot.pause()

                from textual.widgets import DataTable, RichLog
                assert len(app.query_one("#log", RichLog).lines) >= 1
                for _ in range(100):
                    if app.query("#summary-table"):
                        break
                    await asyncio.sleep(0.05)
                table = app.query_one("#summary-table", DataTable)
                assert table.row_count == 1
                # a run with a log but no edr: the check tab shows the FAIL
                for _ in range(100):
                    if app.query_one("#check-body").children:
                        break
                    await asyncio.sleep(0.05)
                assert len(app.query_one("#check-body").children) >= 2

                # the plot body: sparkline fallback, or a real image when
                # matplotlib and textual-image are installed
                try:
                    import matplotlib  # noqa: F401
                    import textual_image.widget  # noqa: F401
                except ImportError:
                    app._show_plot("#energy-body", series, None,
                                   render.stats_markup(series), "#energy-stats")
                    from textual.widgets import Sparkline
                    assert any(isinstance(w, Sparkline) for w in
                               app.query_one("#energy-body").children)
                else:
                    app._show_plot("#energy-body", series, render.plot_png(series),
                                   render.stats_markup(series), "#energy-stats")
                    assert app.query_one("#energy-body").children

        async def scenario_empty():
            from gmxpy.tui.app import GmxpyTUI
            app = GmxpyTUI(Path(tmp) / "nowhere", env=fake_env)
            async with app.run_test(size=(100, 30)) as pilot:
                for _ in range(100):
                    if app._opened:
                        break
                    await asyncio.sleep(0.05)
                assert app._opened and app._target is None
                await pilot.pause()

        asyncio.run(scenario())
        asyncio.run(scenario_empty())


def main():
    slow = ("test_integration", "test_preparation", "test_project",
            "test_analysis_cache", "test_checks_and_cli",
            "test_analysis_engines", "test_free_energy",
            "test_equilibrate", "test_pull_and_wham", "test_remd",
            "test_submit_script")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and k not in slow]
    if "--integration" in sys.argv:
        tests += [test_integration, test_preparation, test_project,
                  test_analysis_cache, test_checks_and_cli,
                  test_analysis_engines, test_free_energy,
                  test_equilibrate, test_pull_and_wham, test_remd,
                  test_submit_script]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
