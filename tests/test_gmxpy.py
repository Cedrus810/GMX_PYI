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
# integration -- needs a working gmx and the test system
# ---------------------------------------------------------------------

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


def main():
    slow = ("test_integration", "test_preparation", "test_project",
            "test_analysis_cache", "test_checks_and_cli",
            "test_analysis_engines", "test_free_energy")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and k not in slow]
    if "--integration" in sys.argv:
        tests += [test_integration, test_preparation, test_project,
                  test_analysis_cache, test_checks_and_cli,
                  test_analysis_engines, test_free_energy]
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
