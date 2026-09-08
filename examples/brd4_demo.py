"""End-to-end MVP demo: prepare, run, select, analyse, report.

Usage:
    python examples/brd4_demo.py [workdir]

Uses the BRD4 + ligand system from the local ABFE benchmark set.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gmxpy import MDP, Simulation, units as u

SYSTEM = Path(os.environ.get("GMXPY_TEST_SYSTEM",
                              "~/abfe-benchmark/parameters/brd4/ligand1")).expanduser()
workdir = Path(sys.argv[1] if len(sys.argv) > 1 else "brd4_demo")
workdir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- input
mdp = MDP.preset(
    "md",
    dt=2 * u.femtosecond,
    nsteps=5000,
    ref_t=300 * u.kelvin,
    tc_grps="System",
    nstenergy=50,
    nstxout_compressed=50,
    gen_vel="yes", gen_temp=300, gen_seed=42, continuation="no",
)

sim = Simulation(
    structure=SYSTEM / "complex.gro",
    topology=SYSTEM / "complex.top",
    mdp=mdp,
    name="prod",
    workdir=workdir,
)

print(sim.env)
print("mdp warnings:", mdp.validate() or "none")

# ------------------------------------------------------------ simulate
sim.prepare(maxwarn=1)
print("command:", sim.command(ntomp=8, nb="gpu"))
result = sim.run(ntomp=8, ntmpi=1, nb="gpu", pme="gpu", bonded="gpu", update="gpu")
print(result, result.summary())

# ----------------------------------------------------------- selection
lig = sim.select.ligand
site = sim.select.protein.within(5 * u.angstrom, lig).by_residue()
print(f"ligand {lig.n_atoms} atoms | site {site.n_atoms} atoms, "
      f"{len(site.residues)} residues")
print("site residues:", [f"{r.resname}{r.resid}" for r in site.residues])

# -------------------------------------------------------------- energy
for term in ("Potential", "Temperature", "Pressure", "Density"):
    series = sim.energy[term]
    print(f"{term:12} {series.mean:12.3f} +- {series.std:8.3f} {sim.energy.unit(term)}")

# ------------------------------------------------------------ analysis
rmsd = sim.analysis.rmsd(sim.select.protein)
rg = sim.analysis.radius_of_gyration(sim.select.protein)
dist = sim.analysis.distance(lig, site)
rdf = sim.analysis.rdf(lig, sim.select.water)
rmsf = sim.analysis.rmsf(sim.select.protein & sim.select.name("CA"), per_residue=True)
hb = sim.analysis.hbonds(sim.select.protein, lig)
for series in (rmsd, rg, dist, rdf, rmsf, hb):
    print(f"{series.name:20} n={len(series):5d} mean={series.mean:10.4f}")

# ------------------------------------------------- the rest of the set
print("SASA         ", round(sim.analysis.sasa(sim.select.protein).mean, 2), "nm^2")
print("min dist     ", round(sim.analysis.mindist(sim.select.protein, lig).mean, 3), "nm")
print("periodic img ", round(sim.analysis.periodic_image_distance().mean, 3), "nm")
print("secondary    ", {s.name: round(s.mean, 1)
                        for s in sim.analysis.secondary_structure()})
pca = sim.analysis.pca(sim.select.protein & sim.select.name("CA"))
print("PCA explained", [round(v, 3) for v in pca.explained])

# trjconv without the group menu: PBC first, then fit (two passes, handled)
fitted = sim.trajectory.process(pbc="mol", center=sim.select.protein,
                                fit="rot+trans", group=sim.select.non_water,
                                skip=5)
print("processed    ", fitted)

# free-energy surface from two observables
from gmxpy import landscape
surface = landscape(rmsd, rg, bins=20)
print("landscape    ", surface)
surface.plot().save(workdir / "landscape.png")

# ------------------------------------------------------ plots + report
sim.energy["Potential"].plot(rolling=5).save(workdir / "potential.png")
rmsd.plot().save(workdir / "rmsd.png")
rdf.plot().save(workdir / "rdf.png")
print("report:", sim.report())
