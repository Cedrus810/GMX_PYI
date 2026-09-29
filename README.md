# gmxpy

An object-oriented Python front-end for GROMACS.  GROMACS still does the
physics; Python does the interface.  Python >= 3.10, GROMACS 2021+.

```python
from gmxpy import Simulation, MDP, units as u

sim = Simulation(structure="complex.gro", topology="complex.top",
                 mdp=MDP.preset("md", nsteps=500_000, ref_t=300 * u.kelvin),
                 name="prod", workdir="run")
sim.prepare(maxwarn=1)
sim.run(ntomp=16, nb="gpu", pme="gpu", bonded="gpu", update="gpu")

lig  = sim.select.ligand
site = sim.select.protein.within(5 * u.angstrom, lig).by_residue()

sim.energy["Potential"].plot()
sim.analysis.rmsd(sim.select.protein).plot()
sim.analysis.rdf(lig, sim.select.water).plot()
sim.report()
```

No group numbers, no selection strings, no XVG, no xmgrace.

## Zero-code post-processing

```bash
gmxpy info   runs/          # what is here: runs, steps, frames, energy terms
gmxpy check  runs/          # thermostat, drift, LINCS, box size -- per run
gmxpy report runs/          # one HTML file with everything
gmxpy energy runs/ Temperature --plot
```

`gmxpy check` on a run that started from `gen_vel` without minimising:

```
=== demo/prod  <CheckReport 4/7 ok, 3 to look at>
[ok  ] finished: mdrun finished normally
[ok  ] constraints: no LINCS warnings
[ok  ] temperature: 303.9 +/- 22.6 K (target 300 K)
[warn] density drift: 8.93% over 20 ps (mean 977.8)
[warn] energy drift: +2.91 kT/ns/atom (+2.923e+05 kJ/mol per ns)
[ok  ] periodic images: closest image 2.357 nm, twice the cut-off is 2 nm
```

## The TUI

```bash
pip install -e ".[tui]"
gmxpy tui runs/              # -n prod picks one run
```

The same library calls the CLI makes, in a terminal interface: the runs
under a directory in a sidebar, and Summary, Check, Energy, Analysis and
Report tabs.  `q` quits, `r` refreshes.  Energy terms come from the
`.edr` header (panedr when installed), Check is `sim.check()` with the
verdicts colour-coded, Analysis offers the common methods with preset or
free-text selections -- for one run, or across every run of a project
(mean +/- sd) -- and Report writes the same self-contained HTML file as
`gmxpy report`.

Plots are real images: matplotlib draws them and
[textual-image](https://pypi.org/project/textual-image/) displays them
through the Kitty graphics protocol or Sixel, falling back to unicode
half-cells on terminals with neither.  kitty, WezTerm, iTerm2, Windows
Terminal 1.22+, foot and recent Konsole/VTE terminals can show them; over
plain SSH the pictures travel with the stream.  tmux needs graphics
passthrough configured, and textual-image needs Python >= 3.12 -- without
either, the TUI still works and the plots fall back to sparklines.

## Install

```bash
git clone https://github.com/Cedrus810/GMX_PYI && cd GMX_PYI
pip install -e ".[analysis]"        # brings the `gmxpy` command
```

The core imports with the standard library alone.  `numpy`/`pandas` are needed
for dataframes, `matplotlib` for plots, `mdtraj` for the fast analysis
backend, `panedr` for direct `.edr` reading, `MDAnalysis` as a second
trajectory engine -- each is imported only where it is used.

`gmx` is taken from explicit sources only, in this order: `executable=`,
`$GMXPY_GMX`, `$GMXBIN`/`$GROMACS_DIR`, `$PATH`, `/usr/local/gromacs*`.
It never globs home directories -- on a shared machine that finds somebody
else's build.  `module load gromacs/2026.3` puts `gmx` on `$PATH`, so that
is enough; otherwise name it once:

```bash
export GMXPY_GMX=/opt/gromacs-2026.3/bin/gmx    # absolute: $HOME differs per node on NFS
```

Shared libraries the build needs but the loader cannot see are repaired
automatically -- `$GMXPY_LIBS`, the interpreter's own libdir, `$HDF5_ROOT`,
`$MKLROOT`, oneAPI, conda/mamba prefixes, `<install root>/library/*/lib`.
That covers the `libhdf5.so.310` and `libmkl_intel_lp64.so.2` cases without
loading those modules. If a library is somewhere else:

```bash
export GMXPY_LIBS=/dir1:/dir2
```

```python
>>> from gmxpy import Environment
>>> Environment.detect()
GROMACS 2026.3 (mixed, AVX2_256, GPU: CUDA) at /opt/gromacs-2026.3/bin/gmx [from $GMXBIN]
```

## Layout

| module | what it owns |
|---|---|
| `units.py` | `5 * u.angstrom`; everything internal is nm/ps/K/bar/kJ per mol |
| `errors.py` | `LincsError`, `TopologyError`, ... parsed out of GROMACS text |
| `environment.py` | executable discovery, capability detection, library repair |
| `command.py` | `GromacsCommand` + one class per subcommand; menus answered via stdin |
| `mdp.py` | `MDP` object, presets (`em`/`nvt`/`npt`/`md`), `validate()` |
| `topology.py` | `Topology` (includes, defines, `[ molecules ]`), `Structure` |
| `selection.py` | selection AST -> GROMACS / MDAnalysis / atom indices / `.ndx` |
| `data.py` | `Energy` (reads the EDR header directly), `Trajectory`, `LogFile`, `Checkpoint`, `Series` |
| `fast.py` | in-memory analysis backend (mdtraj + numpy), GROMACS' definitions |
| `quality.py` | `sim.check()`: thermostat, drift, constraints, box size |
| `__main__.py` | the `gmxpy` command |
| `tui/` | the `gmxpy tui` command: browse runs, checks and plots in the terminal |
| `analysis.py` | rmsd, rmsf, Rg, distance, rdf, msd, hbonds, sasa, mindist, periodic-image, dssp, angle/dihedral, density, cluster, pca, BAR, free-energy landscape |
| `system.py` | `System.from_pdb(...)`: pdb2gmx -> box -> solvate -> add_ions -> minimise |
| `plotting.py` | `result.plot()`, `.save()` |
| `simulation.py` | `Simulation`, `Result`, prepare / run / extend / resume / merge_parts |
| `project.py` | `Project`: many runs at once, fan-out analysis, mean +/- sd |
| `protocol.py` | `system.equilibrate()`: the em -> nvt -> npt chain in one call |
| `sampling.py` | `Pull`, `Umbrella` + `wham`, `TemperatureREMD` + `demux` |
| `hpc.py` | `sim.submit()` / `project.submit()`: PBS/Torque/Slurm job scripts |
| `report.py` | `sim.report()` -> one self-contained HTML file |

## What it covers

```python
# preparation
System.from_pdb(pdb).pdb2gmx(...).box(...).solvate().add_ions().minimise()

# running
sim.prepare(); sim.run(...); sim.extend(extend=10); sim.resume()

# selection
sim.select.protein / water / ions / ligand / backbone / c_alpha / heavy_atoms
          .name(...) .resname(...) .resid(a, b) .chain(...) .expr(raw)
          & | ~ -   .within(d, other)  .by_residue()  .center_of_mass
sim.select.binding_site(ligand="LIG", cutoff=5 * u.angstrom)

# data
sim.energy["Potential"] / .terms / .to_dataframe()
sim.trajectory.n_frames / .load(engine=...) / .process(...) / .frame(time=...)
sim.log.finished / .performance / .converged / .max_force
sim.checkpoint.step / .time

# analysis
rmsd  rmsf  radius_of_gyration  distance  rdf  msd  hbonds
sasa  mindist  periodic_image_distance  secondary_structure
angle  dihedral  density  cluster  pca  convergence
bar(dhdl_files)          # BAR free energy
landscape(x, y)          # -kT ln P(x, y)

# output
series.plot() / .save() / .to_dataframe() / .to_csv()
sim.report()

# equilibration / clusters / enhanced sampling
system.equilibrate(nvt=100 * u.ps, npt=1 * u.ns) .production(...)
sim.submit(scheduler="pbs", ncpus=8, ngpus=1); project.submit(array=True)
Simulation(..., pull=Pull(a, b, k=1000))     # umbrella / steered pulling
Umbrella(pull, values) -> wham() -> PMF      # umbrella + WHAM
TemperatureREMD(mdp, ladder) -> demux()      # T-REMD + exchange trace
```

## Equilibration in one call

```python
eq = system.equilibrate(workdir="eq", nvt=100 * u.ps, npt=1 * u.ns)
eq.ok, eq.check()                       # every stage finished, and well
prod = eq.production(MDP.preset("md", nsteps=5_000_000), name="prod")
prod.run(ntomp=8)
```

The chain hands each stage's output structure to the next and passes the
checkpoint to grompp (`-t`), so velocities survive.  nvt/npt are restrained
(`-DPOSRES` + `-r`) when the topology has a POSRES block -- pdb2gmx writes
one; an Amber-style topology just gets a note and runs free.  Finished
stages are skipped, so an interrupted chain is resumed by the same call.
`Protocol(em=False, nvt={"gen_seed": 7})` customises; durations become
nsteps via `dt`.

## Clusters

```python
script = sim.submit(scheduler="pbs", ncpus=8, ngpus=1, walltime="24:00:00",
                    modules=("gromacs/2026.3",), run_kwargs=dict(nb="gpu"))
script = project.submit(scheduler="pbs", ncpus=8, array=True)   # one array job
```

The mdrun line is the one `sim.command()` prints; `submit=True` hands the
script to `qsub`/`sbatch` and returns the job id, `submit=False` (default)
just writes it.  `pbs` (PBS Pro `select=` syntax), `torque` and `slurm`
templates are built in; a `Project` becomes one array job running a
directory per index -- the shape umbrella windows and replicas want.

## Enhanced sampling

```python
# a pull coordinate: selections in, mdp keys + index groups out
pull = Pull(sim.select.protein, sim.select.ligand, k=1000)      # kJ/mol/nm^2
sim = Simulation(structure=..., topology=..., mdp=mdp, pull=pull, name="md")
sim.prepare()                  # pull-* keys merged, groups written to an ndx
sim.run()                      # -> md_pullx.xvg / md_pullf.xvg

# umbrella windows along it, then WHAM
umb = Umbrella(pull, values=np.arange(0.4, 1.21, 0.05), k=1000,
               structure=..., topology=..., mdp=MDP.preset("md"), workdir="umb")
umb.from_trajectory(traj)      # start each window from the nearest frame
umb.prepare()
umb.project().submit(array=True)        # one PBS array job for all windows
pmf = wham(umb, temperature=300, bootstrap=100)
pmf.plot()

# temperature REMD
remd = TemperatureREMD(mdp, temperature_ladder(300, 340, 8),
                       structure=..., topology=..., workdir="remd")
remd.prepare(); remd.run(replex=100)    # one mdrun -multidir for every replica
trace = remd.demux()                    # which T each directory ran, when
trace.exchange_fractions                # per neighbour pair

sim.run(plumed="plumed.dat")            # validated: file + build support
```

Many trajectories -- replicas, restarts, windows:

```python
project = Project("runs")                    # finds every <name>.tpr underneath
project.summary()                            # did they all finish, how fast
project.energy["Temperature"].plot()         # one line per run
project.analysis.rmsd(sel).plot(band=True)   # mean +/- sd across replicas
project.analysis.sasa(sel).means             # {'rep0': 81.4, 'rep1': 81.0, ...}
project.concatenate(mode="append")           # stack replicas
sim.merge_parts()                            # -noappend chunks back into one
project.report()
```

A `Selection` handed to the project is re-bound to each run, so one selection
object works across every trajectory.

Trajectory processing replaces the trjconv group dance:

```python
sim.trajectory.process(pbc="mol", center=sim.select.protein,
                       fit="rot+trans", group=sim.select.non_water, skip=10)
```

## Speed

`gmx` re-reads the trajectory from disk for every tool it runs.  The
in-memory backend pays one `trjconv -pbc mol` pass plus one load, then does
the arithmetic on frames that are already in RAM:

| analysis | gmx | in-memory | agreement |
|---|---|---|---|
| rmsd (protein) | 0.64 s | 0.10 s | 0.13% |
| rmsf (protein) | 0.58 s | 0.11 s | 0.03% |
| Rg (protein) | 0.58 s | 0.07 s | 0.01% |
| sasa (protein) | 3.44 s | 1.38 s | 2.4% |
| rdf (ligand-water) | 15.89 s | 6.42 s | 2.1% |

(201 frames x 39 689 atoms; one-off cost 2.8 s, shared by every analysis.)

Definitions follow GROMACS, not the library defaults -- RMSD and its
superposition are mass-weighted (a plain `mdtraj.rmsd` call is 13% off),
Rg is mass-weighted about the centre of mass.  The residual 2% on `sasa`
and `rdf` is a genuinely different algorithm; `engine="gmx"` if you need
the old numbers exactly.

Two analyses deliberately stay on gmx even on `engine="auto"`:

- `secondary_structure`: mdtraj's DSSP is *slower* here (1.8 s vs 0.65 s)
  and reports three classes where gmx reports ten.
- `msd`: the two fit the diffusion constant over different windows and
  disagree by ~15%, and D is a number people quote.

## Burn-in detection

```python
>>> density = sim.energy["Density"]
>>> density.mean, density.std
(977.80, 27.89)
>>> production = density.equilibrated()      # drops the burn-in
>>> production.equilibration_time(), production.mean, production.std
(10.0, 999.72, 2.70)
```

## Design notes

**Energy.** Term names come from the `.edr` header (XDR, read directly), so
the numbered menu never appears. Values come from `panedr` when installed,
otherwise from `gmx energy` driven with the numbers we already know --
both paths agree to 1e-3 in the test suite.

**Selection.** `sim.select.protein.within(5 * u.angstrom, lig)` compiles to a
GROMACS selection string and is evaluated by `gmx select`; `.to_mdanalysis()`
emits the MDAnalysis equivalent. Legacy tools that still want an index group
get a single-group `.ndx` written from the AST, so stdin is always `0`.

**Trajectory.** mdtraj cannot read `.tpr` at all, and MDAnalysis 2.10 cannot
read tpx 138 (GROMACS 2026). `load()` therefore tries the tpr, then falls back
to a `.gro` converted from it once and cached in `.gmxpy/`.

**Restart.** `sim.extend(extend=10)` then `sim.resume()`. The checkpoint atom
count is compared against the tpr before mdrun is allowed to start.

**Caching.** Post-processing is interactive, so gmx-backed analyses write to a
scratch file named after a hash of the question (tool, selections, options,
cache format); asking again returns the cached answer, and a newer tpr or
trajectory invalidates it. `gmx rdf` on the demo trajectory: 15.6 s cold,
0.01 s warm. In-memory results are memoised on the `Analysis` object instead,
since they produce no file. Turn both off with `Analysis(sim, cache=False)`.

**GROMACS quirks absorbed, not documented:**

- `-pbc mol` and `-fit rot+trans` cannot run in one trjconv pass; `process()`
  runs two, and keeps every atom in the first so the second pass' index
  numbers still line up.
- A GPU build rejects `-ntomp` without `-ntmpi`; `run(ntomp=8)` supplies
  `-ntmpi 1`.
- grompp reports the real problem in `ERROR n [file x.mdp]:` blocks and then
  says only "There was 1 error in input file(s)" -- the exception carries the
  block, not the count.
- `anaeig -first 1 -last N` writes the projections as consecutive datasets in
  one file, not as columns; `pca()` calls it once per component.
- Energy minimisation has no performance table and its "time" is a step
  count, so `Result` reports `converged` / `potential_energy` / `max_force`
  instead of ns/day.
- `gmx msd` (2026) reports the fitted diffusion constant in the xvg *legend*,
  not on stdout, so `msd()` keeps xmgrace formatting and reads it from there.
- `trjcat` drops duplicate-time frames, which is right for continuation
  chunks and silently wrong for replicas -- `concatenate(mode="append")`
  keeps them all.
- Energy minimisation writes no checkpoint, so the equilibration chain only
  hands a `-t` file to grompp when the previous stage actually produced one.
- Handing grompp any `-n` file hides its built-in index groups, and an mdp
  saying `tc-grps = System` then fails; a pull run therefore starts from a
  `make_ndx` default-groups file (cached in `.gmxpy/`) with the pull groups
  appended.
- A pull group larger than half the box needs `pull-groupX-pbcatom`; `Pull`
  picks each group's centre-nearest atom itself and sets
  `pull-pbc-ref-prev-step-com = yes` to match.
- mdrun names pull output `<name>_pullf.xvg` / `<name>_pullx.xvg` --
  an underscore, unlike every other `-deffnm` output.
- `mdrun -multidir` refuses to run on a thread-MPI build; when a `gmx_mpi`
  sits next to the detected `gmx`, `TemperatureREMD` runs it under
  `mpirun -np <replicas>`.
- GROMACS 2024+ writes replica exchanges as `Replica exchange at step`
  blocks where `Repl ex  0 x  1` marks an accepted swap (the `x` between
  the labels), not the old `Repl t=` lines; `demux()` reads both layouts.

**Paths.** Every path handed to gmx is made absolute (`~` expanded, `..`
normalised, symlinks *not* resolved so automounted NFS paths survive).
Commands run with cwd set to the simulation's workdir, so relative inputs
would otherwise be looked up in the wrong tree:

```python
# cwd = inputs/ , outputs land in ../runs/ , both resolve correctly
sim = Simulation(structure="complex.gro", topology="complex.top",
                 mdp=mdp, name="rel", workdir="../runs")
```

For the same reason, write `$GMXPY_GMX` and any cross-node config as absolute
paths -- `$HOME` is not the same directory on every machine.

## Tests

```bash
python tests/test_gmxpy.py                 # unit, no GROMACS
python tests/test_gmxpy.py --integration   # + real grompp/mdrun/analysis
# or, under pytest:
pytest tests/test_gmxpy.py -k "not integration"        # unit only
pytest tests/test_gmxpy.py::test_remd                   # pick any test
python examples/brd4_demo.py /tmp/demo      # prepare -> run -> analyse -> report
```

The integration tests and the demo need a prepared system (`.gro` + `.top`);
point them at one with `GMXPY_TEST_SYSTEM`. The numbers quoted here come from
a BRD4 + ligand box, 39 689 atoms (protein + ligand + water + ions).

The newer suites, all run against GROMACS 2026.3 (thread-MPI + CUDA) with
`module load gromacs/2026.3`:

- `test_equilibrate` -- the full em -> nvt -> npt chain and the handover to
  production, on a topology without a POSRES block (the restrained path is
  exercised by the pdb2gmx-based tests).
- `test_pull_and_wham` -- a pull run, three umbrella windows, a real
  `gmx wham` roundtrip, and the PBS array script for the windows.
- `test_remd` -- two replicas via `mpirun gmx_mpi -multidir`, then `demux()`
  on the new-format logs.
- `test_submit_script` -- PBS/Torque script text from a live simulation.

## Not built yet

- `gmxapi` backend; the CLI backend is the only one.
- Membrane-specific analyses (`gmx order`, `densmap`), AWH, and the
  GPCR/plugin selectors.

## License

MIT -- see [LICENSE](LICENSE).
