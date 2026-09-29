# GROMACS Python Front-End：项目设计与实施计划

> **实施状态(2026-09-29,v0.2.0)。** 本文档是设计期的计划,API 以
> [README](README.md) 为准。当前进度:P0、P1、P2 全部实现;P3 中
> enhanced sampling 已实现(`Pull`/`Umbrella` + `wham`、`TemperatureREMD`
> + `demux`,见 §31 优先级),GPCR selection、ABFE/RBFE、ML/MM、QM/MM
> 未做。计划之外新增:Textual TUI(`gmxpy tui`)、内存态分析后端
> (`fast.py`,mdtraj + numpy)、`sim.check()` 运行质量检查(`quality.py`)、
> 一键平衡链(`protocol.py`)。本机与集群均按 §19 走版本探测,
> `module load gromacs/2026.3` 即可。

## 1. 项目定位

目标不是重写 GROMACS，也不是再做一个 Snakemake / Nextflow 风格的 workflow engine。

本项目定位为：

> **一个 OPI-style 的现代 GROMACS Python Front-End。**

核心思路：

- GROMACS 继续负责其最擅长的事情：
  - `pdb2gmx`
  - `grompp`
  - `mdrun`
  - checkpoint / restart
  - topology preprocessing
  - 核心 MD 计算
- Python 负责：
  - 输入对象化
  - 命令调用
  - 状态管理
  - trajectory / energy 数据读取
  - selection
  - analysis
  - plotting
  - report
  - 高层 API

最终希望用户看到的是：

```python
from gmxpy import Simulation, units as u

sim = Simulation("prod")

lig = sim.select.ligand
site = sim.select.protein.within(5 * u.angstrom, lig)

sim.energy["Potential"].plot()
sim.analysis.rmsd(site).plot()
sim.analysis.rdf(lig, site).plot()
```

而不是：

```bash
gmx energy
# 输入编号

gmx rms
# 再输入编号

gmx select
# 写 selection language

xmgrace output.xvg
```

---

# 2. 核心问题

GROMACS 本身不是难用在 MD kernel，而是难用在外围交互。

本项目优先解决四类问题。

## 2.1 Interactive numbered menu

典型：

```text
Select a group:
0 System
1 Protein
2 Protein-H
3 C-alpha
...
```

以及：

```text
Select the terms you want:
10 Bond
11 Angle
...
17 Potential
```

目标：

> Python API 中不再让用户处理编号式交互。

例如：

```python
sim.energy["Potential"]
```

而不是：

```bash
echo 17 | gmx energy
```

---

## 2.2 GROMACS selection language

现有语法虽然功能强，但复杂查询很容易变成难以维护的字符串：

```text
protein and within 0.5 of resname LIG
```

本项目使用 Python object / AST selection：

```python
protein = sim.select.protein
ligand = sim.select.resname("LIG")

site = protein.within(5 * u.angstrom, ligand)
```

进一步支持：

```python
sel = (
    sim.select.protein
    & sim.select.name("CA")
    & ~sim.select.resname("GLY")
)
```

selection 不绑定 GROMACS backend。

内部表示：

```text
Selection AST
     │
     ├── evaluate with MDAnalysis
     ├── evaluate with MDTraj
     ├── compile to GROMACS selection
     └── export index.ndx
```

---

## 2.3 XVG / xmgrace

原则：

> **能不产生 XVG，就不产生 XVG。**

传统：

```bash
gmx rms -s md.tpr -f md.xtc -o rmsd.xvg
xmgrace rmsd.xvg
```

改为：

```python
sim.analysis.rmsd("protein").plot()
```

底层：

```text
TPR/XTC
  ↓
MDAnalysis / MDTraj
  ↓
NumPy
  ↓
pandas
  ↓
matplotlib / plotly
```

只有确实需要调用 GROMACS 工具时，才允许内部处理 XVG。

用户默认不接触 XVG。

---

## 2.4 输出文件碎片化

GROMACS 一个 simulation 通常产生：

```text
.tpr
.xtc
.trr
.edr
.log
.cpt
.gro
.xvg
.ndx
```

本项目将它们统一成：

```python
sim.result
sim.energy
sim.trajectory
sim.topology
sim.checkpoint
sim.log
```

例如：

```python
sim.result.finished
sim.result.steps
sim.result.time
sim.result.performance

sim.energy["Potential"]
sim.trajectory.load()
```

---

# 3. 总体架构

```text
                  User Python API
                         │
                         ▼
              ┌─────────────────────┐
              │   Simulation API    │
              │   Project API       │
              └──────────┬──────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
     Input Model     Selection AST     Result Model
          │              │               │
          │              ▼               │
          │        Query Compiler         │
          │       /      |       \        │
          │    MDA     MDTraj     GMX     │
          │                               │
          └──────────────┬────────────────┘
                         │
                         ▼
                  Backend Layer
                /       |        \
             CLI      gmxapi     Python
              │
              ▼
           GROMACS
```

---

# 4. 项目模块

建议目录：

```text
gmxpy/
├── __init__.py
│
├── core/
│   ├── simulation.py
│   ├── project.py
│   ├── run.py
│   ├── result.py
│   └── errors.py
│
├── input/
│   ├── mdp.py
│   ├── structure.py
│   ├── topology.py
│   └── index.py
│
├── command/
│   ├── base.py
│   ├── pdb2gmx.py
│   ├── editconf.py
│   ├── solvate.py
│   ├── genion.py
│   ├── grompp.py
│   ├── mdrun.py
│   └── convert_tpr.py
│
├── backend/
│   ├── cli.py
│   ├── gmxapi.py
│   └── environment.py
│
├── selection/
│   ├── ast.py
│   ├── query.py
│   ├── compiler_gmx.py
│   ├── compiler_mda.py
│   └── presets.py
│
├── data/
│   ├── energy.py
│   ├── trajectory.py
│   ├── topology.py
│   ├── log.py
│   └── checkpoint.py
│
├── analysis/
│   ├── rmsd.py
│   ├── rmsf.py
│   ├── rdf.py
│   ├── distance.py
│   ├── hbond.py
│   ├── gyration.py
│   ├── msd.py
│   └── convergence.py
│
├── plotting/
│   ├── base.py
│   ├── timeseries.py
│   └── free_energy.py
│
├── units/
│   └── __init__.py
│
└── report/
    ├── simulation.py
    └── html.py
```

---

# 5. API 设计

## 5.1 MDP

第一版可以基于 `pydantic` 或 dataclass。

```python
from gmxpy import MDP

mdp = MDP(
    integrator="md",
    dt=0.002,
    nsteps=5_000_000,
    tcoupl="v-rescale",
    ref_t=300,
    constraints="h-bonds",
)

mdp.write("prod.mdp")
```

需要支持：

```python
mdp.dt
mdp.nsteps

mdp.update(
    nsteps=10_000_000,
)

mdp.validate()
```

目标：

- 类型检查
- 合法值检查
- GROMACS version-aware validation
- 单位检查
- 自动写 `.mdp`

---

# 6. Simulation 对象

核心对象：

```python
sim = Simulation(
    structure="eq.gro",
    topology="topol.top",
    mdp=mdp,
    name="prod",
)
```

执行：

```python
sim.prepare()
sim.run()
```

或：

```python
sim.run(
    ntomp=16,
    nb="gpu",
    pme="gpu",
)
```

结果：

```python
sim.result.finished
sim.result.steps
sim.result.simulation_time
sim.result.performance
sim.result.return_code
```

输出：

```python
sim.result.structure
sim.result.trajectory
sim.result.energy
sim.result.checkpoint
sim.result.log
```

---

# 7. Command Layer

第一版只需要可靠 wrapper。

统一基类：

```python
class GromacsCommand:
    executable: str
    arguments: dict

    def build_command(self):
        ...

    def run(self):
        ...
```

例如：

```python
grompp = Grompp(
    mdp="prod.mdp",
    structure="eq.gro",
    topology="topol.top",
    output="prod.tpr",
)

grompp.run()
```

内部：

```bash
gmx grompp \
    -f prod.mdp \
    -c eq.gro \
    -p topol.top \
    -o prod.tpr
```

### 第一版建议支持

```text
pdb2gmx
editconf
solvate
genion
grompp
mdrun
convert-tpr
make_ndx
```

分析类命令不优先。

---

# 8. Topology 策略

## 不做

第一版不要尝试完整解析 GROMACS topology preprocessor。

不要重新实现：

```text
#include
#define
#ifdef
#ifndef
```

不要尝试把所有 `.top/.itp` 映射成完整 Python force-field object。

## 做

```python
top = Topology("topol.top")
```

提供：

```python
top.path
top.includes
top.defines
top.molecule_names
```

如果需要完整 topology 语义：

```python
compiled = top.compile(
    structure="system.gro",
    mdp=mdp,
)
```

底层让 `grompp` 做 preprocessing。

---

# 9. Trajectory

不自己实现 XTC/TRR 二进制 parser。

推荐：

```text
MDTraj
MDAnalysis
```

统一接口：

```python
traj = sim.trajectory.load()
```

可选择 backend：

```python
traj = sim.trajectory.load(engine="mdtraj")
traj = sim.trajectory.load(engine="mdanalysis")
```

方便高层 API：

```python
sim.trajectory.n_frames
sim.trajectory.time
sim.trajectory.dt
```

---

# 10. Energy

不要求用户运行：

```bash
gmx energy
```

优先直接读 EDR。

内部候选：

```text
panedr
MDAnalysis EDR reader
```

接口：

```python
energy = sim.energy
```

查看 terms：

```python
energy.terms
```

访问：

```python
potential = energy["Potential"]
temperature = energy["Temperature"]
pressure = energy["Pressure"]
```

返回优先：

```python
pandas.Series
```

整体：

```python
df = energy.to_dataframe()
```

---

# 11. Selection 系统

这是核心模块之一。

## 11.1 基本 selector

```python
s = sim.select

s.protein
s.backbone
s.water
s.solvent
s.ions
s.ligand
s.heavy_atoms

s.name("CA")
s.resname("LIG")
s.resid(10)
s.resid(10, 30)
s.chain("A")
```

---

## 11.2 布尔组合

```python
sel = s.protein & s.name("CA")
```

```python
sel = s.chain("A") | s.chain("B")
```

```python
sel = s.protein & ~s.resname("GLY")
```

映射：

```text
&   AND
|   OR
~   NOT
```

---

## 11.3 空间 selection

```python
lig = s.resname("LIG")

site = s.protein.within(
    5 * u.angstrom,
    lig,
)
```

默认应该支持：

```python
site.atoms
site.residues
site.by_residue()
```

---

## 11.4 高层语义

```python
site = s.binding_site(
    ligand="LIG",
    cutoff=5 * u.angstrom,
)
```

后续扩展：

```python
s.membrane
s.lipid("POPC")
s.protein_heavy
s.sidechain
s.aromatic
```

GPCR plugin 可进一步：

```python
s.gpcr("3.32")
s.gpcr("6.48")
```

但这不是 MVP。

---

# 12. Selection AST

内部：

```python
And(
    Protein(),
    Within(
        distance=5 * angstrom,
        target=Resname("LIG"),
    ),
)
```

Selection AST 可以：

```python
selection.evaluate()
selection.to_indices()
selection.to_gromacs()
selection.to_mdanalysis()
selection.write_ndx()
```

这使得上层 API 不绑定 GROMACS syntax。

---

# 13. Units

必须避免 nm / Å 混乱。

推荐：

```python
from gmxpy import units as u

5 * u.angstrom
0.5 * u.nanometer
300 * u.kelvin
2 * u.femtosecond
```

内部统一 canonical units。

Selection：

```python
protein.within(
    5 * u.angstrom,
    ligand,
)
```

而不是：

```python
protein.within(0.5, ligand)
```

---

# 14. Analysis

原则：

> 对于已有成熟 Python 实现的分析，不优先调用 `gmx xxx`。

MVP：

```python
sim.analysis.rmsd(...)
sim.analysis.rmsf(...)
sim.analysis.radius_of_gyration(...)
sim.analysis.distance(...)
sim.analysis.rdf(...)
sim.analysis.hbonds(...)
sim.analysis.msd(...)
```

---

## RMSD

```python
result = sim.analysis.rmsd(
    sim.select.protein,
)
```

结果：

```python
result.time
result.values
result.mean
result.plot()
result.to_dataframe()
```

---

## Distance

```python
d = sim.analysis.distance(
    sim.select.residue(100).atom("CA"),
    sim.select.residue(200).atom("CA"),
)

d.plot()
```

---

## RDF

```python
rdf = sim.analysis.rdf(
    sim.select.resname("LIG"),
    sim.select.water,
)

rdf.plot()
```

---

# 15. Plotting

统一结果对象：

```python
result.plot()
```

能量：

```python
sim.energy["Potential"].plot()
```

高级：

```python
sim.energy["Potential"].plot(
    start=100,
    rolling=50,
)
```

输出：

```python
plot = sim.energy["Potential"].plot()

plot.save("potential.png")
```

可选：

```python
result.to_csv("data.csv")
```

---

# 16. Report

目标：

```python
sim.report()
```

自动生成：

```text
Simulation Summary

Status:
✓ Finished normally

Simulation time:
500 ns

Performance:
42.1 ns/day

Temperature
[plot]

Pressure
[plot]

Potential Energy
[plot]

Protein RMSD
[plot]

Radius of Gyration
[plot]
```

第一版：

```text
HTML
```

后续可做：

```text
Markdown
Jupyter
PDF
```

---

# 17. Error Model

不能只把 `subprocess.CalledProcessError` 扔给用户。

建议：

```python
class GromacsError(Exception):
    pass

class GromppError(GromacsError):
    pass

class MdrunError(GromacsError):
    pass

class LincsError(MdrunError):
    pass

class CheckpointError(MdrunError):
    pass

class TopologyError(GromacsError):
    pass
```

解析：

```text
Fatal error
LINCS WARNING
Particle coordinate is nan
Cannot find molecule type
No such moleculetype
GPU initialization failed
```

用户看到：

```python
try:
    sim.run()
except LincsError as e:
    print(e.step)
    print(e.atoms)
```

---

# 18. Restart

这是值得第一版就做好的功能。

```python
sim.run()
```

如果已有：

```text
prod.cpt
```

可以：

```python
sim.resume()
```

甚至：

```python
sim.run(resume=True)
```

内部：

```bash
gmx mdrun -deffnm prod -cpi prod.cpt
```

但必须检查：

- TPR 是否一致
- checkpoint 是否有效
- 输出文件是否冲突
- append / noappend

---

# 19. GROMACS Version Detection

```python
gmx = Environment.detect()

gmx.version
gmx.executable
gmx.mpi
gmx.cuda
```

例如：

```python
print(gmx.version)
# 2026.3
```

用于：

- MDP validation
- command 参数兼容
- output parser
- feature detection

不要只通过版本号硬编码，优先 capability detection。

---

# 20. Backend

## CLI backend

默认。

```text
Python
 ↓
subprocess
 ↓
gmx
```

原因：

- 稳
- 容易 debug
- HPC 友好
- 与现有 GROMACS 使用方式兼容

---

## gmxapi backend

作为可选 backend：

```python
sim.run(backend="gmxapi")
```

不是项目核心依赖。

适合：

- ensemble
- future integration
- in-process execution

---

# 21. Dependencies

MVP 建议：

```text
Python >= 3.11
pydantic
numpy
pandas
matplotlib
MDAnalysis
mdtraj
panedr
```

可选：

```text
plotly
rich
jinja2
gmxapi
pint
```

单位系统可以考虑：

```text
pint
```

或者自己做轻量 Quantity API。

---

# 22. 不做什么

第一版明确不做：

## 不重写 GROMACS

不实现：

```text
MD engine
PME
constraints
neighbor list
integrator
```

---

## 不自己解析所有 binary format

不重新实现：

```text
XTC
TRR
EDR
TPR
CPT
```

优先依赖：

```text
MDAnalysis
MDTraj
panedr
GROMACS
```

---

## 不完整重写 topology parser

尤其不碰：

```text
GROMACS preprocessor
```

---

## 不做完整 workflow manager

暂不复制：

```text
Snakemake
Nextflow
Airflow
```

可以后续提供：

```python
Project
Protocol
Stage
```

但不是 MVP 的核心。

---

# 23. MVP

第一阶段目标：

> 用户可以完全不用编号菜单、selection string、XVG/xmgrace，完成一套常见 GROMACS simulation + analysis。

必须完成：

### Input

- `MDP`
- `Topology`
- `Structure`

### Simulation

- `grompp`
- `mdrun`
- restart
- result object

### Data

- EDR
- XTC/TRR
- TPR topology

### Selection

- protein
- water
- ligand
- resname
- resid
- atom name
- chain
- boolean
- within

### Analysis

- RMSD
- RMSF
- distance
- Rg
- RDF

### Plotting

- energy
- timeseries
- analysis result

---

# 24. MVP 用户体验

完整 demo：

```python
from gmxpy import Simulation, MDP
from gmxpy import units as u

mdp = MDP(
    integrator="md",
    dt=0.002,
    nsteps=50_000_000,
    ref_t=300,
)

sim = Simulation(
    structure="eq.gro",
    topology="topol.top",
    mdp=mdp,
    name="prod",
)

sim.prepare()
sim.run(
    ntomp=16,
    nb="gpu",
    pme="gpu",
)

lig = sim.select.ligand

site = sim.select.protein.within(
    5 * u.angstrom,
    lig,
).by_residue()

sim.energy["Potential"].plot()
sim.energy["Temperature"].plot()

sim.analysis.rmsd(
    sim.select.protein,
).plot()

sim.analysis.rdf(
    lig,
    site,
).plot()

sim.report()
```

如果这段能工作，MVP 就成立。

---

# 25. 第二阶段：Preparation API

加入：

```python
system = System.from_pdb("protein.pdb")

system.pdb2gmx(
    forcefield="amber14sb",
    water="tip3p",
)

system.box(
    shape="dodecahedron",
    padding=1.0 * u.nanometer,
)

system.solvate()

system.add_ions(
    concentration=0.15,
)
```

最终：

```python
system.prepare()
```

---

# 26. 第三阶段：Free Energy

这一步才开始和 ABFE / RBFE pipeline 深度结合。

```python
project = FreeEnergyProject(...)
```

例如：

```python
project.states
project.lambda_states
project.dhdl
```

分析：

```python
project.plot_dhdl()
project.plot_overlap()
project.plot_convergence()
```

后续支持：

```text
ABFE
RBFE
generic perturbation
HREMD
expanded ensemble
```

但 Free Energy 必须建立在前面的通用 Simulation/Data/Selection API 上。

---

# 27. Project 层

后续：

```python
project = Project("h1_ligand")

project.runs
project.states
project.results
```

例如：

```python
project.runs["lambda_00"]
project.runs["lambda_01"]
```

可以：

```python
project.energy.plot(
    "Potential",
    groupby="state",
)
```

---

# 28. HPC

第一版不直接做 PBS/Slurm scheduler。

只保证：

```python
sim.command()
```

能生成：

```bash
gmx mdrun ...
```

后续可以：

```python
sim.submit(
    scheduler="pbs",
    resources={
        "ncpus": 32,
        "ngpus": 1,
    },
)
```

但 scheduler 应该独立成 executor plugin。

---

# 29. Testing

分四层。

## Unit tests

测试：

```text
MDP serialization
Selection AST
unit conversion
command construction
log parser
```

不需要真的跑 MD。

---

## Integration tests

小体系：

```text
water box
alanine dipeptide
small ligand
```

跑：

```text
grompp
100–1000 step mdrun
EDR read
XTC read
analysis
```

---

## Regression tests

固定数据集：

```text
expected energy terms
expected RMSD
expected selection atom indices
```

---

## Version matrix

优先：

```text
GROMACS 2024
GROMACS 2025
GROMACS 2026
```

不追求一开始覆盖所有古老版本。

---

# 30. 开发顺序

## Phase 0 — Skeleton

建立：

```text
package
CI
typing
logging
exceptions
environment detection
```

---

## Phase 1 — Simulation Core

完成：

```text
MDP
grompp
mdrun
result
restart
```

目标：

```python
sim.prepare()
sim.run()
```

---

## Phase 2 — Data Layer

完成：

```text
EDR
XTC/TRR
TPR
log
```

目标：

```python
sim.energy["Potential"]
sim.trajectory.load()
```

---

## Phase 3 — Selection

完成：

```text
Selection AST
basic selectors
boolean queries
within
units
NDX export
```

目标：

```python
site = sim.select.protein.within(
    5 * u.angstrom,
    sim.select.ligand,
)
```

---

## Phase 4 — Analysis

完成：

```text
RMSD
RMSF
distance
Rg
RDF
```

目标：

```python
sim.analysis.rmsd(site).plot()
```

---

## Phase 5 — Plot / Report

完成：

```text
energy plotting
analysis plotting
HTML report
```

目标：

```python
sim.report()
```

---

## Phase 6 — Preparation

完成：

```text
pdb2gmx
editconf
solvate
genion
```

---

## Phase 7 — Free Energy

接入：

```text
ABFE
RBFE
generic perturbation
```

---

# 31. 第一版优先级

## P0

必须有：

```text
Simulation
MDP
grompp
mdrun
restart
EDR
trajectory
Selection AST
RMSD
energy plotting
```

---

## P1

很重要：

```text
RMSF
RDF
distance
Rg
HTML report
pdb2gmx
solvate
genion
```

---

## P2

之后：

```text
H-bond
MSD
free-energy plotting
Project
multi-state
HPC executor
```

---

## P3

插件化：

```text
GPCR selection
ABFE/RBFE
ML/MM
QM/MM
enhanced sampling
```

---

# 32. 核心设计原则

## 32.1 不暴露 GROMACS 心智负担

用户不应该需要知道：

```text
group number
energy term number
XVG
xmgrace
index.ndx
selection string
```

除非主动要求低级接口。

---

## 32.2 Python first

首选：

```python
object.method()
```

而不是：

```python
subprocess.run(...)
```

---

## 32.3 GROMACS remains authoritative

物理计算仍由 GROMACS 完成。

Python 不复制：

```text
integrator
force calculation
topology preprocessing
```

---

## 32.4 Use existing ecosystem

优先整合：

```text
MDAnalysis
MDTraj
panedr
pandas
NumPy
matplotlib
```

---

## 32.5 High-level first, low-level available

默认：

```python
sim.analysis.rmsd("protein")
```

低级用户仍可以：

```python
sim.command.gmx(...)
```

---

# 33. 项目价值

现有 Python/GROMACS 工具大多集中在：

```text
CLI wrapper
workflow automation
analysis library
```

这个项目的区别是：

> **把 GROMACS 当作一个完整 Python-accessible simulation object。**

不是：

```python
gromacs.grompp(...)
gromacs.mdrun(...)
```

而是：

```python
sim.run()

sim.select...
sim.energy...
sim.trajectory...
sim.analysis...
sim.report()
```

真正希望消灭的是：

```text
interactive numbered menus
selection language strings
index group juggling
XVG
xmgrace
manual output bookkeeping
```

---

# 34. 最终目标

理想状态：

```python
from gmxpy import Simulation, units as u

sim = Simulation.open("prod")

lig = sim.select.ligand

site = sim.select.binding_site(
    ligand=lig,
    cutoff=5 * u.angstrom,
)

sim.energy["Potential"].plot()

sim.analysis.rmsd(
    sim.select.protein,
).plot()

sim.analysis.distance(
    lig.center_of_mass,
    site.center_of_mass,
).plot()

sim.report()
```

用户基本不需要知道：

```text
gmx energy
gmx select
gmx rms
gmx distance
gmx make_ndx
XVG
xmgrace
group numbers
```

但底层仍然保留完整 GROMACS 能力。

---

# 35. 一句话定义

> **A modern, object-oriented Python front-end for GROMACS that replaces interactive menus, selection strings, XVG/xmgrace workflows, and fragmented output handling with a unified simulation, selection, analysis, and visualization API.**

