"""HPC job scripts and submission.

``sim.submit()`` wraps the mdrun line ``sim.command()`` already produces into
a batch script, and (with ``submit=True``) hands it to the scheduler.  The
core is pure text generation -- a :class:`JobSpec` plus a template -- so it
is testable and usable on a machine without gmx or a scheduler.

Schedulers: ``pbs`` (PBS Pro, the ``select=`` syntax), ``torque`` (the
``nodes=`` syntax) and ``slurm``.  An array job runs one command per index,
which is what lambda windows, umbrella windows and replicas want.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .errors import HpcError

_SCHEDULERS = ("pbs", "torque", "slurm")
_SUBMIT_CMD = {"pbs": "qsub", "torque": "qsub", "slurm": "sbatch"}


@dataclass
class JobSpec:
    """One command, and what it needs from the machine."""

    command: str                       # the command line to run
    workdir: str = "."                 # cd here first (absolute is safest)
    name: str = "gmxpy"
    ncpus: int = 8
    ngpus: int = 0
    walltime: str = "24:00:00"
    queue: str = ""                    # partition, for slurm
    memory: str = ""                   # e.g. "64gb"; empty = scheduler default
    modules: tuple = ()                # "gromacs/2026.3", ...
    env: dict = field(default_factory=dict)   # extra exports
    pre: tuple = ()                    # lines run before the command
    post: tuple = ()                   # lines run after it


def _module_lines(spec):
    return [f"module load {m}" for m in spec.modules]


def _body(spec, scheduler):
    lines = [f"cd {shlex.quote(str(spec.workdir))}"]
    lines += _module_lines(spec)
    lines += [f"export {key}={shlex.quote(str(value))}"
              for key, value in spec.env.items()]
    lines += list(spec.pre)
    lines.append(spec.command)
    lines += list(spec.post)
    return lines


def _resource_line(spec, scheduler):
    """The one ``#PBS -l`` / resource line that asks for cpu/gpu/memory."""
    if scheduler == "pbs":
        chunk = [f"select=1:ncpus={spec.ncpus}:mpiprocs={spec.ncpus}"]
        if spec.ngpus:
            chunk.append(f"ngpus={spec.ngpus}")
        if spec.memory:
            chunk.append(f"mem={spec.memory}")
        return "#PBS -l " + ":".join(chunk)
    if scheduler == "torque":
        chunk = [f"nodes=1:ppn={spec.ncpus}"]
        if spec.ngpus:
            chunk.append(f"gpus={spec.ngpus}")
        line = "#PBS -l " + ":".join(chunk)
        if spec.memory:
            line += f",mem={spec.memory}"
        return line
    parts = ["--nodes=1", f"--cpus-per-task={spec.ncpus}"]
    if spec.ngpus:
        parts.append(f"--gres=gpu:{spec.ngpus}")
    if spec.memory:
        parts.append(f"--mem={spec.memory}")
    return "#SBATCH " + " ".join(parts)


def render(spec, scheduler="pbs"):
    """A batch script (text) for one :class:`JobSpec`."""
    if scheduler not in _SCHEDULERS:
        raise HpcError(f"unknown scheduler {scheduler!r} "
                       f"(one of {', '.join(_SCHEDULERS)})")
    out = str(Path(spec.workdir) / f"{spec.name}.out")
    if scheduler == "slurm":
        header = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.name}",
            _resource_line(spec, scheduler),
            f"#SBATCH --time={spec.walltime}",
            f"#SBATCH --partition={spec.queue}" if spec.queue else "",
            f"#SBATCH --output={out}-%j",
        ]
    else:
        header = [
            "#!/bin/bash",
            f"#PBS -N {spec.name}",
            _resource_line(spec, scheduler),
            f"#PBS -l walltime={spec.walltime}",
            f"#PBS -q {spec.queue}" if spec.queue else "",
            "#PBS -j oe",
            f"#PBS -o {out}",
        ]
    return "\n".join([h for h in header if h] + _body(spec, scheduler)) + "\n"


def _array_index_var(scheduler):
    # one expression that resolves the current index on every scheduler
    fallback = "${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-${SLURM_ARRAY_TASK_ID:-0}}}"
    return {"pbs": "${PBS_ARRAY_INDEX:-0}",
            "torque": "${PBS_ARRAYID:-0}",
            "slurm": "${SLURM_ARRAY_TASK_ID:-0}"}.get(scheduler, fallback)


def render_array(specs, scheduler="pbs", name="gmxpy"):
    """One array job running every spec, one per array index.

    The specs share the header of the first (they are windows of the same
    calculation -- if their resources differ, the first one's ask is what
    the job gets, so keep them equal).
    """
    if not specs:
        raise HpcError("render_array needs at least one JobSpec")
    spec = specs[0]
    if scheduler not in _SCHEDULERS:
        raise HpcError(f"unknown scheduler {scheduler!r}")
    n = len(specs) - 1
    out = str(Path(spec.workdir) / f"{name}.out")
    array_flag = {
        "pbs": f"#PBS -J 0-{n}",
        "torque": f"#PBS -t 0-{n}",
        "slurm": f"#SBATCH --array=0-{n}",
    }[scheduler]
    if scheduler == "slurm":
        header = ["#!/bin/bash", f"#SBATCH --job-name={name}",
                  _resource_line(spec, scheduler),
                  f"#SBATCH --time={spec.walltime}",
                  f"#SBATCH --partition={spec.queue}" if spec.queue else "",
                  array_flag, f"#SBATCH --output={out}-%A-%a"]
    else:
        header = ["#!/bin/bash", f"#PBS -N {name}",
                  _resource_line(spec, scheduler),
                  f"#PBS -l walltime={spec.walltime}",
                  f"#PBS -q {spec.queue}" if spec.queue else "",
                  array_flag, "#PBS -j oe", f"#PBS -o {out}"]
    header = [h for h in header if h]

    dirs = [f"  {shlex.quote(str(Path(s.workdir)))}" for s in specs]
    cmds = [f"  {shlex.quote(s.command)}" for s in specs]
    body = _body(JobSpec(command="", workdir=spec.workdir, modules=spec.modules,
                         env=spec.env, pre=spec.pre, post=spec.post), scheduler)
    body = [line for line in body if line != ""]   # drop the empty command
    body += [
        "DIRS=(",
        *dirs,
        ")",
        "CMDS=(",
        *cmds,
        ")",
        f"i={_array_index_var(scheduler)}",
        'cd "${DIRS[$i]}"',
        'eval "${CMDS[$i]}"',
    ]
    return "\n".join(header + body) + "\n"


def write(spec, path=None, scheduler="pbs", array=None, name="gmxpy"):
    """Render and write one script (or an array script).  Returns its path."""
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    if array is None:
        array = len(specs) > 1
    if array:
        if len(specs) < 2:
            raise HpcError("an array job needs at least two specs")
        text = render_array(specs, scheduler, name=name or specs[0].name)
    else:
        text = render(specs[0], scheduler)
    path = Path(path) if path else Path(specs[0].workdir) / (
        f"{name or specs[0].name}.array.sh" if array else f"{specs[0].name}.sh")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def submit_script(path, scheduler="pbs"):
    """Hand a written script to the scheduler.  Returns the job id."""
    binary = shutil.which(_SUBMIT_CMD[scheduler])
    if not binary:
        raise HpcError(
            f"{_SUBMIT_CMD[scheduler]} not found -- submit from a node that "
            "can reach the scheduler, or keep the script and submit by hand")
    proc = subprocess.run([binary, str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise HpcError(f"{_SUBMIT_CMD[scheduler]} failed: "
                       + (proc.stderr or proc.stdout).strip()[-500:])
    return (proc.stdout or "").strip()
