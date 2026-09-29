"""Command line: post-processing without writing a script.

    python -m gmxpy check   [dir]           what went wrong, if anything
    python -m gmxpy report  [dir]           one HTML file with everything
    python -m gmxpy info    [dir]           what is in this directory
    python -m gmxpy energy  [dir] TERM      print or plot an energy term
    python -m gmxpy tui     [dir]           browse all of it in the terminal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _open(path, name=None):
    """A Simulation or a Project, whichever this directory looks like."""
    from .project import Project
    from .simulation import Simulation

    path = Path(path)
    if name:
        return Simulation.open(name, workdir=path)
    here = sorted(path.glob("*.tpr"))
    if len(here) == 1:
        return Simulation.open(here[0].stem, workdir=path)
    if len(here) > 1:
        raise SystemExit("several runs here: " + ", ".join(t.stem for t in here)
                         + "\nchoose one with -n NAME")
    return Project(path)


def _each(target):
    from .project import Project
    return list(target) if isinstance(target, Project) else [target]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gmxpy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action",
                        choices=["check", "report", "info", "energy", "tui"])
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("term", nargs="?", help="energy term, for 'energy'")
    parser.add_argument("-n", "--name", help="run name (deffnm) to pick one run")
    parser.add_argument("-o", "--output", help="output file")
    parser.add_argument("--plot", action="store_true", help="write a png")
    args = parser.parse_args(argv)

    if args.action == "tui":
        from .tui import run_tui
        try:
            run_tui(args.path, name=args.name)
        except ImportError as exc:
            print(f"the TUI needs extra dependencies ({exc}):\n"
                  '  pip install "gmxpy[tui]"')
            return 2
        return 0

    target = _open(args.path, args.name)

    if args.action == "info":
        print(target)
        for sim in _each(target):
            traj = sim.trajectory
            print(f"  {sim.name:12} {'finished' if sim.result.finished else 'incomplete':11}"
                  f" {sim.result.steps or '?':>9} steps"
                  f"  {traj.n_frames if traj.exists else 0:>6} frames")
        first = _each(target)[0]
        try:
            print("  energy terms:", ", ".join(first.energy.terms))
        except Exception as exc:
            print("  energy terms: unavailable:", exc)
        return 0

    if args.action == "check":
        failed = False
        for sim in _each(target):
            report = sim.check()
            print(f"=== {sim.workdir.name}/{sim.name}  {report!r}")
            print(report)
            failed = failed or not report.ok
        return 1 if failed else 0

    if args.action == "report":
        print(target.report(args.output))
        return 0

    if args.action == "energy":
        if not args.term:
            print("energy terms:", ", ".join(_each(target)[0].energy.terms))
            return 0
        for sim in _each(target):
            series = sim.energy[args.term]
            print(f"{sim.workdir.name}/{sim.name}: {series.name} = "
                  f"{series.mean:.6g} +/- {series.std:.4g} "
                  f"{sim.energy.unit(args.term)}  (n={len(series)})")
            if args.plot:
                out = args.output or f"{sim.name}_{args.term.replace(' ', '_')}.png"
                print("  ->", series.plot().save(out))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
