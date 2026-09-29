"""gmxpy -- an object-oriented Python front-end for GROMACS.

    from gmxpy import Simulation, MDP, units as u

    sim = Simulation(structure="eq.gro", topology="topol.top",
                     mdp=MDP.preset("md"), name="prod")
    sim.prepare()
    sim.run(ntomp=16, nb="gpu")

    site = sim.select.protein.within(5 * u.angstrom, sim.select.ligand)
    sim.energy["Potential"].plot()
    sim.analysis.rmsd(site).plot()
    sim.report()

No group numbers, no selection strings, no XVG.
"""

from . import units
from .analysis import Analysis, bar, landscape
from .command import GromacsCommand, gmx
from .data import Checkpoint, Energy, LogFile, Series, Trajectory
from .environment import Environment
from .errors import (AnalysisError, CheckpointError, GmxpyError, GromacsError,
                     GromacsNotFound, GromppError, HpcError, LincsError,
                     MdpError, MdrunError, SelectionError, TopologyError)
from .hpc import JobSpec, render, render_array
from .mdp import MDP
from .plotting import plot
from .protocol import Equilibration, Protocol
from .quality import CheckReport, check
from .sampling import (Pull, ReplTrace, TemperatureREMD, Umbrella, demux,
                       temperature_ladder, wham)
from .selection import Selection, SelectionContext, Selector
from .project import Project, SeriesGroup
from .simulation import Result, Simulation
from .system import System
from .topology import Structure, Topology

__version__ = "0.2.0"

__all__ = [
    "Simulation", "Result", "System", "Project", "SeriesGroup",
    "MDP", "Topology", "Structure",
    "Protocol", "Equilibration",
    "Pull", "Umbrella", "wham", "TemperatureREMD", "temperature_ladder",
    "demux", "ReplTrace",
    "JobSpec", "render", "render_array",
    "Energy", "Trajectory", "LogFile", "Checkpoint", "Series",
    "Selection", "Selector", "SelectionContext", "Analysis",
    "bar", "landscape",
    "Environment", "GromacsCommand", "gmx", "plot", "units",
    "check", "CheckReport",
    "GmxpyError", "GromacsError", "GromacsNotFound", "GromppError",
    "MdrunError", "LincsError", "CheckpointError", "TopologyError",
    "SelectionError", "AnalysisError", "MdpError", "HpcError",
    "__version__",
]
