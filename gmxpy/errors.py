"""Typed errors.

GROMACS reports failures as free text on stderr / in the .log file.  This
module turns that text into an exception the user can actually catch,
instead of a bare ``CalledProcessError``.
"""

from __future__ import annotations

import re


class GmxpyError(Exception):
    """Base class for everything this package raises."""


class GromacsNotFound(GmxpyError):
    """No usable ``gmx`` executable."""


class GromacsError(GmxpyError):
    """A gmx command failed."""

    def __init__(self, message, command=None, returncode=None, output=""):
        super().__init__(message)
        self.message = message
        self.command = command
        self.returncode = returncode
        self.output = output

    def __str__(self):
        head = self.message.strip() or "gmx command failed"
        if self.command:
            head += "\n  command: " + " ".join(str(c) for c in self.command)
        if self.returncode is not None:
            head += f"\n  exit code: {self.returncode}"
        return head


class GromppError(GromacsError):
    pass


class MdrunError(GromacsError):
    pass


class LincsError(MdrunError):
    """Constraint failure -- usually a blown-up system."""

    def __init__(self, message, step=None, atoms=(), **kw):
        super().__init__(message, **kw)
        self.step = step
        self.atoms = list(atoms)


class CheckpointError(MdrunError):
    pass


class TopologyError(GromacsError):
    pass


class SelectionError(GmxpyError):
    pass


class AnalysisError(GmxpyError):
    pass


class MdpError(GmxpyError):
    pass


_FATAL = re.compile(
    r"-{5,}\s*\n\s*(?:Program:.*?\n\s*)?"
    r"(?:Fatal error|Error in user input|Software inconsistency error|Program error)\s*:?\s*\n"
    r"(.*?)(?:\n\s*For more information and tips|\n-{5,}|\Z)",
    re.S,
)


_ERROR_BLOCK = re.compile(
    r"^ERROR\s+\d+\s*\[([^\]]*)\]:\s*\n((?:[ \t]+\S.*\n?)+)", re.M)


def error_blocks(text):
    """grompp reports the real problem in ``ERROR n [file x.mdp]:`` blocks;
    the Fatal error line that follows only counts them."""
    out = []
    for where, body in _ERROR_BLOCK.findall(text or ""):
        message = re.sub(r"\s+", " ", body).strip()
        out.append(f"{message} [{where}]" if where else message)
    return out


def fatal_message(text):
    """The most informative failure message GROMACS printed."""
    blocks = error_blocks(text)
    if blocks:
        return "; ".join(blocks)
    m = _FATAL.search(text or "")
    if m:
        return re.sub(r"\n\s*", " ", m.group(1)).strip()
    for line in (text or "").splitlines():
        if line.startswith(("Fatal error", "Error in user input")):
            return line.strip()
    return ""


# (regex, exception class) -- first match wins, most specific first
_RULES = [
    (re.compile(r"LINCS WARNING|Constraint error|constraint deviation|"
                r"There are inconsistent shifts", re.I), LincsError),
    (re.compile(r"coordinate is nan|X particles communicated to PME|"
                r"box.*too small|Water molecule starting at atom .* can not be settled", re.I), LincsError),
    (re.compile(r"checkpoint", re.I), CheckpointError),
    (re.compile(r"moleculetype|molecule type|Atomtype .* not found|"
                r"number of coordinates in coordinate file.*does not match topology|"
                r"No default .* types", re.I), TopologyError),
]


def classify(text, command=None, returncode=None, subcommand=""):
    """Map GROMACS output onto the right exception instance."""
    msg = fatal_message(text) or (text or "").strip().splitlines()[-1:] or [""]
    msg = msg if isinstance(msg, str) else msg[0]
    kw = dict(command=command, returncode=returncode, output=text or "")

    for pattern, cls in _RULES:
        if pattern.search(text or ""):
            if cls is LincsError:
                step = re.search(r"step\s+(\d+)", text or "", re.I)
                atoms = re.findall(r"atoms?\s+(\d+)[\s,]+(\d+)", text or "")
                return LincsError(
                    msg, step=int(step.group(1)) if step else None,
                    atoms=[int(a) for pair in atoms for a in pair], **kw)
            if cls is TopologyError and subcommand not in ("grompp", "pdb2gmx", ""):
                break
            return cls(msg, **kw)

    cls = {"grompp": GromppError, "mdrun": MdrunError}.get(subcommand, GromacsError)
    return cls(msg, **kw)
