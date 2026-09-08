"""GROMACS installation discovery and capability detection.

Prefer capability detection over version-number branching: everything the
rest of the package needs to know is read out of ``gmx --version``.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .errors import GromacsNotFound

# libraries a GROMACS build may need that are not on the default loader path
# (HDF5/H5MD builds are the usual offender on module-based clusters).
_LIB_HINT_DIRS = (
    "$HDF5_ROOT/lib", "$HDF5_DIR/lib",
    "$MKLROOT/lib", "$MKLROOT/lib/intel64",
    "$ONEAPI_ROOT/mkl/latest/lib", "$ONEAPI_ROOT/mkl/latest/lib/intel64",
    "/opt/intel/oneapi/mkl/latest/lib*", "$HOME/intel/oneapi/mkl/latest/lib*",
    "$CONDA_PREFIX/lib",
    "$MAMBA_ROOT_PREFIX/lib", "$MAMBA_ROOT_PREFIX/envs/*/lib",
    "$HOME/miniforge3/lib", "$HOME/miniforge3/envs/*/lib",
    "$HOME/mambaforge/lib", "$HOME/mambaforge/envs/*/lib",
    "$HOME/miniconda3/lib", "$HOME/miniconda3/envs/*/lib",
    "$HOME/library/*/lib", "$HOME/library/*/lib64",
)

_MISSING_LIB = re.compile(r"error while loading shared libraries: ([^:]+):")


def _expand(pattern):
    return glob.glob(os.path.expandvars(os.path.expanduser(pattern)))


@dataclass
class Environment:
    """A usable GROMACS installation."""

    executable: str
    version: str = ""
    prefix: str = ""
    precision: str = ""
    mpi: str = ""
    gpu: str = ""
    simd: str = ""
    source: str = ""
    features: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict, repr=False)

    _detected = None

    # -- discovery -----------------------------------------------------
    @classmethod
    def detect(cls, executable=None, refresh=False):
        """Find and probe a gmx binary.  Result is cached per process."""
        if cls._detected is not None and executable is None and not refresh:
            return cls._detected

        tried, why = [], []
        for cand, source in cls._candidates(executable):
            if cand in tried:
                continue
            tried.append(cand)
            env, problem = cls._working_env(cand)
            if env is None:
                why.append(f"{cand} (from {source}): {problem}")
                continue
            self = cls(executable=cand, env=env, source=source)
            self._probe()
            if not self.version:
                why.append(f"{cand} (from {source}): does not look like gmx")
                continue
            if executable is None:
                cls._detected = self
            return self

        raise GromacsNotFound(
            "no working gmx executable found.\n  "
            + "\n  ".join(why or [
                "nothing to try: gmx is not on $PATH and $GMXBIN/$GMXPY_GMX are unset"])
            + "\n\nFix with one of:"
              "\n  export GMXPY_GMX=/path/to/gmx        # pick the installation"
              "\n  export GMXPY_LIBS=/dir1:/dir2        # where the missing .so live"
              "\n  module load <your gromacs module>    # then rerun")

    @staticmethod
    def _candidates(executable):
        """Only explicit sources -- never guess by globbing home directories.

        A shared machine has other people's GROMACS builds lying around; the
        one you want is the one your environment names.
        """
        out = []
        if executable:
            out.append((str(executable), "executable="))
        if os.environ.get("GMXPY_GMX"):
            out.append((os.environ["GMXPY_GMX"], "$GMXPY_GMX"))
        for var in ("GMXBIN", "GROMACS_DIR", "GMXPREFIX"):
            base = os.environ.get(var)
            if base:
                out += [(str(Path(base) / "gmx"), f"${var}"),
                        (str(Path(base) / "bin" / "gmx"), f"${var}")]
        for name in ("gmx", "gmx_mpi", "gmx_d"):
            found = shutil.which(name)
            if found:
                out.append((found, "$PATH"))
        for path in sorted(_expand("/usr/local/gromacs*/bin/gmx"), reverse=True):
            out.append((path, "/usr/local/gromacs"))
        return [(c, why) for c, why in out
                if os.path.isfile(c) and os.access(c, os.X_OK)]

    @classmethod
    def _working_env(cls, exe):
        """Return ``(environ, None)`` in which ``exe --version`` runs, else ``(None, why)``.

        Resolves missing shared libraries by searching the install tree and
        a few conventional roots, so users need not ``module load`` first.
        """
        env = dict(os.environ)
        prefix = Path(exe).resolve().parent.parent
        extra = [str(prefix / d) for d in ("lib64", "lib") if (prefix / d).is_dir()]
        cls._prepend(env, extra)

        # also look next to the install tree itself, e.g. <root>/library/HDF5/lib
        roots = [str(prefix.parent / "library" / "*" / lib) for lib in ("lib", "lib64")]

        for _ in range(6):  # one round per missing library
            proc = subprocess.run([exe, "--version"], capture_output=True,
                                  text=True, env=env)
            if proc.returncode == 0:
                return env, None
            output = (proc.stderr or "") + (proc.stdout or "")
            m = _MISSING_LIB.search(output)
            if not m:
                return None, (output.strip().splitlines() or ["failed to run"])[-1]
            soname = m.group(1).strip()
            found = cls._find_library(soname, roots)
            if not found:
                return None, f"missing shared library {soname} (set GMXPY_LIBS to its directory)"
            cls._prepend(env, [found])
        return None, "could not resolve shared libraries"

    @staticmethod
    def _find_library(soname, extra_roots=()):
        # explicit override first, then the interpreter's own libdir (gmx
        # builds with Python support link against it), then conventions
        patterns = [p for p in os.environ.get("GMXPY_LIBS", "").split(":") if p]
        import sysconfig
        patterns += [d for d in (sysconfig.get_config_var("LIBDIR"),
                                 os.environ.get("LD_RUN_PATH")) if d]
        patterns += list(_LIB_HINT_DIRS) + list(extra_roots)
        for pattern in patterns:
            for d in _expand(pattern):
                if os.path.exists(os.path.join(d, soname)):
                    return d
        return None

    @staticmethod
    def _prepend(env, dirs):
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [d for d in dirs if d] + [p for p in current.split(":") if p]
        seen, uniq = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        env["LD_LIBRARY_PATH"] = ":".join(uniq)

    # -- capabilities --------------------------------------------------
    def _probe(self):
        out = subprocess.run([self.executable, "--version"], capture_output=True,
                             text=True, env=self.env).stdout
        get = lambda key: (re.search(rf"^{key}:\s*(.+)$", out, re.M) or [None, ""])[1].strip()
        self.version = get("GROMACS version")
        self.precision = get("Precision")
        self.mpi = get("MPI library")
        self.gpu = get("GPU support")
        self.simd = get("SIMD instructions")
        self.prefix = get("Data prefix") or str(Path(self.executable).parent.parent)
        self.features = {
            "gpu": self.gpu.lower() not in ("", "disabled"),
            "mpi": "thread_mpi" not in self.mpi.lower() and self.mpi.lower() != "none",
            "double": self.precision.startswith("double"),
            "plumed": get("Plumed support").lower().startswith("enabled"),
            "colvars": get("Colvars support").lower().startswith("enabled"),
            "cp2k": get("CP2K support").lower().startswith("enabled"),
            "tng": get("TNG support").lower().startswith("enabled"),
            "h5md": "h5md" in out.lower(),
        }

    @property
    def version_tuple(self):
        nums = re.findall(r"\d+", self.version)
        return tuple(int(n) for n in nums[:2]) or (0,)

    def has(self, feature):
        return bool(self.features.get(feature))

    def __str__(self):
        return (f"GROMACS {self.version} ({self.precision}, {self.simd}, "
                f"GPU: {self.gpu or 'none'}) at {self.executable}"
                + (f" [from {self.source}]" if self.source else ""))
