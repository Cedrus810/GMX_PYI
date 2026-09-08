"""Minimal unit system.

Canonical units are GROMACS units: nm, ps, K, bar, kJ/mol, degree.
Anything the rest of gmxpy receives is normalised through :func:`value_in`,
so a user can pass ``5 * u.angstrom`` or a bare float (assumed canonical).
"""

from __future__ import annotations

import math

# unit name -> (dimension, factor to canonical)
_UNITS = {
    "nanometer": ("length", 1.0),
    "nm": ("length", 1.0),
    "angstrom": ("length", 0.1),
    "picometer": ("length", 1e-3),
    "pm": ("length", 1e-3),
    "picosecond": ("time", 1.0),
    "ps": ("time", 1.0),
    "femtosecond": ("time", 1e-3),
    "fs": ("time", 1e-3),
    "nanosecond": ("time", 1e3),
    "ns": ("time", 1e3),
    "microsecond": ("time", 1e6),
    "us": ("time", 1e6),
    "kelvin": ("temperature", 1.0),
    "K": ("temperature", 1.0),
    "bar": ("pressure", 1.0),
    "atmosphere": ("pressure", 1.01325),
    "atm": ("pressure", 1.01325),
    "kJ_per_mol": ("energy", 1.0),
    "kJ_mol": ("energy", 1.0),
    "kcal_per_mol": ("energy", 4.184),
    "kcal_mol": ("energy", 4.184),
    "degree": ("angle", 1.0),
    "radian": ("angle", 180.0 / math.pi),
    "dimensionless": ("dimensionless", 1.0),
}

_CANONICAL = {
    "length": "nm",
    "time": "ps",
    "temperature": "K",
    "pressure": "bar",
    "energy": "kJ_per_mol",
    "angle": "degree",
    "dimensionless": "",
}


class UnitError(ValueError):
    pass


class Quantity:
    """A number with a unit.  Immutable, comparable, printable."""

    __slots__ = ("magnitude", "unit")

    def __init__(self, magnitude, unit="dimensionless"):
        if unit not in _UNITS:
            raise UnitError(f"unknown unit {unit!r}")
        self.magnitude = float(magnitude)
        self.unit = unit

    # -- introspection -------------------------------------------------
    @property
    def dimension(self):
        return _UNITS[self.unit][0]

    @property
    def canonical(self):
        """Magnitude expressed in GROMACS canonical units."""
        return self.magnitude * _UNITS[self.unit][1]

    def to(self, unit):
        if unit not in _UNITS:
            raise UnitError(f"unknown unit {unit!r}")
        dim, factor = _UNITS[unit]
        if dim != self.dimension:
            raise UnitError(f"cannot convert {self.dimension} to {dim}")
        return Quantity(self.canonical / factor, unit)

    def value_in(self, unit):
        return self.to(unit).magnitude

    # -- arithmetic ----------------------------------------------------
    def _same(self, other):
        if not isinstance(other, Quantity):
            raise UnitError(f"cannot combine Quantity with {type(other).__name__}")
        if other.dimension != self.dimension:
            raise UnitError(f"cannot combine {self.dimension} with {other.dimension}")
        return other.to(self.unit).magnitude

    def __mul__(self, other):
        if isinstance(other, Quantity):
            raise UnitError("quantity * quantity is not supported")
        return Quantity(self.magnitude * other, self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return self.canonical / other.canonical  # ratio, dimensionless
        return Quantity(self.magnitude / other, self.unit)

    def __add__(self, other):
        return Quantity(self.magnitude + self._same(other), self.unit)

    def __sub__(self, other):
        return Quantity(self.magnitude - self._same(other), self.unit)

    def __neg__(self):
        return Quantity(-self.magnitude, self.unit)

    def __abs__(self):
        return Quantity(abs(self.magnitude), self.unit)

    def __float__(self):
        return self.canonical

    def __eq__(self, other):
        return isinstance(other, Quantity) and self.canonical == other.canonical
    def __lt__(self, other):
        return self.magnitude < self._same(other)
    def __le__(self, other):
        return self.magnitude <= self._same(other)
    def __gt__(self, other):
        return self.magnitude > self._same(other)
    def __ge__(self, other):
        return self.magnitude >= self._same(other)
    def __hash__(self):
        return hash((self.dimension, self.canonical))

    def __repr__(self):
        return f"{self.magnitude:g} {self.unit}"

    def __format__(self, spec):
        return format(self.magnitude, spec) + f" {self.unit}"


def value_in(value, unit):
    """Normalise ``value`` to a float in ``unit``.

    Bare numbers are assumed to already be in ``unit`` -- this keeps the
    low-level command layer usable without importing the unit system.
    """
    if isinstance(value, Quantity):
        return value.value_in(unit)
    if value is None:
        return None
    return float(value)


def canonical(value):
    """Float in canonical GROMACS units (nm/ps/K/bar/kJ per mol)."""
    return float(value) if isinstance(value, Quantity) else float(value)


# module-level unit constants: ``5 * u.angstrom``
for _name in _UNITS:
    globals()[_name] = Quantity(1.0, _name)

__all__ = ["Quantity", "UnitError", "value_in", "canonical"] + list(_UNITS)
