"""Plotting.

``result.plot()`` everywhere; matplotlib is imported lazily so the rest of
the package works without it.
"""

from __future__ import annotations

from pathlib import Path


def _pyplot():
    try:
        import matplotlib
    except ImportError:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from None
    import os
    if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() not in ("agg",):
        matplotlib.use("Agg")   # headless nodes are the normal case here
    import matplotlib.pyplot as plt
    return plt


class Plot:
    """Thin handle on a matplotlib figure."""

    def __init__(self, fig, ax):
        self.figure = fig
        self.ax = ax

    def save(self, path, dpi=150, **kwargs):
        self.figure.savefig(str(path), dpi=dpi, bbox_inches="tight", **kwargs)
        return Path(path)

    def show(self):
        _pyplot().show()
        return self

    def png_bytes(self, dpi=110):
        import io
        buf = io.BytesIO()
        self.figure.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()

    def close(self):
        _pyplot().close(self.figure)

    def _repr_html_(self):        # notebooks
        import base64
        return f'<img src="data:image/png;base64,{base64.b64encode(self.png_bytes()).decode()}"/>'

    def __repr__(self):
        return f"<Plot {self.ax.get_title() or self.ax.get_ylabel()!r}>"


def plot(series, ax=None, start=None, rolling=None, title=None, label=None,
         figsize=(7, 3.6), grid=True, **kwargs):
    """Plot one :class:`~gmxpy.data.Series` or a list of them."""
    plt = _pyplot()
    items = list(series) if isinstance(series, (list, tuple)) else [series]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for item in items:
        data = item
        if start is not None:
            data = data.since(start)
        if rolling:
            ax.plot(data.x, data.y, alpha=0.25, linewidth=0.8,
                    color=kwargs.get("color"))
            data = data.rolling(rolling)
        ax.plot(data.x, data.y, label=label or item.name, **kwargs)

    ax.set_xlabel(items[0].xlabel)
    ax.set_ylabel(items[0].ylabel if len(items) == 1 else "")
    if title or len(items) == 1:
        ax.set_title(title or items[0].name)
    if len(items) > 1:
        ax.legend()
    if grid:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return Plot(fig, ax)
