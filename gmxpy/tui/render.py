"""Turning gmxpy results into widgets the TUI can show.

Two rendering paths:

- a real image -- matplotlib draws the figure, ``Plot.png_bytes()`` renders
  it, and ``textual_image`` displays it through the Sixel or Kitty graphics
  protocol (unicode half-cells on terminals without either);
- sparklines -- textual's built-in widget, when textual-image is missing
  or the figure cannot be drawn.

matplotlib is not thread-safe and the TUI plots from worker threads, so
figure creation is serialised with a lock.
"""

from __future__ import annotations

import threading

from rich.markup import escape

from ..data import Series
from ..plotting import Plot, plot as _plot
from ..project import SeriesGroup

_MPL_LOCK = threading.Lock()

_MAX_FALLBACK_PANELS = 8


def plot_png(result, dpi=110):
    """Any analysis result -> png bytes, or None when it cannot be plotted."""
    with _MPL_LOCK:
        plot = _as_plot(result)
        if plot is None:
            return None
        try:
            return plot.png_bytes(dpi=dpi)
        finally:
            plot.close()


def _as_plot(result):
    if isinstance(result, Plot):
        return result
    if isinstance(result, Series):
        return result.plot()
    if isinstance(result, SeriesGroup):
        return result.plot()
    if (isinstance(result, (list, tuple)) and result
            and all(isinstance(item, Series) for item in result)):
        return _plot(list(result))
    if hasattr(result, "plot"):
        candidate = result.plot()
        if isinstance(candidate, Plot):
            return candidate
    return None


def image_widget(png_bytes):
    """A png as a terminal image (Sixel / Kitty / unicode half-cells)."""
    import io
    from PIL import Image as _PILImage
    from textual_image.widget import Image
    return Image(_PILImage.open(io.BytesIO(png_bytes)))


def fallback_widget(result):
    """Sparkline panels with numbers, when terminal images are unavailable."""
    from textual.containers import Vertical
    from textual.widgets import Sparkline, Static

    panels: list = []
    note = ""
    if isinstance(result, Series):
        panels = [(result.name or "series", result)]
    elif isinstance(result, SeriesGroup):
        try:
            panels = [(f"{result.name} (mean of {len(result)})",
                       result.mean_series())]
        except Exception:
            panels = []
    elif isinstance(result, (list, tuple)):
        panels = [(item.name or f"series {i}", item)
                  for i, item in enumerate(result) if isinstance(item, Series)]
        if len(panels) > _MAX_FALLBACK_PANELS:
            note = f"(+{len(panels) - _MAX_FALLBACK_PANELS} more)"
            panels = panels[:_MAX_FALLBACK_PANELS]
    if not panels:
        return Vertical(Static(escape(repr(result)), markup=True))

    widgets = []
    for name, series in panels:
        widgets.append(Static(f"[b]{escape(name)}[/]", markup=True))
        widgets.append(Sparkline(list(series.y)) if len(series) >= 2
                       else Static("no data", markup=True))
        widgets.append(Static(_one_line_stats(series), markup=True))
    if note:
        widgets.append(Static(escape(note), markup=True))
    return Vertical(*widgets)


def stats_markup(result):
    """One rich-markup line (or a few) of numbers, shown above the plot."""
    if isinstance(result, Series):
        return _one_line_stats(result)
    if isinstance(result, SeriesGroup):
        try:
            items = list(result.means.items())[:4]
            means = ", ".join(f"{escape(k)}={v:.4g}" for k, v in items)
        except Exception:
            means = "?"
        more = ", ... more runs" if len(result) > 4 else ""
        return f"[b]{escape(result.name)}[/] over {len(result)} runs: {means}{more}"
    if isinstance(result, (list, tuple)):
        lines = [_one_line_stats(item) for item in result[:6]
                 if isinstance(item, Series)]
        if len(result) > 6:
            lines.append(f"... (+{len(result) - 6} more)")
        return "\n".join(lines)
    return escape(repr(result))


def _one_line_stats(series):
    return (f"[b]{escape(series.name or 'series')}[/] = "
            f"{series.mean:.6g} +/- {series.std:.4g}  "
            f"({escape(series.ylabel)}), n={len(series)}")
