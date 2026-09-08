"""One-call simulation report.

``sim.report()`` -> a self-contained HTML file: status, performance, the
energy terms that matter, and a couple of structural observables.  Panels
that cannot be produced (missing group, missing trajectory) are skipped
with a note rather than killing the report.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

_CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,sans-serif;
     margin:0;padding:2rem;background:#fafafa;color:#1a1a1a}
h1{font-size:1.5rem;margin:0 0 .25rem}
h2{font-size:1.05rem;margin:2rem 0 .5rem;color:#444}
.sub{color:#777;margin-bottom:1.5rem}
.cards{display:flex;flex-wrap:wrap;gap:.75rem;margin-bottom:1rem}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;
      padding:.7rem 1rem;min-width:130px}
.card .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:#888}
.card .v{font-size:1.15rem;font-weight:600;margin-top:.15rem}
.ok{color:#1a7f37}.bad{color:#b42318}
img{max-width:100%;background:#fff;border:1px solid #e3e3e3;border-radius:8px}
table{border-collapse:collapse;background:#fff;border:1px solid #e3e3e3;
      border-radius:8px;font-size:.9rem}
td,th{padding:.4rem .8rem;text-align:left;border-bottom:1px solid #eee}
.note{color:#999;font-size:.85rem;font-style:italic}
"""


def _card(key, value, cls=""):
    return (f'<div class="card"><div class="k">{html.escape(key)}</div>'
            f'<div class="v {cls}">{value}</div></div>')


def _embed(plot):
    data = base64.b64encode(plot.png_bytes()).decode()
    plot.close()
    return f'<img src="data:image/png;base64,{data}"/>'


def _panel(title, build):
    try:
        return f"<h2>{html.escape(title)}</h2>" + _embed(build())
    except Exception as exc:                     # a missing group is not fatal
        return (f"<h2>{html.escape(title)}</h2>"
                f'<p class="note">not available: {html.escape(str(exc)[:200])}</p>')


def html_report(sim, path=None, energy_terms=("Potential", "Temperature",
                                              "Pressure", "Density"),
                analyses=("rmsd", "gyration"), selection=None, open_browser=False):
    """Build the report.  Returns the written path."""
    path = Path(path or sim.workdir / f"{sim.name}_report.html")
    result = sim.result
    log = sim.log

    finished = result.finished
    time = result.simulation_time
    cards = [
        _card("Status", "finished" if finished else "incomplete",
              "ok" if finished else "bad"),
        _card("Steps", result.steps if result.steps is not None else "-"),
        _card("Simulated", f"{float(time):.4g} ps" if time is not None else "-"),
        _card("Performance", f"{result.performance:g} ns/day"
              if result.performance else "-"),
        _card("Wall time", f"{result.wall_time:g} s" if result.wall_time else "-"),
    ]

    body = [f"<h1>{html.escape(sim.name)}</h1>",
            f'<div class="sub">{html.escape(str(sim.workdir.resolve()))} &middot; '
            f'GROMACS {html.escape(sim.env.version)} &middot; '
            f'{datetime.now():%Y-%m-%d %H:%M}</div>',
            '<div class="cards">' + "".join(cards) + "</div>"]

    energy = sim.energy
    available = []
    try:
        available = [t for t in energy_terms if t in energy]
    except Exception as exc:
        body.append(f'<p class="note">no energy file: {html.escape(str(exc)[:200])}</p>')

    if available:
        rows = "".join(
            f"<tr><td>{html.escape(t)}</td><td>{energy[t].mean:.4f}</td>"
            f"<td>{energy[t].std:.4f}</td><td>{html.escape(energy.unit(t))}</td></tr>"
            for t in available)
        body.append("<h2>Energy averages</h2><table><tr><th>Term</th><th>Mean</th>"
                    f"<th>SD</th><th>Unit</th></tr>{rows}</table>")
        for term in available:
            body.append(_panel(term, lambda t=term: energy[t].plot()))

    if sim.trajectory.exists:
        target = selection if selection is not None else _default_selection(sim)
        if "rmsd" in analyses:
            body.append(_panel("RMSD", lambda: sim.analysis.rmsd(target).plot()))
        if "gyration" in analyses:
            body.append(_panel("Radius of gyration",
                               lambda: sim.analysis.radius_of_gyration(target).plot()))
        if "rmsf" in analyses:
            body.append(_panel("RMSF", lambda: sim.analysis.rmsf(
                target, per_residue=True).plot()))
    else:
        body.append('<p class="note">no trajectory -- structural panels skipped</p>')

    notes = log.notes[:5] if log.exists else []
    if notes:
        body.append("<h2>mdrun notes</h2><pre>"
                    + html.escape("\n".join(notes)) + "</pre>")

    path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(sim.name)} report"
        f"</title><style>{_CSS}</style>" + "".join(body))
    if open_browser:                       # pragma: no cover
        import webbrowser
        webbrowser.open(path.resolve().as_uri())
    return path


def _default_selection(sim):
    """Protein if there is one, otherwise the whole system."""
    try:
        protein = sim.select.protein
        if protein.n_atoms:
            return protein
    except Exception:
        pass
    return sim.select.all


def project_report(project, path=None, energy_terms=("Potential", "Temperature",
                                                     "Pressure", "Density"),
                   analyses=("rmsd", "radius_of_gyration"), selection=None,
                   band=True):
    """One HTML page for a set of runs: status table plus overlaid observables."""
    path = Path(path or project.workdir / f"{project.name}_report.html")

    rows = []
    for row in project.summary():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values())
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(k)}</th>"
                      for k in project.summary()[0]) if project.summary() else ""

    done = sum(1 for sim in project if sim.result.finished)
    cards = [_card("Runs", len(project)),
             _card("Finished", f"{done}/{len(project)}",
                   "ok" if done == len(project) else "bad")]

    body = [f"<h1>{html.escape(project.name)}</h1>",
            f'<div class="sub">{html.escape(str(project.workdir))} &middot; '
            f'GROMACS {html.escape(project.env.version)} &middot; '
            f'{datetime.now():%Y-%m-%d %H:%M}</div>',
            '<div class="cards">' + "".join(cards) + "</div>",
            f"<h2>Runs</h2><table><tr>{headers}</tr>{''.join(rows)}</table>"]

    energy = project.energy
    for term in energy_terms:
        if term in energy:
            body.append(_panel(term, lambda t=term: energy[t].plot()))

    target = selection if selection is not None else _default_selection(project.first)
    for name in analyses:
        body.append(_panel(
            name.replace("_", " "),
            lambda n=name: getattr(project.analysis, n)(target).plot(band=band)))

    path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(project.name)}"
        f" report</title><style>{_CSS}</style>" + "".join(body))
    return path
