"""The gmxpy TUI: browse runs, checks, energies and analyses in the terminal.

    gmxpy tui runs/

Left: the runs found under the directory (a run is a directory containing
``<name>.tpr``).  Right: Summary, Check, Energy, Analysis and Report tabs.
Plots appear as terminal images where the terminal speaks Sixel or the
Kitty graphics protocol, sparklines otherwise.

Every tab shows what the library already exposes -- the same calls the CLI
makes -- but the library is synchronous and blocking, so each of them runs
in a worker thread and hands its result back to the UI thread through
``call_from_thread``.  DOM access happens on the UI thread only.
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header,
                             Input, Label, LoadingIndicator, OptionList,
                             RichLog, Select, Static, TabbedContent, TabPane)

from ..errors import GmxpyError
from ..quality import FAIL, OK, SKIP, WARN
from . import render

METHODS = [
    ("rmsd", "rmsd"),
    ("rmsf", "rmsf"),
    ("radius_of_gyration", "radius of gyration"),
    ("sasa", "sasa"),
    ("msd", "msd"),
    ("rdf", "rdf  (A against reference B)"),
    ("distance", "distance  (A to B)"),
    ("mindist", "mindist  (A to B)"),
    ("hbonds", "hbonds  (donors A, acceptors B)"),
    ("density", "density profile"),
    ("secondary_structure", "secondary structure"),
    ("pca", "pca"),
]

SELECTIONS = ["protein", "water", "ligand", "backbone", "c_alpha", "sidechain",
              "ions", "non_water", "heavy_atoms", "protein_heavy", "all"]

_STATUS = {OK: ("green", "ok  "), WARN: ("yellow", "warn"),
           FAIL: ("red", "FAIL"), SKIP: ("dim", "--  ")}

_CUSTOM = "__custom__"
# textual 8 blanked Select with Select.NULL; older versions used BLANK
_BLANK = getattr(Select, "NULL", Select.BLANK)


class SelectionPicker(Vertical):
    """One selection: a preset dropdown, or a free-text expression."""

    DEFAULT_CSS = """
    SelectionPicker { height: auto; }
    SelectionPicker Label { color: $text-muted; }
    SelectionPicker Input { display: none; }
    SelectionPicker.custom Input { display: block; }
    """

    def __init__(self, title, **kwargs):
        self._title = title
        super().__init__(**kwargs)

    def compose(self):
        yield Label(self._title)
        yield Select([(s, s) for s in SELECTIONS] + [("custom...", _CUSTOM)],
                     prompt="preset...", allow_blank=True)
        yield Input(placeholder="selection expression, e.g. resname LIG")

    @property
    def value(self):
        """The expression to use: custom text when shown, else the preset."""
        if self.has_class("custom"):
            return self.query_one(Input).value.strip() or None
        chosen = self.query_one(Select).value
        return chosen if isinstance(chosen, str) and chosen else None

    def clear(self):
        self.remove_class("custom")
        self.query_one(Select).value = _BLANK
        self.query_one(Input).value = ""

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value == _CUSTOM:
            self.add_class("custom")
            self.query_one(Input).focus()
        else:
            self.remove_class("custom")
        event.stop()


class GmxpyTUI(App):
    TITLE = "gmxpy"

    CSS = """
    #body { height: 1fr; }
    #sidebar { width: 34; padding: 0 1; border-right: solid $panel; }
    #sidebar-title { text-style: bold; color: $accent; }
    #runs { height: auto; max-height: 60%; background: transparent;
            border: none; }
    #env-line { color: $text-muted; margin-top: 1; }
    #log { height: 10; border-top: solid $panel; }
    #controls { height: auto; }
    .row > Select, .row > SelectionPicker { width: 1fr; }
    .row > Checkbox { width: auto; margin-left: 1; }
    .row > Button { margin-right: 1; }
    .check-line { padding: 0 1; }
    .section-title { text-style: bold; margin-top: 1; }
    #energy-term, #analysis-method { width: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, path=".", name=None, env=None):
        super().__init__()
        self._path = Path(path)
        self._name = name
        self._env = env
        self._target = None          # Simulation or Project
        self._names = []
        self._sim_name = None
        self.sim = None              # the run the tabs currently look at
        self._opened = False         # bootstrap settled (used by tests)

    # -- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("runs", id="sidebar-title")
                yield OptionList(id="runs")
                yield Static("", id="env-line")
            with TabbedContent():
                with TabPane("Summary", id="tab-summary"):
                    yield VerticalScroll(id="summary-body")
                with TabPane("Check", id="tab-check"):
                    yield VerticalScroll(id="check-body")
                with TabPane("Energy", id="tab-energy"):
                    with Vertical():
                        yield Select([], prompt="energy term...",
                                     allow_blank=True, id="energy-term")
                        yield Static("", id="energy-stats", markup=True)
                        yield VerticalScroll(id="energy-body")
                with TabPane("Analysis", id="tab-analysis"):
                    with Vertical():
                        with Horizontal(classes="row"):
                            yield Select([(label, key) for key, label in METHODS],
                                         prompt="method...", allow_blank=True,
                                         id="analysis-method")
                            yield Checkbox("mean +/- sd across all runs",
                                           id="across-runs")
                        with Horizontal(classes="row"):
                            yield SelectionPicker("selection A", id="sel-a")
                            yield SelectionPicker("selection B (reference)",
                                                  id="sel-b")
                        yield Static("", id="analysis-stats", markup=True)
                        yield VerticalScroll(id="analysis-body")
                with TabPane("Report", id="tab-report"):
                    with Vertical():
                        yield Label("one self-contained HTML file for this target")
                        yield Input(placeholder="output path (blank = default)",
                                    id="report-path")
                        with Horizontal(classes="row"):
                            yield Button("Generate", id="report-generate",
                                         variant="primary")
                            yield Button("Open in browser", id="report-open")
        yield RichLog(id="log", markup=True, highlight=True, wrap=True,
                      max_lines=2000)
        yield Footer()

    def on_mount(self):
        self.sub_title = str(self._path)
        self._bootstrap()

    # -- opening ---------------------------------------------------------
    @work(thread=True, group="bootstrap", exclusive=True)
    def _bootstrap(self):
        from ..project import Project
        from ..simulation import Simulation
        try:
            if self._name:
                target = Simulation.open(self._name, workdir=self._path,
                                         env=self._env)
            else:
                target = Project(self._path, env=self._env)
        except Exception as exc:
            self.call_from_thread(self._show_open_error, exc)
            return
        self.call_from_thread(self._use_target, target)

    def _show_open_error(self, exc):
        self._target = None
        self.sim = None
        self._opened = True
        self._names = []
        self.query_one("#runs", OptionList).clear_options()
        self._log(f"[red]cannot open {escape(str(self._path))}: [/]"
                  f"{escape(str(exc))}")
        self.notify(str(exc), title="cannot open", severity="error", timeout=10)

    def _use_target(self, target):
        from ..project import Project
        self._target = target
        self._opened = True
        self._names = target.names if isinstance(target, Project) \
            else [target.name]
        runs = self.query_one("#runs", OptionList)
        runs.clear_options()
        runs.add_options(self._names)
        env = getattr(target, "env", None)
        self.query_one("#env-line", Static).update(
            escape(str(env)) if env else "")
        self._log(f"[b]{len(self._names)}[/] run(s) under "
                  f"{escape(str(self._path))}")
        self._load_summary()
        if self._names:
            self._select_run(self._names[0])

    # -- run switching ----------------------------------------------------
    @on(OptionList.OptionSelected, "#runs")
    def _run_picked(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        if 0 <= index < len(self._names):
            self._select_run(self._names[index])

    def _select_run(self, name):
        from ..project import Project
        self._sim_name = name
        self.sim = self._target[name] if isinstance(self._target, Project) \
            else self._target
        self._start("#check-body")
        self._start("#energy-body")
        self.query_one("#energy-stats", Static).update("")
        self._start("#analysis-body")
        self.query_one("#analysis-stats", Static).update("")
        self._load_check()
        self._load_terms()

    def action_refresh(self):
        if self._target is None:
            return
        self._load_summary()
        if self.sim is not None:
            self._load_check()
            self._load_terms()

    # -- summary tab -------------------------------------------------------
    @work(thread=True, group="summary", exclusive=True)
    def _load_summary(self):
        from ..project import Project
        target = self._target
        if target is None:
            return
        try:
            if isinstance(target, Project):
                rows = target.summary()
            else:
                rows = [dict(target.result.summary(), run=target.name)]
        except Exception as exc:
            self.call_from_thread(self._log_error, "summary", exc)
            return
        self.call_from_thread(self._show_summary, rows)

    def _show_summary(self, rows):
        body = self.query_one("#summary-body", VerticalScroll)
        body.remove_children()
        if not rows:
            body.mount(Static("nothing here", markup=True))
            return
        keys: dict = {}
        for row in rows:
            keys.update({k: None for k in row})
        ordered = sorted(keys, key=lambda k: k != "run")
        table = DataTable(id="summary-table", cursor_type="row",
                          zebra_stripes=True)
        body.mount(table)
        for key in ordered:
            table.add_column(key, key=str(key))
        for row in rows:
            table.add_row(*[_cell(row.get(k)) for k in ordered])

    # -- check tab ---------------------------------------------------------
    @work(thread=True, group="check", exclusive=True)
    def _load_check(self):
        sim, name = self.sim, self._sim_name
        if sim is None:
            return
        try:
            report = sim.check()
        except Exception as exc:
            self.call_from_thread(self._log_error, "check", exc)
            return
        self.call_from_thread(self._show_check, name, report)

    def _show_check(self, name, report):
        body = self.query_one("#check-body", VerticalScroll)
        body.remove_children()
        overall = "green" if report.ok else "yellow"
        body.mount(Static(f"[b]{escape(name)}[/]  [{overall}]"
                          f"{escape(repr(report))}[/{overall}]",
                          classes="section-title"))
        for check in report.checks:
            style, mark = _STATUS.get(check.status, ("white", "??"))
            body.mount(Static(
                f"[{style}] [{mark}] [/{style}]"
                f"{escape(check.name)}: {escape(check.message)}",
                classes="check-line"))

    # -- energy tab --------------------------------------------------------
    @work(thread=True, group="energy-terms", exclusive=True)
    def _load_terms(self):
        sim = self.sim
        if sim is None:
            return
        try:
            terms = sim.energy.terms
        except Exception as exc:
            self.call_from_thread(self._terms_failed, exc)
            return
        self.call_from_thread(self._show_terms, terms)

    def _terms_failed(self, exc):
        self.query_one("#energy-term", Select).set_options([])
        self.query_one("#energy-stats", Static).update("")
        self._log_error("energy", exc)

    def _show_terms(self, terms):
        select = self.query_one("#energy-term", Select)
        select.set_options([(t, t) for t in terms])
        if isinstance(select.value, str) and select.value:
            self._load_energy_series(select.value)

    @on(Select.Changed, "#energy-term")
    def _term_picked(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and event.value:
            self._load_energy_series(event.value)

    @work(thread=True, group="energy", exclusive=True)
    def _load_energy_series(self, term):
        sim = self.sim
        if sim is None:
            return
        try:
            series = sim.energy[term]
            png = render.plot_png(series) if self._images() else None
            stats = render.stats_markup(series)
        except Exception as exc:
            self.call_from_thread(self._log_error, f"energy {term}", exc)
            return
        self.call_from_thread(self._show_plot, "#energy-body", series, png,
                              stats, "#energy-stats")

    # -- analysis tab ------------------------------------------------------
    @on(Select.Changed, "#analysis-method")
    def _method_picked(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and event.value:
            self._dispatch_analysis()

    @on(Checkbox.Changed, "#across-runs")
    def _across_changed(self, event: Checkbox.Changed) -> None:
        self._dispatch_analysis()

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        if isinstance(event.input.parent, SelectionPicker):
            self._dispatch_analysis()
        elif event.input.id == "report-path":
            self._make_report(event.input.value.strip(), open_browser=False)

    def _dispatch_analysis(self):
        method = self.query_one("#analysis-method", Select).value
        if not isinstance(method, str) or not method:
            return
        if self.sim is None:
            self.notify("open a run first", severity="warning")
            return
        across = self.query_one("#across-runs", Checkbox).value
        sel_a = self.query_one("#sel-a", SelectionPicker).value
        sel_b = self.query_one("#sel-b", SelectionPicker).value
        self._start("#analysis-body")
        self.query_one("#analysis-stats", Static).update("")
        self._run_analysis(str(method), sel_a, sel_b, across)

    @work(thread=True, group="analysis", exclusive=True)
    def _run_analysis(self, method, sel_a, sel_b, across):
        # a Project fans the analysis out over every run; a Selection is
        # re-bound per run by the library, so one expression covers all
        target = self._target if across else self.sim
        if target is None:
            return
        try:
            result = self._compute(target, method, sel_a, sel_b)
            png = render.plot_png(result) if self._images() else None
            stats = render.stats_markup(result)
        except Exception as exc:
            self.call_from_thread(self._log_error, f"analysis {method}", exc)
            return
        self.call_from_thread(self._show_plot, "#analysis-body", result, png,
                              stats, "#analysis-stats")

    @staticmethod
    def _compute(target, method, sel_a, sel_b):
        analysis = target.analysis
        a = target.select.expr(sel_a) if sel_a else None
        b = target.select.expr(sel_b) if sel_b else None
        if method == "rdf":
            return analysis.rdf(b, a)
        if method == "distance":
            return analysis.distance(a, b)
        if method == "mindist":
            return analysis.mindist(a, b)
        if method == "hbonds":
            return analysis.hbonds(a, b)
        return getattr(analysis, method)(a)

    # -- report tab --------------------------------------------------------
    @on(Button.Pressed, "#report-generate")
    def _report_generate(self):
        self._make_report(self.query_one("#report-path", Input).value.strip(),
                          open_browser=False)

    @on(Button.Pressed, "#report-open")
    def _report_generate_open(self):
        self._make_report(self.query_one("#report-path", Input).value.strip(),
                          open_browser=True)

    @work(thread=True, group="report", exclusive=True)
    def _make_report(self, path, open_browser=False):
        target = self._target
        if target is None:
            return
        try:
            written = target.report(path or None)
        except Exception as exc:
            self.call_from_thread(self._log_error, "report", exc)
            return
        self.call_from_thread(self._report_done, written, open_browser)

    def _report_done(self, written, open_browser):
        self._log(f"[green]report written:[/] {escape(str(written))}")
        self.notify(f"report: {written}")
        if open_browser:
            import webbrowser
            webbrowser.open(Path(written).resolve().as_uri())

    # -- shared bits -------------------------------------------------------
    @staticmethod
    def _images():
        """Terminal images, unless textual-image is not importable."""
        try:
            import textual_image.widget  # noqa: F401
            return True
        except ImportError:
            return False

    def _start(self, body_sel):
        """Clear a plot area and show a spinner while a worker runs."""
        body = self.query_one(body_sel, VerticalScroll)
        body.remove_children()
        body.mount(LoadingIndicator())

    def _show_plot(self, body_sel, result, png, stats, stats_sel):
        body = self.query_one(body_sel, VerticalScroll)
        body.remove_children()
        widget = None
        if png is not None:
            try:
                widget = render.image_widget(png)
            except ImportError:
                widget = None
        if widget is None:
            widget = render.fallback_widget(result)
        body.mount(widget)
        if stats_sel:
            self.query_one(stats_sel, Static).update(stats or "")

    def _log(self, markup):
        self.query_one("#log", RichLog).write(markup)

    def _log_error(self, context, exc):
        self._log(f"[red]{escape(context)} failed:[/] {escape(str(exc))}")
        self.notify(f"{context}: {exc}", title="error", severity="error",
                    timeout=10)


def _cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
