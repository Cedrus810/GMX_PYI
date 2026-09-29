"""Terminal user interface (textual): ``gmxpy tui [dir]``.

An optional extra -- ``pip install "gmxpy[tui]"``.  Nothing else in the
package imports this, so the core stays standard-library only.  Plots show
as terminal images (Sixel / Kitty graphics protocol) where the terminal
supports them, sparklines everywhere else.
"""

__all__ = ["run_tui"]


def run_tui(path=".", name=None, env=None):
    """Launch the TUI over a directory of runs."""
    import os
    os.environ.setdefault("MPLBACKEND", "agg")     # figures are made off-screen
    try:
        # asks the terminal what it can draw; must happen before the app
        # starts, because textual's own threads swallow the answer
        import textual_image.renderable  # noqa: F401
    except ImportError:
        pass
    from .app import GmxpyTUI
    GmxpyTUI(path, name=name, env=env).run()
