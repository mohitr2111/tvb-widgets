# Xircuits Integration — Architecture Notes & Open Questions
# GSoC 2026 pre-bonding exploration
# Author: Mohit R.

## What I traced

Spent time reading the full tvb-ext-xircuits codebase, specifically
nb_generator.py, xai_components/base_tvb.py, and xai_components/xai_connectivity/.

Key findings:

### 1. PhasePlaneWidget is NOT a React literal component
The proposal mentioned "React JS literal component registration following
the PhasePlaneWidget pattern." After reading the source, PhasePlaneWidget
integration is entirely on the Python side — nb_generator.py generates
a notebook cell that instantiates the widget. The React/TypeScript layer
(src/) is the generic Xircuits canvas infrastructure, not TVB-specific.
So "wiring a widget into xircuits" means:
  (a) a Python component class subclassing ComponentWithWidget
  (b) a NotebookGenerator subclass that generates the instantiation cell
  (c) a dispatch branch in NotebookFactory.get_notebook_for_component()

### 2. Dispatch is by class name prefix (fragile but consistent)
NotebookFactory routes to generators using .startswith() on the
component class name. Adding Connectivity3DViewer and AnimatedSurface3DViewer
follows this pattern with two new elif branches.

A cleaner alternative: add a `notebook_generator_class` class attribute
to ComponentWithWidget subclasses and let the factory read that attribute
instead of pattern-matching names. This would make the factory closed to
modification when new widgets are added. Worth discussing.

### 3. The core open question: how to pass non-primitive types to generators
PhasePlaneNotebookGenerator gets component_inputs as a flat dict of
primitive values (float/str) that map directly to model constructor kwargs.
The generated notebook re-constructs the model from scratch using those values.

Connectivity3DWidget and AnimatedSurface3DWidget take structured TVB objects
(Connectivity, Surface) and numpy arrays — not primitives. Two approaches:

  Option A — File path threading:
    The generated notebook reloads the object from the same file path used
    in the workflow. Simple, works for the default tvb-data case.
    Breaks if connectivity was built programmatically upstream.

  Option B — Serialization handoff (like TimeSeriesVbi):
    The simulation/connectivity node serializes its output to disk during
    execute(), and the generated notebook loads it from a known path
    (derived from xircuits_filename, like TimeSeriesVbiNotebookGenerator does).

Option B is more robust and already has precedent in the codebase.
But it requires agreeing on:
  - Where files land (NOTEBOOKS_DIR vs output_<xircuits_filename>?)
  - Whether the path is explicit (OutArg[str] wired to the viewer node)
    or implicit (derived from xircuits_filename inside the generator)

## Files in this sketch

- xai_tvb_3d_widgets.py   — Component stubs (Connectivity3DViewer, AnimatedSurface3DViewer)
- nb_generator_additions.py — Generator subclasses + proposed dispatch change
