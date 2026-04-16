# -*- coding: utf-8 -*-
#
# "TheVirtualBrain - Widgets" package
#
# (c) 2022-2025, TVB Widgets Team
#
# GSoC 2026 sketch: xircuits component stubs for Connectivity3DWidget
# and AnimatedSurface3DWidget. These are DRAFT components — not yet
# wired into NotebookFactory dispatch. See ARCHITECTURE_QUESTION.md.
#

import numpy as np

from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface

from xai_components.base import InArg, OutArg, xai_component
from xai_components.base_tvb import ComponentWithWidget


@xai_component(color='rgb(85, 37, 130)')
class Connectivity3DViewer(ComponentWithWidget):
    """
    Xircuits component that wraps Connectivity3DWidget.

    When this node is double-clicked in the xircuits canvas, NotebookFactory
    should route to Connectivity3DNotebookGenerator, which generates a notebook
    cell that instantiates Connectivity3DWidget and calls add_datatype().

    Input ports
    -----------
    connectivity : InArg[Connectivity]
        A TVB Connectivity object, typically wired from ConnectivityFromFile.

    colormap : InArg[str]
        Optional. Colormap name passed to the widget (default: 'viridis').
        Lets researchers pre-select a colormap from the workflow level.

    ARCHITECTURE QUESTION FOR LIA:
    --------------------------------
    PhasePlaneNotebookGenerator receives component_inputs as a flat dict of
    primitive values (floats, strings) that map directly to model constructor
    kwargs. But for Connectivity3DViewer, the key input is a Connectivity
    *object* — a TVB datatype that has already been constructed by an upstream
    ConnectivityFromFile node and stored in xircuits context (ctx).

    The PhasePlane pattern re-constructs the model from scratch inside the
    generated notebook using the exported primitive params. For Connectivity3D
    that doesn't apply — we can't serialize a full Connectivity object into
    notebook code the same way.

    Two options I can see:
      Option A: The generated notebook loads connectivity from the same file
                path that ConnectivityFromFile used (we thread file_path through
                component_inputs). This works for the default case but breaks
                if connectivity was built programmatically upstream.
      Option B: The widget node only appears at the END of a workflow (after
                simulation), and we serialize the connectivity to a temp .zip
                and reload it in the notebook.

    Which pattern does the team prefer? Is there precedent in the existing
    components for passing non-primitive types through to notebook generators?
    """

    connectivity: InArg[Connectivity]
    colormap: InArg[str]

    def __init__(self):
        self.done = False
        self.connectivity = InArg(None)
        self.colormap = InArg('viridis')

    def execute(self, ctx) -> None:
        # At workflow runtime, this component just passes through —
        # the actual widget is launched via the generated notebook,
        # not inline during execute().
        connectivity = self.connectivity.value
        if connectivity is None:
            raise ValueError("Connectivity3DViewer: no connectivity object received. "
                             "Wire a ConnectivityFromFile node into this component.")
        # Store in context so downstream components can access if needed.
        ctx['connectivity_3d_viewer_data'] = connectivity


@xai_component(color='rgb(0, 116, 200)')
class AnimatedSurface3DViewer(ComponentWithWidget):
    """
    Xircuits component that wraps AnimatedSurface3DWidget.

    Intended to sit after a simulation node in the workflow. Takes the
    cortical surface and the simulation timeseries output, and generates
    a notebook that plays the spatiotemporal animation.

    Input ports
    -----------
    surface : InArg[Surface]
        A TVB Surface object (e.g. from WhiteMatterSurface node).

    timeseries_path : InArg[str]
        Path to a .npy or .npz file containing the (T, N_vertices) float32
        timeseries array produced by the simulation.

    ARCHITECTURE QUESTION FOR LIA (same core issue as Connectivity3DViewer):
    -------------------------------------------------------------------------
    AnimatedSurface3DWidget takes a live numpy array at construction time.
    The simulation timeseries is computed at workflow runtime, not available
    at notebook-generation time.

    The current resolution I'm assuming: we follow the TimeSeriesVbiNotebookGenerator
    pattern — we save the timeseries to disk during execute(), and the generated
    notebook loads it from that path. This is consistent with how VBI handles it.

    But this means we need to agree on:
      1. Where the timeseries gets saved (same NOTEBOOKS_DIR structure, or output/?
      2. Whether the file path is passed as a component_input or derived from
         xircuits_filename (like VBI does it).

    Happy to sketch both versions if that helps the discussion.
    """

    surface: InArg[Surface]
    timeseries_path: InArg[str]

    def __init__(self):
        self.done = False
        self.surface = InArg(None)
        self.timeseries_path = InArg(None)

    def execute(self, ctx) -> None:
        surface = self.surface.value
        timeseries_path = self.timeseries_path.value

        if surface is None:
            raise ValueError("AnimatedSurface3DViewer: no surface object received.")
        if timeseries_path is None:
            raise ValueError("AnimatedSurface3DViewer: no timeseries_path received. "
                             "This should point to the .npy file saved by the simulation node.")

        ctx['animated_surface_3d_viewer_data'] = {
            'surface': surface,
            'timeseries_path': timeseries_path,
        }
