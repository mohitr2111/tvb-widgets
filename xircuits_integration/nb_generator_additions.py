# -*- coding: utf-8 -*-
#
# "TheVirtualBrain - Widgets" package
#
# (c) 2022-2025, TVB Widgets Team
#
# GSoC 2026 sketch: NotebookGenerator subclasses for the two new 3D widgets.
# This file shows where these classes slot into nb_generator.py and what
# the dispatch change in NotebookFactory looks like.
#
# STATUS: Draft / proof-of-concept. Not yet merged into nb_generator.py.
# Open questions are marked with # QUESTION: comments.
#

import os
import nbformat

# --- Paste these two classes into nb_generator.py after PhasePlaneNotebookGenerator ---


class Connectivity3DNotebookGenerator(NotebookGenerator):
    """
    Generates a JupyterLab notebook cell that instantiates Connectivity3DWidget
    and calls add_datatype() with the connectivity loaded from a file path.

    This follows Option A from the architecture discussion: the generated notebook
    re-loads connectivity from a file path that was threaded through component_inputs.

    QUESTION FOR LIA:
    Is file_path the right thing to pass through component_inputs here?
    The PhasePlane pattern passes model primitive params (floats/strings).
    Connectivity is a structured object — loading from file is the simplest
    workaround, but it breaks if connectivity was built programmatically upstream.
    Is there a precedent in existing components for this kind of handoff?
    """

    def get_notebook(self):
        title = "# Interactive 3D Connectivity Viewer"
        self.add_markdown_cell(title)

        intro = (
            "#### Run the cell below to display the Connectivity3DWidget.\n"
            "#### You can threshold edges, switch colormaps, and toggle hemispheres interactively.\n"
            "*All controls mutate k3d traitlets in-place — no plot rebuild on interaction.*"
        )
        self.add_markdown_cell(intro)

        code = self._connectivity_3d_cell()
        self.add_code_cell(code)
        return self.notebook

    def _connectivity_3d_cell(self):
        file_path = self.component_inputs.get('file_path', 'connectivity_76.zip')
        colormap = self.component_inputs.get('colormap', 'viridis')

        # QUESTION: Should colormap be wired as a workflow-level input port,
        # or should it always be left to the researcher to set inside the widget UI?
        # PhasePlane disables the model dropdown (widget-internal concern stays internal).
        # Not sure whether colormap preselection is a workflow-level or widget-level concern.

        code = (
            "from tvb.datatypes.connectivity import Connectivity\n"
            "from tvbwidgets.api import Connectivity3DWidget\n"  # QUESTION: confirm import path once merged
            "from IPython.core.display_functions import display\n"
            "\n"
            "# Load connectivity from the path that was used in the workflow\n"
            "connectivity = Connectivity.from_file('{file_path}')\n"
            "connectivity.configure()\n"
            "\n"
            "w = Connectivity3DWidget()\n"
            "w.add_datatype(connectivity)\n"
            "display(w)\n"
        ).format(file_path=file_path, colormap=colormap)

        return code


class AnimatedSurface3DNotebookGenerator(NotebookGenerator):
    """
    Generates a JupyterLab notebook cell that instantiates AnimatedSurface3DWidget
    and loads a pre-saved timeseries numpy array for animation.

    Follows the TimeSeriesVbiNotebookGenerator pattern: the simulation node saves
    timeseries to disk, and the generated notebook loads it from that path.

    QUESTION FOR LIA:
    TimeSeriesVbiNotebookGenerator derives the output path from xircuits_filename
    (via get_base_dir_web() + "output_" + xircuits_filename). Should we follow
    exactly that convention, or should the simulation node that produces the
    timeseries export the path explicitly as an OutArg[str] that gets wired
    into AnimatedSurface3DViewer as timeseries_path?

    The explicit wiring feels cleaner and more xircuits-idiomatic, but it
    requires the simulation node to be updated to expose a timeseries_path
    OutArg. Is that a change the team would accept, or should we stay with
    the path-derivation convention?
    """

    def get_notebook(self):
        title = "# Animated Cortical Surface 3D Viewer"
        self.add_markdown_cell(title)

        intro = (
            "#### Run the cell below to animate the simulation timeseries on the cortical surface.\n"
            "#### Use Play/Pause to control animation, scrub the slider, and adjust speed and colormap.\n"
            "*Frames are rendered by mutating mesh.attribute in-place — no mesh rebuild per frame.*"
        )
        self.add_markdown_cell(intro)

        code = self._animated_surface_3d_cell()
        self.add_code_cell(code)
        return self.notebook

    def _animated_surface_3d_cell(self):
        timeseries_path = self.component_inputs.get('timeseries_path', '')
        surface_file = self.component_inputs.get('surface_file', 'cortex_16384.zip')

        code = (
            "import numpy as np\n"
            "from tvb.datatypes.surfaces import WhiteMatterSurface\n"
            "from tvbwidgets.api import AnimatedSurface3DWidget\n"  # QUESTION: confirm import path
            "from IPython.core.display_functions import display\n"
            "\n"
            "# Load surface\n"
            "surface = WhiteMatterSurface.from_file(source_file='{surface_file}')\n"
            "surface.configure()\n"
            "\n"
            "# Load timeseries — shape (T, N_vertices), dtype float32\n"
            "# If the simulation saved float64, the widget will cast it with a warning.\n"
            "timeseries = np.load('{timeseries_path}')  # update path if needed\n"
            "\n"
            "w = AnimatedSurface3DWidget()\n"
            "w.add_datatype(surface)\n"
            "w.add_datatype(timeseries)\n"
            "display(w)\n"
        ).format(surface_file=surface_file, timeseries_path=timeseries_path)

        return code

    def edit_cell(self):
        # Allow researcher to edit the path if they used a remote run
        return True


# --- Change to make in NotebookFactory.get_notebook_for_component() ---
#
# CURRENT (lines 101-107 in nb_generator.py):
#
#   if component_class.__name__.startswith('StoreResults'):
#       return TimeSeriesNotebookGenerator(...)
#   elif component_class.__name__.startswith('SamplePosterior'):
#       return SamplePosteriorVbiNotebookGenerator(...)
#   elif component_class.__name__.startswith('SimulationRunner'):
#       return TimeSeriesVbiNotebookGenerator(...)
#   return PhasePlaneNotebookGenerator(...)   # <-- catch-all
#
# PROPOSED ADDITION (two new elif branches before the catch-all):
#
#   elif component_class.__name__.startswith('Connectivity3DViewer'):
#       return Connectivity3DNotebookGenerator(component_class, component_id, component_inputs).get_notebook()
#   elif component_class.__name__.startswith('AnimatedSurface3DViewer'):
#       return AnimatedSurface3DNotebookGenerator(component_class, component_id, component_inputs).get_notebook()
#   return PhasePlaneNotebookGenerator(...)   # catch-all still last
#
# DESIGN NOTE: this dispatch-by-class-name pattern is already established
# in the codebase. Adding two more branches is consistent. However, a
# cleaner long-term design might be a class attribute on the component
# itself (e.g. notebook_generator_class = Connectivity3DNotebookGenerator)
# so the factory doesn't need to know component names at all.
# Worth discussing with Lia whether to refactor the dispatch or stay consistent.
