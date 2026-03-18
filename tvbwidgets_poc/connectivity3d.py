# -*- coding: utf-8 -*-
#
# TVB Widgets PoC — GSoC 2026
# Connectivity3DWidget: interactive 3D brain connectivity visualisation.
#
# Architecture:
#   Connectivity3DWidget(ipywidgets.VBox, TVBWidgetPOC)
#
# Phase 1: k3d 3D render — nodes (3dSpecular spheres), edges (uint32-indexed lines)
# Phase 2: interactive control panel — threshold slider, node-size slider,
#           colormap dropdown, hemisphere toggle, live info label.
#
# k3d live-update API notes (from k3d 2.16.1 source inspection):
#   RQ1: k3d.lines supports a `colors` attribute — an array of uint32
#        per-VERTEX (not per-edge), shape (N_vertices,). For segment-mode
#        lines sharing the vertices array (76 centres), we assign 76 colors —
#        one per brain region.  When `colors` is set it overrides `color`.
#   RQ2: For live update of `lines.indices`, the array must be flat uint32
#        shape (E*2,) — each consecutive pair [i, j] defines one segment.
#        column_stack(...).flatten() produces this correctly.

import logging

import ipywidgets
import k3d
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
from tvb.datatypes.connectivity import Connectivity

from tvbwidgets_poc.base_widget import TVBWidgetPOC

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormap → representative single-color mapping for node highlights
# ---------------------------------------------------------------------------
_CMAP_NODE_COLORS = {
    'viridis':  0x21918c,
    'plasma':   0xf89540,
    'hot':      0xff4500,
    'coolwarm': 0x3b4cc0,
    'rainbow':  0x00aa00,
}


class Connectivity3DWidget(ipywidgets.VBox, TVBWidgetPOC):
    """
    Interactive 3D visualisation of a TVB Connectivity object.

    Renders brain regions as illuminated 3D spheres (k3d ``'3dSpecular'``
    shader) and inter-region connections as 3D line segments weighted by the
    connectivity matrix.

    **Interactive controls (Phase 2):**

    - Weight threshold slider — hide connections below a normalised weight
    - Node size slider — change the diameter of brain-region spheres live
    - Colormap dropdown — recolour nodes and edges via matplotlib colormaps
    - Hemisphere toggle — show left / right / both hemispheres
    - Info label — shows how many connections are currently visible

    Parameters
    ----------
    connectivity : tvb.datatypes.connectivity.Connectivity, optional
        TVB Connectivity object to visualise.  Can also be supplied later
        via :meth:`add_datatype`.
    width : int
        k3d viewport width in pixels.  Default 1000.
    height : int
        k3d viewport height in pixels.  Default 600.
    **kwargs
        Forwarded to ``ipywidgets.VBox.__init__``.
    """

    def __init__(self, connectivity=None, width=1000, height=600, **kwargs):
        # ----------------------------------------------------------------
        # Output container — k3d renders inside here so the canvas sits
        # correctly within the VBox DOM node (not in raw cell output).
        # ----------------------------------------------------------------
        self.output = ipywidgets.Output(
            layout=ipywidgets.Layout(
                width=str(width) + 'px',
                height=str(height) + 'px',
            )
        )

        # ----------------------------------------------------------------
        # TVBWidgetPOC.__init__ — sets self.logger
        # ----------------------------------------------------------------
        TVBWidgetPOC.__init__(self)

        # ----------------------------------------------------------------
        # Connectivity data and derived render state
        # ----------------------------------------------------------------
        self.connectivity = None
        self._k3d_points = None
        self._k3d_lines  = None
        self._edge_rows   = None   # int64 (1560,)
        self._edge_cols   = None   # int64 (1560,)
        self._weights_norm = None  # float64 (1560,) ∈ [0, 1]

        # Hemisphere index arrays (built once from region_labels)
        self._left_idx  = None    # uint32 indices of left-hemi regions
        self._right_idx = None    # uint32 indices of right-hemi regions

        # ----------------------------------------------------------------
        # k3d plot
        # ----------------------------------------------------------------
        self.plot = k3d.plot(
            grid_visible=False,
            background_color=self.PLOT_BG_COLOR,
        )

        # ----------------------------------------------------------------
        # Build the control panel (widgets only — not wired to data yet)
        # ----------------------------------------------------------------
        self._controls = self._build_controls()

        # ----------------------------------------------------------------
        # VBox: controls above the 3D viewport.
        # **kwargs not *kwargs — explicit fix over original HeadWidget bug.
        # ----------------------------------------------------------------
        super(Connectivity3DWidget, self).__init__(
            children=[self._controls, self.output],
            layout=ipywidgets.Layout(**self.DEFAULT_BORDER),
            **kwargs,
        )

        # ----------------------------------------------------------------
        # Render data if provided at construction time
        # ----------------------------------------------------------------
        if connectivity is not None:
            self.add_datatype(connectivity)

        # Capture k3d canvas output inside the Output widget
        with self.output:
            self.plot.display()

        self.logger.debug(
            f"Connectivity3DWidget initialised (width={width}, height={height})."
        )

    # ====================================================================
    # Public API
    # ====================================================================

    def add_datatype(self, connectivity):
        # type: (Connectivity) -> None
        """Load a TVB Connectivity object and render it."""
        if not self._validate_connectivity(connectivity):
            return

        self.connectivity = connectivity
        self.connectivity.configure()

        self.logger.info(
            f"Connectivity loaded: {self.connectivity.number_of_regions} regions."
        )

        # Pre-compute hemisphere index arrays (needed by callbacks)
        # TVB 76-region default connectivity uses 'r' prefix for right
        # hemisphere (indices 0-37) and 'l' prefix for left hemisphere
        # (indices 38-75). NOT 'rh'/'lh' — confirmed by inspecting labels.
        labels = list(self.connectivity.region_labels)
        self._left_idx  = numpy.array(
            [i for i, lbl in enumerate(labels) if lbl.startswith('l')],
            dtype=numpy.uint32,
        )
        self._right_idx = numpy.array(
            [i for i, lbl in enumerate(labels) if lbl.startswith('r')],
            dtype=numpy.uint32,
        )

        self._render_connectivity()
        self._refresh_info_label()

    def get_connectivity_info(self):
        """Return a summary dict for the loaded connectivity."""
        if self.connectivity is None:
            self.logger.warning("No connectivity loaded.")
            return {}

        weights = self.connectivity.weights
        n_regions = self.connectivity.number_of_regions
        nonzero = weights != 0

        return {
            'n_regions':      n_regions,
            'n_connections':  int(nonzero.sum()),
            'weight_min':     float(weights[nonzero].min()) if nonzero.any() else 0.0,
            'weight_max':     float(weights.max()),
            'region_labels':  list(self.connectivity.region_labels),
        }

    # ====================================================================
    # Control panel builder
    # ====================================================================

    def _build_controls(self):
        """
        Build and return the ipywidgets control panel VBox.

        Wires all Observer callbacks to the widget controls.
        Stores control widgets as instance attributes for later access
        by ``_get_active_mask`` and the callbacks.
        """
        # --- Row 1: Filtering -------------------------------------------
        self._threshold_slider = ipywidgets.FloatSlider(
            description='Threshold',
            min=0.0, max=1.0, step=0.01, value=0.0,
            style={'description_width': '80px'},
            layout=ipywidgets.Layout(width='300px'),
        )
        self._node_size_slider = ipywidgets.FloatSlider(
            description='Node size',
            min=1.0, max=20.0, step=0.5, value=8.0,
            style={'description_width': '80px'},
            layout=ipywidgets.Layout(width='250px'),
        )
        row1 = ipywidgets.HBox([self._threshold_slider, self._node_size_slider])

        # --- Row 2: Appearance ------------------------------------------
        self._colormap_dropdown = ipywidgets.Dropdown(
            description='Colormap',
            options=self.COLORMAP_OPTIONS,
            value='viridis',
            style={'description_width': '80px'},
            layout=ipywidgets.Layout(width='210px'),
        )
        self._hemisphere_toggle = ipywidgets.ToggleButtons(
            description='Hemisphere',
            options=['Both', 'Left', 'Right'],
            value='Both',
            button_style='',
            style={'description_width': '80px', 'button_width': '70px'},
            layout=ipywidgets.Layout(width='310px'),
        )
        self._info_label = ipywidgets.HTML(
            value='<span style="color:#888; font-size:12px;">No connectivity loaded.</span>'
        )
        row2 = ipywidgets.HBox(
            [self._colormap_dropdown, self._hemisphere_toggle, self._info_label]
        )

        # --- Header -----------------------------------------------------
        header = ipywidgets.HTML(
            value='<b style="font-size:13px; color:#555;">Connectivity Controls</b>'
        )

        # --- Wire callbacks ---------------------------------------------
        self._threshold_slider.observe(self._on_threshold_change,   names='value')
        self._node_size_slider.observe(self._on_node_size_change,   names='value')
        self._colormap_dropdown.observe(self._on_colormap_change,   names='value')
        self._hemisphere_toggle.observe(self._on_hemisphere_change, names='value')

        return ipywidgets.VBox(
            [header, row1, row2],
            layout=ipywidgets.Layout(padding='8px', border='1px solid #ddd'),
        )

    # ====================================================================
    # Combined filter mask
    # ====================================================================

    def _get_active_mask(self, hemi_override=None):
        """
        Return a boolean mask over all 1560 edges reflecting the current
        state of BOTH the threshold slider AND the hemisphere toggle.

        Parameters
        ----------
        hemi_override : str, optional
            If provided, use this hemisphere value instead of reading from
            ``self._hemisphere_toggle.value``.  Required when called from
            inside a callback *before* the widget value has been committed
            (e.g. direct unit-test calls).

        Both ``_on_threshold_change`` and ``_on_hemisphere_change`` call
        this to avoid duplicating filter logic and to ensure consistent
        behaviour when both filters are active simultaneously.
        """
        if self._weights_norm is None:
            return None

        # Threshold component
        threshold = self._threshold_slider.value
        mask = self._weights_norm >= threshold

        # Hemisphere component — use override if provided (needed for
        # direct callback invocations before toggle.value is committed)
        hemi = hemi_override if hemi_override is not None else self._hemisphere_toggle.value
        if hemi == 'Left':
            # Hide any edge touching the right hemisphere
            right_set = set(self._right_idx.tolist())
            mask &= ~numpy.isin(self._edge_rows, list(right_set))
            mask &= ~numpy.isin(self._edge_cols, list(right_set))
        elif hemi == 'Right':
            left_set = set(self._left_idx.tolist())
            mask &= ~numpy.isin(self._edge_rows, list(left_set))
            mask &= ~numpy.isin(self._edge_cols, list(left_set))

        return mask

    # ====================================================================
    # Callbacks — all mutate k3d traitlets in-place (no plot rebuild)
    # ====================================================================

    def _on_threshold_change(self, change):
        """Live-filter edges by normalised weight threshold."""
        if self._k3d_lines is None:
            return
        # No hemi_override needed: threshold slider doesn't change hemisphere
        mask = self._get_active_mask()
        self._apply_edge_mask(mask)

    def _on_node_size_change(self, change):
        """Live-update the diameter of all brain-region spheres."""
        if self._k3d_points is None:
            return
        self._k3d_points.point_size = change['new']

    def _on_colormap_change(self, change):
        """
        Recolour nodes and edges using the selected matplotlib colormap.

        Nodes: single representative color per colormap (stored in
        ``_CMAP_NODE_COLORS``).

        Edges: per-vertex colors derived from ``_weights_norm`` passed
        through the selected colormap. k3d.lines `colors` is per-vertex
        (one uint32 per entry in the `vertices` array, i.e. 76 values for
        76 brain regions).  We use the mean normalised weight of all edges
        incident to each region as the per-vertex value.
        """
        if self._k3d_points is None or self._k3d_lines is None:
            return

        cmap_name = change['new']

        # Node color — representative solid color per palette
        node_color = _CMAP_NODE_COLORS.get(cmap_name, 0x6aa84f)
        self._k3d_points.color = node_color

        # Edge per-vertex coloring via matplotlib colormap.
        # k3d.lines `colors` is per-VERTEX (one uint32 per entry in the
        # vertices array = 76 for 76 brain regions, not per-edge).
        # We use mean normalised incident-edge weight per region as the
        # per-vertex colour value.
        n_regions = self.connectivity.number_of_regions
        vertex_weights = numpy.zeros(n_regions, dtype=numpy.float64)
        counts = numpy.zeros(n_regions, dtype=numpy.int64)

        for i in range(len(self._edge_rows)):
            r_idx, c_idx = int(self._edge_rows[i]), int(self._edge_cols[i])
            w = self._weights_norm[i]
            vertex_weights[r_idx] += w
            vertex_weights[c_idx] += w
            counts[r_idx] += 1
            counts[c_idx] += 1

        mask_nonzero = counts > 0
        vertex_weights[mask_nonzero] /= counts[mask_nonzero]

        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(vertex_weights)                            # (76, 4) float 0-1
        r_ch = (rgba[:, 0] * 255).astype(numpy.uint32)
        g_ch = (rgba[:, 1] * 255).astype(numpy.uint32)
        b_ch = (rgba[:, 2] * 255).astype(numpy.uint32)
        colors_uint32 = numpy.array(
            (r_ch << 16) | (g_ch << 8) | b_ch, dtype=numpy.uint32
        )                                                      # (76,) uint32

        self._k3d_lines.colors = colors_uint32

        self.logger.debug(f"Colormap changed to '{cmap_name}'.")

    def _on_hemisphere_change(self, change):
        """Show Left, Right, or Both hemispheres — filters nodes and edges."""
        if self._k3d_points is None or self._k3d_lines is None:
            return

        hemi = change['new']

        # --- Node visibility via per-point colors -----------------------
        # k3d Points has no per-point visibility flag.  We implement it by
        # setting invisible nodes to PLOT_BG_COLOR (0x1a1a2e — dark navy,
        # same as the k3d background) making them effectively invisible.
        # This is flicker-free and fully reversible without plot rebuild.
        n = self.connectivity.number_of_regions
        cmap_name = self._colormap_dropdown.value
        node_color = _CMAP_NODE_COLORS.get(cmap_name, 0x6aa84f)

        node_colors = numpy.full(n, node_color, dtype=numpy.uint32)

        if hemi == 'Left' and self._right_idx is not None and len(self._right_idx) > 0:
            node_colors[self._right_idx] = self.PLOT_BG_COLOR
        elif hemi == 'Right' and self._left_idx is not None and len(self._left_idx) > 0:
            node_colors[self._left_idx] = self.PLOT_BG_COLOR

        self._k3d_points.colors = node_colors

        # --- Edge filtering via combined mask ---------------------------
        # Pass hemi explicitly: when this callback fires, _hemisphere_toggle
        # has already been updated by ipywidgets, but we pass it explicitly
        # anyway so direct test calls also work correctly.
        mask = self._get_active_mask(hemi_override=hemi)
        self._apply_edge_mask(mask)

        self.logger.debug(f"Hemisphere filter set to '{hemi}'.")

    # ====================================================================
    # Internal helpers
    # ====================================================================

    def _apply_edge_mask(self, mask):
        """
        Apply a boolean *mask* over the full edge set to live-update the
        k3d lines indices.

        Parameters
        ----------
        mask : numpy.ndarray of bool, shape (1560,)
            True for edges that should be visible.
        """
        if mask is None:
            return

        n_visible = int(mask.sum())

        if n_visible == 0:
            self._k3d_lines.indices = numpy.array([], dtype=numpy.uint32)
        else:
            rows = self._edge_rows[mask]
            cols = self._edge_cols[mask]
            # k3d lines (indices_type='segment') expects a FLAT uint32 array
            # of shape (E*2,) — each consecutive pair [i, j] is one segment.
            self._k3d_lines.indices = (
                numpy.column_stack([rows, cols])
                .astype(numpy.uint32)
                .flatten()
            )

        self._refresh_info_label(n_visible)

    def _refresh_info_label(self, n_visible=None):
        """Update the info HTML label with current connection count."""
        if self._weights_norm is None:
            return

        total = len(self._weights_norm)
        if n_visible is None:
            # Called after first load — all connections visible
            n_visible = total

        info = self.get_connectivity_info()
        n_regions = info.get('n_regions', '?')

        self._info_label.value = (
            f'<span style="color:#555; font-size:12px;">'
            f'<b>{n_regions}</b> regions &nbsp;|&nbsp; '
            f'<b>{n_visible}</b> / {total} connections visible'
            f'</span>'
        )

    # ====================================================================
    # Core rendering (called once by add_datatype)
    # ====================================================================

    def _render_connectivity(self):
        """
        Build k3d nodes and edges and add them to the plot.

        Called once by ``add_datatype``.  Subsequent changes are applied
        via traitlet mutation in the callbacks — no full redraw needed.
        """
        self.logger.debug("Rendering connectivity…")

        centres_f32 = self._to_float32(self.connectivity.centres)

        # ---- Nodes -------------------------------------------------------
        points = k3d.points(
            positions=centres_f32,
            point_size=self._node_size_slider.value,
            shader='3dSpecular',
            color=_CMAP_NODE_COLORS['viridis'],
            name='BrainRegions',
        )
        self._k3d_points = points

        # ---- Edges -------------------------------------------------------
        rows, cols = numpy.nonzero(self.connectivity.weights)

        edge_indices = (
            numpy.column_stack([rows, cols])
            .astype(numpy.uint32)
            .flatten()          # (E*2,) flat — required shape for live updates
        )

        weights_flat = self.connectivity.weights[rows, cols]
        weight_range = weights_flat.max() - weights_flat.min()
        weights_norm = (weights_flat - weights_flat.min()) / (weight_range + 1e-8)

        lines = k3d.lines(
            vertices=centres_f32,
            indices=edge_indices,
            indices_type='segment',
            shader='simple',
            color=0xaaaaaa,
            width=1.5,
            name='Connections',
        )
        self._k3d_lines  = lines
        self._edge_rows   = rows
        self._edge_cols   = cols
        self._weights_norm = weights_norm

        # ---- Add to plot -------------------------------------------------
        self.plot += self._k3d_points
        self.plot += self._k3d_lines

        self.logger.info(
            f"Rendered {len(rows)} edges across "
            f"{self.connectivity.number_of_regions} regions. "
            f"Edge indices shape: {edge_indices.shape}, dtype: {edge_indices.dtype}."
        )
