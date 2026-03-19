# -*- coding: utf-8 -*-

import logging

import ipywidgets
import k3d
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
from tvb.datatypes.connectivity import Connectivity

from tvbwidgets_poc.base_widget import TVBWidgetPOC

_logger = logging.getLogger(__name__)

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
        self.output = ipywidgets.Output(
            layout=ipywidgets.Layout(
                width=str(width) + 'px',
                height=str(height) + 'px',
            )
        )

        TVBWidgetPOC.__init__(self)

        self.connectivity = None
        self._k3d_points = None
        self._k3d_lines  = None
        self._edge_rows   = None   # int64 (1560,)
        self._edge_cols   = None   # int64 (1560,)
        self._weights_norm = None  # float64 (1560,) ∈ [0, 1]

        self._left_idx  = None    # uint32 indices of left-hemi regions
        self._right_idx = None    # uint32 indices of right-hemi regions

        self._node_sizes     = None   # float32 (N,) per-node sizes [4, 18]
        self._node_strengths = None   # float64 (N,) normalised row-sum
        self._node_colors_base = None # uint32 (N,) viridis colors

        self._k3d_labels = []   # list of active k3d.text objects in the plot

        # ----------------------------------------------------------------
        # k3d plot
        # ----------------------------------------------------------------
        self.plot = k3d.plot(
            grid_visible=False,
            background_color=self.PLOT_BG_COLOR,
        )

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

        if connectivity is not None:
            self.add_datatype(connectivity)

        with self.output:
            self.plot.display()

        self.logger.debug(
            f"Connectivity3DWidget initialised (width={width}, height={height})."
        )

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

    def _build_controls(self):
        """
        Build and return the ipywidgets control panel VBox.
        """
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
        self._label_toggle = ipywidgets.ToggleButton(
            value=False,
            description='Labels',
            icon='tag',
            layout=ipywidgets.Layout(width='100px', height='32px'),
        )
        self._info_label = ipywidgets.HTML(
            value='<span style="color:#888; font-size:12px;">No connectivity loaded.</span>'
        )
        row2 = ipywidgets.HBox(
            [self._colormap_dropdown, self._hemisphere_toggle,
             self._label_toggle, self._info_label]
        )

        header = ipywidgets.HTML(
            value='<b style="font-size:13px; color:#555;">Connectivity Controls</b>'
        )

        self._threshold_slider.observe(self._on_threshold_change,   names='value')
        self._node_size_slider.observe(self._on_node_size_change,   names='value')
        self._colormap_dropdown.observe(self._on_colormap_change,   names='value')
        self._hemisphere_toggle.observe(self._on_hemisphere_change, names='value')
        self._label_toggle.observe(self._on_label_toggle,           names='value')

        return ipywidgets.VBox(
            [header, row1, row2],
            layout=ipywidgets.Layout(padding='8px', border='1px solid #ddd'),
        )

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
        """
        if self._weights_norm is None:
            return None

        # Threshold component
        threshold = self._threshold_slider.value
        mask = self._weights_norm >= threshold

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

    def _on_threshold_change(self, change):
        """Live-filter edges by normalised weight threshold."""
        if self._k3d_lines is None:
            return
        mask = self._get_active_mask()
        self._apply_edge_mask(mask)

    def _on_node_size_change(self, change):
        """Live-scale all node sizes while preserving relative hub/peripheral sizing."""
        if self._k3d_points is None or self._node_sizes is None:
            return
        scale = change['new'] / 8.0
        self._k3d_points.point_sizes = (self._node_sizes * scale).astype(numpy.float32)

    def _on_colormap_change(self, change):
        """
        Recolour nodes and edges using the selected matplotlib colormap.
        """
        if self._k3d_points is None or self._k3d_lines is None:
            return

        cmap_name = change['new']
        cmap = plt.get_cmap(cmap_name)

        if self._node_strengths is not None:
            rgba_n = cmap(self._node_strengths)              # (N, 4) float
            r_n = (rgba_n[:, 0] * 255).astype(numpy.uint32)
            g_n = (rgba_n[:, 1] * 255).astype(numpy.uint32)
            b_n = (rgba_n[:, 2] * 255).astype(numpy.uint32)
            node_colors = numpy.array(
                (r_n << 16) | (g_n << 8) | b_n, dtype=numpy.uint32
            )
            self._node_colors_base = node_colors
            self._k3d_points.colors = node_colors
        else:
            node_color = _CMAP_NODE_COLORS.get(cmap_name, 0x6aa84f)
            self._k3d_points.color = node_color

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

        rgba_e = cmap(vertex_weights)                        # (76, 4) float
        r_ch = (rgba_e[:, 0] * 255).astype(numpy.uint32)
        g_ch = (rgba_e[:, 1] * 255).astype(numpy.uint32)
        b_ch = (rgba_e[:, 2] * 255).astype(numpy.uint32)
        colors_uint32 = numpy.array(
            (r_ch << 16) | (g_ch << 8) | b_ch, dtype=numpy.uint32
        )                                                    # (76,) uint32

        self._k3d_lines.colors = colors_uint32

        self.logger.debug(f"Colormap changed to '{cmap_name}'.")

    def _on_hemisphere_change(self, change):
        """Show Left, Right, or Both hemispheres — filters nodes and edges."""
        if self._k3d_points is None or self._k3d_lines is None:
            return

        hemi = change['new']

        n = self.connectivity.number_of_regions
        cmap_name = self._colormap_dropdown.value
        node_color = _CMAP_NODE_COLORS.get(cmap_name, 0x6aa84f)

        node_colors = numpy.full(n, node_color, dtype=numpy.uint32)

        if hemi == 'Left' and self._right_idx is not None and len(self._right_idx) > 0:
            node_colors[self._right_idx] = self.PLOT_BG_COLOR
        elif hemi == 'Right' and self._left_idx is not None and len(self._left_idx) > 0:
            node_colors[self._left_idx] = self.PLOT_BG_COLOR

        self._k3d_points.colors = node_colors

        mask = self._get_active_mask(hemi_override=hemi)
        self._apply_edge_mask(mask)

        self.logger.debug(f"Hemisphere filter set to '{hemi}'.")

    def _on_label_toggle(self, change):
        """Show or hide region labels for the top-10 hub nodes."""
        if self._k3d_points is None or self._node_strengths is None:
            return

        if change['new']:
            top_hub_indices = numpy.argsort(self._node_strengths)[-10:]
            for idx in top_hub_indices:
                pos = self._to_float32(self.connectivity.centres[idx])
                label = k3d.text(
                    str(self.connectivity.region_labels[idx]),
                    position=tuple(float(x) for x in pos),
                    color=0xffffff,
                    size=0.5,
                    label_box=False,
                    name=f'label_{idx}',
                )
                self.plot += label
                self._k3d_labels.append(label)
            self.logger.debug(f"Labels shown for {len(self._k3d_labels)} hub regions.")
        else:
            for label in self._k3d_labels:
                self.plot -= label
            self._k3d_labels = []
            self.logger.debug("Labels cleared.")

            self.logger.debug("Labels cleared.")

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
            n_visible = total

        info = self.get_connectivity_info()
        n_regions = info.get('n_regions', '?')

        self._info_label.value = (
            f'<span style="color:#555; font-size:12px;">'
            f'<b>{n_regions}</b> regions &nbsp;|&nbsp; '
            f'<b>{n_visible}</b> / {total} connections visible'
            f'</span>'
        )

    def _render_connectivity(self):
        """
        Build k3d nodes and edges and add them to the plot.

        Called once by ``add_datatype``.  Subsequent changes are applied
        via traitlet mutation in the callbacks — no full redraw needed.
        """
        self.logger.debug("Rendering connectivity…")

        centres_f32 = self._to_float32(self.connectivity.centres)

        node_strength = self.connectivity.weights.sum(axis=1)   # (N,) float64
        strength_norm = (
            (node_strength - node_strength.min())
            / (node_strength.max() - node_strength.min() + 1e-8)
        )                                                        # (N,) in [0,1]
        node_sizes = (4.0 + strength_norm * 14.0).astype(numpy.float32)  # [4, 18]
        self._node_sizes     = node_sizes
        self._node_strengths = strength_norm

        cmap_init = plt.get_cmap('viridis')
        rgba_init = cmap_init(strength_norm)                     # (N, 4)
        r_i = (rgba_init[:, 0] * 255).astype(numpy.uint32)
        g_i = (rgba_init[:, 1] * 255).astype(numpy.uint32)
        b_i = (rgba_init[:, 2] * 255).astype(numpy.uint32)
        node_colors = numpy.array(
            (r_i << 16) | (g_i << 8) | b_i, dtype=numpy.uint32
        )                                                        # (N,) uint32
        self._node_colors_base = node_colors

        points = k3d.points(
            positions=centres_f32,
            point_sizes=node_sizes,          # per-node hub sizing
            colors=node_colors,              # per-node viridis coloring
            shader='3dSpecular',
            name='BrainRegions',
        )
        self._k3d_points = points

        rows, cols = numpy.nonzero(self.connectivity.weights)

        edge_indices = (
            numpy.column_stack([rows, cols])
            .astype(numpy.uint32)
            .flatten()
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

        self.plot += self._k3d_points
        self.plot += self._k3d_lines

        self.logger.info(
            f"Rendered {len(rows)} edges across "
            f"{self.connectivity.number_of_regions} regions. "
            f"Edge indices shape: {edge_indices.shape}, dtype: {edge_indices.dtype}."
        )
