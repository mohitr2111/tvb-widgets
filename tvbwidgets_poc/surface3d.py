# -*- coding: utf-8 -*-

import logging

import ipywidgets
import k3d
import numpy
from tvb.datatypes.surfaces import CorticalSurface

from tvbwidgets_poc.base_widget import TVBWidgetPOC

_logger = logging.getLogger(__name__)

_DIVERGING_CMAPS = ['coolwarm', 'RdBu', 'seismic', 'bwr', 'PRGn']


class AnimatedSurface3DWidget(ipywidgets.VBox, TVBWidgetPOC):
    """
    Interactive 3D cortical surface renderer with animated timeseries overlay.

    Renders a TVB ``CorticalSurface`` as a k3d mesh and drives per-vertex
    colour from a ``(T, N_vertices)`` float32 timeseries array.  Animation
    is controlled by an ``ipywidgets.Play`` counter linked (via ``jslink``)
    to an ``IntSlider``, with a Python observer on the slider that mutates
    ``mesh.attribute`` in-place at up to ~12 fps.

    Parameters
    ----------
    surface : tvb.datatypes.surfaces.CorticalSurface, optional
        TVB cortical surface object.  Can be supplied later via
        :meth:`add_datatype`.
    timeseries : numpy.ndarray, optional
        Float32 array of shape ``(T, N_vertices)``.  If ``None`` and a
        surface is provided, synthetic timeseries is generated automatically.
    width : int
        k3d viewport width in pixels.  Default 1000.
    height : int
        k3d viewport height in pixels.  Default 600.

    Examples
    --------
    >>> from tvb.datatypes.surfaces import CorticalSurface
    >>> from tvbwidgets_poc import AnimatedSurface3DWidget
    >>> surface = CorticalSurface.from_file()
    >>> surface.configure()
    >>> widget = AnimatedSurface3DWidget(surface)
    >>> widget   # display in JupyterLab cell
    """

    def __init__(self, surface=None, timeseries=None, width=1000, height=600, **kwargs):
        self.output = ipywidgets.Output(
            layout=ipywidgets.Layout(
                width=str(width) + 'px',
                height=str(height) + 'px',
            )
        )

        TVBWidgetPOC.__init__(self)

        self.surface     = None
        self.timeseries  = None
        self._k3d_mesh   = None
        self._current_frame = 0
        self._is_playing    = False

        self.plot = k3d.plot(
            grid_visible=False,
            background_color=self.PLOT_BG_COLOR,
        )

        self._controls = self._build_controls()

        super(AnimatedSurface3DWidget, self).__init__(
            children=[self._controls, self.output],
            layout=ipywidgets.Layout(**self.DEFAULT_BORDER),
            **kwargs,
        )

        with self.output:
            self.plot.display()

        if surface is not None:
            self.add_datatype(surface, timeseries)

        self.logger.debug(
            f"AnimatedSurface3DWidget initialised (width={width}, height={height})."
        )

    def add_datatype(self, surface, timeseries=None):
        # type: (CorticalSurface, numpy.ndarray | None) -> None
        """
        Load a TVB CorticalSurface and optional timeseries, then render.

        Parameters
        ----------
        surface : tvb.datatypes.surfaces.CorticalSurface
        timeseries : numpy.ndarray, optional
            Float32 array of shape ``(T, N_vertices)``.  Generated
            synthetically if not provided.
        """
        if surface is None:
            self.logger.error("Surface is None — cannot render.")
            return

        if not hasattr(surface, 'vertices') or surface.vertices.shape[1] != 3:
            self.logger.error("Invalid surface: vertices must have shape (N, 3).")
            return

        surface.configure()
        self.surface  = surface
        n_vertices    = len(surface.vertices)

        self.logger.info(
            f"Surface loaded: {n_vertices} vertices, "
            f"{len(surface.triangles)} triangles."
        )

        if timeseries is None:
            self.logger.info("No timeseries provided — generating synthetic data.")
            self.timeseries = self._generate_synthetic_timeseries()
        else:
            t, n = timeseries.shape
            if n != n_vertices:
                self.logger.error(
                    f"Timeseries vertex count ({n}) ≠ surface vertex count ({n_vertices})."
                )
                return
            # Store as float32 at load time — never cast inside _on_frame_change
            self.timeseries = numpy.asarray(timeseries, dtype=numpy.float32)

        self._render_surface()
        self._update_controls_state(enabled=True)

    def _generate_synthetic_timeseries(self, n_frames=120):
        """
        Generate ``(n_frames, N_vertices)`` float32 synthetic neural activity.

        Mathematical design
        -------------------
        * Divide the N vertices evenly into 8 pseudo-regions.
        * Each region ``r`` oscillates at a distinct frequency:
          ``freq_r = 0.5 + r × 0.3`` Hz  (0.5, 0.8, … 2.6 Hz).
        * Phase offset: ``phase_r = r × π/4`` — staggered so regions do
          not all peak simultaneously.
        * Time vector: ``t = linspace(0, 4, n_frames)`` (4 seconds).
        * Signal per vertex: ``sin(2π × freq_r × t + phase_r)``.
        * Small Gaussian noise (σ=0.05) added for visual texture.

        Result: at each frame different brain regions are at different
        phases of their oscillation — creating spatially varied colour
        patterns that resemble actual neural activity.
        """
        n_vertices = len(self.surface.vertices)
        n_regions  = 8

        region_id  = numpy.arange(n_vertices) // (n_vertices // n_regions)
        region_id  = numpy.clip(region_id, 0, n_regions - 1)

        t          = numpy.linspace(0, 4, n_frames, dtype=numpy.float32)
        timeseries = numpy.zeros((n_frames, n_vertices), dtype=numpy.float32)

        for r in range(n_regions):
            freq    = 0.5 + r * 0.3
            phase   = r * numpy.pi / 4
            signal  = numpy.sin(2 * numpy.pi * freq * t + phase).astype(numpy.float32)
            mask    = region_id == r
            # Broadcast signal (n_frames,) onto all vertices in this region
            timeseries[:, mask] = signal[:, numpy.newaxis]

        # Add small noise for visual texture across all vertices
        rng   = numpy.random.default_rng(seed=42)
        noise = rng.standard_normal((n_frames, n_vertices)).astype(numpy.float32) * 0.05
        timeseries += noise

        self.logger.info(
            f"Synthetic timeseries generated: shape={timeseries.shape}, "
            f"dtype={timeseries.dtype}, "
            f"range=[{timeseries.min():.3f}, {timeseries.max():.3f}]"
        )
        return timeseries

    def _render_surface(self):
        """
        Build the k3d mesh with the first frame as initial attribute data
        and add it to the already-displayed plot.

        Never calls ``plot.display()`` — that was done once in ``__init__``.
        k3d's live canvas updates when objects are added via ``plot +=``.
        """
        self.logger.debug("Rendering cortical surface mesh…")

        vertices  = self._to_float32(self.surface.vertices)
        triangles = self.surface.triangles.astype(numpy.uint32)

        # First frame drives initial per-vertex colour
        first_frame = self.timeseries[0]   # (N_vertices,) float32 — no copy

        # coolwarm: diverging blue→white→red, ideal for signals that swing +/-
        color_map = getattr(k3d.matplotlib_color_maps, 'coolwarm')

        self._k3d_mesh = k3d.mesh(
            vertices=vertices,
            indices=triangles,
            attribute=first_frame,
            color_map=color_map,
            color_range=[-1.3, 1.3],    # covers full sine range ± noise (observed: ~±1.22)
            side='double',              # render both faces (no z-fighting on thin mesh)
            opacity=0.9,
            name='CorticalSurface',
        )

        self.plot += self._k3d_mesh

        self.logger.info(
            f"Mesh rendered: {len(vertices)} vertices, {len(triangles)} triangles, "
            f"attribute dtype={first_frame.dtype}."
        )

    def _build_controls(self):
        """
        Build the ipywidgets control panel and wire all observers.

        Play ↔ Slider sync: ``jslink`` (client-side) handles the UI
        counter without a kernel round-trip.  A separate Python ``.observe``
        on the slider fires ``_on_frame_change`` which mutates
        ``mesh.attribute`` — this MUST run in the Python kernel.
        ``jslink`` alone cannot mutate Python objects, hence both are needed.
        """
        n_frames = 120    # default; updated when data loads if different

        self._play = ipywidgets.Play(
            min=0, max=n_frames - 1, step=1, value=0,
            interval=67,        # ≈ 15 fps
            description='▶',
            disabled=True,      # enabled after data loads
        )
        self._frame_slider = ipywidgets.IntSlider(
            min=0, max=n_frames - 1, step=1, value=0,
            description='Frame',
            layout=ipywidgets.Layout(width='350px'),
            disabled=True,
        )

        ipywidgets.jslink((self._play, 'value'), (self._frame_slider, 'value'))
        self._frame_slider.observe(self._on_frame_change, names='value')

        row1 = ipywidgets.HBox([self._play, self._frame_slider])

        self._colormap_dropdown = ipywidgets.Dropdown(
            description='Colormap',
            options=_DIVERGING_CMAPS,
            value='coolwarm',
            style={'description_width': '80px'},
            layout=ipywidgets.Layout(width='210px'),
        )
        self._speed_slider = ipywidgets.FloatSlider(
            description='Speed',
            min=0.25, max=4.0, step=0.25, value=1.0,
            style={'description_width': '60px'},
            layout=ipywidgets.Layout(width='250px'),
        )
        self._frame_label = ipywidgets.HTML(
            value='<span style="font-size:12px; color:#888;">No data loaded.</span>'
        )

        self._colormap_dropdown.observe(self._on_colormap_change, names='value')
        self._speed_slider.observe(self._on_speed_change,         names='value')

        row2 = ipywidgets.HBox(
            [self._colormap_dropdown, self._speed_slider, self._frame_label]
        )

        header = ipywidgets.HTML(
            value='<b style="font-size:13px; color:#555;">Surface Animation Controls</b>'
        )

        return ipywidgets.VBox(
            [header, row1, row2],
            layout=ipywidgets.Layout(padding='8px', border='1px solid #ddd'),
        )

    def _on_frame_change(self, change):
        """
        Hot path — called up to ~12 times per second during playback.

        Mutates ``mesh.attribute`` in-place (k3d 2.16.x traitlet mutation,
        no plot rebuild).  ``self.timeseries[frame]`` is already float32 —
        no ``astype()`` call here to avoid per-frame allocation.
        """
        if self._k3d_mesh is None:
            return   # guard: slider fires before data loads

        frame = change['new']
        self._current_frame  = frame
        self._k3d_mesh.attribute = self.timeseries[frame]
        self._frame_label.value  = (
            f'<span style="font-size:12px;">'
            f'Frame {frame + 1} / {len(self.timeseries)}'
            f'</span>'
        )

    def _on_colormap_change(self, change):
        """Live-swap the diverging colormap on the mesh."""
        if self._k3d_mesh is None:
            return
        cmap_name = change['new']
        self._k3d_mesh.color_map = getattr(k3d.matplotlib_color_maps, cmap_name)
        self.logger.debug(f"Colormap changed to '{cmap_name}'.")

    def _on_speed_change(self, change):
        """Adjust playback speed by mutating Play.interval (ms per frame)."""
        self._play.interval = int(67 / change['new'])

    def _update_controls_state(self, enabled):
        """
        Enable or disable playback controls.

        Called with ``enabled=False`` before data loads and
        ``enabled=True`` after ``_render_surface()`` completes.
        """
        self._play.disabled         = not enabled
        self._frame_slider.disabled = not enabled

        if enabled and self.timeseries is not None:
            n_frames = len(self.timeseries)
            self._play.max         = n_frames - 1
            self._frame_slider.max = n_frames - 1
            self._frame_label.value = (
                f'<span style="font-size:12px;">'
                f'Frame 1 / {n_frames} &nbsp;|&nbsp; '
                f'{len(self.surface.vertices):,} vertices'
                f'</span>'
            )
