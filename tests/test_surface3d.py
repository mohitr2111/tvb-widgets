# -*- coding: utf-8 -*-
"""
Unit tests for AnimatedSurface3DWidget.

Run with:
    pytest tests/test_surface3d.py -v
"""
import numpy as np
import pytest
from tvb.datatypes.surfaces import CorticalSurface

from tvbwidgets_poc import AnimatedSurface3DWidget


@pytest.fixture(scope="module")
def surf():
    """Load and configure the default TVB cortical surface once per module."""
    s = CorticalSurface.from_file()
    s.configure()
    return s


@pytest.fixture(scope="module")
def widget(surf):
    """Instantiated widget with surface and synthetic timeseries."""
    return AnimatedSurface3DWidget(surf)


def test_widget_instantiates_with_surface(surf):
    """Widget must render successfully when given a valid surface."""
    w = AnimatedSurface3DWidget(surf)
    assert w.surface is not None
    assert w.timeseries is not None
    assert w._k3d_mesh is not None


def test_synthetic_timeseries_shape_and_dtype(widget, surf):
    """Synthetic timeseries must be (120, N_vertices) float32 with sane range."""
    n_vertices = len(surf.vertices)
    assert widget.timeseries.shape == (120, n_vertices), (
        f"Expected shape (120, {n_vertices}), got {widget.timeseries.shape}"
    )
    assert widget.timeseries.dtype == np.float32, (
        f"Expected float32, got {widget.timeseries.dtype}"
    )
    assert widget.timeseries.min() > -2.0, "Timeseries min unexpectedly low"
    assert widget.timeseries.max() <  2.0, "Timeseries max unexpectedly high"


def test_custom_timeseries_accepted(surf):
    """Widget must accept and store a user-supplied timeseries correctly."""
    n_vertices = len(surf.vertices)
    ts = np.random.default_rng(0).standard_normal((60, n_vertices)).astype(np.float32)

    w = AnimatedSurface3DWidget(surf, timeseries=ts)

    assert w.timeseries.shape == (60, n_vertices), (
        f"Expected (60, {n_vertices}), got {w.timeseries.shape}"
    )
    assert w.timeseries.dtype == np.float32
    assert w._play.max == 59, f"Play max should be 59, got {w._play.max}"


def test_frame_change_updates_mesh_attribute(widget):
    """_on_frame_change must update mesh.attribute to a different frame."""
    frame_0_attr = np.array(widget._k3d_mesh.attribute, copy=True)

    widget._on_frame_change({'new': 60})

    assert widget._current_frame == 60
    assert not np.allclose(widget._k3d_mesh.attribute, frame_0_attr), (
        "mesh.attribute should differ between frame 0 and frame 60"
    )

    # Reset to frame 0
    widget._on_frame_change({'new': 0})


def test_speed_change_updates_play_interval(widget):
    """_on_speed_change must correctly mutate Play.interval as int(83/speed)."""
    widget._on_speed_change({'new': 2.0})
    assert widget._play.interval == 41, (
        f"Speed 2.0 → expected 41 ms, got {widget._play.interval}"
    )

    widget._on_speed_change({'new': 0.5})
    assert widget._play.interval == 166, (
        f"Speed 0.5 → expected 166 ms, got {widget._play.interval}"
    )

    widget._on_speed_change({'new': 1.0})
    assert widget._play.interval == 83, (
        f"Speed 1.0 → expected 83 ms, got {widget._play.interval}"
    )
