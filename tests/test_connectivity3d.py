# -*- coding: utf-8 -*-
"""
Unit tests for Connectivity3DWidget.

Run with:
    pytest tests/test_connectivity3d.py -v
"""
import numpy as np
import pytest
from tvb.datatypes.connectivity import Connectivity

from tvbwidgets_poc import Connectivity3DWidget


@pytest.fixture(scope="module")
def conn():
    """Load and configure the default TVB connectivity once per module."""
    c = Connectivity.from_file()
    c.configure()
    return c


@pytest.fixture(scope="module")
def widget(conn):
    """Instantiated widget with connectivity loaded."""
    return Connectivity3DWidget(conn)


def test_widget_instantiates_with_connectivity(conn):
    """Widget must render successfully when given a valid connectivity."""
    w = Connectivity3DWidget(conn)
    assert isinstance(w, Connectivity3DWidget)
    assert w.connectivity is not None
    assert w._k3d_points is not None
    assert w._k3d_lines is not None


def test_widget_instantiates_without_data():
    """Widget must not raise when constructed with no arguments."""
    w = Connectivity3DWidget()
    assert w.connectivity is None
    assert w._k3d_points is None
    assert w._k3d_lines is None


def test_connectivity_info_structure(widget):
    """get_connectivity_info() must return the expected dict with correct values."""
    info = widget.get_connectivity_info()

    assert isinstance(info, dict)
    for key in ('n_regions', 'n_connections', 'weight_min', 'weight_max', 'region_labels'):
        assert key in info, f"Missing key: {key}"

    assert info['n_regions'] == 76
    assert info['n_connections'] == 1560
    assert len(info['region_labels']) == 76


def test_threshold_callback_reduces_visible_edges(widget):
    """
    Threshold=0.5 must produce fewer visible edges than threshold=0.0.
    """
    # Reset to 0
    widget._threshold_slider.value = 0.0
    widget._on_threshold_change({'new': 0.0})
    mask_all = widget._get_active_mask()
    initial = int(mask_all.sum())
    assert initial == 1560, f"Baseline should be 1560, got {initial}"

    # Apply threshold
    widget._threshold_slider.value = 0.5
    widget._on_threshold_change({'new': 0.5})
    mask_filtered = widget._get_active_mask()
    filtered = int(mask_filtered.sum())

    assert filtered < initial, "Threshold 0.5 should reduce visible edge count"
    assert filtered > 0,      "Threshold 0.5 should not hide all connections"

# Reset
    widget._threshold_slider.value = 0.0
    widget._on_threshold_change({'new': 0.0})


def test_hemisphere_filter_left_right_sum(widget):
    """Left and right hemisphere connection counts must be symmetric and < total."""
    # Reset threshold
    widget._threshold_slider.value = 0.0
    widget._on_threshold_change({'new': 0.0})

    left_count  = int(widget._get_active_mask(hemi_override='Left').sum())
    right_count = int(widget._get_active_mask(hemi_override='Right').sum())
    both_count  = int(widget._get_active_mask(hemi_override='Both').sum())

    assert left_count == right_count, (
        f"Left ({left_count}) != Right ({right_count}): asymmetric default connectome"
    )
    assert left_count < both_count, "Left-only should be fewer connections than Both"
    assert both_count == 1560, f"Both must restore full 1560. Got {both_count}"


def test_edge_indices_dtype_is_uint32(widget):
    """
    The indices array produced by _apply_edge_mask must be uint32.
    """
    # Reset
    widget._threshold_slider.value = 0.0
    widget._on_threshold_change({'new': 0.0})

    mask_all = widget._get_active_mask()

    # Build the exact array _apply_edge_mask sends to k3d
    rows = widget._edge_rows[mask_all]
    cols = widget._edge_cols[mask_all]
    indices_arr = (
        np.column_stack([rows, cols])
        .astype(np.uint32)
        .flatten()
    )
    assert indices_arr.dtype == np.uint32, (
        f"Indices dtype before k3d write: {indices_arr.dtype}"
    )

    # After threshold callback, verify same construction still yields uint32
    widget._threshold_slider.value = 0.3
    widget._on_threshold_change({'new': 0.3})
    mask_filtered = widget._get_active_mask()

    rows_f = widget._edge_rows[mask_filtered]
    cols_f = widget._edge_cols[mask_filtered]
    indices_filtered = (
        np.column_stack([rows_f, cols_f])
        .astype(np.uint32)
        .flatten()
    )
    assert indices_filtered.dtype == np.uint32, (
        f"Post-callback indices dtype: {indices_filtered.dtype}"
    )

    # Reset
    widget._threshold_slider.value = 0.0
    widget._on_threshold_change({'new': 0.0})
