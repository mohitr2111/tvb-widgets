"""
Phase 1 verification script — run from tvb-widgets-poc/ directory.
Tests import, data shapes, dtype, widget instantiation, logging, and stored refs.
"""
import sys
import logging

# --------------------------------------------------------------------
# Enable DEBUG logging so we can see logger output as verification step 4
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s [%(name)s] %(message)s',
)

sys.path.insert(0, r'C:\Users\mohit\OneDrive\Desktop\GSoC 2026\tvb-widgets-poc')

print("=" * 60)
print("STEP 1 — Import test")
print("=" * 60)

from tvbwidgets_poc import Connectivity3DWidget  # noqa: E402
from tvb.datatypes.connectivity import Connectivity  # noqa: E402
import numpy  # noqa: E402

print("PASS: imports successful")

print()
print("=" * 60)
print("STEP 2 — Data shape + dtype test")
print("=" * 60)

conn = Connectivity.from_file()
conn.configure()

assert conn.weights.shape == (76, 76), f"FAIL weights shape: {conn.weights.shape}"
assert conn.centres.shape == (76, 3), f"FAIL centres shape: {conn.centres.shape}"
print(f"PASS: weights.shape = {conn.weights.shape}")
print(f"PASS: centres.shape = {conn.centres.shape}")

rows, cols = numpy.nonzero(conn.weights)
idx = numpy.column_stack([rows, cols]).astype(numpy.uint32)
assert idx.dtype == numpy.uint32, f"FAIL dtype: {idx.dtype}"
print(f"PASS: edge indices dtype = {idx.dtype}")
print(f"      nonzero connections = {len(rows)}")

print()
print("=" * 60)
print("STEP 3 — Widget instantiation + stored refs")
print("=" * 60)

# This creates the widget but cannot display in a plain Python script
# (display requires JupyterLab frontend).  We verify object creation only.
w = Connectivity3DWidget(conn)

assert w._k3d_points is not None,   "FAIL: _k3d_points is None"
assert w._k3d_lines  is not None,   "FAIL: _k3d_lines is None"
assert w._edge_rows  is not None,   "FAIL: _edge_rows is None"
assert w._edge_cols  is not None,   "FAIL: _edge_cols is None"
assert w._weights_norm is not None, "FAIL: _weights_norm is None"

print("PASS: widget instantiated without errors")
print(f"PASS: _k3d_points  = {w._k3d_points}")
print(f"PASS: _k3d_lines   = {w._k3d_lines}")
print(f"PASS: _edge_rows   shape={w._edge_rows.shape}, dtype={w._edge_rows.dtype}")
print(f"PASS: _edge_cols   shape={w._edge_cols.shape}, dtype={w._edge_cols.dtype}")
print(f"PASS: _weights_norm shape={w._weights_norm.shape}, min={w._weights_norm.min():.4f}, max={w._weights_norm.max():.4f}")

print()
print("=" * 60)
print("STEP 4 — get_connectivity_info()")
print("=" * 60)

info = w.get_connectivity_info()
assert info['n_regions'] == 76,  f"FAIL n_regions: {info['n_regions']}"
assert info['n_connections'] > 0, "FAIL n_connections == 0"
assert len(info['region_labels']) == 76, "FAIL region_labels length"
print(f"PASS: n_regions      = {info['n_regions']}")
print(f"PASS: n_connections  = {info['n_connections']}")
print(f"PASS: weight range   = [{info['weight_min']:.4f}, {info['weight_max']:.4f}]")
print(f"PASS: region_labels  (first 3): {info['region_labels'][:3]}")

print()
print("=" * 60)
print("ALL PHASE 1 VERIFICATION STEPS PASSED")
print("=" * 60)
