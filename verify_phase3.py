"""
Phase 3 verification script — run from tvb-widgets-poc/ directory.
Tests all 7 verification steps from the Phase 3 spec.
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s] %(message)s')
sys.path.insert(0, r'C:\Users\mohit\OneDrive\Desktop\GSoC 2026\tvb-widgets-poc')

from tvb.datatypes.surfaces import CorticalSurface
import numpy

print("=" * 60)
print("STEP 1 — Surface load test (TQ3 confirmation)")
print("=" * 60)

s = CorticalSurface.from_file()
s.configure()
print(f"type:             {type(s)}")
print(f"vertices.shape:   {s.vertices.shape}")
print(f"triangles.shape:  {s.triangles.shape}")
print(f"vertices.dtype:   {s.vertices.dtype}")
print(f"triangles.dtype:  {s.triangles.dtype}")
assert s.vertices.shape[1] == 3,    "FAIL: vertices should be (N, 3)"
assert s.triangles.shape[1] == 3,   "FAIL: triangles should be (M, 3)"
print("PASS: surface loaded successfully")

print()
print("=" * 60)
print("STEP 7 — Import test (__init__.py update)")
print("=" * 60)
from tvbwidgets_poc import AnimatedSurface3DWidget
print("PASS: from tvbwidgets_poc import AnimatedSurface3DWidget")

print()
print("=" * 60)
print("STEP 2 — Synthetic timeseries")
print("=" * 60)

w = AnimatedSurface3DWidget()
w.add_datatype(s)
ts = w.timeseries
print(f"ts.shape:  {ts.shape}")
print(f"ts.dtype:  {ts.dtype}")
print(f"ts.min():  {ts.min():.4f}")
print(f"ts.max():  {ts.max():.4f}")
assert ts.shape == (120, s.vertices.shape[0]), f"FAIL shape: {ts.shape}"
assert ts.dtype == numpy.float32,              f"FAIL dtype: {ts.dtype}"
assert ts.min() > -1.25,                       "FAIL: min out of expected range"
assert ts.max() <  1.25,                        "FAIL: max out of expected range"
print("PASS: timeseries shape, dtype, range all correct")

print()
print("=" * 60)
print("STEP 3 — Mesh render test")
print("=" * 60)

assert w._k3d_mesh is not None,              "FAIL: _k3d_mesh is None"
print(f"type(_k3d_mesh):  {type(w._k3d_mesh)}")
print(f"attribute.shape:  {w._k3d_mesh.attribute.shape}")
print(f"attribute.dtype:  {w._k3d_mesh.attribute.dtype}")
assert w._k3d_mesh.attribute.shape == (s.vertices.shape[0],), "FAIL: attribute shape"
# k3d internally coerces float32 — dtype check is informational
print("PASS: mesh rendered with correct attribute shape")

print()
print("=" * 60)
print("STEP 4 — Frame change test (headless)")
print("=" * 60)

w._on_frame_change({'new': 60})
assert w._current_frame == 60,                          "FAIL: _current_frame not updated"
assert w._k3d_mesh.attribute.shape == (s.vertices.shape[0],), "FAIL: attribute shape after frame change"
print(f"Frame 60 attribute range: [{w._k3d_mesh.attribute.min():.4f}, {w._k3d_mesh.attribute.max():.4f}]")
print("PASS: frame change updates current_frame and mesh attribute")

print()
print("=" * 60)
print("STEP 5 — Controls test")
print("=" * 60)

assert hasattr(w, '_play'),              "FAIL: no _play"
assert hasattr(w, '_frame_slider'),      "FAIL: no _frame_slider"
assert hasattr(w, '_colormap_dropdown'), "FAIL: no _colormap_dropdown"
assert w._play.disabled == False,        "FAIL: _play still disabled after data load"
assert w._frame_slider.disabled == False,"FAIL: _frame_slider still disabled"
print(f"PASS: _play.disabled={w._play.disabled}, _frame_slider.disabled={w._frame_slider.disabled}")
print("PASS: all control attributes present and enabled")

print()
print("=" * 60)
print("STEP 6 — Speed change test")
print("=" * 60)

w._on_speed_change({'new': 2.0})
expected_interval = int(83 / 2.0)
print(f"Speed 2x → interval = {w._play.interval} ms/frame (expected ~{expected_interval})")
assert w._play.interval == expected_interval, f"FAIL: interval={w._play.interval}, expected {expected_interval}"
print("PASS: speed change correctly mutates Play.interval")

print()
print("=" * 60)
print("STEP 7b — Colormap change test")
print("=" * 60)

for cm in ['RdBu', 'seismic', 'bwr', 'PRGn', 'coolwarm']:
    w._on_colormap_change({'new': cm})
    print(f"PASS: colormap '{cm}' — no error")

print()
print("=" * 60)
print("ALL PHASE 3 VERIFICATION STEPS PASSED")
print("=" * 60)
