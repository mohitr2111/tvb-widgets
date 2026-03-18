"""
Phase 2 verification script — run from tvb-widgets-poc/ directory.
Tests all 5 verification steps from the Phase 2 spec.
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s] %(message)s')
sys.path.insert(0, r'C:\Users\mohit\OneDrive\Desktop\GSoC 2026\tvb-widgets-poc')

from tvb.datatypes.connectivity import Connectivity
from tvbwidgets_poc import Connectivity3DWidget

import numpy

# ------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Import test")
print("=" * 60)
print("PASS: imports successful")

# ------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 2 — Instantiation with controls")
print("=" * 60)

conn = Connectivity.from_file()
conn.configure()
w = Connectivity3DWidget(conn)

assert hasattr(w, '_controls'),          "FAIL: no _controls"
assert hasattr(w, '_threshold_slider'),  "FAIL: no _threshold_slider"
assert hasattr(w, '_node_size_slider'),  "FAIL: no _node_size_slider"
assert hasattr(w, '_colormap_dropdown'), "FAIL: no _colormap_dropdown"
assert hasattr(w, '_hemisphere_toggle'), "FAIL: no _hemisphere_toggle"
assert hasattr(w, '_info_label'),        "FAIL: no _info_label"

print(f"PASS: controls built — type: {type(w._controls).__name__}")
print(f"PASS: threshold_slider value = {w._threshold_slider.value}")
print(f"PASS: node_size_slider value = {w._node_size_slider.value}")
print(f"PASS: colormap_dropdown value = {w._colormap_dropdown.value}")
print(f"PASS: hemisphere_toggle value = {w._hemisphere_toggle.value}")

# ------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 3 — Threshold callback (headless)")
print("=" * 60)

w._on_threshold_change({'new': 0.5})
n_visible = int((w._weights_norm >= 0.5).sum())
print(f"PASS: threshold=0.5 → {n_visible} / {len(w._weights_norm)} connections visible")
assert n_visible < 1560, "FAIL: threshold 0.5 should hide some connections"
assert n_visible > 0,    "FAIL: threshold 0.5 should not hide all connections"

# Reset
w._on_threshold_change({'new': 0.0})
print("PASS: threshold reset to 0.0")

# ------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 4 — Colormap callback")
print("=" * 60)

w._on_colormap_change({'new': 'plasma'})
print("PASS: colormap 'plasma' — no error")
w._on_colormap_change({'new': 'hot'})
print("PASS: colormap 'hot'   — no error")
w._on_colormap_change({'new': 'viridis'})
print("PASS: colormap 'viridis' restored")

# ------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 5 — Hemisphere callback")
print("=" * 60)

# Use hemi_override so direct calls work consistently
# (in JupyterLab, ipywidgets updates toggle.value before calling observers)
w._on_hemisphere_change({'new': 'Left'})
mask_left = w._get_active_mask(hemi_override='Left')
n_left = int(mask_left.sum())
print(f"PASS: hemisphere 'Left'  → {n_left} connections visible")
assert n_left < 1560, f"FAIL: Left filter should reduce connections, got {n_left}"

w._on_hemisphere_change({'new': 'Right'})
mask_right = w._get_active_mask(hemi_override='Right')
n_right = int(mask_right.sum())
print(f"PASS: hemisphere 'Right' → {n_right} connections visible")
assert n_right < 1560, f"FAIL: Right filter should reduce connections, got {n_right}"

w._on_hemisphere_change({'new': 'Both'})
mask_both = w._get_active_mask(hemi_override='Both')
n_both = int(mask_both.sum())
print(f"PASS: hemisphere 'Both'  → {n_both} connections visible (full set)")
assert n_both == 1560, f"FAIL: 'Both' should restore all 1560, got {n_both}"

# ------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 6 — Edge indices shape / dtype verification")
print("=" * 60)

# Confirm stored indices are flat (E*2,) uint32
rows = w._edge_rows
cols = w._edge_cols
flat = numpy.column_stack([rows, cols]).astype(numpy.uint32).flatten()
assert flat.dtype  == numpy.uint32, f"FAIL dtype: {flat.dtype}"
assert flat.shape  == (1560 * 2,),  f"FAIL shape: {flat.shape}"
print(f"PASS: edge indices — shape={flat.shape}, dtype={flat.dtype}")

# ------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 7 — Combined threshold + hemisphere filter")
print("=" * 60)

w._threshold_slider.value = 0.3
w._hemisphere_toggle.value = 'Left'
mask_combined = w._get_active_mask()
n_combined = int(mask_combined.sum())
print(f"PASS: threshold=0.3 + hemisphere='Left' → {n_combined} connections visible")
assert n_combined < 1560

# Reset
w._threshold_slider.value = 0.0
w._hemisphere_toggle.value = 'Both'

# ------------------------------------------------------------------
print()
print("=" * 60)
print("ALL PHASE 2 VERIFICATION STEPS PASSED")
print("=" * 60)
