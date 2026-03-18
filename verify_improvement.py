"""Verification for Improvement pass — hub sizing + label toggle."""
import sys
sys.path.insert(0, r'C:\Users\mohit\OneDrive\Desktop\GSoC 2026\tvb-widgets-poc')

from tvb.datatypes.connectivity import Connectivity
from tvbwidgets_poc import Connectivity3DWidget
import numpy as np

print("=" * 60)
print("IMPROVEMENT VERIFICATION")
print("=" * 60)

conn = Connectivity.from_file()
conn.configure()
w = Connectivity3DWidget(conn)

print()
print("--- Hub sizing check ---")
print(f"_node_sizes dtype:  {w._node_sizes.dtype}")
print(f"_node_sizes range:  {w._node_sizes.min():.2f} – {w._node_sizes.max():.2f}")
assert w._node_sizes.min() >= 3.9, f"Min size too low: {w._node_sizes.min()}"
assert w._node_sizes.max() <= 18.1, f"Max size too high: {w._node_sizes.max()}"
assert not np.allclose(w._node_sizes, w._node_sizes[0]), "All sizes are identical — hub sizing not working"
print("PASS: node sizes vary [~4, ~18] — hub regions are larger")

print()
print("--- Node colors check ---")
print(f"_k3d_points.colors dtype: {np.array(w._k3d_points.colors).dtype}")
print(f"_k3d_points.colors len:   {len(np.array(w._k3d_points.colors))}")
assert np.array(w._k3d_points.colors).dtype == np.uint32, "Node colors not uint32"
assert len(np.array(w._k3d_points.colors)) == 76, "Node colors length != 76"
print("PASS: node colors are uint32, length 76 (viridis-mapped)")

print()
print("--- Label toggle check ---")
w._on_label_toggle({'new': True})
print(f"Labels added: {len(w._k3d_labels)}")
assert len(w._k3d_labels) == 10, f"Expected 10 labels, got {len(w._k3d_labels)}"
print("PASS: 10 hub labels added")

w._on_label_toggle({'new': False})
print(f"Labels cleared: {len(w._k3d_labels)}")
assert len(w._k3d_labels) == 0, f"Expected 0 labels after clear, got {len(w._k3d_labels)}"
print("PASS: labels cleared to 0")

print()
print("--- Node size slider scales relative hub sizes ---")
w._on_node_size_change({'new': 16.0})  # 2x multiplier from baseline 8.0
scaled = np.array(w._k3d_points.point_sizes)
print(f"After 2x scale — sizes range: {scaled.min():.2f} – {scaled.max():.2f}")
assert not np.allclose(scaled, scaled[0]), "Scaled sizes lost hub hierarchy"
print("PASS: hub hierarchy preserved after size slider")

print()
print("=" * 60)
print("ALL IMPROVEMENT VERIFICATIONS PASSED")
print("=" * 60)
