"""Full k3d API details for improvement."""
import sys, inspect
sys.path.insert(0, r'C:\Users\mohit\OneDrive\Desktop\GSoC 2026\tvb-widgets-poc')
import k3d, numpy as np

print("=== k3d.points FULL SIGNATURE ===")
print(inspect.signature(k3d.points))

print()
print("=== k3d.text FULL SIGNATURE ===")
print(inspect.signature(k3d.text))

print()
print("=== k3d.label FULL SIGNATURE ===")
print(inspect.signature(k3d.label))

print()
print("=== test point_sizes array ===")
positions = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
sizes = np.array([4.0, 9.0, 16.0], dtype=np.float32)
try:
    p = k3d.points(positions=positions, point_sizes=sizes)
    print(f"point_sizes (array) ACCEPTED — type: {type(p.point_sizes)}")
except Exception as e:
    print(f"point_sizes REJECTED: {e}")

print()
print("=== test k3d.text creation ===")
try:
    t = k3d.text("Test", position=(0.0, 0.0, 0.0), color=0xffffff, size=0.5)
    print(f"k3d.text created OK — type: {type(t)}")
except Exception as e:
    print(f"k3d.text failed: {e}")
    # Try with label_box
    try:
        t = k3d.text("Test", position=(0.0, 0.0, 0.0), color=0xffffff, size=0.5, label_box=False)
        print(f"k3d.text with label_box=False OK")
    except Exception as e2:
        print(f"k3d.text with label_box=False also failed: {e2}")
