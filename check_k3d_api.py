"""Pre-code API checks for Improvement pass."""
import sys, inspect
sys.path.insert(0, r'C:\Users\mohit\OneDrive\Desktop\GSoC 2026\tvb-widgets-poc')

print("=" * 60)
print("CHECK 1 — k3d.points() point_size: scalar or array?")
print("=" * 60)
import k3d, numpy as np

sig = inspect.signature(k3d.points)
print("k3d.points signature:", sig)

# Try array point_size
positions = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
sizes = np.array([4.0, 9.0, 16.0], dtype=np.float32)
try:
    p = k3d.points(positions=positions, point_size=sizes)
    stored = p.point_size
    print(f"Array point_size: ACCEPTED — stored type: {type(stored)}, value: {stored}")
except Exception as e:
    print(f"Array point_size: REJECTED — {e}")

print()
print("=" * 60)
print("CHECK 2 — k3d.text() / k3d.label() existence")
print("=" * 60)

if hasattr(k3d, 'text'):
    print("k3d.text: EXISTS")
    try:
        sig2 = inspect.signature(k3d.text)
        print("k3d.text signature:", sig2)
    except Exception as e:
        print("k3d.text signature error:", e)
else:
    print("k3d.text: DOES NOT EXIST")

if hasattr(k3d, 'label'):
    print("k3d.label: EXISTS")
    try:
        sig3 = inspect.signature(k3d.label)
        print("k3d.label signature:", sig3)
    except Exception as e:
        print("k3d.label signature error:", e)
else:
    print("k3d.label: DOES NOT EXIST")

text_attrs = [a for a in dir(k3d) if 'text' in a.lower() or 'label' in a.lower()]
print("All k3d text/label attrs:", text_attrs)

print()
print("CHECK done.")
