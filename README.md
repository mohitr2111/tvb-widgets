# tvb-widgets-poc

New 3D graphical widgets for The Virtual Brain JupyterLab ecosystem — GSoC 2026 Project #11 proof-of-concept.

---

## Overview

This repository contains two new widgets for the [`tvb-widgets`](https://github.com/the-virtual-brain/tvb-widgets)
ecosystem: a 3D connectivity viewer and an animated cortical surface renderer.
Both extend the existing `tvb-widgets` architecture — using the same `k3d`
rendering backend and `ipywidgets` control pattern — while adding new capabilities
requested in GSoC 2026 Project #11. Both widgets run locally using data bundled
with `tvb-data`; no EBRAINS account or `CLB_AUTH` token is required.

---

## Widgets

| Widget | TVB datatype | Key features |
|--------|-------------|--------------|
| `Connectivity3DWidget` | `Connectivity` | 3D region spheres at real MNI coordinates, weighted edges, live threshold / colormap / hemisphere / node-size controls |
| `AnimatedSurface3DWidget` | `CorticalSurface` | Cortical mesh (~16k vertices), animated per-vertex timeseries, Play / scrub / speed / colormap controls |

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/tvb-widgets-poc
cd tvb-widgets-poc
pip install -e .
jupyter lab
```

Open `notebooks/demo_widgets.ipynb` and press **Run All**.

---

## Usage

### Connectivity3DWidget

```python
from tvb.datatypes.connectivity import Connectivity
from tvbwidgets_poc import Connectivity3DWidget

conn = Connectivity.from_file()
conn.configure()
w = Connectivity3DWidget(conn)
display(w)
```

### AnimatedSurface3DWidget

```python
from tvb.datatypes.surfaces import CorticalSurface
from tvbwidgets_poc import AnimatedSurface3DWidget

surf = CorticalSurface.from_file()
surf.configure()
w = AnimatedSurface3DWidget(surf)   # synthetic timeseries auto-generated
display(w)
```

To animate real TVB simulation output:

```python
# ts must be (T, N_vertices) float32
ts = your_simulation_result.astype('float32')
w = AnimatedSurface3DWidget(surf, timeseries=ts)
display(w)
```

---

## Architecture

Both widgets follow the existing `tvb-widgets` multiple-inheritance pattern:
`ipywidgets.VBox` provides the DOM container; `TVBWidgetPOC` (defined in
`tvbwidgets_poc/base_widget.py`) provides shared constants, logging, and helper
utilities including `_to_float32()` (ensures k3d-compatible float32 vertex
arrays) and `_validate_connectivity()` (guards against malformed TVB data
objects). Each widget renders a `k3d.plot()` inside an `ipywidgets.Output()`
for correct JupyterLab DOM placement and exposes `add_datatype(HasTraits)` as
its public data-loading API.

---

## Tests

```bash
pytest tests/ -v
```

11 tests covering widget instantiation, data loading, callback correctness
(threshold, colormap, hemisphere, frame change, speed), and dtype guarantees
for k3d index and attribute arrays.

---

## Relation to tvb-widgets

This PoC extends the [`tvb-widgets`](https://github.com/the-virtual-brain/tvb-widgets)
ecosystem. `HeadWidget` and `SpaceTimeWidget` serve as architectural precedents —
both use the same `k3d` + `ipywidgets.Output` pattern adopted here. Both new
widgets are also designed for integration into
[`tvb-ext-xircuits`](https://github.com/the-virtual-brain/tvb-ext-xircuits):
their `add_datatype()` entry point maps directly to an xircuits input port,
following the established `PhasePlaneWidget` integration pattern.

---

## GSoC 2026

This repository is a proof-of-concept for
[GSoC 2026 Project #11](https://neurostars.org/t/gsoc-2026-project-11-the-virtual-brain-new-graphical-widget-s-for-jupyterlab/35570)
with The Virtual Brain / INCF.

Mentors: Lia Domide, Paula Prodan, Teodora Misan.
