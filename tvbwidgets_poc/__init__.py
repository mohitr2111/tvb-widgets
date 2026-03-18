# -*- coding: utf-8 -*-
#
# TVB Widgets PoC — GSoC 2026
#
"""
tvbwidgets_poc — public API

Exposes the Phase 1 widget at the package level so users can import as:

    from tvbwidgets_poc import Connectivity3DWidget
"""

from tvbwidgets_poc.connectivity3d import Connectivity3DWidget
from tvbwidgets_poc.surface3d import AnimatedSurface3DWidget

__all__ = [
    "Connectivity3DWidget",
    "AnimatedSurface3DWidget",
]
