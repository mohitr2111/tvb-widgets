# -*- coding: utf-8 -*-
#
# TVB Widgets PoC — GSoC 2026
# Base widget class for all TVB PoC widgets.
#
# Architectural note:
# We define TVBWidgetPOC(object) — distinct name from the installed tvbwidgets.TVBWidget —
# to prevent any import shadowing if both packages coexist in the same environment.
# Logging uses stdlib (logging.getLogger) rather than tvbwidgets.core.logger.builder
# to keep our package self-contained with no internal tvbwidgets dependencies.

import logging

import numpy
from ipywidgets import DOMWidget
from tvb.basic.neotraits.api import HasTraits


class TVBWidgetPOC(object):
    """
    Abstract base class for all TVB PoC widgets.

    Provides shared constants, logging setup, dtype helpers, and validation
    utilities that every concrete widget inherits.
    """

    # -----------------------------------------------------------------------
    # Layout / style constants — mirror the originals for visual consistency
    # -----------------------------------------------------------------------
    DEFAULT_BORDER = {
        'border': '2px solid lightgray',
        'padding': '10px',
    }

    BUTTON_STYLE = {
        'height': '40px',
        'width': '150px',
    }

    # -----------------------------------------------------------------------
    # Visual constants — deliberate improvements over the original grey bg
    # -----------------------------------------------------------------------
    # Dark navy background: more contrast against green brain-region nodes
    PLOT_BG_COLOR = 0x1a1a2e

    # Standard colormap options exposed across all widgets
    COLORMAP_OPTIONS = ['viridis', 'plasma', 'hot', 'coolwarm', 'rainbow']

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------
    def __init__(self, **kwargs):
        # Use stdlib logging so TVBWidgetPOC has zero dependency on
        # tvbwidgets internals — our package stays standalone.
        self.logger = logging.getLogger(self.__class__.__name__)

    # -----------------------------------------------------------------------
    # Widget interface
    # -----------------------------------------------------------------------
    def get_widget(self):
        """Return self if we are a DOMWidget, otherwise raise."""
        if isinstance(self, DOMWidget):
            return self
        self.logger.error(
            "Not a valid widget. Subclass must also inherit a DOMWidget "
            "(e.g. ipywidgets.VBox) or override get_widget()."
        )
        raise RuntimeWarning("Not a valid widget!")

    def add_datatype(self, datatype):
        # type: (HasTraits) -> None
        """Load a TVB datatype into the widget. Must be overridden."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement add_datatype(datatype)."
        )

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------
    def _validate_connectivity(self, connectivity):
        """
        Validate that *connectivity* is a usable TVB Connectivity object.

        Checks:
          - Not None
          - Has a 2-D numpy weights array
          - Has centres with shape (N, 3)

        Returns True if valid, False (with a logged error) if not.
        """
        if connectivity is None:
            self.logger.error("Connectivity is None — cannot render.")
            return False

        weights = getattr(connectivity, 'weights', None)
        if weights is None or not isinstance(weights, numpy.ndarray) or weights.ndim != 2:
            self.logger.error(
                "connectivity.weights must be a 2-D numpy array. "
                f"Got: {type(weights)}"
            )
            return False

        centres = getattr(connectivity, 'centres', None)
        if centres is None or not isinstance(centres, numpy.ndarray) or centres.ndim != 2 or centres.shape[1] != 3:
            self.logger.error(
                "connectivity.centres must be a numpy array of shape (N, 3). "
                f"Got shape: {getattr(centres, 'shape', 'N/A')}"
            )
            return False

        self.logger.debug(
            f"Connectivity valid: {centres.shape[0]} regions, "
            f"weights {weights.shape}."
        )
        return True

    # -----------------------------------------------------------------------
    # Array helpers
    # -----------------------------------------------------------------------
    def _to_float32(self, arr):
        """
        Convert *arr* to numpy float32.

        k3d requires float32 for all position / vertex arrays.
        Centralising this here ensures every widget uses the correct dtype
        without scattering .astype() calls throughout rendering code.
        """
        return numpy.asarray(arr, dtype=numpy.float32)
