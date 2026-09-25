"""Local browser studio for every TeoTensor model.

``open_studio()`` serves one page on localhost. The page loads tables, edits
``get_params``, fits, and plays films when the model provides them. The math
stays in the model.
"""

from __future__ import annotations

from teotensor.studio.server import open_studio

__all__ = ["open_studio"]
