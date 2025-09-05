"""Terminal frontend package with Rich/Textual interface."""

from .client import TerminalClient
from .components import *
from .widgets import *

__all__ = ["TerminalClient"]