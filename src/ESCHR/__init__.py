from importlib.metadata import version

from . import pl, tl, _read_write_utils

__all__ = ["pl", "tl", "_read_write_utils"]

__version__ = version("ESCHR")
