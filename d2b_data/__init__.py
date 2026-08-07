"""d2b_data — data framework for marketing APIs.

Submodules are imported explicitly (``from d2b_data.Google_GA4 import Google_GA4``)
so that installing the package never forces the import of every third-party SDK.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("d2b_data")
except PackageNotFoundError:  # el paquete se está usando sin instalar
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
