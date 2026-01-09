from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from . import cli, mpl, pl, terminal  # ty: ignore[unresolved-import]
    from .tqdm_rich import tqdm_rich as tqdmr  # ty: ignore[unresolved-import]

__all__ = ['cli', 'mpl', 'pl', 'terminal', 'tqdmr']


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return importlib.import_module(f'.{name}', __name__)

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
