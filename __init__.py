from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from . import cli, mpl, pl, terminal
    from ._tqdm_rich import tqdm

__all__ = ['cli', 'mpl', 'pl', 'terminal', 'tqdm']


def __getattr__(name: str) -> ModuleType:
    if name == 'tqdm':
        module = importlib.import_module('._tqdm_rich', __name__)
        return module.tqdm

    if name in __all__:
        return importlib.import_module(f'.{name}', __name__)

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
