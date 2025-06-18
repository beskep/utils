from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterable


class _MissingModule:
    def __init__(self, module: str, dependencies: Iterable[str]) -> None:
        self.module = module
        self.dependencies = sorted(dependencies)
        self.msg = f'Module `{module}` requires following packages: {self.dependencies}'

    def __getattr__(self, name: str) -> NoReturn:
        raise ImportError(self.msg)


if TYPE_CHECKING:
    from . import cli, mpl, pl, terminal
else:
    try:
        from . import cli
    except ImportError:
        cli = _MissingModule('cli', ['cyclopts'])

    try:
        from . import mpl
    except ImportError:
        mpl = _MissingModule('mpl', ['cmap', 'matplotlib', 'seaborn'])

    try:
        from . import pl
    except ImportError:
        pl = _MissingModule('pl', ['polars', 'pyarrow', 'whenever', 'xlsxwriter'])

    try:
        from . import terminal
    except ImportError:
        terminal = _MissingModule('terminal', ['loguru', 'rich'])


__all__ = ['cli', 'mpl', 'pl', 'terminal']
