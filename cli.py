"""cyclopts.App"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, overload

import attrs
import cyclopts

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    'REGISTERED_ORDER',
    'REMOVE_PREFIX',
    'App',
    'RegisteredOrder',
    'RemovePrefix',
]

F = TypeVar('F', bound=Callable)


def _is_helper(name: str | Iterable[str] | None) -> bool:
    if name is None:
        return False

    names = [name] if isinstance(name, str) else name
    return any(n in {'-h', '--help', '--version'} for n in names)


class RegisteredOrder(enum.Enum):
    REGISTERED_ORDER = 0


class RemovePrefix(enum.Enum):
    REMOVE_PREFIX = 0


REGISTERED_ORDER = RegisteredOrder.REGISTERED_ORDER
REMOVE_PREFIX = RemovePrefix.REMOVE_PREFIX


@attrs.define
class App(cyclopts.App):
    _count: itertools.count = attrs.field(factory=itertools.count)

    def _remove_prefix(self, s: str) -> str:
        t = s.removeprefix(self.name[0]) if self.name else s
        t = cyclopts.default_name_transform(t)
        return t or cyclopts.default_name_transform(s)

    @overload  # type: ignore[override]
    def command(
        self,
        obj: F,
        name: str | Iterable[str] | None = ...,
        sort_key: Any = ...,  # noqa: ANN401
        name_transform: Callable[[str], str] | RemovePrefix | None = ...,
        **kwargs: object,
    ) -> F: ...

    @overload
    def command(
        self,
        obj: None = None,
        name: str | Iterable[str] | None = ...,
        sort_key: Any = ...,  # noqa: ANN401
        name_transform: Callable[[str], str] | RemovePrefix | None = ...,
        **kwargs: object,
    ) -> Callable[[F], F]: ...

    def command(
        self,
        obj: Callable | None = None,
        name: str | Iterable[str] | None = None,
        sort_key: Any = REGISTERED_ORDER,
        name_transform: Callable[[str], str] | RemovePrefix | None = REMOVE_PREFIX,
        **kwargs: object,
    ):
        if _is_helper(name):
            sort_key = None
        elif sort_key is REGISTERED_ORDER:
            sort_key = next(self._count)

        if isinstance(obj, cyclopts.App):
            obj.sort_key = sort_key
        else:
            if name_transform is REMOVE_PREFIX:
                name_transform = self._remove_prefix

            kwargs['sort_key'] = sort_key
            kwargs['name_transform'] = name_transform

        return super().command(obj=obj, name=name, **kwargs)
