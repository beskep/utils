from __future__ import annotations

import json
import sysconfig
import tomllib
import warnings
from pathlib import Path
from subprocess import check_output
from typing import TYPE_CHECKING, Literal

import rich
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ['RuffRules', 'find_ruff_bin', 'ruff_linter', 'ruff_rule']


def find_ruff_bin() -> Path:
    exe = f'ruff{sysconfig.get_config_var("EXE")}'
    scripts = sysconfig.get_path('scripts')

    if (path := Path(scripts) / exe).is_file():
        return path

    scheme = sysconfig.get_preferred_scheme('user')
    scripts = sysconfig.get_path('scripts', scheme=scheme)
    if (path := Path(scripts) / exe).is_file():
        return path

    raise FileNotFoundError(path)


def _find_ruff_bin() -> Path | Literal['ruff']:
    try:
        return find_ruff_bin()
    except OSError as e:
        logger.warning(str(e))

    return 'ruff'


def ruff_linter() -> Iterable[tuple[str, str]]:
    try:
        out = check_output([_find_ruff_bin(), 'linter'])
    except OSError as e:
        msg = '`ruff linter`를 실행할 수 없음'
        raise RuntimeError(msg) from e

    for linter in out.decode().split('\n'):
        if not linter:
            continue

        code, desc = linter.strip().split(' ', maxsplit=1)
        for c in code.split('/'):
            yield c, desc


def ruff_rule() -> Iterable[tuple[str, str, str]]:
    try:
        out = check_output([
            _find_ruff_bin(),
            'rule',
            '--all',
            '--output-format',
            'json',
        ]).decode()
    except OSError as e:
        msg = '`ruff rule`을 실행할 수 없음'
        raise RuntimeError(msg) from e

    for rule in json.loads(out):
        yield rule['code'], rule['linter'], rule['name']


class RuffRules:
    SETTINGS = (
        'select',
        'ignore',
        'fixable',
        'unfixable',
        'extend-select',
        'extend-fixable',
        'extend-safe-fixes',
        'extend-unsafe-fixes',
    )

    def __init__(self) -> None:
        self._linter = dict(ruff_linter())
        self._rules = self._linter | {x[0]: f'{x[1]}: {x[2]}' for x in ruff_rule()}

    def describe(self, code: str) -> str | None:
        if code == 'ALL':
            return None

        if d := self._rules.get(code, None):
            return d

        for k, v in self._linter.items():
            if code.startswith(k):
                return f'{v}: {code}*'

        return None

    @staticmethod
    def from_toml(d: str | Path | None = None, /) -> dict | None:
        if d is None:
            d = Path.cwd() / 'pyproject.toml'

        s = d if isinstance(d, str) else d.read_text()
        config = tomllib.loads(s)

        try:
            return config['tool']['ruff']['lint']  # pyproject.toml
        except KeyError:
            pass

        try:
            return config['lint']  # ruff.toml
        except KeyError:
            pass

        return None

    @staticmethod
    def fmt(code: str) -> str:
        return f'    "{code}",'

    def annotate(
        self,
        key: str | None = None,
        codes: Iterable[str] | None = None,
    ) -> Iterable[str]:
        if codes is None:
            codes = self._linter

        yield '[' if key is None else f'{key} = ['

        w = max(len(x) for x in codes) + len(self.fmt(''))
        for code in sorted(codes):
            c = self.fmt(code)
            if (desc := self.describe(code)) is None:
                yield c
            else:
                yield f'{c:<{w}}  # {desc}'

        yield ']'

    def print_linters(self) -> None:
        console = rich.get_console()
        for line in self.annotate():
            console.print(line)

    def print_settings(self, toml: str | Path | None = None) -> None:
        console = rich.get_console()

        if (config := self.from_toml(toml)) is None:
            warnings.warn('ruff 세팅을 찾을 수 없음', stacklevel=2)
            return

        for key in self.SETTINGS:
            if key not in config:
                continue

            for line in self.annotate(key, config[key]):
                console.print(line)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-s',
        '--setting',
        action='store_const',
        dest='cmd',
        const='setting',
        default='setting',
    )
    group.add_argument(
        '-l',
        '--linter',
        action='store_const',
        dest='cmd',
        const='linter',
    )

    ruff = RuffRules()
    match cmd := parser.parse_args().cmd:
        case 'setting':
            ruff.print_settings()
        case 'linter':
            ruff.print_linters()
        case _:
            raise ValueError(cmd)
