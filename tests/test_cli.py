from __future__ import annotations

import _cli


def test_cyclopts():
    app = _cli.App()

    @app.command
    def b():
        pass

    @app.command
    def a():
        pass

    app('-h')
    app('b')
    app('a')
