from __future__ import annotations

import cli


def test_cyclopts():
    app = cli.App()

    @app.command
    def b():
        pass

    @app.command
    def a():
        pass

    app('-h')
    app('b')
    app('a')
