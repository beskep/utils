from .. import cli


def test_cyclopts():
    app = cli.App(result_action='return_none')

    @app.command
    def b():
        pass

    @app.command
    def a():
        pass

    app('-h')
    app('b')
    app('a')
