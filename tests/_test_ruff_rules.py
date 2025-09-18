import ruff_rules


def test_ruff():
    rr = ruff_rules.RuffRules()
    rr.print_settings()
    rr.print_linters()
