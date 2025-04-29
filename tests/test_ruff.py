from __future__ import annotations

import _ruff


def test_ruff():
    rr = _ruff.RuffRules()
    rr.print_settings()
    rr.print_linters()
