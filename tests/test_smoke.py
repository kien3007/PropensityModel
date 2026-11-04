"""Offline package smoke test."""
from tecd_propensity import __version__
from tecd_propensity.__main__ import main


def test_smoke(capsys):
    main()
    assert capsys.readouterr().out.strip() == f"tecd-propensity {__version__}"
