from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "detector" / "GroundingDINO" / "groundingdino"

__path__ = [str(_PACKAGE_ROOT)]
