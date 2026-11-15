"""The organisers' own submission validator (in evaluation/possession_metrics.py), imported from
a clone of their repository."""

import importlib.util
import os

from . import paths

_PM = None


def metrics_module():
    global _PM
    if _PM is None:
        path = os.path.join(paths.ORGANISERS_DIR, "evaluation", "possession_metrics.py")
        spec = importlib.util.spec_from_file_location("possession_metrics", path)
        _PM = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PM)
    return _PM
