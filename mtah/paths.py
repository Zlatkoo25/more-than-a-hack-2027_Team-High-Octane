"""Where things live. Every location can be overridden with an environment variable."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The Metrica sample games: the data/ folder of a clone of github.com/metrica-sports/sample-data.
METRICA_DIR = os.environ.get("MTAH_METRICA_DIR",
                             os.path.join(ROOT, "public_data", "metrica", "data"))

# The DFL games, one converted .npz each, written by `python -m mtah.dfl`.
DFL_DIR = os.environ.get("MTAH_DFL_DIR", os.path.join(ROOT, "public_data", "dfl"))

# Parsed Metrica tracking, rebuilt on demand.
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR", os.path.join(ROOT, "cache"))

# The trained models: scripts/train.py writes them, scripts/predict.py reads them.
MODEL_DIR = os.environ.get("MTAH_MODEL_DIR", os.path.join(ROOT, "models"))

# The organisers' repository, used only for its submission validator.
ORGANISERS_DIR = os.environ.get(
    "MTAH_ORGANISERS_DIR", os.path.join(os.path.dirname(ROOT), "more-than-a-hack-2027"))
