"""Read the challenge match's tracking (Metrica FIFA/EPTS export) into the same arrays as the
sample games. Paths come from the command line or MTAH_DATA_DIR; nothing is copied anywhere.

Facts used here are from the organisers' public documentation (DATA.md, games.py,
starter/load_atd.py), not from the data:
  * 199319 records, frame_id 1..199319, video frame = frame_id - 1
  * pitch 105.3 x 68.0 m
  * frame 100000 is halftime (no play); the first half kicks off at 11185, the second at 113742
"""

import os

import numpy as np

from .metrica import arrays_from_dataset

PITCH = (105.3, 68.0)
N_FRAMES = 199319
HALFTIME_FRAME = 100000
KICKOFF_1, KICKOFF_2 = 11185, 113742
# docs/PRIMER.md, 'Frames that don't count': only these spans are scored, and the second one
# starts 2 minutes before the kick-off at KICKOFF_2, all of it D.
SCORED = ((11185, 90652), (110808, 190198))
META = os.path.join("tactical", "BAR-ATH_tactical_atd.xml")
RAW = os.path.join("tactical", "BAR-ATH_tactical_atd.txt")


def paths(data_dir=None):
    data_dir = data_dir or os.environ.get("MTAH_DATA_DIR")
    if not data_dir:
        raise SystemExit("set MTAH_DATA_DIR to the folder holding tactical/ (or pass --data)")
    meta, raw = os.path.join(data_dir, META), os.path.join(data_dir, RAW)
    for path in (meta, raw):
        if not os.path.exists(path):
            raise SystemExit("not found: " + path)
    return meta, raw


def load(data_dir=None):
    """(ball, players, team, period) for the challenge match, row = video frame."""
    from kloppy import metrica
    meta, raw = paths(data_dir)
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    ball, players, team, _ = arrays_from_dataset(dataset, PITCH)
    if len(ball) < N_FRAMES:  # trailing frames with no record: pad
        pad = N_FRAMES - len(ball)
        ball = np.concatenate([ball, np.full((pad, 2), np.nan, ball.dtype)])
        players = np.concatenate([players, np.full((pad,) + players.shape[1:], np.nan,
                                                   players.dtype)])
    ball, players = ball[:N_FRAMES], players[:N_FRAMES]
    period = np.where(np.arange(N_FRAMES) < HALFTIME_FRAME, 1, 2).astype(np.int8)
    return ball, players, team, period
