"""Load the three public Metrica sample games into one plain, challenge-like shape.

Every game comes back as a Game with:

    ball     float32 (T, 2)    metres, NaN where the tracking has no ball
    players  float32 (T, P, 2) metres, NaN where a slot is empty
    team     int8    (P,)      0 = home, 1 = away
    period   int8    (T,)      1 or 2
    events   DataFrame         team (0/1/-1), type, subtype, period, start, end (row indices)

Row i is tracking frame_id i + 1, the same convention the challenge uses (video frame =
frame_id - 1). Event frames are converted to rows the same way. Coordinates are put in metres
with x along the pitch (0..105) and y across it (0..68), using the raw Metrica orientation (y
grows downwards in the source; nothing here depends on its sign).

Tracking is read through kloppy (the Game 3 EPTS export maps 35 players onto 22 channels with a
different map in each half, which is not worth re-implementing). The result is cached as .npz.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths

PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)


@dataclass
class Game:
    game: int
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    period: np.ndarray
    events: pd.DataFrame

    @property
    def n_frames(self):
        return len(self.period)


# -- events ------------------------------------------------------------------------------------

def _subtype_name(value):
    """Game 3 JSON stores subtypes as null, one {name}, or a list of them. Join with '-' like the
    CSV games do, so 'ON TARGET-GOAL' means the same thing everywhere."""
    if not value:
        return ""
    if isinstance(value, dict):
        return value["name"]
    return "-".join(item["name"] for item in value)


def load_events(game, sample_dir=None):
    sample_dir = sample_dir or paths.METRICA_DIR
    folder = os.path.join(sample_dir, "Sample_Game_{}".format(game))
    if game in (1, 2):
        raw = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
        out = pd.DataFrame({
            "team": raw["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
            "type": raw["Type"].str.upper(),
            "subtype": raw["Subtype"].fillna("").str.upper(),
            "period": raw["Period"].astype(int),
            "start": raw["Start Frame"].astype(int) - 1,
            "end": raw["End Frame"].astype(int) - 1,
            "player": raw["From"].fillna(""),
        })
    else:
        with open(os.path.join(folder, "Sample_Game_3_events.json")) as handle:
            data = json.load(handle)["data"]
        home_name = "Team A"  # FIFATMA, the LocalTeam in the metadata
        out = pd.DataFrame({
            "team": [(0 if e["team"]["name"] == home_name else 1) if e.get("team") else -1
                     for e in data],
            "type": [e["type"]["name"].upper() for e in data],
            "subtype": [_subtype_name(e.get("subtypes")).upper() for e in data],
            "period": [int(e["period"]) for e in data],
            "start": [int(e["start"]["frame"]) - 1 for e in data],
            "end": [int(e["end"]["frame"]) - 1 if e["end"]["frame"] is not None
                    else int(e["start"]["frame"]) - 1 for e in data],
            "player": [(e.get("from") or {}).get("name", "") for e in data],
        })
    # A stable order: by start frame, keeping the provider's order within a frame.
    out = out.reset_index().sort_values(["start", "index"], kind="stable").drop(columns="index")
    return out.reset_index(drop=True)


# -- tracking ----------------------------------------------------------------------------------

def _load_tracking_kloppy(game, sample_dir):
    from kloppy import metrica
    folder = os.path.join(sample_dir, "Sample_Game_{}".format(game))
    if game in (1, 2):
        stem = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{{}}_Team.csv".format(game))
        dataset = metrica.load_tracking_csv(home_data=stem.format("Home"),
                                            away_data=stem.format("Away"))
    else:
        dataset = metrica.load_tracking_epts(
            meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
            raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"))
    return arrays_from_dataset(dataset, PITCH)


def arrays_from_dataset(dataset, pitch):
    """(ball, players, team, period) arrays in metres from any kloppy TrackingDataset.

    Row = frame_id - 1. Slots are every player kloppy lists, home team first; team 0 = the first
    team in the metadata (home, by the FIFA/EPTS convention the challenge also follows).
    """
    frame = dataset.to_df()

    ids, teams = [], []
    for team_index, team in enumerate(dataset.metadata.teams[:2]):
        for player in team.players:
            if "{}_x".format(player.player_id) in frame.columns:
                ids.append(player.player_id)
                teams.append(team_index)

    frame_ids = frame["frame_id"].to_numpy().astype(np.int64)
    n = int(frame_ids.max())
    rows = frame_ids - 1

    def column(name):
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)

    # kloppy's coordinate system flips y relative to the raw Metrica files; flip it back so both
    # the CSV and the EPTS games agree with the raw orientation. Then scale to metres.
    length, width = pitch
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[rows, 0] = column("ball_x") * length
    ball[rows, 1] = (1.0 - column("ball_y")) * width
    players = np.full((n, len(ids), 2), np.nan, dtype=np.float32)
    for slot, player_id in enumerate(ids):
        players[rows, slot, 0] = column("{}_x".format(player_id)) * length
        players[rows, slot, 1] = (1.0 - column("{}_y".format(player_id))) * width
    period = np.zeros(n, dtype=np.int8)
    period[rows] = frame["period_id"].to_numpy().astype(np.int8)
    return ball, players, np.array(teams, dtype=np.int8), period


def load(game, sample_dir=None, use_cache=True):
    """One sample game as a Game. The first call per game takes ~15-60 s; later calls ~1 s."""
    sample_dir = sample_dir or paths.METRICA_DIR
    cache = os.path.join(paths.CACHE_DIR, "sample_game_{}.npz".format(game))
    if use_cache and os.path.exists(cache):
        stored = np.load(cache)
        ball, players, team, period = (stored["ball"], stored["players"], stored["team"],
                                       stored["period"])
    else:
        ball, players, team, period = _load_tracking_kloppy(game, sample_dir)
        os.makedirs(paths.CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, ball=ball, players=players, team=team, period=period)
    return Game(game, ball, players, team, period, load_events(game, sample_dir))
