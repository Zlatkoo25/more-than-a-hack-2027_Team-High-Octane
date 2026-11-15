"""Seven more public games: the DFL / IDSSE Bundesliga 2022-23 open dataset (CC BY 4.0).

    Bassek, M., Rein, R., Weber, H. et al. An integrated dataset of spatiotemporal and event data
    in elite soccer. Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

    python -m mtah.dfl            # download, convert and cache all seven (about 2.6 GB of traffic)

Each game comes back in the same shape as metrica.Game (metres, 25 Hz, NaN where absent), plus a
per-frame truth: the provider's own BallStatus (0 = dead) and BallPossession (1 = home, 2 = away)
flags, which human operators set. That is a different labelling process from the Metrica events,
which is the point: a second convention and seven more games.

The raw position files are 350-420 MB each. Each is streamed to a temporary file, parsed line by
line, saved as a small .npz under public_data/dfl/, and deleted.
"""

import os
import re
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import H, A, D, paths
from .decode import merge_short_spells

GAMES = {  # match id -> figshare file ids (meta, tracking)
    "J03WMX": (51643475, 51643514),   # Koeln - Bayern (Bundesliga)
    "J03WN1": (51643472, 51643517),   # Bochum - Leverkusen (Bundesliga)
    "J03WPY": (51643487, 51643526),   # Duesseldorf - Nuernberg (2. Bundesliga)
    "J03WOH": (51643478, 51643520),   # Duesseldorf - Regensburg
    "J03WQQ": (51643484, 51643529),   # Duesseldorf - St. Pauli
    "J03WOY": (51643481, 51643523),   # Duesseldorf - Rostock
    "J03WR9": (51643490, 51643532),   # Duesseldorf - Kaiserslautern
}
URL = "https://ndownloader.figshare.com/files/{}"
DIR = paths.DFL_DIR
PITCH = (105.0, 68.0)

_SET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"')
_BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
_TEAM = re.compile(r'<Team [^>]*>')


@dataclass
class Game:
    game: str
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    period: np.ndarray
    truth_raw: np.ndarray      # the provider's flags as H/A/D

    @property
    def n_frames(self):
        return len(self.period)


def _fetch(file_id, target, tries=30):
    """Download with a timeout, resuming where a stalled connection left off."""
    done = 0
    open(target, "wb").close()
    for _ in range(tries):
        headers = {"User-Agent": "python-urllib"}
        if done:
            headers["Range"] = "bytes=%d-" % done
        try:
            request = urllib.request.Request(URL.format(file_id), headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                if done and response.status != 206:
                    done = 0
                with open(target, "r+b") as out:
                    out.seek(done)
                    out.truncate()
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            return
                        out.write(block)
                        done += len(block)
        except OSError:
            continue
    raise OSError("download of figshare file %s kept failing" % file_id)


def _home_away(meta_path):
    with open(meta_path, encoding="utf-8") as handle:
        text = handle.read()
    roles = {}
    for tag in _TEAM.findall(text):
        team_id = re.search(r'TeamId="([^"]+)"', tag)
        role = re.search(r'Role="(\w+)"', tag)
        if team_id and role:
            roles[team_id.group(1)] = 0 if role.group(1).lower() == "home" else 1
    if sorted(roles.values()) != [0, 1]:
        raise ValueError("could not read home/guest teams from " + meta_path)
    return roles


def _parse(track_path, roles):
    """Stream the positions file. Returns the arrays of a Game."""
    sets = {}                     # (section, person) -> (team, [N], [x], [y], [poss], [status])
    current = None
    with open(track_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FRAME.match(line)
                current[1].append(int(m.group(1)))
                current[2].append(float(m.group(2)))
                current[3].append(float(m.group(3)))
                if current[0] == "BALL":
                    b = _BALL.search(line)
                    current[4].append(int(b.group(1)))
                    current[5].append(int(b.group(2)))
            elif line.startswith("<FrameSet "):
                m = _SET.match(line)
                current = (m.group(2), [], [], [], [], [])
                sets[(m.group(1), m.group(3))] = current

    sections = [s for s in ("firstHalf", "secondHalf") if any(k[0] == s for k in sets)]
    span = {}
    for s in sections:
        ball = next(v for k, v in sets.items() if k[0] == s and v[0] == "BALL")
        span[s] = (min(ball[1]), max(ball[1]) + 1)
    offset, total = {}, 0
    for s in sections:
        offset[s] = total
        total += span[s][1] - span[s][0]

    persons = sorted({(roles[v[0]], k[1]) for k, v in sets.items() if v[0] in roles})
    slot = {person: i for i, (_, person) in enumerate(persons)}
    team = np.array([t for t, _ in persons], dtype=np.int8)
    ball = np.full((total, 2), np.nan, dtype=np.float32)
    players = np.full((total, len(persons), 2), np.nan, dtype=np.float32)
    period = np.zeros(total, dtype=np.int8)
    truth = np.full(total, D, dtype=np.int8)
    for (s, person), (team_id, n, x, y, poss, status) in sets.items():
        lo, hi = span[s]
        n = np.asarray(n)
        keep = (n >= lo) & (n < hi)
        rows = offset[s] + n[keep] - lo
        xy = np.column_stack([np.asarray(x)[keep] + PITCH[0] / 2, np.asarray(y)[keep] + PITCH[1] / 2])
        if team_id == "BALL":
            ball[rows] = xy
            period[rows] = 1 + sections.index(s)
            poss, status = np.asarray(poss)[keep], np.asarray(status)[keep]
            truth[rows] = np.where(status == 0, D, np.where(poss == 1, H, A))
        elif team_id in roles:           # some files also carry the referees
            players[rows, slot[person]] = xy
    return ball, players, team, period, truth


def load(game):
    """One DFL game, converted on first use."""
    cache = os.path.join(DIR, game + ".npz")
    if not os.path.exists(cache):
        os.makedirs(DIR, exist_ok=True)
        meta_id, track_id = GAMES[game]
        # Staged next to the output, not in the system temp folder, and removed even when the
        # run fails; a run that is killed outright leaves them here, and the next run deletes them.
        meta = os.path.join(DIR, "_download_meta.xml")
        track = os.path.join(DIR, "_download_positions.xml")
        try:
            _fetch(meta_id, meta)
            _fetch(track_id, track)
            ball, players, team, period, truth = _parse(track, _home_away(meta))
        finally:
            for f in (meta, track):
                if os.path.exists(f):
                    os.remove(f)
        np.savez_compressed(cache, ball=ball, players=players, team=team, period=period,
                            truth=truth)
    z = np.load(cache)
    return Game(game, z["ball"], z["players"], z["team"], z["period"], z["truth"])


def truth(G, min_hold=50):
    """H/A/D truth in the challenge's spirit: a spell that takes the ball off the other team but
    lasts under min_hold frames goes back to the previous team (no control). min_hold=0 keeps the
    provider's flags as they are."""
    if not min_hold:
        return G.truth_raw
    return merge_short_spells(G.truth_raw, G.period, min_hold)


if __name__ == "__main__":
    for name in GAMES:
        g = load(name)
        y = g.truth_raw
        print("%s  frames %d  slots %d  ball seen %.3f  H %.3f A %.3f D %.3f" % (
            name, g.n_frames, g.players.shape[1], float(np.mean(~np.isnan(g.ball[:, 0]))),
            np.mean(y == H), np.mean(y == A), np.mean(y == D)))
        sys.stdout.flush()
