"""Damage clean public tracking so that it looks like automatic, uncorrected tracking.

The organisers' DATA.md says the match's ball is missing on about 41% of graded frames and that
players come as anonymous tracklets with a team. How the missing ball splits between live and
dead play, and how long the gaps are, is not stated. So every training copy of a game gets its
own random scenario (draw), and the models learn to work across the whole family:

    alive_rate   fraction of live-ball frames to hide (in bursts)
    burst_mean   mean burst length in frames (geometric lengths)
    speed_bias   0 = bursts start anywhere; 1 = burst starts weighted by ball speed, standing in
                 for lofted balls and fast passes that optical tracking loses
    noise_m     Gaussian noise added to the visible ball position, in metres
    player_drop  fraction of player-frames to hide, in bursts of ~2 s (tracklet breaks)
    dead_rate    fraction of visible dead-ball frames to hide (DFL games keep the ball while dead)

Dead-ball frames stay as the samples have them: mostly hidden, sometimes visible (a ball carried
back for a restart). Revealing a hidden dead ball is impossible, as there is no position to show.

The live/dead truth used to place the bursts comes from the event-derived labels; that is fine
because the simulation only produces inputs, never answers.
"""

from dataclasses import dataclass

import numpy as np

from . import D

FPS = 25


@dataclass(frozen=True)
class Scenario:
    name: str
    alive_rate: float = 0.0
    burst_mean: float = 50.0
    speed_bias: float = 0.0
    noise_m: float = 0.0
    player_drop: float = 0.0
    tracklet_s: float = 0.0     # 0 keeps identities; else mean tracklet length in seconds
    team_noise: float = 0.0     # fraction of tracklets put in a slot of the wrong team
    dead_rate: float = 0.0      # fraction of visible dead-ball frames to hide (for sources that
                                # keep tracking the ball while it is dead, unlike Metrica)
    seed: int = 0


def draw(rng, seed):
    """A random scenario for one training copy of a game: how much live play is hidden, burst
    length, ball noise, player gaps, and whether players come as anonymous tracklets."""
    tracklets = rng.random() < 0.5
    return Scenario(
        "random", alive_rate=rng.uniform(0.0, 0.5),
        burst_mean=float(np.exp(rng.uniform(np.log(20), np.log(250)))),
        speed_bias=rng.uniform(0.0, 1.0), noise_m=rng.uniform(0.0, 1.5),
        player_drop=rng.uniform(0.0, 0.08), tracklet_s=60.0 if tracklets else 0.0,
        team_noise=rng.uniform(0.0, 0.08) if tracklets else 0.0, seed=seed)


def _ball_speed(ball):
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    step = np.concatenate(([0.0], step))
    return np.nan_to_num(step, nan=0.0)


def _fill(values):
    """Linear interpolation across NaN, flat at the ends."""
    ok = ~np.isnan(values)
    if not ok.any():
        return np.full(len(values), 52.5)
    idx = np.arange(len(values))
    return np.interp(idx, idx[ok], values[ok])


def _bursts(mask_allowed, target_frames, mean_len, weights, rng):
    """Boolean mask with about target_frames True, in geometric bursts confined to mask_allowed."""
    n = len(mask_allowed)
    hidden = np.zeros(n, dtype=bool)
    if target_frames <= 0:
        return hidden
    candidates = np.flatnonzero(mask_allowed)
    if not len(candidates):
        return hidden
    p = weights[candidates]
    p = p / p.sum()
    p_geo = 1.0 / max(mean_len, 1.0)
    while hidden.sum() < target_frames:
        starts = rng.choice(candidates, size=64, p=p)
        lengths = rng.geometric(p_geo, size=64)
        for start, length in zip(starts, lengths):
            stop = min(n, start + length)
            segment = mask_allowed[start:stop]
            # stop the burst at the first frame where it would leave the allowed region
            cut = np.flatnonzero(~segment)
            if len(cut):
                stop = start + cut[0]
            hidden[start:stop] = True
            if hidden.sum() >= target_frames:
                break
    return hidden


def to_tracklets(players, team, mean_s, team_noise, rng, n_slots=46):
    """Cut every player's track into tracklets (geometric lengths, mean mean_s seconds) and pack
    them into n_slots anonymous slots, half per team, like the challenge's ATD. With probability
    team_noise a tracklet goes into a slot of the other team. Returns (players, slot_team)."""
    n = len(players)
    half = n_slots // 2
    slot_team = np.array([0] * half + [1] * (n_slots - half), dtype=np.int8)
    out = np.full((n, n_slots, 2), np.nan, dtype=players.dtype)
    busy_until = np.zeros(n_slots, dtype=np.int64)
    pieces = []
    for p in range(players.shape[1]):
        present = ~np.isnan(players[:, p, 0])
        edges = np.diff(np.concatenate(([0], present.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            t = a
            while t < b:
                length = int(rng.geometric(1.0 / max(mean_s * FPS, 1.0)))
                pieces.append((t, min(b, t + length), p))
                t += length
    pieces.sort()
    for a, b, p in pieces:
        k = team[p] if rng.random() >= team_noise else 1 - team[p]
        free = np.flatnonzero((slot_team == k) & (busy_until <= a))
        if not len(free):
            free = np.flatnonzero(busy_until <= a)
            if not len(free):
                continue
        slot = free[0]
        out[a:b, slot] = players[a:b, p]
        busy_until[slot] = b
    return out, slot_team


def degrade(ball, players, truth, scenario, team=None):
    """(ball, players, team) copies degraded according to the scenario."""
    rng = np.random.default_rng(scenario.seed)
    ball = ball.copy()
    players = players.copy()
    visible = ~np.isnan(ball[:, 0])
    alive = truth != D

    if scenario.alive_rate > 0:
        speed = _ball_speed(ball)
        weights = 1.0 + scenario.speed_bias * (speed / (np.mean(speed[alive & visible]) + 1e-9) - 1.0)
        weights = np.clip(weights, 0.05, None)
        target = int(scenario.alive_rate * np.sum(alive))
        already = int(np.sum(alive & ~visible))
        hide = _bursts(alive & visible, max(0, target - already), scenario.burst_mean, weights,
                       rng)
        ball[hide] = np.nan

    if scenario.dead_rate > 0:
        shown = ~alive & ~np.isnan(ball[:, 0])
        hide = _bursts(shown, int(min(scenario.dead_rate, 0.95) * shown.sum()), 4 * FPS,
                       np.ones(len(ball)), rng)
        ball[hide] = np.nan

    if scenario.noise_m > 0:
        ok = ~np.isnan(ball[:, 0])
        ball[ok] += rng.normal(0.0, scenario.noise_m, size=(int(ok.sum()), 2)).astype(ball.dtype)

    if scenario.player_drop > 0:
        n, slots = players.shape[:2]
        for slot in range(slots):
            present = ~np.isnan(players[:, slot, 0])
            target = int(scenario.player_drop * present.sum())
            hide = _bursts(present, target, 2 * FPS, np.ones(n), rng)
            players[hide, slot] = np.nan

    if scenario.tracklet_s > 0:
        players, team = to_tracklets(players, team, scenario.tracklet_s, scenario.team_noise, rng)
    return ball, players, team
