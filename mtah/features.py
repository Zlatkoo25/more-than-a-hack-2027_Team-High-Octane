"""Per-frame features computed only from tracking (ball, players, team) -- the inputs the challenge
also has. Nothing here reads events or labels."""

import numpy as np

FPS = 25
PITCH = (105.0, 68.0)


def team_ball_distance(ball, players, team):
    """(T, 2) distance in metres from the ball to the nearest visible player of each team."""
    out = np.full((len(ball), 2), np.nan, dtype=np.float32)
    delta = players - ball[:, None, :]
    dist = np.sqrt((delta ** 2).sum(axis=2))  # (T, P), NaN where either is missing
    for k in (0, 1):
        sub = dist[:, team == k]
        with np.errstate(all="ignore"):
            any_ok = ~np.all(np.isnan(sub), axis=1)
            out[any_ok, k] = np.nanmin(sub[any_ok], axis=1)
    return out


def speed(xy, smooth=5):
    """Speed in m/s of a (T, 2) or (T, P, 2) track, centred difference, moving-average smoothed."""
    diff = np.full(xy.shape, np.nan, dtype=np.float32)
    diff[1:-1] = (xy[2:] - xy[:-2]) / 2.0
    v = np.sqrt((diff ** 2).sum(axis=-1)) * FPS
    if smooth > 1:
        v = moving_nanmean(v, smooth)
    return v


def moving_nanmean(values, window):
    """Centred moving mean along axis 0 that ignores NaN (NaN only where the window is all NaN)."""
    values = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0)
    pad = [(window // 2, window - 1 - window // 2)] + [(0, 0)] * (values.ndim - 1)
    c_val = np.cumsum(np.pad(filled, pad), axis=0)
    c_ok = np.cumsum(np.pad(ok.astype(np.float64), pad), axis=0)
    c_val = np.concatenate([np.zeros((1,) + values.shape[1:]), c_val])
    c_ok = np.concatenate([np.zeros((1,) + values.shape[1:]), c_ok])
    s = c_val[window:] - c_val[:-window]
    k = c_ok[window:] - c_ok[:-window]
    with np.errstate(all="ignore"):
        return np.where(k > 0, s / k, np.nan)


def gap_lengths(missing):
    """For each frame: length of the missing-run it sits in (0 if visible), frames since the last
    visible frame, frames until the next visible frame."""
    n = len(missing)
    idx = np.arange(n)
    last_seen = np.where(~missing, idx, -1)
    last_seen = np.maximum.accumulate(last_seen)
    next_seen = np.where(~missing, idx, n)
    next_seen = np.minimum.accumulate(next_seen[::-1])[::-1]
    since = np.where(last_seen >= 0, idx - last_seen, n)
    until = np.where(next_seen < n, next_seen - idx, n)
    run = np.where(missing, np.minimum(since + until - 1, n), 0)
    return run, since, until, last_seen, next_seen


def velocity(xy, half=2):
    """Velocity (m/s) by centred difference over +/- half frames; NaN where either end is NaN."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[half:-half] = (xy[2 * half:] - xy[:-2 * half]) * (FPS / (2.0 * half))
    return v


def touch_features(ball, players, team, touch_dv=5.0, touch_m=2.0, control_m=1.5,
                   comove_mps=2.0):
    """Ball-trajectory evidence of who touches and who carries the ball.

    touch_h / touch_a   1 where the ball's velocity changes sharply (|v_after - v_before| above
                        touch_dv m/s) with that team's player nearest and within touch_m -- a
                        kick, a reception, an interception or a deflection.
    carry_h / carry_a   1 where the nearest player is within control_m and moves with the ball
                        (relative speed under comove_mps) -- dribbling, i.e. control.
    """
    n = len(ball)
    vb = velocity(ball, 2)
    before = np.full_like(vb, np.nan)
    after = np.full_like(vb, np.nan)
    before[3:] = vb[:-3]
    after[:-3] = vb[3:]
    dv = np.sqrt(((after - before) ** 2).sum(axis=1))

    delta = players - ball[:, None, :]
    dist = np.sqrt((delta ** 2).sum(axis=2))
    dist_f = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist_f.argmin(axis=1)
    dnear = dist_f[np.arange(n), nearest]
    tnear = team[nearest]
    vp = velocity(players, 2)
    rel = np.sqrt(((vp[np.arange(n), nearest] - vb) ** 2).sum(axis=1))

    touch = np.nan_to_num(dv, nan=0.0) > touch_dv
    touch &= dnear < touch_m
    carry = (dnear < control_m) & (np.nan_to_num(rel, nan=99.0) < comove_mps)
    return {
        "touch_h": (touch & (tnear == 0)).astype(np.float64),
        "touch_a": (touch & (tnear == 1)).astype(np.float64),
        "carry_h": (carry & (tnear == 0)).astype(np.float64),
        "carry_a": (carry & (tnear == 1)).astype(np.float64),
        "ball_dv": np.nan_to_num(dv, nan=0.0),
        "rel_speed": np.nan_to_num(np.minimum(rel, 30.0), nan=30.0),
    }


def row_percentile(values, q):
    """Nearest-rank percentile along axis 1, ignoring NaN. Much faster than np.nanpercentile."""
    s = np.sort(np.where(np.isnan(values), np.inf, values), axis=1)
    k = np.sum(~np.isnan(values), axis=1)
    idx = np.clip(np.round(q / 100.0 * (k - 1)).astype(int), 0, values.shape[1] - 1)
    out = s[np.arange(len(values)), idx]
    return np.where(k > 0, out, np.nan)


def attack_direction(cx_own, cx_other, period):
    """+1 where the team attacks towards larger x in that half, judged from where the blocks sit."""
    out = np.zeros(len(period))
    for p in np.unique(period[period > 0]):
        m = period == p
        out[m] = np.sign(np.nanmean(cx_other[m]) - np.nanmean(cx_own[m])) or 1.0
    return out


def shape_features(players, team, period):
    """Team-shape cues to who has the ball, needing no ball. Oriented by attacking direction.

    centroid x (territory), mean forward velocity, block length and width (the side in possession
    spreads out), deepest outfield player (defensive line), and how far forward home sits
    relative to away. Returns (X (T, F) float32, names).
    """
    v = velocity(players, 2)
    with np.errstate(all="ignore"):
        cx = {k: np.nanmean(players[:, team == k, 0], axis=1) for k in (0, 1)}
    direction = {0: attack_direction(cx[0], cx[1], period),
                 1: attack_direction(cx[1], cx[0], period)}
    cols = {}
    for k, tag in ((0, "h"), (1, "a")):
        P = players[:, team == k]
        d = direction[k][:, None]
        x = (P[:, :, 0] - PITCH[0] / 2) * d
        with np.errstate(all="ignore"):
            cols["cx_" + tag] = np.nanmean(x, axis=1)
            cols["vx_" + tag] = np.nanmean(v[:, team == k, 0] * d, axis=1)
        cols["len_" + tag] = row_percentile(x, 90) - row_percentile(x, 10)
        cols["wid_" + tag] = row_percentile(P[:, :, 1], 90) - row_percentile(P[:, :, 1], 10)
        srt = np.sort(np.where(np.isnan(x), np.inf, x), axis=1)
        cols["line_" + tag] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)
    cols["gap"] = cols["cx_h"] + cols["cx_a"]
    names = list(cols)
    return np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names]), names


def frame_features(ball, players, team, period):
    """Dict of (T,) arrays."""
    missing = np.isnan(ball[:, 0])
    dist = team_ball_distance(ball, players, team)
    ball_v = speed(ball, smooth=3)
    player_v = speed(players, smooth=5)  # (T, P)
    with np.errstate(all="ignore"):
        team_v = np.stack([np.nanmean(player_v[:, team == k], axis=1) for k in (0, 1)], axis=1)
        centroid = np.stack([np.nanmean(players[:, team == k, :], axis=1) for k in (0, 1)],
                            axis=1)  # (T, 2 teams, 2)
        spread = np.stack([np.nanmean(np.nanstd(players[:, team == k, :], axis=1), axis=1)
                           for k in (0, 1)], axis=1)
    run, since, until, last_seen, next_seen = gap_lengths(missing)
    out_of_pitch = (~missing) & ((ball[:, 0] < 0) | (ball[:, 0] > PITCH[0]) |
                                 (ball[:, 1] < 0) | (ball[:, 1] > PITCH[1]))
    return {
        "missing": missing,
        "d_home": dist[:, 0], "d_away": dist[:, 1],
        "ball_speed": ball_v,
        "speed_all": np.nanmean(team_v, axis=1),
        "speed_home": team_v[:, 0], "speed_away": team_v[:, 1],
        "centroid_home_x": centroid[:, 0, 0], "centroid_away_x": centroid[:, 1, 0],
        "spread_home": spread[:, 0], "spread_away": spread[:, 1],
        "gap_run": run, "gap_since": since, "gap_until": until,
        "out_of_pitch": out_of_pitch,
        "period": period,
    }
