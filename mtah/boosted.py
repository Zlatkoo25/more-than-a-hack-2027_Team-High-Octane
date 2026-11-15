"""v3: gradient-boosted trees on tracking-only features, one score per frame for H, A and D.

Features are window statistics at several scales, so the classifier sees context: who was near
the ball over the last and next few seconds, touches and carries, how fast everyone is moving,
how long the ball has been gone and where it was last seen, and the shape of both teams.

Home/away symmetry comes from training on each game twice, once with the teams swapped (labels H
and A swapped with them), so the model cannot learn "home has the ball more".
"""

import numpy as np

from . import H, A, features

WINDOWS = (12, 50, 125, 250)
CONTROL_M = 1.5
CLIP_GAP = 1500


def _dist_to_boundary(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([x, features.PITCH[0] - x, y, features.PITCH[1] - y])


def _direction(cx_own, cx_other, period):
    """+1 where the team attacks towards larger x in that half, from where the two blocks sit."""
    out = np.zeros(len(period))
    for p in np.unique(period[period > 0]):
        m = period == p
        out[m] = np.sign(np.nanmean(cx_other[m]) - np.nanmean(cx_own[m])) or 1.0
    return out


def design(feat, ball, touch):
    """(T, F) float32 matrix and the column names, from features.frame_features and
    features.touch_features."""
    f = feat
    n = len(f["missing"])
    vis = ~f["missing"]
    dh = np.nan_to_num(f["d_home"], nan=30.0)
    da = np.nan_to_num(f["d_away"], nan=30.0)
    ctrl_h = (vis & (dh < CONTROL_M) & (dh < da)).astype(np.float64)
    ctrl_a = (vis & (da < CONTROL_M) & (da < dh)).astype(np.float64)
    near_h = (vis & (dh < da)).astype(np.float64)
    near_a = (vis & (da < dh)).astype(np.float64)
    spread_diff = f["spread_home"] - f["spread_away"]
    dir_h = _direction(f["centroid_home_x"], f["centroid_away_x"], f["period"])
    block_x = (f["centroid_home_x"] + f["centroid_away_x"]) / 2.0 - features.PITCH[0] / 2.0
    block_adv_h = block_x * dir_h  # > 0: play is in the away half, i.e. home attacking
    ball_x = np.where(vis, ball[:, 0] - features.PITCH[0] / 2.0, np.nan) * dir_h
    bspeed = np.where(vis, f["ball_speed"], np.nan)

    # Ball last seen before / first seen after this frame: which team was nearest, and how
    # close the ball was to a line (a ball vanishing at the touchline has usually gone out).
    run, since, until, last_seen, next_seen = features.gap_lengths(f["missing"])
    last_ok = last_seen >= 0
    next_ok = next_seen < n
    li = np.clip(last_seen, 0, n - 1)
    ni = np.clip(next_seen, 0, n - 1)
    side_last = np.where(last_ok, near_h[li] - near_a[li], 0.0)
    side_next = np.where(next_ok, near_h[ni] - near_a[ni], 0.0)
    bnd = _dist_to_boundary(np.nan_to_num(ball, nan=-99.0))
    bnd_last = np.where(last_ok, bnd[li], -1.0)
    bnd_next = np.where(next_ok, bnd[ni], -1.0)

    cols = {
        "missing": f["missing"].astype(np.float64),
        "dh": dh, "da": da, "dd": np.clip(dh - da, -20, 20), "dmin": np.minimum(dh, da),
        "ball_speed": np.nan_to_num(bspeed, nan=-1.0),
        "speed_all": f["speed_all"], "speed_h": f["speed_home"], "speed_a": f["speed_away"],
        "spread_h": f["spread_home"], "spread_a": f["spread_away"], "spread_diff": spread_diff,
        "block_adv_h": block_adv_h, "ball_x_h": np.nan_to_num(ball_x, nan=0.0),
        "gap_run": np.minimum(run, CLIP_GAP), "gap_since": np.minimum(since, CLIP_GAP),
        "gap_until": np.minimum(until, CLIP_GAP),
        "side_last": side_last, "side_next": side_next,
        "bnd_last": bnd_last, "bnd_next": bnd_next,
        "out_of_pitch": f["out_of_pitch"].astype(np.float64),
    }
    for w in WINDOWS:
        win = 2 * w + 1
        cols["ctrl_h_%d" % w] = features.moving_nanmean(ctrl_h, win)
        cols["ctrl_a_%d" % w] = features.moving_nanmean(ctrl_a, win)
        cols["near_diff_%d" % w] = features.moving_nanmean(near_h - near_a, win)
        cols["vis_%d" % w] = features.moving_nanmean(vis.astype(np.float64), win)
        cols["speed_all_%d" % w] = features.moving_nanmean(f["speed_all"], win)
        cols["spread_diff_%d" % w] = features.moving_nanmean(spread_diff, win)
        cols["bspeed_%d" % w] = np.nan_to_num(features.moving_nanmean(bspeed, win), nan=-1.0)
        cols["oop_%d" % w] = features.moving_nanmean(f["out_of_pitch"].astype(np.float64), win)
        # one-sided: before and after, so the model can see a change of owner
        back = features.moving_nanmean(np.concatenate((np.full(w, np.nan), near_h - near_a))[:n], w)
        fwd = features.moving_nanmean(np.concatenate(((near_h - near_a)[w:], np.full(w, np.nan))), w)
        cols["near_back_%d" % w] = np.nan_to_num(back, nan=0.0)
        cols["near_fwd_%d" % w] = np.nan_to_num(fwd, nan=0.0)
    cols["ball_dv"] = touch["ball_dv"]
    cols["rel_speed"] = touch["rel_speed"]
    th_ta = touch["touch_h"] - touch["touch_a"]
    carry = touch["carry_h"] - touch["carry_a"]
    for w in (12, 25, 50, 125):
        win = 2 * w + 1
        cols["touch_%d" % w] = features.moving_nanmean(th_ta, win) * win
        cols["carry_%d" % w] = features.moving_nanmean(carry, win)
        back = np.concatenate((np.zeros(w), th_ta))[:n]
        fwd = np.concatenate((th_ta[w:], np.zeros(w)))
        cols["touch_back_%d" % w] = features.moving_nanmean(back, w) * w
        cols["touch_fwd_%d" % w] = features.moving_nanmean(fwd, w) * w
        cb = np.concatenate((np.zeros(w), carry))[:n]
        cf = np.concatenate((carry[w:], np.zeros(w)))
        cols["carry_back_%d" % w] = features.moving_nanmean(cb, w)
        cols["carry_fwd_%d" % w] = features.moving_nanmean(cf, w)
    names = list(cols)
    X = np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names])
    return X, names


def build(ball, players, team, period):
    """The (T, 121) feature matrix for one match from tracking alone, and the column names.
    Training and prediction both call this, so they use exactly the same features."""
    feat = features.frame_features(ball, players, team, period)
    X, names = design(feat, ball, features.touch_features(ball, players, team))
    S, snames = features.shape_features(players, team, period)
    S = np.concatenate([S] + [features.moving_nanmean(S, 2 * w + 1).astype(np.float32)
                              for w in (25, 125)], axis=1)
    X = np.concatenate([X, S], axis=1)
    names = names + ["%s_w%d" % (n_, w) for w in (0, 25, 125) for n_ in snames]
    return X, names


def swap_labels(y):
    out = y.copy()
    out[y == H] = A
    out[y == A] = H
    return out


def fit(train_sets):
    """train_sets: list of (X, y), already thinned by the caller (neighbouring frames are
    near-duplicates)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs = np.concatenate([X for X, _ in train_sets])
    ys = np.concatenate([y for _, y in train_sets])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(Xs, ys)
    return model


def frame_logp(model, X):
    """(T, 3) log-probabilities for H, A, D."""
    return np.log(np.clip(model.predict_proba(X), 1e-6, 1.0))
