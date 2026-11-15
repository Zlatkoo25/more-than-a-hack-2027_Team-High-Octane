"""Structured decoding: the best H/A/D path given per-frame scores and a price on every change.

A possession is an object with a start and an end; frame-wise argmax ignores that and flickers.
Viterbi with a switching penalty chooses the path that maximises summed per-frame log-score minus
a fixed cost per label change -- the number of possessions it creates is controlled directly by
the penalties, which is what spellF1 rewards.
"""

import numpy as np

from . import H, A, D


def viterbi(logp, period, switch_team=8.0, to_dead=6.0, from_dead=6.0):
    """logp (T, 3) per-frame log-scores for H, A, D. Returns int8 codes (T,).

    Penalties are in the same units as logp (nats). Decoding restarts at every period boundary,
    and each half starts and ends in D for free (the scorer's halves do).
    """
    n = len(logp)
    cost = np.array([[0.0, switch_team, to_dead],
                     [switch_team, 0.0, to_dead],
                     [from_dead, from_dead, 0.0]])
    out = np.full(n, D, dtype=np.int8)
    bounds = np.flatnonzero(np.diff(period.astype(int)) != 0) + 1
    for lo, hi in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [n]))):
        if period[lo] == 0:
            continue
        out[lo:hi] = _viterbi_block(logp[lo:hi], cost)
    return out


def merge_short_spells(codes, period, min_len):
    """Give a spell that took the ball from the other team, but held it under min_len frames,
    back to the previous team -- the prediction-side version of 'a change needs control'."""
    out = codes.copy()
    n = len(out)
    t = 1
    while t < n:
        prev, cur = out[t - 1], out[t]
        if cur != prev and cur != D and prev != D and period[t] == period[t - 1]:
            stop = t
            while stop < n and out[stop] == cur and period[stop] == period[t]:
                stop += 1
            if stop - t < min_len:
                out[t:stop] = prev
            t = stop
        else:
            t += 1
    return out


def _viterbi_block(logp, cost):
    n = len(logp)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logp[0].copy()
    # transitions: new[j] = max_i score[i] - cost[i, j]
    for t in range(1, n):
        cand = score[:, None] - cost          # (from, to)
        best_from = cand.argmax(axis=0)
        back[t] = best_from
        score = cand[best_from, np.arange(3)] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
