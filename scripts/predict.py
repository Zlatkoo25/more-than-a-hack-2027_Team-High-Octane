"""Label the challenge match with the models in models/ and write the submission file.

    python scripts/predict.py --data "<folder holding tactical\\>" --out <team_name>.csv

Recipe: the average of v3 and the set model, the set model itself averaged over 2 seeds x 4 views
of the match (as is, pitch mirrored, teams swapped, both), decoded with a price of 25 on every
label change. Nothing about the challenge match chooses anything: models, weights and the price
were fixed on other games.

One line is printed: that it finished, the row count and whether the file is valid. An error
prints its type and code location; the full trace goes to <out>_error.txt.

--sample 1 runs the same code on Metrica sample game 1 instead: a check that the pipeline works
without the challenge data. Its labels mean nothing, the models were trained on that game.
"""

import argparse
import os
import pickle
import sys
import traceback
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import D, LABELS, boosted, challenge, features, paths, score  # noqa: E402
from mtah.decode import viterbi  # noqa: E402

# Fixed from held-out public games, never from the challenge match.
PRICE = 25.0
SEEDS = (0, 1)
SET_WEIGHT = 0.5


def pad_slots(players, team):
    """At least 11 slots per team (the set model reads 11 a side), extra ones always empty."""
    for k in (0, 1):
        short = 11 - int(np.sum(team == k))
        if short > 0:
            players = np.concatenate(
                [players, np.full((len(players), short, 2), np.nan, players.dtype)], axis=1)
            team = np.concatenate([team, np.full(short, k, team.dtype)])
    return players, team


def frame_scores(ball, players, team, period):
    """(v3 scores, set-model scores), each (T, 3) log-probabilities for H, A, D."""
    import torch
    from mtah import setnet

    with open(os.path.join(paths.MODEL_DIR, "final_v3.pkl"), "rb") as handle:
        v3 = boosted.frame_logp(pickle.load(handle), boosted.build(ball, players, team, period)[0])
    nets = []
    for seed in SEEDS:
        net = setnet.SetNet().to(setnet.DEVICE)
        net.load_state_dict(torch.load(
            os.path.join(paths.MODEL_DIR, "final_set_s%d.pt" % seed), map_location=setnet.DEVICE))
        nets.append(net)
    views = []
    for mirror in (False, True):
        b, p = ball.copy(), players.copy()
        if mirror:
            b[:, 1] = features.PITCH[1] - b[:, 1]
            p[:, :, 1] = features.PITCH[1] - p[:, :, 1]
        for swap in (False, True):
            X = setnet.raw_design(b, p, team, period, swap)[::setnet.STEP]
            for net in nets:
                lp = setnet.predict(net, X, len(period))
                views.append(lp[:, [1, 0, 2]] if swap else lp)
    return v3, np.mean(views, axis=0)


def run(args):
    if args.sample:
        from mtah import metrica
        G = metrica.load(args.sample)
        ball, players, team, period = G.ball, G.players, G.team, G.period
        scored = live = np.ones(len(ball), bool)
    else:
        ball, players, team, period = challenge.load(args.data)
        (a1, b1), (a2, b2) = challenge.SCORED
        scored = np.zeros(len(ball), bool)
        scored[a1:b1] = scored[a2:b2] = True
        live = scored.copy()
        live[a2:challenge.KICKOFF_2] = False
    players, team = pad_slots(players, team)
    v3, st = frame_scores(ball, players, team, period)
    # Known from the organisers' primer, not from the match: outside the scored spans nothing
    # counts, and the scored frames before the second kick-off are all D. Both models are held to
    # D there, so the decode enters each half from a dead ball; unscored frames are written as X.
    v3, st = v3.copy(), st.copy()
    for lp in (v3, st):
        lp[~live, :D] = -1e6
    codes = viterbi(SET_WEIGHT * st + (1 - SET_WEIGHT) * v3, period, PRICE, PRICE, PRICE)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (i, LABELS[c] if ok else "X")
                          for i, (c, ok) in enumerate(zip(codes, scored)))
    if not os.path.isdir(os.path.join(paths.ORGANISERS_DIR, "evaluation")):
        return len(codes), "not validated: organisers' repository not found"
    pm = score.metrics_module()
    problem = pm.validate_submission(pm.read_label_csv(args.out), len(codes))
    return len(codes), "valid" if problem is None else "INVALID: %s" % problem


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="folder holding tactical/ (default: MTAH_DATA_DIR)")
    parser.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    parser.add_argument("--sample", type=int, choices=(1, 2, 3),
                        help="run on a Metrica sample game instead (pipeline check)")
    args = parser.parse_args()
    warnings.simplefilter("ignore")      # e.g. numpy's 'mean of empty slice' on empty frames
    stem = os.path.splitext(args.out)[0]
    code = 0
    try:
        n, status = run(args)
        print("finished: %d rows written to %s (%s)" % (n, args.out, status))
    except BaseException as error:  # noqa: BLE001
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(stem + "_error.txt", "w", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        where = [f for f in traceback.extract_tb(error.__traceback__)
                 if f.filename.startswith(ROOT) and ".venv" not in f.filename]
        last = where[-1] if where else traceback.extract_tb(error.__traceback__)[-1]
        print("failed: %s at %s:%d in %s (full trace in %s)" % (
            type(error).__name__, os.path.relpath(last.filename, ROOT), last.lineno, last.name,
            stem + "_error.txt"))
        code = 1
    sys.stdout.flush()
    os._exit(code)      # skip CUDA teardown, which has crashed the interpreter on exit here


if __name__ == "__main__":
    main()
