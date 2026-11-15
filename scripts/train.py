"""Train the three models the submission uses, on public games only.

    python scripts/train.py v3                 # boosted trees, CPU, about 10 minutes
    python scripts/train.py set --seed 0       # set model, GPU, about 35 minutes
    python scripts/train.py set --seed 1       # the second seed

Training games: the 3 Metrica sample games and the 7 DFL games, each as several randomly damaged
copies and in both team orders (mtah/training.py). The challenge match is never read here.
Models are written to models/.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import boosted, paths, training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("v3", "set"))
    parser.add_argument("--seed", type=int, default=0, help="set model only")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(paths.MODEL_DIR, exist_ok=True)
    t0 = time.time()
    if args.kind == "set":
        import torch
        from mtah import setnet
    seqs = []
    for game, (ball, players, team, period, y) in training.copies():
        for swap in (False, True):
            yy = boosted.swap_labels(y) if swap else y
            if args.kind == "set":
                X = setnet.raw_design(ball, players, team, period, swap)[::setnet.STEP]
                seqs.append((X.astype(np.float16), yy[::setnet.STEP].astype(np.int64)))
            else:
                # neighbouring frames are near-duplicates: every 4th frame of a Metrica copy,
                # every 8th of a DFL copy
                thin = 4 if game in training.METRICA else 8
                tm = (1 - team).astype(np.int8) if swap else team
                seqs.append((boosted.build(ball, players, tm, period)[0][::thin],
                             yy[::thin].copy()))
    print("%d training sequences (%.0fs)" % (len(seqs), time.time() - t0))
    if args.kind == "set":
        net = setnet.fit(seqs, seed=args.seed)
        out = os.path.join(paths.MODEL_DIR, "final_set_s%d.pt" % args.seed)
        torch.save(net.state_dict(), out + ".partial")
        os.replace(out + ".partial", out)
        print("saved %s (%.0fs, GPU peak %.2f GiB)" % (out, time.time() - t0, setnet.gpu_peak()))
        sys.stdout.flush()
        os._exit(0)     # skip CUDA teardown, which has crashed the interpreter on exit here
    model = boosted.fit(seqs)
    out = os.path.join(paths.MODEL_DIR, "final_v3.pkl")
    with open(out + ".partial", "wb") as handle:
        pickle.dump(model, handle)
    os.replace(out + ".partial", out)
    print("saved %s (%.0fs)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
