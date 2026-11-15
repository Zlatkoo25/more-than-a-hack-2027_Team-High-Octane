"""The training data: every public game, as several randomly damaged copies.

Metrica sample games 1-3 (truth from their event data, mtah/labels.py) and the 7 DFL games (truth
from the provider's own possession flags, mtah/dfl.py). The challenge match is never read here.
"""

from dataclasses import replace

import numpy as np

from . import dfl, labels, metrica, simulate

METRICA = (1, 2, 3)
DFL = tuple(dfl.GAMES)
COPIES = {"metrica": 8, "dfl": 4}


def load(game):
    """(ball, players, team, period, truth) of one clean public game."""
    if game in METRICA:
        G = metrica.load(game)
        touch = labels.events_to_labels(G.events, G.period)
        # a change of possession needs control: a 2 s hold or a completed pass
        return G.ball, G.players, G.team, G.period, labels.controlled(
            touch, G.events, G.period, min_hold=50, pass_counts=True)
    G = dfl.load(game)
    return G.ball, G.players, G.team, G.period, dfl.truth(G)


def copies():
    """Yields (game, (ball, players, team, period, truth)) for every damaged training copy.
    The seeds are fixed, so the copies are the same on every run."""
    rng = np.random.default_rng(11)
    for g in METRICA:
        ball, players, team, period, y = load(g)
        for k in range(COPIES["metrica"]):
            scenario = simulate.draw(rng, 9000 + 100 * g + k)
            yield g, (*simulate.degrade(ball, players, y, scenario, team), period, y)
    rng = np.random.default_rng(12)
    for i, g in enumerate(DFL):
        # the DFL games track the ball while play is dead; Metrica's do not. Hiding a random share
        # of the dead ball teaches the models both conventions.
        scenarios = [replace(simulate.draw(rng, 9500 + 100 * i + k),
                             dead_rate=rng.uniform(0.2, 0.95)) for k in range(COPIES["dfl"])]
        ball, players, team, period, y = load(g)
        for scenario in scenarios:
            yield g, (*simulate.degrade(ball, players, y, scenario, team), period, y)
