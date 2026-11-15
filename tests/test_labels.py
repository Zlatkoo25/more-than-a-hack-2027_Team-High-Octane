import numpy as np
import pandas as pd

from mtah import H, A, D
from mtah.labels import events_to_labels


def _events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


def _expect(n, spans):
    out = np.full(n, D, dtype=np.int8)
    for start, stop, code in spans:
        out[start:stop] = code
    return out


def test_one_half_with_every_kind_of_stoppage():
    n = 400
    period = np.ones(n, dtype=np.int8)
    events = _events([
        (0, "SET PIECE", "KICK OFF", 1, 10, 10),
        (0, "PASS", "", 1, 10, 25),
        (0, "BALL LOST", "INTERCEPTION", 1, 30, 40),
        (1, "RECOVERY", "INTERCEPTION", 1, 40, 40),
        (1, "BALL OUT", "", 1, 50, 60),          # dead from the end frame
        (0, "SET PIECE", "THROW IN", 1, 100, 100),
        (0, "PASS", "", 1, 100, 120),
        (0, "FAULT RECEIVED", "", 1, 150, 150),  # foul, but play runs on ...
        (1, "CHALLENGE", "TACKLE-FAULT-LOST", 1, 150, 150),
        (0, "PASS", "", 1, 160, 170),            # ... until this pass is received
        (0, "SET PIECE", "FREE KICK", 1, 200, 200),
        (1, "RECOVERY", "INTERCEPTION", 1, 210, 210),
        (1, "SHOT", "ON TARGET-GOAL", 1, 220, 230),
        (0, "SET PIECE", "KICK OFF", 1, 300, 300),
        (0, "BALL LOST", "END HALF", 1, 350, 355),
    ])
    expected = _expect(n, [(10, 40, H), (40, 60, A), (100, 171, H),
                           (200, 210, H), (210, 230, A), (300, 355, H)])
    np.testing.assert_array_equal(events_to_labels(events, period), expected)


def test_challenge_alone_does_not_move_possession():
    period = np.ones(100, dtype=np.int8)
    events = _events([
        (0, "SET PIECE", "KICK OFF", 1, 0, 0),
        (1, "CHALLENGE", "GROUND-WON", 1, 20, 20),
        (0, "PASS", "", 1, 30, 40),
        (0, "BALL OUT", "", 1, 50, 60),
    ])
    expected = _expect(100, [(0, 60, H)])
    np.testing.assert_array_equal(events_to_labels(events, period), expected)


def test_untagged_restart_after_ball_out():
    period = np.ones(200, dtype=np.int8)
    events = _events([
        (0, "SET PIECE", "KICK OFF", 1, 0, 0),
        (0, "BALL OUT", "", 1, 50, 60),
        (1, "PASS", "", 1, 120, 130),     # a throw-in nobody tagged as a set piece
        (1, "BALL OUT", "", 1, 150, 160),
    ])
    expected = _expect(200, [(0, 60, H), (120, 160, A)])
    np.testing.assert_array_equal(events_to_labels(events, period), expected)


def test_halves_are_independent():
    period = np.array([1] * 100 + [2] * 100, dtype=np.int8)
    events = _events([
        (0, "SET PIECE", "KICK OFF", 1, 5, 5),
        (0, "BALL LOST", "END HALF", 1, 90, 95),
        (1, "SET PIECE", "KICK OFF", 2, 110, 110),
        (1, "BALL LOST", "END HALF", 2, 190, 195),
    ])
    expected = _expect(200, [(5, 95, H), (110, 195, A)])
    np.testing.assert_array_equal(events_to_labels(events, period), expected)
