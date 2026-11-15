"""Turn Metrica sample-game events into per-frame H/A/D labels: a stand-in ground truth.

Follows Metrica's own event definitions: a team keeps possession until a player of the other
team takes over (an insignificant touch does not count), or the ball goes out of play.

The construction is anchored on RESTARTS, because restarts are the one thing the event feed tags
reliably: every SET PIECE (kick-off, throw-in, corner, free kick, penalty), every GOAL KICK, and
the first on-ball event after the ball left the pitch when no set piece was tagged.

For each restart, look back to what stopped play:

* the ball left the pitch -- BALL OUT, a SHOT whose subtype says OUT or GOAL, END HALF: dead from
  that event's END frame (the moment the ball crosses the line);
* an infringement -- FAULT RECEIVED, a CHALLENGE whose subtype says FAULT, OFFSIDE, HAND BALL,
  or a BALL LOST / FORCED with nothing on the ball after it: the whistle often comes late (play
  runs on, and the provider then tags a FORCED loss), so dead from just after the last on-ball
  event before the restart, but never before the infringement itself;
* nothing tagged (an injury, a drop ball): dead from just after the last on-ball event.

Between a restart and the next stoppage the ball is alive and belongs to the team of the latest
on-ball event (PASS, CARRY, RECOVERY, BALL LOST, SHOT, SET PIECE, BALL OUT). A misplaced pass
stays with the passer's team while it is in the air. CHALLENGE events never move possession on
their own: the duel winner's next action does.

Events are stamped at the touch, so these labels are approximate at the level of a few frames to
about a second. controlled() then applies the challenge's rule that a change of possession needs
control.
"""

import numpy as np

from . import H, A, D

ON_BALL = {"PASS", "CARRY", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE", "BALL OUT"}


def _parts(subtype):
    return set(subtype.split("-")) if subtype else set()


def is_out_of_play(event):
    kind, parts = event.type, _parts(event.subtype)
    return (kind == "BALL OUT" or "END HALF" in parts or
            (kind == "SHOT" and bool(parts & {"OUT", "GOAL"})))


def is_infringement(event):
    kind, parts = event.type, _parts(event.subtype)
    return (kind == "FAULT RECEIVED" or (kind == "CHALLENGE" and "FAULT" in parts) or
            "OFFSIDE" in parts or "HAND BALL" in parts or
            (kind == "BALL LOST" and "FORCED" in parts))


def is_tagged_restart(event):
    return event.team in (0, 1) and (event.type == "SET PIECE" or
                                     "GOAL KICK" in _parts(event.subtype))


def _team_code(team):
    return H if team == 0 else A


def events_to_labels(events, period):
    """Per-row H/A/D codes for one game.

    events  DataFrame from metrica.load_events (row-indexed start/end, team 0/1/-1)
    period  int array (T,), the half of each row; label D outside the play of a half.
    """
    n = len(period)
    codes = np.full(n, D, dtype=np.int8)
    spans = []

    for half in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == half)
        half_start, half_stop = int(rows[0]), int(rows[-1]) + 1
        ev = list(events[events.period == half].sort_values("start", kind="stable")
                  .itertuples(index=False))
        on_ball = [e for e in ev if e.type in ON_BALL and e.team in (0, 1)]
        if not on_ball:
            continue

        # 1. restarts: tagged ones, plus the first on-ball event after an untagged out-of-play.
        restart_frames = {int(e.start) for e in ev if is_tagged_restart(e)}
        restart_frames.add(int(on_ball[0].start))  # the kick-off, tagged or not
        for e in ev:
            if is_out_of_play(e) and "END HALF" not in _parts(e.subtype):
                after = [o for o in on_ball if o.start > e.end]
                if after and not any(e.end < r <= after[0].start for r in restart_frames):
                    restart_frames.add(int(after[0].start))
        restarts = sorted(restart_frames)

        # 2. for each stretch from one restart to the next, find where play stopped.
        bounds = restarts + [half_stop]
        for k in range(len(restarts)):
            start, next_restart = bounds[k], bounds[k + 1]
            inside = [e for e in ev if start <= e.start < next_restart]
            plays = [e for e in inside if e.type in ON_BALL and e.team in (0, 1)]
            last_touch = max(int(e.end) for e in plays) if plays else start

            outs = [e for e in inside if is_out_of_play(e)]
            fouls = [e for e in inside if is_infringement(e)]
            if outs:
                dead = int(outs[-1].end)
            elif fouls:
                dead = max(int(fouls[0].start), last_touch + 1)
            else:
                dead = last_touch + 1
            dead = min(max(dead, start + 1), next_restart)

            # 3. possession inside the alive part.
            current, since = None, start
            for e in plays:
                frame = int(e.start)
                if frame >= dead:
                    break
                code = _team_code(e.team)
                if current is None:
                    current = code
                elif code != current:
                    spans.append((since, frame, current))
                    current, since = code, frame
            if current is not None and dead > since:
                spans.append((since, dead, current))
            if next_restart > dead:
                spans.append((dead, next_restart, D))
        del half_start

    for start, stop, code in spans:
        codes[start:stop] = code
    return codes


def controlled(codes, events, period, min_hold=50, pass_counts=True):
    """Keep a change of possession only if the new team CONTROLS the ball.

    The challenge (CHALLENGE.md) counts a change of possession only when the other team gains
    controlled possession: a simple touch, a deflection, or an aerial duel won without follow-up
    control does not count, and a loose ball stays with the last team that had it. Its suggested
    proxies are a player keeping the ball for a few seconds or a run of passes.

    Here a spell that takes the ball from the other team (not one that starts at a restart) is
    kept if it lasts at least min_hold frames, or (pass_counts) contains a completed PASS by that
    team; otherwise it is relabelled as the previous team's. Applied left to right, so merges chain.
    """
    out = codes.copy()
    passes = events[(events.type == "PASS") & events.team.isin([0, 1])]
    pass_frames = {0: np.sort(passes[passes.team == 0].start.to_numpy()),
                   1: np.sort(passes[passes.team == 1].start.to_numpy())}
    n = len(out)
    t = 1
    while t < n:
        prev, cur = out[t - 1], out[t]
        if cur != prev and cur != D and prev != D and period[t] == period[t - 1]:
            stop = t
            while stop < n and out[stop] == cur and period[stop] == period[t]:
                stop += 1
            team = 0 if cur == H else 1
            pf = pass_frames[team]
            has_pass = pass_counts and np.any((pf >= t) & (pf < stop))
            if stop - t < min_hold and not has_pass:
                out[t:stop] = prev
                # the spell after may now continue prev's possession; re-check from here
            t = stop
        else:
            t += 1
    return out
