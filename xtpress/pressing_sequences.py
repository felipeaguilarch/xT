"""Pressing detection and team-level pressing value.

A StatsBomb ``Pressure`` event fires when a defender closes to roughly five
yards of the player in possession (up to ten for goalkeepers). It carries a
location, a duration, an optional ``counterpress`` flag, and -- the part that
matters most -- ``related_events``.

Measured on three matches (WC22, Euro24, WWC23), as calibration targets:
    264-438 pressure events per match
    93%     of pressures have related_events pointing at an under_pressure action
    17.9%   of pressures precede a possession change to the pressing team <8s
    7.8%    same figure for counterpresses only
    7.4-12.3  PPDA across the six teams
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from .loaders import _flag, _timestamp_seconds, _xy
from . import config


def to_possession_frame(
    x: pd.Series | np.ndarray, y: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a defensive event into the possession team's coordinate frame.

        x, y -> 120 - x, 80 - y

    StatsBomb normalises every event to the frame where the ACTING team attacks
    toward x=120 (trap #2). For a pass or carry the actor IS the possession
    team, so nothing is needed. For a Pressure the frames are mirrored --
    measured median distance 4.4 assuming mirrored vs 97.6 assuming shared.

    Use this when joining a press to the possession it interrupted. Do NOT use
    it when asking how high the press was relative to the goal the pressing
    team attacks -- the raw x already answers that.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return config.PITCH_LENGTH - x, config.PITCH_WIDTH - y


def extract_pressures(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per Pressure event with the fields Phase 06 needs.

    Columns: id, match_id, period, time_seconds, team, team_id, player,
    player_id, x, y, duration, counterpress, possession, possession_team,
    related_events.

    ``id`` is the event's own uuid. ``link_pressures_to_actions`` needs it to
    say which pressure a link belongs to. ``related_events`` holds the ids of
    the events this pressure points at, and is passed through untouched: an
    array from the parquet cache, a list from a fresh fetch.

    ``x, y`` are in the PRESSING team's frame -- the frame of the team that
    performed the event -- so they must go through ``to_possession_frame``
    before being compared with the possession team's coordinates.

    ``time_seconds`` restarts at zero each period. Any join or sort on time
    must include ``period``, or a pressure at 10:00 of the first half is
    treated as simultaneous with one at 10:00 of the second.

    Note ``duration`` is how long the press LASTED (median 0.47s) -- it is not
    a window for finding affected events (trap #13).
    """

    pr = events[events["type"] == "Pressure"].copy()

    xy = pd.DataFrame(pr["location"].map(_xy).tolist(), index=pr.index, columns=["x", "y"])

    out = pd.DataFrame({
        "id": pr["id"],
        "match_id": pr["match_id"],
        "period": pr["period"],
        "time_seconds": _timestamp_seconds(pr["timestamp"]),
        "team": pr["team"],
        "team_id": pr["team_id"],
        "player": pr["player"],
        "player_id": pr["player_id"].astype(int),
        "x": xy["x"],
        "y": xy["y"],
        "duration": pr["duration"],
        "counterpress": _flag(pr, "counterpress"),
        "possession": pr["possession"],
        "possession_team": pr["possession_team"],
        "related_events": pr["related_events"],
    })
    return out.reset_index(drop=True)


def _link_by_related_events(pressures, events):
    """Pressures whose ``related_events`` point at an opponent event: one row each."""
    links = pressures[["id", "team_id", "related_events"]].explode("related_events")
    links = links.dropna(subset=["related_events"])

    target = events[["id", "team_id", "type", "player_id", "location"]]
    m = links.merge(target, left_on="related_events", right_on="id",
                    suffixes=("_press", "_action"))
    m = m[m["team_id_press"] != m["team_id_action"]]
    m = m.drop_duplicates("id_press", keep="first")

    return m.rename(columns={"id_press": "id", "id_action": "action_id",
                             "type": "action_type", "player_id": "action_player_id",
                             "location": "action_location"})[
        ["id", "action_id", "action_type", "action_player_id", "action_location"]]


def _link_by_time_window(pressures, events):
    """Pressures matched to the nearest opponent under_pressure event in time."""
    ev = events.assign(time_seconds=_timestamp_seconds(events["timestamp"]))

    teams = ev[["match_id", "team_id"]].dropna().drop_duplicates()
    opp = pressures[["id", "match_id", "team_id"]].merge(teams, on="match_id",
                                                         suffixes=("", "_opp"))
    opp = opp[opp["team_id"] != opp["team_id_opp"]][["id", "team_id_opp"]]

    left = pressures.merge(opp, on="id")[
        ["id", "match_id", "period", "time_seconds", "team_id_opp"]]
    left = left.sort_values("time_seconds")

    cand = ev[_flag(ev, "under_pressure")][
        ["id", "match_id", "period", "time_seconds", "team_id", "type",
         "player_id", "location"]]
    cand = cand.rename(columns={"id": "action_id", "type": "action_type",
                                "player_id": "action_player_id",
                                "location": "action_location"})
    cand = cand.sort_values("time_seconds")

    out = pd.merge_asof(
        left, cand, on="time_seconds",
        left_by=["match_id", "period", "team_id_opp"],
        right_by=["match_id", "period", "team_id"],
        direction="nearest", tolerance=config.PRESS_MATCH_WINDOW_SECONDS)

    return out.dropna(subset=["action_id"])[
        ["id", "action_id", "action_type", "action_player_id", "action_location"]]


def link_pressures_to_actions(
    pressures: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Attach the opponent action each pressure affected.

    Returns ``pressures`` with the same rows in the same order, plus
    ``action_id``, ``action_type``, ``action_player_id``, ``action_x``,
    ``action_y``, ``link_method`` ("related", "window" or missing) and
    ``link_distance``.

    Primary path: resolve ``related_events`` against the event frame and keep
    the first related event belonging to the OTHER team. Measured on 64 WC 2022
    matches this links 90.5% of pressures (80-98% by match).

    Fallback for the rest: the nearest opponent ``under_pressure`` event within
    ``config.PRESS_MATCH_WINDOW_SECONDS``, matched within the same match AND
    period. The window is two-sided because the gap is often negative (46% of
    true links): a Carry is timestamped when it starts, and the press arrives
    partway through (trap #15). This rescues a further 5.0%, leaving 4.4%
    unlinked.

    The fallback is reliable about WHO was pressed and WHERE, not about WHICH
    action: forced onto pressures whose true link is known, it picked the same
    player 99.3% of the time but the same event only 59.9% (usually the same
    player's adjacent Pass, Carry or Ball Receipt). Use ``link_method`` to
    report or exclude window links.

    ``link_distance`` is the distance between the press, mirrored into the
    possession team's frame by ``to_possession_frame``, and the action. This
    is the one place the mirror is applied. Median 3.3 for related links, but
    roughly 74 on the sample match if the mirror is forgotten.
    """
    primary = _link_by_related_events(pressures, events)
    rest = pressures[~pressures["id"].isin(primary["id"])]
    fallback = _link_by_time_window(rest, events)

    links = pd.concat([primary.assign(link_method="related"),
                       fallback.assign(link_method="window")], ignore_index=True)
    out = pressures.merge(links, on="id", how="left")

    xy = np.array(out["action_location"].map(_xy).tolist())
    out["action_x"], out["action_y"] = xy[:, 0], xy[:, 1]

    press_x, press_y = to_possession_frame(out["x"], out["y"])
    out["link_distance"] = np.hypot(press_x - out["action_x"], press_y - out["action_y"])
    return out.drop(columns="action_location")


def build_press_sequences(
    pressures: pd.DataFrame,
    gap_seconds: float = config.PRESS_SEQUENCE_GAP_SECONDS,
) -> pd.DataFrame:
    """Cluster individual pressures into pressing sequences.

    A single Pressure event is not a press. Within one team and match, start a
    new sequence whenever the gap to the previous pressure exceeds
    ``gap_seconds`` OR the ``possession`` id changes. A new match, team or
    period also starts one: ``time_seconds`` restarts each period, so pressures
    must be ordered by period BEFORE time, or the two halves interleave.

    One row per sequence: ``sequence_id`` (unique across the frame), match_id,
    period, team, team_id, start_time, end_time (both seconds within the
    period), n_pressures, mean_x, any_counterpress, possession and
    possession_team (the possession pressed against, and the team that held it).

    ``mean_x`` is the RAW x, in the pressing team's own attacking frame, so it
    answers "how high did they press" directly. Do not mirror it.

    Measured on 64 WC 2022 matches at the default 5 s gap: a median of 94
    sequences per team per match (61-163), not the 40-80 this docstring once
    quoted. 76% are a single pressure and the median duration is 0 s, so most
    "sequences" are isolated presses. The count depends on the threshold (2 s:
    14,588 sequences; 12 s: 10,394), so report it together with the choice.
    None spans a possession or a period.
    """
    p = pressures.sort_values(["match_id", "team_id", "period", "time_seconds"]).reset_index(drop=True)

    gap = p.groupby(["match_id", "team_id", "period"])["time_seconds"].diff()
    new = gap.isna() | (gap > gap_seconds) | (p["possession"] != p["possession"].shift())
    p["sequence_id"] = new.cumsum()

    seq = p.groupby("sequence_id").agg(
        match_id=("match_id", "first"),
        period=("period", "first"),
        team=("team", "first"),
        team_id=("team_id", "first"),
        start_time=("time_seconds", "min"),
        end_time=("time_seconds", "max"),
        n_pressures=("id", "size"),
        mean_x=("x", "mean"),
        any_counterpress=("counterpress", "any"),
        possession=("possession", "first"),
        possession_team=("possession_team", "first"),
    ).reset_index()
    return seq


def label_sequence_outcomes(
    sequences: pd.DataFrame, events: pd.DataFrame,
    window: float = config.PRESS_REGAIN_WINDOW_SECONDS,
    high_regain_min_x: float = config.PRESS_REGAIN_MIN_X,
) -> pd.DataFrame:
    """Mark each sequence successful if the pressing team won the ball.

    A sequence succeeded if the NEXT possession -- the pressed possession's id
    plus one; ids are consecutive within a match -- belongs to the pressing
    team and its first event comes within ``window`` seconds of the sequence's
    last pressure.

    Do not hunt for the Ball Recovery event itself. A recovery that ends a
    possession is often still logged INSIDE the old possession block, so it
    carries the old id (trap #16). The first event of the next possession is
    always performed by the team that now has the ball, so read the time and
    the location from that event.

    All three must hold: the next possession's owner is the pressing team; the
    pressing team did not already own the pressed possession; and the next
    possession starts within ``window`` after the sequence ends. The second
    guard matters: 14.8% of sequences (1,840 of 12,424 on 64 WC 2022 matches)
    are pressures logged during the pressing team's OWN possession, typically
    17 s into it. They cannot win a ball the team already holds, and without
    the guard about 4% of them would be counted as wins.

    Two levels of success, because they answer different questions:
      * ``won_possession``    -- the ball was won at all
      * ``won_high``          -- won AND in the opponent half, i.e. the turnover
                                 location has x >= ``high_regain_min_x``

    Keep both. "Won the ball" measures whether the press worked; "won it high"
    measures whether it worked somewhere that mattered, and a team that only
    ever regains on its own edge of the box is not really pressing. Report the
    high-regain rate in the headline and the raw rate alongside it.

    The turnover event is performed by the pressing team, so its x is already
    in their attacking frame -- do NOT mirror it here (trap #2).

    Adds: won_possession (bool), won_high (bool), won_possession_id,
    seconds_to_regain, regain_x, regain_y. The last four are NaN for sequences
    that did not win the ball. ``seconds_to_regain`` is measured from the
    sequence's ``end_time``.

    Measured on 64 WC 2022 matches at the default 8 s window: 2,024 of 12,424
    sequences won the ball (16.3%) and 435 won it high (3.5%), a median 2.5 s
    after the last pressure, at a median regain x of 37. On the sample match,
    31 of 208 (Canada 17, Morocco 14), 2 of them high.
    """
    ev = events.assign(time_seconds=_timestamp_seconds(events["timestamp"])).sort_values(["match_id", "index"])
    xy = pd.DataFrame(ev["location"].map(_xy).tolist(), index=ev.index, columns=["x", "y"])
    ev = ev.join(xy)

    # one row per possession: its FIRST event (rows are in event order)
    starts = (ev.groupby(["match_id", "possession"])
                .agg(period=("period", "first"), t0=("time_seconds", "first"),
                    owner=("possession_team_id", "first"), x=("x", "first"), y=("y", "first"))
                .reset_index()
                .rename(columns={"possession": "won_possession_id", "period": "next_period"}))

    # the possession right after the one being pressed
    out = sequences.assign(won_possession_id=sequences["possession"] + 1).merge(
        starts, on=["match_id", "won_possession_id"], how="left")

    dt = out["t0"] - out["end_time"]
    ok = ((out["owner"] == out["team_id"])             # the next possession is the pressing team's
        & (out["possession_team"] != out["team"])   # ...and they did not already have the ball
        & (out["next_period"] == out["period"])     # time restarts each period
        & dt.between(0, window))

    out["won_possession"] = ok
    out["won_high"] = ok & (out["x"] >= high_regain_min_x)
    out["seconds_to_regain"] = dt.where(ok)
    out["regain_x"] = out["x"].where(ok)
    out["regain_y"] = out["y"].where(ok)
    out["won_possession_id"] = out["won_possession_id"].where(ok)
    return out.drop(columns=["next_period", "t0", "owner", "x", "y"])


def sequence_value(
    sequences: pd.DataFrame, rated_actions: pd.DataFrame, actions: pd.DataFrame
) -> pd.DataFrame:
    """Attach the xT the won possession went on to generate.

    For each successful sequence (``won_possession``), take the possession the
    pressing team won and compute, over THAT TEAM's actions in it:
      * xt_gained     -- sum of ``xt_delta`` over every move (net: backward
                         moves count against it)
      * reached_shot  -- bool, at least one shot
      * shot_xg       -- sum of the shots' xG
      * start_zone_xt -- the xT value of where the ball was won; half the story

    Needs both frames, and the reason is a trap. ``rated_actions`` (from
    ``ExpectedThreat.rate_actions``) holds only successful passes and carries:
    it has the xT but NO shots, so ``is_shot`` or ``xg`` cannot be read from it.
    Shots come from ``actions``.

    Filter to the pressing team before summing: a possession block also holds
    some actions by the other side (2% of the rated moves in the won
    possessions of 64 WC 2022 matches), and they are not the press's payoff.

    Sequences that did not win the ball keep NaN, and ``reached_shot`` False.
    They are never dropped: dropping them would make the denominator the count
    of successful presses, which flatters a team that presses rarely but well.
    A WON sequence with no rated move at all (3.2%) is different: it earned
    nothing, so ``xt_gained`` and ``shot_xg`` are 0, not NaN.

    ``start_zone_xt`` is the ``xt_start`` of the first rated move of the won
    possession. It needs no model, and tracks xT at the regain location closely
    (correlation 0.98, mean absolute difference 0.0003). NaN when there is no
    move to read it from.

    Measured on 64 WC 2022 matches, per won sequence: mean ``xt_gained`` 0.0139
    (median 0.0044), 10.4% reach a shot, mean ``shot_xg`` 0.0111, mean
    ``start_zone_xt`` 0.0083. Sample match: 31 won sequences, mean
    ``xt_gained`` 0.0117, 3 reach a shot.
    """
    raise NotImplementedError


def ppda(events: pd.DataFrame) -> pd.DataFrame:
    """Passes per defensive action, per team per match. Lower = more intense.

    Trainor's definition. The physical region is the opponent's own defensive
    60% of the pitch, but because coordinates are in the ACTING team's frame
    the two sides of the ratio use different thresholds for the same grass:

        numerator   opponent passes with x <= config.PPDA_OPPONENT_PASS_MAX_X
        denominator your Duels / Interceptions / Fouls Committed
                    with x >= config.PPDA_DEFENSIVE_ACTION_MIN_X

    Calibration: 7.4-12.3 on the three sample matches. If you land outside
    6-15, check the zone filter before anything else.
    """
    raise NotImplementedError


def pressures_per_opponent_half_touch(events: pd.DataFrame) -> pd.DataFrame:
    """Pressing intensity that does not depend on the opponent choosing to pass.

    Denominator is opponent touches in their own half. Arguably a better
    intensity measure than PPDA for exactly that reason.
    """
    raise NotImplementedError


def team_pressing_summary(
    sequences: pd.DataFrame, events: pd.DataFrame, rated_actions: pd.DataFrame
) -> pd.DataFrame:
    """One row per team: intensity and payoff side by side.

    Columns to end up with:
        ppda, pressures_per_opp_half_touch, n_sequences, regain_rate,
        high_regain_rate, xt_per_100_pressures, xt_per_successful_sequence,
        shots_from_presses

    ``xt_per_100_pressures`` is YOUR metric and the contribution of this
    project. PPDA says a team presses a lot; nobody is publishing what the
    pressing earns, in the same units as the rest of the model.
    """
    raise NotImplementedError
