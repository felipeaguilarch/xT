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
    raise NotImplementedError


def extract_pressures(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per Pressure event with the fields Phase 06 needs.

    Columns: match_id, period, time_seconds, team, team_id, player, player_id,
    x, y, duration, counterpress, possession, possession_team, related_events.

    Note ``duration`` is how long the press LASTED (median 0.47s) -- it is not
    a window for finding affected events (trap #13).
    """
    raise NotImplementedError


def link_pressures_to_actions(
    pressures: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Attach the opponent action each pressure affected.

    Resolve ``related_events`` ids against the event frame and keep the first
    related event belonging to the OTHER team. This works for 93% of pressures
    -- do not build a time-window join as the primary path.

    Fallback for the remaining 7%: a TWO-SIDED window of
    ``config.PRESS_MATCH_WINDOW_SECONDS``. The gap runs from about -1.2s to
    +0.9s because a Carry is timestamped at its start while the press arrives
    partway through, so a forward-only window loses most matches (trap #15).
    """
    raise NotImplementedError


def build_press_sequences(
    pressures: pd.DataFrame,
    gap_seconds: float = config.PRESS_SEQUENCE_GAP_SECONDS,
) -> pd.DataFrame:
    """Cluster individual pressures into pressing sequences.

    A single Pressure event is not a press. Within one team and match, start a
    new sequence whenever the gap to the previous pressure exceeds
    ``gap_seconds`` OR the ``possession`` id changes.

    One row per sequence: start/end time, n_pressures, mean_x (how high),
    any_counterpress, possession id pressed against, team.

    Expect roughly 40-80 sequences per team per match, and none spanning a
    possession change.
    """
    raise NotImplementedError


def label_sequence_outcomes(
    sequences: pd.DataFrame, events: pd.DataFrame,
    window: float = config.PRESS_REGAIN_WINDOW_SECONDS,
    high_regain_min_x: float = config.PRESS_REGAIN_MIN_X,
) -> pd.DataFrame:
    """Mark each sequence successful if the pressing team won the ball.

    Walk forward from the sequence for up to ``window`` seconds and find the
    first event whose ``possession`` id differs; the sequence succeeded if that
    new possession belongs to the pressing team.

    Careful: a Ball Recovery that ends a possession is often still logged
    INSIDE the old possession block, so do not assume the recovery event itself
    carries the new id (trap #16).

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
    seconds_to_regain, regain_x, regain_y.
    """
    raise NotImplementedError


def sequence_value(
    sequences: pd.DataFrame, rated_actions: pd.DataFrame
) -> pd.DataFrame:
    """Attach the xT the won possession went on to generate.

    For each successful sequence, take the possession the pressing team won and
    compute:
      * xt_gained   -- sum of xt_delta over every move in that possession
      * reached_shot (bool) and shot_xg (sum)
      * start_zone_xt -- the xT value of where the ball was won; half the story

    Requires ``rated_actions`` from ``ExpectedThreat.rate_actions``.
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
