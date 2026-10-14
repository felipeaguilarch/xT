"""Player-level press resistance -- the standout metric.

Flip the lens from Phase 06: who retains and progresses the ball when the
press arrives?

The headline output is deliberately TWO axes, not one composite:
    x  retention rate under pressure      -- does the ball survive?
    y  mean xT delta per pressured action -- does it go somewhere useful?

Quadrants name themselves: safe recyclers, risk-takers, the genuinely
press-resistant, and the liability. The composite score exists only to sort
the table underneath that scatter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def pressured_actions(
    actions: pd.DataFrame, include_receipt_carryover: bool = False
) -> pd.DataFrame:
    """Select the actions a player took while under pressure.

    Key off the ACTION, not the receipt. Measured on one match: 236 Carries
    and 100 Passes carry ``under_pressure``, but only 47 Ball Receipts --
    keying off receipts discards roughly 80% of the sample (trap #14).

        PRESSURED = actions.under_pressure == True
        CANDIDATE = actions.action_type.isin(["pass", "carry", "dribble"])

    ``include_receipt_carryover`` widens the net to actions whose immediately
    preceding Ball Receipt by the same player was flagged. Report which
    definition you used -- the leaderboard shifts under it.

    Sanity: the pressured sample should be roughly 8-12% of all actions.
    """
    raise NotImplementedError


def classify_outcome(pressured: pd.DataFrame) -> pd.DataFrame:
    """Label what happened to each pressured action.

        retained_progressed   successful move, xt_delta > 0
        retained_flat         successful move, xt_delta <= 0
        retained_foul_won     Foul Won under pressure -- a legitimate escape
        lost                  Dispossessed, Miscontrol, or incomplete pass

    Adds a boolean ``retained`` and a categorical ``outcome``.
    League-wide retention should land around 65-80%.
    """
    raise NotImplementedError


def zone_baselines(pressured: pd.DataFrame) -> pd.DataFrame:
    """League-average retention rate and xT delta per grid zone, under pressure.

    This is the input to the adjustment that makes the metric original. Uses
    the same grid as the xT model, so import ``zone_index`` from
    ``expected_threat`` rather than re-binning here.
    """
    raise NotImplementedError


def player_press_resistance(
    pressured: pd.DataFrame,
    baselines: pd.DataFrame,
    minutes: pd.DataFrame | None = None,
    min_actions: int = config.MIN_PRESSURED_ACTIONS,
) -> pd.DataFrame:
    """The leaderboard. One row per player.

    Raw columns:
        n_pressured, retention_rate, mean_xt_delta, xt_retained_per_90

    Adjusted columns -- THE STEP THAT MAKES THIS YOURS:
        retention_above_expected, xt_delta_above_expected

    A raw ratio mostly measures WHERE a player receives the ball. A centre-back
    pressured on his own six-yard line and a winger pressured in the final
    third face completely different tasks, and an uncorrected metric ranks them
    as if they did the same job. Score each player on actual minus expected
    given the zone they received in, using ``baselines``. Then "+4.2% retention
    above expectation" means something a ratio never could, and it survives the
    first hard question an interviewer asks.

    Guards: minimum ``min_actions`` pressured actions, and carry a position
    label on every row so the reader can compare like with like. Consider
    reporting a per-position percentile rather than a raw rank.

    Check: the adjusted ranking must differ visibly from the raw one. If it
    does not, the adjustment is not actually wired in.
    """
    raise NotImplementedError


def press_resistance_score(
    table: pd.DataFrame, w_retention: float = 0.5, w_threat: float = 0.5
) -> pd.Series:
    """Collapse the two adjusted axes into one sortable score.

    Standardise each axis before weighting -- they are in different units.
    Keep the weights as arguments and state them in the README; the whole
    reason the scatter leads is that these weights are a judgement call rather
    than a result.
    """
    raise NotImplementedError
