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
from .expected_threat import zone_index


def pressured_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Select the actions a player took while under pressure.

    Key off the ACTION, not the receipt. Measured on one match: 236 Carries
    and 100 Passes carry ``under_pressure``, but only 47 Ball Receipts --
    keying off receipts discards roughly 80% of the sample (trap #14).

    A pressured action is one with ``under_pressure`` set whose ``action_type``
    is a pass, a carry, a dribble, or one of ``config.LOSS_EVENT_TYPES``
    (dispossessed, miscontrol). The loss events have to be in the sample: a
    StatsBomb Carry is never recorded as failed (all 13,700 pressured carries
    are "successful"), and losing the ball under pressure is a separate
    Dispossessed or Miscontrol event. Leave them out and retention reads 86.5%
    instead of 80.1%. Shots are left out (238 pressured): a shot is not a
    decision to keep the ball.

    Dribbles and dispossessions are flagged ``under_pressure`` every time, so
    they are always in; passes are flagged 11.5% of the time, carries 25.5%.

    Columns pass through unchanged. If ``classify_outcome`` is to use the
    rating, join ``xt_start``, ``xt_end`` and ``xt_delta`` onto ``actions``
    first (``rate_actions`` keeps the original index, so a plain ``join``
    works). Only successful passes and carries are rated, so everything else
    has NaN there.

    The receipt carry-over option from the original stub is not implemented.
    The action frame has no Ball Receipt events, and on the 64 WC 2022 matches
    only 5 actions (0.02%) follow a pressured receipt without carrying the flag
    themselves, so it would change nothing.

    Sanity, as a share of ACTIONS (not of events): 19.6% over the 64 matches
    (25,270 of 128,711) and 22.0% on the sample match. As a share of all events
    the pass and carry part is about 10%, which is where the 8-12% in the
    original brief comes from. Far below that either way means you keyed off
    Ball Receipts.
    """
    candidates = config.MOVE_TYPES + ("dribble",) + config.LOSS_EVENT_TYPES
    mask = actions["under_pressure"] & actions["action_type"].isin(candidates)
    return actions[mask].copy()


def classify_outcome(pressured: pd.DataFrame) -> pd.DataFrame:
    """Label what happened to each pressured action.

        retained_progressed   successful pass or carry, xt_delta > 0
        retained_flat         successful pass or carry, xt_delta <= 0
        retained_dribble      successful dribble
        lost                  Dispossessed, Miscontrol, an incomplete pass or
                              an incomplete dribble

    Adds a boolean ``retained`` (the action's ``successful``) and a categorical
    ``outcome`` with those four categories, in that order. The input is not
    modified.

    Two departures from the original list of categories.

    A dribble has no end location, so ``rate_actions`` never rates it and its
    ``xt_delta`` is NaN. It is retained when it succeeds, but it can be neither
    "progressed" nor "flat", and folding it into "flat" would mislabel 937
    actions (3.7%), so it gets its own label.

    ``retained_foul_won`` is gone. Foul Won is not in the action frame, which
    keeps only six event types. The raw events hold 1,674 pressured Foul Wons
    (6.6% on top of the sample); counting them as retained would lift retention
    from 80.1% to 81.3%. It is a known gap, not something this function can fix.

    Needs ``xt_delta``: join the rating onto the actions before calling
    ``pressured_actions``. A successful pass or carry with no ``xt_delta`` means
    that join was skipped, and without a check every one of them would quietly
    become ``retained_flat``.

    Measured on 64 WC 2022 matches (25,270 pressured actions): retention 80.1%
    (89.0% for the same event types when not pressured; 90.2% is passes and
    carries alone, which is not the same population). Progressed 27.2%, flat
    49.2%, dribble 3.7%, lost 19.9%. Of the retained moves 36% have an ``xt_delta`` of
    exactly zero (the move stayed inside one zone) and 28% are negative. Sample
    match: 81.2%.
    """
    if "xt_delta" not in pressured.columns:
        raise ValueError("join the rating (xt_delta) onto the actions first")
    moves = pressured["action_type"].isin(config.MOVE_TYPES) & pressured["successful"]
    if pressured.loc[moves, "xt_delta"].isna().any():
        raise ValueError("a successful pass or carry has no xt_delta: the rating was not joined onto every action")

    retained = pressured["successful"]
    is_dribble = pressured["action_type"] == "dribble"

    outcome = np.select(
        [~retained, is_dribble & retained, retained & (pressured["xt_delta"] > 0)],
        ["lost", "retained_dribble", "retained_progressed"],
        default="retained_flat",
    )
    categories = ["retained_progressed", "retained_flat", "retained_dribble", "lost"]
    return pressured.assign(retained=retained, outcome=pd.Categorical(outcome, categories=categories))


def zone_baselines(pressured: pd.DataFrame) -> pd.DataFrame:
    """League-average retention rate and xT delta per grid zone, under pressure.

    This is the input to the adjustment that makes the metric original. Uses
    the same grid as the xT model, so import ``zone_index`` from
    ``expected_threat`` rather than re-binning here.

    One row per zone, every zone: ``zone`` (the flat index, z = row * cols +
    col), ``n`` (pressured actions that started there), ``expected_retention``
    and ``expected_xt_delta``. A zone nobody was pressured in still gets a row
    (n = 0, NaN values), so a merge on ``zone`` never drops an action without
    saying so. On WC 2022 every zone has data: the smallest has 28 actions, the
    median 251 and the largest 584, and only two have fewer than 50 (0.2% of
    the sample). So the baselines are not smoothed.

    The zone is where the action STARTED, that is where the player had the ball
    when the press arrived. A dribble has no end location, so the start is the
    only place all of them share.

    ``expected_xt_delta`` is the mean over retained passes and carries, the only
    rated actions. A lost ball or a dribble has NaN there and is skipped, not
    counted as zero. That keeps the two axes apart: retention says whether the
    ball survived, the delta says whether it went somewhere when it did.

    Measured on 64 WC 2022 matches. Across zones with at least 50 actions
    retention runs from 0.53 to 0.89 (median 0.82; 0.83 in the own third, 0.83
    in the middle, 0.74 in the final third) and the mean xT delta from -0.008 to
    +0.012. Zone alone explains only 2.5% of the variance in whether an action
    is retained and 8.6% of the xT delta, so the adjustment is real but modest.
    """
    zone = zone_index(pressured["start_x"], pressured["start_y"], config.GRID_COLS, config.GRID_ROWS)
    valid = zone >= 0
    table = (pressured[valid].assign(zone=zone[valid])
             .groupby("zone")
             .agg(n=("retained", "count"),
                  expected_retention=("retained", "mean"),
                  expected_xt_delta=("xt_delta", "mean")))
    return (table.reindex(range(config.GRID_COLS * config.GRID_ROWS))
                .rename_axis("zone").reset_index()
                .assign(n=lambda t: t["n"].fillna(0).astype(int)))


def player_press_resistance(pressured: pd.DataFrame, baselines: pd.DataFrame,
                            minutes: pd.DataFrame | None = None,
                            min_actions: int = config.MIN_PRESSURED_ACTIONS) -> pd.DataFrame:
    """The leaderboard. One row per player.

    Raw columns:
        n_pressured, retention_rate, mean_xt_delta, xt_retained_per_90

    Adjusted columns -- THE STEP THAT MAKES THIS YOURS:
        retention_above_expected, xt_delta_above_expected

    A raw ratio mostly measures WHERE a player receives the ball. A centre-back
    pressured on his own six-yard line and a winger pressured in the final third
    face completely different tasks, and an uncorrected metric ranks them as if
    they did the same job. Each pressured action is therefore compared with the
    league average for the zone it started in (``baselines``, from
    ``zone_baselines``). ``retention_above_expected`` is the player's retention
    minus the mean of those zone expectations over his actions;
    ``xt_delta_above_expected`` is the mean, over his retained passes and
    carries, of the delta minus the zone's expected delta. Then "+4.2% retention
    above expectation" means something a ratio never could.

    ``xt_retained_per_90`` is the summed ``xt_delta`` of the retained moves per 90
    minutes. ``minutes`` has one row per match, so sum it per player first, and
    note that ``player_id`` is a float in the action frame and an int in the
    minutes table. The minutes cover regulation only, so actions in extra time
    (2.2% of the pressured sample) are left out of this column, though not of the
    others. NaN if ``minutes`` is None or the player has no row in it.

    Guards: at least ``min_actions`` pressured actions (50 keeps 179 of the 663
    players on WC 2022), and every row carries ``position`` (the player's most
    common label) and ``position_group`` (GK, DF, MF or FW). Raw positions are
    too fine to compare within (23 labels), so each adjusted axis also gets its
    percentile within the group: ``retention_pct`` and ``threat_pct``. The groups
    hold 71 MF, 66 FW, 40 DF and only 2 GK, so the goalkeepers' percentiles mean
    little. Position still matters after the zone adjustment: a goalkeeper is
    third on retention above expectation (Unai Simon, +0.117).

    The baseline includes the player's own actions. Leaving them out gives the
    same ranking (Spearman 0.9999, largest difference 0.011), so it is not
    corrected.

    Check: the adjusted ranking must differ visibly from the raw one. It does:
    Spearman 0.935 between raw and adjusted retention. Retention above
    expectation runs from -0.158 to +0.120 (median +0.014), and the two adjusted
    axes correlate at -0.18, so they measure different things.
    """
    # 1. compare every action with its zone's league average
    zone = zone_index(pressured["start_x"], pressured["start_y"], config.GRID_COLS, config.GRID_ROWS)
    d = pressured.assign(zone=zone).merge(
        baselines[["zone", "expected_retention", "expected_xt_delta"]], on="zone", how="left")
    d["delta_above"] = d["xt_delta"] - d["expected_xt_delta"]
    d["player_id"] = d["player_id"].astype(int)

    # 2. one row per player
    g = d.groupby("player_id")
    table = pd.DataFrame({
        "player": g["player"].first(),
        "position": g["position"].agg(lambda s: s.mode().iat[0]),
        "n_pressured": g.size(),
        "retention_rate": g["retained"].mean(),
        "mean_xt_delta": g["xt_delta"].mean(),
        "retention_above_expected": g["retained"].mean() - g["expected_retention"].mean(),
        "xt_delta_above_expected": g["delta_above"].mean(),
    })

    # 3. xT of the retained moves per 90 minutes -- regulation only, like the minutes table
    table["xt_retained_per_90"] = np.nan
    if minutes is not None:
        reg = d[d["period"].isin(config.REGULATION_PERIODS)].groupby("player_id")["xt_delta"].sum()
        table["xt_retained_per_90"] = reg / minutes.groupby("player_id")["minutes"].sum() * 90

    # 4. position group
    pos = table["position"]
    table["position_group"] = np.select(
        [pos.str.contains("keeper", case=False), pos.str.contains("back", case=False),
         pos.str.contains("mid", case=False)],
        ["GK", "DF", "MF"], default="FW")

    # 5. the sample guard and the within-group percentiles
    table = table[table["n_pressured"] >= min_actions].copy()
    for col, name in [("retention_above_expected", "retention_pct"), ("xt_delta_above_expected", "threat_pct")]:
        table[name] = table.groupby("position_group")[col].rank(pct=True)
    return table.reset_index()


def press_resistance_score(
    table: pd.DataFrame, w_retention: float = 0.5, w_threat: float = 0.5
) -> pd.Series:
    """Collapse the two adjusted axes into one sortable score.

    Each axis -- ``retention_above_expected`` and ``xt_delta_above_expected`` --
    is standardised to a z-score across the table (mean 0, sample standard
    deviation 1), then the two are blended as
    ``w_retention * z_retention + w_threat * z_threat``. Standardising first is
    what makes the weights mean anything: retention varies by ~0.054 between
    players and threat by ~0.0015, so a raw blend would be 99.9% retention
    whatever the weights said.

    The weights are arguments, not constants, and the README states the ones
    used; the whole reason the scatter leads is that they are a judgement call
    rather than a result. The two axes are mildly *opposed* (Spearman -0.18:
    players who keep the ball tend to do it less threateningly), so no weighting
    is neutral.

    Returns a Series aligned to ``table.index``; the input is not modified.
    """
    axis_r = table["retention_above_expected"]
    axis_t = table["xt_delta_above_expected"]

    z_r = (axis_r - axis_r.mean()) / axis_r.std()     # z-score: centre, then scale
    z_t = (axis_t - axis_t.mean()) / axis_t.std()

    return w_retention * z_r + w_threat * z_t
