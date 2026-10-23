"""The acceptance checks from the plan, as executable tests.

They are the definition of done for each phase. Tests for implemented code
run and pass; tests for the phases still to come are skipped, and each skip
reason names the function to implement. The loop: implement, delete the skip
mark, run the suite, and let it tell you when the phase is finished rather than
eyeballing a notebook.

    python -m pytest -q

Run it with pytest. ``python tests/test_invariants.py`` defines the tests and
calls none of them, so it prints nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from xtpress import config


# -- Phase 00 ---------------------------------------------------------------

def test_config_is_coherent():
    assert config.N_ZONES == config.GRID_COLS * config.GRID_ROWS
    assert config.PITCH_LENGTH == 120.0 and config.PITCH_WIDTH == 80.0
    for key, (comp, season) in config.COMPETITIONS.items():
        assert isinstance(comp, int) and isinstance(season, int), key
    for key in config.FIT_COMPETITIONS + [config.FOCUS_COMPETITION]:
        assert key in config.COMPETITIONS, key


def test_ppda_zone_thresholds_describe_the_same_strip():
    """The two thresholds must be mirror images: 120 - 72 == 48."""
    assert (
        config.PITCH_LENGTH - config.PPDA_OPPONENT_PASS_MAX_X
        == config.PPDA_DEFENSIVE_ACTION_MIN_X
    )


# -- Phase 01 ---------------------------------------------------------------

def test_actions_are_on_the_pitch(actions):
    assert actions.start_x.between(0, config.PITCH_LENGTH).all()
    assert actions.start_y.between(0, config.PITCH_WIDTH).all()
    moves = actions[actions.action_type.isin(config.MOVE_TYPES)]
    assert moves.end_x.notna().all()


def test_pass_completion_is_plausible(actions):
    """Trap #1: a COMPLETED pass has no outcome. Inverting this gives ~0.05."""
    passes = actions[actions.action_type == "pass"]
    assert 0.78 <= passes.successful.mean() <= 0.84


def test_shootout_is_excluded(actions):
    assert not actions.period.isin(config.EXCLUDED_PERIODS).any()


# -- Phase 02 ---------------------------------------------------------------

def test_zone_centres_round_trip_through_zone_index():
    """Binning a zone's own centre must return that zone, for every zone.

    Catches a flipped ``row * cols + col`` convention -- which transposes every
    heatmap without raising -- and any off-by-one at the edges. It does NOT
    catch a missing ``+ 0.5``: a cell's own corner bins back into that cell, so
    corner points round-trip perfectly. That needs the test below.
    """
    from xtpress.expected_threat import zone_centres, zone_index

    centres = zone_centres()
    assert centres.shape == (config.N_ZONES, 2)
    assert np.array_equal(zone_index(*centres.T), np.arange(config.N_ZONES))


def test_zone_centres_are_centres_and_not_corners():
    """The ``+ 0.5``. Without it every plotted label sits half a cell adrift.

    Two independent angles: each centre stands half a cell from its own edges,
    and the set of centres mirrors onto itself about the halfway line -- true
    of a uniform grid's centres, false of its corners, which bunch against the
    0 edge and leave nothing at 120.
    """
    from xtpress.expected_threat import zone_centres

    centres = zone_centres()
    cell_w = config.PITCH_LENGTH / config.GRID_COLS
    cell_h = config.PITCH_WIDTH / config.GRID_ROWS

    assert np.allclose(centres[:, 0] % cell_w, cell_w / 2)
    assert np.allclose(centres[:, 1] % cell_h, cell_h / 2)

    assert np.allclose(np.sort(centres[:, 0]), np.sort(config.PITCH_LENGTH - centres[:, 0]))
    assert np.allclose(np.sort(centres[:, 1]), np.sort(config.PITCH_WIDTH - centres[:, 1]))


@pytest.mark.parametrize("cols,rows", [(12, 8), (16, 12), (24, 16)])
def test_zone_centres_round_trip_on_other_grids(cols, rows):
    """The sensitivity-check grids must obey the same convention."""
    from xtpress.expected_threat import zone_centres, zone_index

    centres = zone_centres(cols, rows)
    binned = zone_index(*centres.T, cols=cols, rows=rows)
    assert np.array_equal(binned, np.arange(cols * rows))


def test_zone_index_handles_edges_and_missing_coordinates():
    """Trap #7 at both far edges, and -1 rather than a silent zone 0."""
    from xtpress.expected_threat import zone_index

    last = config.N_ZONES - 1
    x = np.array([0.0, 119.9, config.PITCH_LENGTH, np.nan, 60.0])
    y = np.array([0.0, 79.9, config.PITCH_WIDTH, 40.0, np.nan])
    assert list(zone_index(x, y)) == [0, last, last, -1, -1]


def test_unreachable_zones_detects_a_self_absorbing_zone():
    """m=1 and T[z,z]=1 -- the zone the ball enters and never leaves."""
    from xtpress.expected_threat import assert_solvable, unreachable_zones

    n = 4
    T = np.full((n, n), 1 / n)
    T[2] = 0.0
    T[2, 2] = 1.0                      # zone 2 only ever returns to itself
    m = np.array([0.9, 0.9, 1.0, 0.9])  # and is never shot from

    assert unreachable_zones(m, T).tolist() == [2]
    with pytest.raises(ValueError, match="no unique solution"):
        assert_solvable(m, T)


def test_unreachable_zones_detects_a_closed_group():
    """Two zones passing only to each other. No row is all-zero."""
    from xtpress.expected_threat import unreachable_zones

    n = 5
    T = np.full((n, n), 1 / n)
    T[1] = 0.0; T[1, 3] = 1.0
    T[3] = 0.0; T[3, 1] = 1.0
    m = np.array([0.9, 1.0, 0.9, 1.0, 0.9])

    A = np.eye(n) - m[:, None] * T
    assert not (np.abs(A).sum(axis=1) < 1e-12).any(), "no all-zero row to find"
    assert unreachable_zones(m, T).tolist() == [1, 3]


def test_a_healthy_system_passes_and_solves():
    from xtpress.expected_threat import assert_solvable, unreachable_zones

    n = 4
    T = np.full((n, n), 1 / n)
    m = np.full(n, 0.9)

    assert unreachable_zones(m, T).tolist() == []
    assert_solvable(m, T)
    xt = np.linalg.solve(np.eye(n) - m[:, None] * T, np.full(n, 0.01))
    assert np.isfinite(xt).all()


def test_transition_matrix_is_row_stochastic(model):
    assert np.allclose(model.T.sum(axis=1), 1.0, atol=1e-9)


def test_shot_and_move_probabilities_sum_to_one(model):
    assert np.allclose(model.s + model.m, 1.0, atol=1e-12)


def test_iterative_and_closed_form_agree(model):
    """Each solver is a free correctness check on the other."""
    iterative, _ = model.solve_iterative()
    assert np.allclose(iterative, model.solve_closed_form(), atol=1e-9)


def test_grid_has_no_nans(model):
    assert np.isfinite(model.xt).all()


def test_completion_rates_known_answer():
    """Hand-countable: three zones across the pitch, one row.

    Zone 0 (x < 40): 4 move attempts, 3 arrive              -> 0.75
    Zone 1 (40-80):  1 attempt, 1 arrives                   -> 1.00
    Zone 2 (x > 80): no attempts -> the OVERALL rate, 4 / 5 -> 0.80

    The shot and the Dispossessed sit in zone 0 and must not count as move
    attempts: this is the denominator of m(z), so it is passes and carries only.
    """
    from xtpress.expected_threat import completion_rates

    rows = [
        ("pass", 10, True), ("pass", 10, True), ("carry", 10, True),
        ("pass", 10, False),
        ("shot", 10, True), ("dispossessed", 10, False),
        ("pass", 60, True),
    ]
    frame = pd.DataFrame(rows, columns=["action_type", "start_x", "successful"])
    frame["start_y"] = 40.0

    c = completion_rates(frame, cols=3, rows=1)
    assert np.allclose(c, [0.75, 1.0, 0.8])


# -- Phase 03 ---------------------------------------------------------------

def test_value_increases_toward_the_goal(model):
    """Column means must rise monotonically from own half to opponent goal."""
    col_means = model.grid().mean(axis=0)
    assert np.all(np.diff(col_means) > 0)


def test_magnitudes_match_the_calibration_table(model):
    grid = model.grid()
    assert 0.002 <= grid[:, 0].mean() <= 0.020   # own defensive third
    assert 0.15 <= grid.max() <= 0.35            # six-yard box peak


def test_agrees_with_karun_singhs_published_grid(model):
    from xtpress.expected_threat import load_reference_grid

    reference = load_reference_grid()
    r = np.corrcoef(model.grid().ravel(), reference.ravel())[0, 1]
    assert r > 0.97


def test_loss_modelling_is_what_recovers_the_published_surface(full_actions):
    """Ablation: the same data and grid, with and without the completion factor.

    The textbook recursion has rows of T summing to 1, so a failed move never
    ends the chain and value mixes across the pitch until every zone is nearly
    equal. Measured on WC 2022 at 12x8:

        model_loss=False   r = 0.860   goal-end / own-end = 1.4x
        model_loss=True    r = 0.996   goal-end / own-end = 18.7x
        published grid                                    12.6x

    If the two come out alike, ``fit`` is not applying ``c`` -- or the factor
    was wired into only one of the two solvers.
    """
    from xtpress.expected_threat import ExpectedThreat, load_reference_grid

    reference = load_reference_grid()
    with_loss = ExpectedThreat().fit(full_actions, model_loss=True).grid()
    baseline = ExpectedThreat().fit(full_actions, model_loss=False).grid()

    def r(grid):
        return np.corrcoef(grid.ravel(), reference.ravel())[0, 1]

    def goal_to_own(grid):
        col_means = grid.mean(axis=0)
        return col_means[-1] / col_means[0]

    assert r(with_loss) > 0.97
    assert r(baseline) < 0.90
    assert r(with_loss) - r(baseline) > 0.05

    assert goal_to_own(baseline) < 3
    assert goal_to_own(with_loss) > 8


def test_both_solvers_apply_the_completion_factor(full_actions):
    """Wiring check for the loss model.

    An iteration that used ``m`` alone would converge to the BASELINE surface,
    and a closed form that did the same would agree with it -- both wrong, in
    step. So check the default fit against its own iteration AND against the
    baseline, which must differ.
    """
    from xtpress.expected_threat import ExpectedThreat

    headline = ExpectedThreat().fit(full_actions, model_loss=True)
    baseline = ExpectedThreat().fit(full_actions, model_loss=False)

    iterative, _ = headline.solve_iterative()
    assert np.allclose(iterative, headline.xt, atol=1e-9)
    assert not np.allclose(headline.xt, baseline.xt, atol=1e-3)
    assert headline.c.min() < 1.0
    assert np.all(baseline.c == 1.0)

def test_save_load_round_trip(model, tmp_path):
    from xtpress.expected_threat import ExpectedThreat

    path = model.save(tmp_path / "model.npy")
    loaded = ExpectedThreat.load(path)

    assert np.array_equal(loaded.xt, model.xt)
    assert np.array_equal(loaded.grid(), model.grid())
    assert loaded.metadata == model.metadata
    assert loaded.c is None


def test_load_rejects_a_mismatched_pair(model, tmp_path):
    from xtpress.expected_threat import ExpectedThreat

    np.save(tmp_path / "bad.npy", np.zeros((6, 8)))
    (tmp_path / "bad.json").write_text(json.dumps(model.metadata))
    with pytest.raises(ValueError, match="mismatched"):
        ExpectedThreat.load(tmp_path / "bad.npy")


def test_variants_save_to_different_default_files(full_actions, tmp_path, monkeypatch):
    from xtpress.expected_threat import ExpectedThreat

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    headline = ExpectedThreat().fit(full_actions).save()
    baseline = ExpectedThreat().fit(full_actions, model_loss=False).save()
    assert headline.name == "xt_grid_12x8.npy"
    assert baseline.name == "xt_grid_12x8_baseline.npy"

def test_rate_actions_known_answer():
    """Three zones, xt = [0.1, 0.2, 0.4]; hand-countable.

    Forward +0.3 and backward -0.3 are kept; a failed pass and a shot are not
    rated at all. Reversing start and end flips both signs.
    """
    from xtpress.expected_threat import ExpectedThreat

    model = ExpectedThreat(cols=3, rows=1)
    model.xt = np.array([0.1, 0.2, 0.4])

    nan = np.nan
    frame = pd.DataFrame(
        [
            ("pass",  True,  10.0, 100.0),   # forward
            ("pass",  False, 10.0, 100.0),   # failed -> excluded
            ("carry", True, 100.0,  10.0),   # backward -> kept
            ("shot",  True, 100.0,   nan),   # NaN end -> excluded
        ],
        columns=["action_type", "successful", "start_x", "end_x"],
    )
    frame["start_y"] = 40.0
    frame["end_y"] = 40.0
    frame.loc[3, "end_y"] = nan

    rated = model.rate_actions(frame)

    assert list(rated.index) == [0, 2]
    assert np.allclose(rated.xt_delta, [0.3, -0.3])
    assert np.allclose(rated.xt_delta_clipped, [0.3, 0.0])

def test_minutes_played_known_answer():
    """One match, two 45-minute periods, hand-countable.

    A1 plays 90. A2 is replaced 15 minutes into period 2 -> 60, and A3 plays
    the remaining 30. B1 is sent off 20 minutes into period 2 -> 65. Events in
    period 3 (extra time) and period 5 (shootout) must not lengthen the match.
    """
    from xtpress import loaders

    def lineup(*players):
        return {"lineup": [{"player": {"id": i, "name": n}} for i, n in players]}

    rows = [
        (1, "Starting XI", "00:00:00.000", "A", None, {"tactics": lineup((1, "A1"), (2, "A2"))}),
        (1, "Starting XI", "00:00:00.000", "B", None, {"tactics": lineup((10, "B1"))}),
        (1, "Half End", "00:45:00.000", "A", None, {}),
        (2, "Substitution", "00:15:00.000", "A", 2,
            {"substitution_replacement": "A3", "substitution_replacement_id": 3.0}),
        (2, "Foul Committed", "00:20:00.000", "B", 10, {"foul_committed_card": "Red Card"}),
        (2, "Half End", "00:45:00.000", "A", None, {}),
        (3, "Pass", "00:10:00.000", "A", 1, {}),
        (5, "Pass", "00:02:00.000", "A", 1, {}),
    ]
    frame = pd.DataFrame(
        [dict(match_id=1, period=p, type=t, timestamp=ts, team=tm, player_id=pid, **x)
         for p, t, ts, tm, pid, x in rows]
    )

    got = loaders.minutes_played(frame).set_index("player").minutes.to_dict()
    assert got == {"A1": 90.0, "A2": 60.0, "A3": 30.0, "B1": 65.0}, got


# -- Phase 06 ---------------------------------------------------------------

@pytest.mark.skip(reason="implement pressing_sequences.to_possession_frame")
def test_frame_mirror_is_an_involution():
    """Trap #2: applying the mirror twice must return the original point."""
    from xtpress.pressing_sequences import to_possession_frame

    x, y = np.array([47.1]), np.array([46.8])
    assert np.allclose(to_possession_frame(*to_possession_frame(x, y)), (x, y))


@pytest.mark.skip(reason="implement pressing_sequences.ppda")
def test_ppda_lands_in_the_published_band(ppda_table):
    """Measured 7.4-12.3 on three sample matches. Outside 6-15, check the zone."""
    assert ppda_table.ppda.between(6, 15).mean() > 0.9


@pytest.mark.skip(reason="implement pressing_sequences.build_press_sequences")
def test_sequences_never_span_a_possession_change(sequences):
    assert sequences.possession.notna().all()


# -- Phase 07 ---------------------------------------------------------------

@pytest.mark.skip(reason="implement press_resistance.pressured_actions")
def test_pressured_sample_is_the_right_size(actions, pressured):
    """Far below 8% means you keyed off Ball Receipts instead of the action."""
    assert 0.08 <= len(pressured) / len(actions) <= 0.12


@pytest.mark.skip(reason="implement press_resistance.classify_outcome")
def test_retention_rate_is_plausible(pressured):
    assert 0.65 <= pressured.retained.mean() <= 0.80


@pytest.mark.skip(reason="implement press_resistance.player_press_resistance")
def test_zone_adjustment_actually_changes_the_ranking(resistance_table):
    """If raw and adjusted rank identically, the adjustment is not wired in."""
    raw = resistance_table.retention_rate.rank()
    adjusted = resistance_table.retention_above_expected.rank()
    assert raw.corr(adjusted, method="spearman") < 0.99
