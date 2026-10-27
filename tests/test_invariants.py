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

def test_to_possession_frame_known_answer():
    """The corner maps to the opposite corner; the centre spot to itself."""
    from xtpress.pressing_sequences import to_possession_frame

    x, y = to_possession_frame(np.array([0.0, 60.0]), np.array([0.0, 40.0]))
    assert np.allclose(x, [120.0, 60.0])
    assert np.allclose(y, [80.0, 40.0])

def test_frame_mirror_is_an_involution():
    """Trap #2: applying the mirror twice must return the original point."""
    from xtpress.pressing_sequences import to_possession_frame

    x, y = np.array([47.1]), np.array([46.8])
    assert np.allclose(to_possession_frame(*to_possession_frame(x, y)), (x, y))

def test_extract_pressures_on_the_sample_match(events):
    """Canada v Morocco: 289 Pressure events, 58 of them counterpresses."""
    from xtpress.pressing_sequences import extract_pressures

    p = extract_pressures(events)

    assert list(p.columns) == [
        "id", "match_id", "period", "time_seconds", "team", "team_id", "player",
        "player_id", "x", "y", "duration", "counterpress", "possession",
        "possession_team", "related_events",
    ]
    assert len(p) == (events["type"] == "Pressure").sum() == 289
    assert p["id"].is_unique and p["id"].notna().all()
    assert p["counterpress"].dtype == bool and p["counterpress"].sum() == 58
    assert p[["x", "y"]].notna().all().all()
    assert p["x"].between(0, 120).all() and p["y"].between(0, 80).all()
    assert abs(p["duration"].median() - 0.47) < 0.01

def test_link_pressures_known_answer():
    """Three pressures against four events, each built to trip one mistake.

    p1 links through related_events to e1. p2 has no related events, so it
    falls back to the window and must pick e2, the OPPONENT event 0.5 s BEFORE
    it:
      * a forward-only window misses e2 (trap #15)
      * e3 is at the same clock time but in period 2, so a join without
        ``period`` picks it instead
      * e4, 0.2 s after, belongs to the pressing team's own side
    p3 has nothing within the window and must stay in the output, unlinked.
    """
    from xtpress.pressing_sequences import link_pressures_to_actions

    events = pd.DataFrame(
        [
            ("e1", 1, "00:10:00.000", 2, 22, "Carry", True, [30.0, 40.0]),
            ("e2", 1, "00:19:59.500", 2, 23, "Pass", True, [50.0, 40.0]),
            ("e3", 2, "00:20:00.000", 2, 24, "Pass", True, [10.0, 10.0]),
            ("e4", 1, "00:20:00.200", 1, 11, "Carry", True, [99.0, 5.0]),
        ],
        columns=["id", "period", "timestamp", "team_id", "player_id", "type",
                 "under_pressure", "location"],
    )
    events.insert(1, "match_id", 1)
    pressures = pd.DataFrame(
        [
            ("p1", 1, 600.3, 1, 90.0, 40.0, ["e1"]),
            ("p2", 1, 1200.0, 1, 70.0, 40.0, None),
            ("p3", 1, 3000.0, 1, 60.0, 40.0, None),
        ],
        columns=["id", "period", "time_seconds", "team_id", "x", "y", "related_events"],
    )
    pressures.insert(1, "match_id", 1)

    out = link_pressures_to_actions(pressures, events)

    assert len(out) == 3
    assert list(out["link_method"].fillna("none")) == ["related", "window", "none"]
    assert list(out["action_id"].fillna("-")) == ["e1", "e2", "-"]
    assert list(out["action_player_id"].fillna(-1)) == [22, 23, -1]
    assert np.allclose(out["link_distance"].fillna(-1), [0.0, 0.0, -1.0])


def test_link_pressures_on_the_sample_match(events):
    from xtpress.pressing_sequences import extract_pressures, link_pressures_to_actions

    pressures = extract_pressures(events)
    linked = link_pressures_to_actions(pressures, events)

    assert len(linked) == len(pressures) and linked["id"].is_unique
    assert linked["link_method"].value_counts().to_dict() == {"related": 254, "window": 18}
    assert linked["link_method"].isna().sum() == 17
    related = linked[linked["link_method"] == "related"]
    assert related["link_distance"].median() < 6      # about 73 if the mirror is forgotten

def test_build_press_sequences_known_answer():
    """Team A has five pressures that must split into four sequences; team B
    has two that form one. Rows are deliberately shuffled.

    a1-a3  one sequence (gaps 2 s, possession 5)
    b1     25 s later: the GAP breaks it (11 s > 5 s)
    c1     1 s after b1 but possession 6: the POSSESSION breaks it
    d1     period 2 at t=13: sorted by time alone it would land between a2
           and a3, and without a period break it would join c1
    e1-e2  the other team, so a separate sequence
    """
    from xtpress.pressing_sequences import build_press_sequences

    rows = [
        ("e2", 1, 11.0, 7, 30.0, False, 2), ("a3", 1, 14.0, 5, 90.0, False, 1),
        ("c1", 1, 26.0, 6, 50.0, False, 1), ("a1", 1, 10.0, 5, 70.0, False, 1),
        ("d1", 2, 13.0, 6, 40.0, False, 1), ("b1", 1, 25.0, 5, 60.0, False, 1),
        ("a2", 1, 12.0, 5, 80.0, True, 1),  ("e1", 1, 10.0, 7, 20.0, False, 2),
    ]
    pressures = pd.DataFrame(rows, columns=["id", "period", "time_seconds", "possession",
                                            "x", "counterpress", "team_id"])
    pressures["match_id"] = 1
    pressures["team"] = pressures["team_id"].map({1: "A", 2: "B"})
    pressures["possession_team"] = pressures["team_id"].map({1: "B", 2: "A"})

    s = build_press_sequences(pressures)

    assert list(s["n_pressures"]) == [3, 1, 1, 1, 2]
    assert np.allclose(s["mean_x"], [80, 60, 50, 40, 25])        # raw x, not mirrored
    assert list(s["any_counterpress"]) == [True, False, False, False, False]
    assert list(s["possession"]) == [5, 5, 6, 6, 7]
    assert list(s["period"]) == [1, 1, 1, 2, 1]
    assert np.allclose(s["start_time"], [10, 25, 26, 13, 10])
    assert np.allclose(s["end_time"], [14, 25, 26, 13, 11])


def test_build_press_sequences_on_the_sample_match(pressures, sequences):
    assert list(sequences.columns) == [
        "sequence_id", "match_id", "period", "team", "team_id", "start_time",
        "end_time", "n_pressures", "mean_x", "any_counterpress", "possession",
        "possession_team",
    ]
    assert sequences["sequence_id"].is_unique
    assert sequences["n_pressures"].sum() == len(pressures) == 289
    assert sequences.groupby("team").size().to_dict() == {"Canada": 101, "Morocco": 107}
    assert (sequences["n_pressures"] == 1).sum() == 144
    assert sequences["any_counterpress"].sum() == 46

@pytest.mark.skip(reason="implement pressing_sequences.ppda")
def test_ppda_lands_in_the_published_band(ppda_table):
    """Measured 7.4-12.3 on three sample matches. Outside 6-15, check the zone."""
    assert ppda_table.ppda.between(6, 15).mean() > 0.9


def test_label_sequence_outcomes_known_answer():
    """Team A (id 1) presses in six hand-built sequences, team B in one; only S1 and S4 win.

    S1  possession 1 (B) -> possession 2 is A's, 2 s later at x=40:  WON, not high
    S2  possession 3 (B) -> possession 4 is A's, but 15 s later:      too slow
    S3  possession 4 is A's OWN, and possession 5 is A's too:         the guard
    S4  possession 6 (B) -> possession 7 is A's, 3 s later at x=75:   WON, high
    S5  possession 8 (B) -> the next possession is in period 2:       not won
    S6  possession 10, the last of the match:                         no next
    S7  team B presses possession 4 (A's) and A keeps the ball in 5:  not won

    Each possession gets a decoy second event at x=5, and the rows are
    reversed, so the answer depends on picking the FIRST event by ``index``.
    """
    from xtpress.pressing_sequences import label_sequence_outcomes

    def stamp(t):
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"

    # (possession, owner id, period, first-event time, first-event x)
    possessions = [
        (1, 2, 1, 0.0, 60.0), (2, 1, 1, 12.0, 40.0), (3, 2, 1, 20.0, 50.0),
        (4, 1, 1, 60.0, 70.0), (5, 1, 1, 72.0, 80.0), (6, 2, 1, 76.0, 30.0),
        (7, 1, 1, 83.0, 75.0), (8, 2, 1, 90.0, 55.0), (9, 1, 2, 1.0, 65.0),
        (10, 2, 2, 50.0, 45.0),
    ]
    rows = []
    for k, (poss, owner, period, t, x) in enumerate(possessions):
        rows.append((2 * k, poss, owner, period, stamp(t), [x, 40.0]))
        rows.append((2 * k + 1, poss, owner, period, stamp(t + 1.0), [5.0, 5.0]))
    events = pd.DataFrame(rows[::-1], columns=["index", "possession", "possession_team_id",
                                               "period", "timestamp", "location"])
    events["match_id"] = 1

    names = {1: "A", 2: "B"}
    #        team_id, possession, period, end_time
    seqs = [(1, 1, 1, 10.0), (1, 3, 1, 45.0), (1, 4, 1, 70.0), (1, 6, 1, 80.0),
            (1, 8, 1, 2700.0), (1, 10, 2, 60.0), (2, 4, 1, 62.0)]
    sequences = pd.DataFrame(seqs, columns=["team_id", "possession", "period", "end_time"])
    sequences["match_id"] = 1
    sequences["team"] = sequences["team_id"].map(names)
    owner_of = {p[0]: p[1] for p in possessions}
    sequences["possession_team"] = sequences["possession"].map(owner_of).map(names)

    out = label_sequence_outcomes(sequences, events, window=8.0, high_regain_min_x=60.0)

    assert list(out["won_possession"]) == [True, False, False, True, False, False, False]
    assert list(out["won_high"]) == [False, False, False, True, False, False, False]
    assert out.loc[[0, 3], "seconds_to_regain"].tolist() == [2.0, 3.0]
    assert out.loc[[0, 3], "regain_x"].tolist() == [40.0, 75.0]       # first event, not the decoy
    assert out.loc[[0, 3], "won_possession_id"].tolist() == [2, 7]
    lost = out[~out["won_possession"]]
    assert lost[["seconds_to_regain", "regain_x", "regain_y", "won_possession_id"]].isna().all().all()


def test_sequences_never_span_a_possession_change(pressures, sequences):
    # 1. Nothing lost, nothing counted twice.
    assert sequences.n_pressures.sum() == len(pressures)

    # 2. Put every pressure back into a sequence with the SAME possession id.
    key = ["match_id", "team_id", "period", "possession"]
    m = pressures.merge(sequences, on=key, suffixes=("", "_seq"))
    inside = m[(m["time_seconds"] >= m["start_time"]) & (m["time_seconds"] <= m["end_time"])]

    # every pressure lands in exactly one sequence, and none are left over
    assert inside.groupby("id").size().max() == 1
    assert inside["id"].nunique() == len(pressures)


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
