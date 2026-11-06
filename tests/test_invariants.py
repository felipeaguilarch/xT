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

import json

import numpy as np
import pandas as pd
import pytest

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

def test_ppda_lands_in_the_published_band(ppda_table):
    """Measured on all 64 WC 2022 matches: median 11.7, range 4.2-26.3, 77% in 6-15.

    The band was first measured on three matches (7.4-12.3), where it looked
    tight. Over 64 it is not: deep-defending sides (Japan 26.3, Morocco 25.2,
    Costa Rica 23.6) sit far above 15 legitimately. So this checks the centre
    of the distribution, not that nearly every team is inside 6-15. A wrong
    frame or a lowercase mismatch moves the median to 22, 7 or NaN.
    """
    assert len(ppda_table) == 128
    assert ppda_table.ppda.notna().all()
    assert 10 <= ppda_table.ppda.median() <= 13
    assert ppda_table.ppda.between(6, 15).mean() > 0.7


def test_ppda_on_the_sample_match(events):
    from xtpress.pressing_sequences import ppda

    t = ppda(events).set_index("team")
    assert t.loc["Morocco", ["opp_passes", "def_actions"]].tolist() == [407, 33]
    assert t.loc["Canada", ["opp_passes", "def_actions"]].tolist() == [327, 28]
    assert round(t.loc["Morocco", "ppda"], 2) == 12.33
    assert round(t.loc["Canada", "ppda"], 2) == 11.68


def test_ppda_known_answer():
    """Two matches, hand-counted. Thresholds are 72 (opponent passes) and 48 (your actions).

    Match 1  A (id 1) passes at x = 10, 20, 72, 80:   3 in the zone (72 counts)
             B (id 2) passes at x = 30, 72, 73, 100:  2 in the zone (73 does not)
             A defends: Duel 48 (counts), Interception 60, Foul Committed 47.9
                        (does NOT), Duel 100 (counts)       -> 3
             B defends: Interception 50                      -> 1
             A.ppda = B's 2 passes / A's 3 actions;  B.ppda = A's 3 / B's 1
    Match 2  D (id 4) has events but no defensive action in the zone, so its
             PPDA is NaN, not infinity, and the row must still exist.
    """
    from xtpress.pressing_sequences import ppda

    def ev(match, team_id, team, kind, x):
        return (match, team_id, team, kind, None if x is None else [x, 40.0])

    rows = [ev(1, 1, "A", "Starting XI", None), ev(1, 2, "B", "Starting XI", None)]
    rows += [ev(1, 1, "A", "Pass", x) for x in (10.0, 20.0, 72.0, 80.0)]
    rows += [ev(1, 2, "B", "Pass", x) for x in (30.0, 72.0, 73.0, 100.0)]
    rows += [ev(1, 1, "A", "Duel", 48.0), ev(1, 1, "A", "Interception", 60.0),
             ev(1, 1, "A", "Foul Committed", 47.9), ev(1, 1, "A", "Duel", 100.0),
             ev(1, 1, "A", "Pressure", 90.0), ev(1, 2, "B", "Interception", 50.0)]
    rows += [ev(2, 3, "C", "Pass", 5.0)] * 6 + [ev(2, 4, "D", "Pass", 5.0)] * 3
    rows += [ev(2, 3, "C", "Duel", 60.0)] * 2 + [ev(2, 4, "D", "Duel", 10.0)]
    events = pd.DataFrame(rows[::-1], columns=["match_id", "team_id", "team", "type", "location"])

    t = ppda(events)

    assert list(t["team"]) == ["A", "B", "C", "D"]
    assert list(t["opp_passes"]) == [2, 3, 3, 6]
    assert list(t["def_actions"]) == [3, 1, 2, 0]
    assert np.allclose(t["ppda"], [2 / 3, 3.0, 1.5, np.nan], equal_nan=True)
    assert not np.isinf(t["ppda"]).any()
    assert "x" not in events.columns                                        # input left alone


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


def test_sequence_value_known_answer():
    """Five hand-built sequences; each one earns its answer for a different reason.

    S1  match 1, won possession 2 (team 1): moves +0.010 then -0.004, two shots
        (xG 0.10 + 0.05)           -> xt_gained 0.006, start 0.02, shot_xg 0.15
        The same possession also holds an OTHER-team move (+0.5) and shot (0.9):
        they must not count.
    S2  won possession 7, nothing happened                -> 0, 0, False; start NaN
    S3  NOT won (possession id NaN)                       -> everything NaN, False
    S4  match 1, won possession 4 (team 2): one move +0.02, no shot -> 0.02, 0.0
    S5  match 2, won possession 2 (team 1): one move +0.1 -> 0.1; the same possession
        number as S1 in another match must not mix with it

    The rated rows are reversed, so ``start_zone_xt`` is only right if the moves
    are put in event order (``index``) before taking the first.
    """
    from xtpress.pressing_sequences import sequence_value

    # (match_id, index, possession, team_id, xt_start, xt_delta) -- later index first
    moves = [
        (1, 30, 2, 2, 0.30, 0.500),
        (1, 9, 2, 1, 0.05, -0.004),
        (1, 5, 2, 1, 0.02, 0.010),
        (1, 12, 4, 2, 0.01, 0.020),
        (2, 8, 2, 1, 0.03, 0.100),
    ]
    rated = pd.DataFrame(moves, columns=["match_id", "index", "possession", "team_id", "xt_start", "xt_delta"])

    # (match_id, possession, team_id, action_type, xg); passes carry no xG
    acts = [
        (1, 2, 1, "shot", 0.05), (1, 2, 1, "shot", 0.10), (1, 2, 2, "shot", 0.90),
        (1, 2, 1, "pass", np.nan), (1, 4, 2, "pass", np.nan), (2, 2, 1, "pass", np.nan),
    ]
    actions = pd.DataFrame(acts, columns=["match_id", "possession", "team_id", "action_type", "xg"])

    # (match_id, team_id, won_possession, won_possession_id)
    seqs = [(1, 1, True, 2.0), (1, 1, True, 7.0), (1, 1, False, np.nan), (1, 2, True, 4.0), (2, 1, True, 2.0)]
    sequences = pd.DataFrame(seqs, columns=["match_id", "team_id", "won_possession", "won_possession_id"])

    out = sequence_value(sequences, rated, actions)

    assert len(out) == 5 and list(out["match_id"]) == [1, 1, 1, 1, 2]       # nothing dropped, order kept
    assert np.allclose(out["xt_gained"], [0.006, 0.0, np.nan, 0.02, 0.1], equal_nan=True)
    assert np.allclose(out["start_zone_xt"], [0.02, np.nan, np.nan, 0.01, 0.03], equal_nan=True)
    assert np.allclose(out["shot_xg"], [0.15, 0.0, np.nan, 0.0, 0.0], equal_nan=True)
    assert list(out["reached_shot"]) == [True, False, False, False, False]
    assert out["reached_shot"].dtype == bool


def test_pressures_per_opp_half_touch_on_the_sample_match(events):
    from xtpress.pressing_sequences import pressures_per_opponent_half_touch

    t = pressures_per_opponent_half_touch(events).set_index("team")
    assert t.loc["Morocco", ["pressures", "opp_touches"]].tolist() == [52, 837]
    assert t.loc["Canada", ["pressures", "opp_touches"]].tolist() == [91, 821]
    assert round(t.loc["Morocco", "pressures_per_opp_half_touch"], 3) == 0.062
    assert round(t.loc["Canada", "pressures_per_opp_half_touch"], 3) == 0.111


def test_pressures_per_opp_half_touch_known_answer():
    """Hand-counted. The halfway line is x = 60 and counts on both sides.

    Match 1  A presses at x = 70, 61, 60, 100 (4 in the opponent's half; 59.9 is not)
             B presses at x = 80, 20                   (1)
             A's touches in A's half: Pass 10, Carry 30, Ball Receipt* 60,
                Shot 5                                 -> 4
                (Pass 60.1, a Duel and a Pressure do NOT count)
             B's touches in B's half: Pass 40, Ball Receipt* 50, Clearance 10
                                                       -> 3   (Dribble 70 does not)
             A.ratio = 4 pressures / B's 3 touches;  B.ratio = 1 / A's 4
    Match 2  D has no touch in its own half, so C's ratio is NaN, not infinity.
             D presses nothing, so D keeps its row with 0 pressures and ratio 0.
    """
    from xtpress.pressing_sequences import pressures_per_opponent_half_touch

    def ev(match, team_id, team, kind, x):
        return (match, team_id, team, kind, None if x is None else [x, 40.0])

    rows = [ev(1, 1, "A", "Starting XI", None)]
    rows += [ev(1, 1, "A", "Pressure", x) for x in (70.0, 61.0, 60.0, 59.9, 100.0)]
    rows += [ev(1, 2, "B", "Pressure", x) for x in (80.0, 20.0)]
    rows += [ev(1, 1, "A", "Pass", 10.0), ev(1, 1, "A", "Carry", 30.0), ev(1, 1, "A", "Ball Receipt*", 60.0),
             ev(1, 1, "A", "Shot", 5.0), ev(1, 1, "A", "Pass", 60.1), ev(1, 1, "A", "Duel", 20.0)]
    rows += [ev(1, 2, "B", "Pass", 40.0), ev(1, 2, "B", "Ball Receipt*", 50.0),
             ev(1, 2, "B", "Clearance", 10.0), ev(1, 2, "B", "Dribble", 70.0)]
    rows += [ev(2, 3, "C", "Pressure", 90.0)] * 3 + [ev(2, 3, "C", "Pass", 20.0)] * 2
    rows += [ev(2, 4, "D", "Pass", 90.0), ev(2, 4, "D", "Carry", 100.0)]
    events = pd.DataFrame(rows[::-1], columns=["match_id", "team_id", "team", "type", "location"])

    t = pressures_per_opponent_half_touch(events)

    assert list(t["team"]) == ["A", "B", "C", "D"]
    assert list(t["pressures"]) == [4, 1, 3, 0]
    assert list(t["opp_touches"]) == [3, 4, 0, 2]
    assert np.allclose(t["pressures_per_opp_half_touch"], [4 / 3, 0.25, np.nan, 0.0], equal_nan=True)
    assert not np.isinf(t["pressures_per_opp_half_touch"]).any()
    assert "x" not in events.columns                                        # input left alone


def test_team_pressing_summary_known_answer():
    """Three teams over two matches (A v B, A v C), every number counted by hand.

    PPDA is opponent passes with x <= 72 over your defensive actions with x >= 48:
      A  M1: B's 6 passes / A's 2 actions = 3.0   M2: C's 4 / A's 4 = 1.0
         pooled 10 / 6 = 1.667    (the mean of the two ratios would be 2.0)
      B  3 / 1 = 3.0           C  2 / 1 = 2.0
    pressures_per_opp_half_touch is pressures with x >= 60 over the opponent's touches with x <= 60:
      A  M1: 3 / 4   M2: 1 / 3   pooled 4 / 7 = 0.571   (mean of ratios 0.542)
      B  2 / 2 = 1.0           C  1 / 1 = 1.0
    Sequences (n_pressures in brackets): A has five (12 pressures), three of them won,
    and A1 and A2 win the SAME possession of match 1 -- it must be credited once. A4 wins
    possession 5 of match 2: the same number in another match, so it is a separate credit.
      A  xT credited 0.10 + 0.20 = 0.30 over 12 pressures -> 2.5 per 100; one shot, not two
      B  never won a ball: keeps its row, 0 xT, 0 shots, no per-sequence mean
      C  one sequence, won high: 0.05 over 1 pressure -> 5.0 per 100
    """
    from xtpress.pressing_sequences import team_pressing_summary

    def ev(match, team_id, team, kind, x):
        return (match, team_id, team, kind, [x, 40.0])

    rows = []
    for x in (10.0, 30.0, 65.0):
        rows.append(ev(1, 1, "A", "Pass", x))
    for x in (5.0, 20.0, 40.0, 55.0, 66.0, 70.0):
        rows.append(ev(1, 2, "B", "Pass", x))
    rows += [ev(1, 1, "A", "Duel", 60.0), ev(1, 1, "A", "Interception", 50.0), ev(1, 2, "B", "Interception", 80.0)]
    rows += [ev(1, 1, "A", "Pressure", x) for x in (70.0, 80.0, 90.0, 20.0)]
    rows += [ev(1, 2, "B", "Pressure", x) for x in (75.0, 62.0, 10.0)]
    for x in (20.0, 62.0):
        rows.append(ev(2, 1, "A", "Pass", x))
    for x in (10.0, 20.0, 30.0, 70.0):
        rows.append(ev(2, 3, "C", "Pass", x))
    rows += [ev(2, 1, "A", "Duel", 55.0)] * 4 + [ev(2, 3, "C", "Duel", 70.0)]
    rows += [ev(2, 1, "A", "Pressure", 61.0), ev(2, 3, "C", "Pressure", 100.0)]
    events = pd.DataFrame(rows[::-1], columns=["match_id", "team_id", "team", "type", "location"])

    cols = ["sequence_id", "match_id", "team_id", "team", "n_pressures", "won_possession", "won_high",
            "won_possession_id", "seconds_to_regain", "xt_gained", "reached_shot"]
    nan = np.nan
    seqs = [
        (1, 1, 1, "A", 3, True, False, 5, 2.0, 0.10, True),     # A1  wins possession 5 ...
        (2, 1, 1, "A", 2, True, True, 5, 1.0, 0.10, True),      # A2  ... and so does A2
        (3, 1, 1, "A", 1, False, False, nan, nan, nan, False),
        (4, 2, 1, "A", 4, True, False, 5, 4.0, 0.20, False),    # A4  possession 5 of ANOTHER match
        (5, 2, 1, "A", 2, False, False, nan, nan, nan, False),
        (6, 1, 2, "B", 2, False, False, nan, nan, nan, False),
        (7, 1, 2, "B", 3, False, False, nan, nan, nan, False),
        (8, 2, 3, "C", 1, True, True, 9, 3.0, 0.05, False),
    ]
    sequences = pd.DataFrame(seqs, columns=cols)

    t = team_pressing_summary(sequences, events).set_index("team")

    assert list(t.index) == ["A", "B", "C"]
    assert np.allclose(t["ppda"], [10 / 6, 3.0, 2.0])
    assert np.allclose(t["pressures_per_opp_half_touch"], [4 / 7, 1.0, 1.0])
    assert list(t["n_sequences"]) == [5, 2, 1]
    assert np.allclose(t["regain_rate"], [0.6, 0.0, 1.0])
    assert np.allclose(t["high_regain_rate"], [0.2, 0.0, 1.0])
    assert np.allclose(t["xt_per_100_pressures"], [2.5, 0.0, 5.0])
    assert np.allclose(t["xt_per_successful_sequence"], [0.15, np.nan, 0.05], equal_nan=True)
    assert list(t["shots_from_presses"]) == [1, 0, 0]


def test_team_pressing_summary_over_the_competition(team_summary):
    """Measured on 64 WC 2022 matches. The telling number is the last one: how hard a
    team presses (PPDA) says almost nothing about what its pressing earns."""
    t = team_summary.set_index("team")

    assert len(t) == 32 and not t.isna().any().any()
    assert abs(t["ppda"].median() - 11.27) < 0.02
    assert abs(t["xt_per_100_pressures"].median() - 0.159) < 0.003
    assert t["shots_from_presses"].sum() == 202                 # 211 if a won possession is counted twice
    assert abs(t.loc["Canada", "xt_per_100_pressures"] - 0.325) < 0.005
    assert abs(t.loc["Brazil", "xt_per_100_pressures"] - 0.324) < 0.005
    assert abs(t.loc["Mexico", "ppda"] - 7.42) < 0.02
    assert -0.3 < t["ppda"].corr(t["xt_per_100_pressures"], method="spearman") < 0.1


def test_zone_baselines_known_answer():
    """Eight pressured actions in four zones, one with no coordinates. 12x8 grid, z = row * 12 + col.

    zone 0   (x=5,  y=5):    retained +0.02, -0.01, +0.08, and a lost -> n 4, retention 3/4, delta 0.03 (the median is 0.02)
    zone 1   (x=15, y=5):    two retained dribbles (no rating)       -> n 2, retention 1.0, delta NaN
    zone 12  (x=5,  y=15):   lost                                    -> n 1, retention 0.0, delta NaN
    zone 95  (x=120, y=80):  retained +0.1, on the very edge         -> n 1, retention 1.0, delta 0.1
    and one action with a missing start location, which belongs nowhere
    """
    from xtpress.press_resistance import zone_baselines

    nan = np.nan
    rows = [(5.0, 5.0, True, 0.02), (5.0, 5.0, True, -0.01), (5.0, 5.0, True, 0.08), (5.0, 5.0, False, nan),
            (15.0, 5.0, True, nan), (15.0, 5.0, True, nan),
            (5.0, 15.0, False, nan),
            (120.0, 80.0, True, 0.1),
            (nan, nan, True, 0.5)]
    pressured = pd.DataFrame(rows[::-1], columns=["start_x", "start_y", "retained", "xt_delta"])
    before = pressured.copy()

    t = zone_baselines(pressured)

    assert list(t.columns) == ["zone", "n", "expected_retention", "expected_xt_delta"]
    assert list(t["zone"]) == list(range(96))                          # every zone, in order
    assert t["n"].dtype == int and t["n"].sum() == 8                   # the unlocated action is dropped
    assert t.loc[[0, 1, 12, 95], "n"].tolist() == [4, 2, 1, 1]
    assert np.allclose(t.loc[[0, 1, 12, 95], "expected_retention"], [0.75, 1.0, 0.0, 1.0])
    assert np.allclose(t.loc[[0, 1, 12, 95], "expected_xt_delta"], [0.03, nan, nan, 0.1], equal_nan=True)
    empty = t[t["n"] == 0]
    assert len(empty) == 92 and empty[["expected_retention", "expected_xt_delta"]].isna().all().all()
    pd.testing.assert_frame_equal(pressured, before)                   # the input is left alone


def test_zone_baselines_over_the_competition(pressured_all):
    from xtpress import config
    from xtpress.press_resistance import zone_baselines

    t = zone_baselines(pressured_all)
    assert len(t) == 96 and not t.isna().any().any()
    assert t["n"].sum() == 25270 and t["n"].min() == 28 and t["n"].max() == 584
    assert abs((t["n"] * t["expected_retention"]).sum() / t["n"].sum() - 0.8009) < 0.0005    # the overall rate
    z = t.set_index("zone")
    assert (z.loc[8, "n"], round(z.loc[8, "expected_retention"], 3)) == (584, 0.805)
    assert (z.loc[59, "n"], round(z.loc[59, "expected_retention"], 3)) == (28, 0.607)
    assert (z.loc[73, "n"], round(z.loc[73, "expected_retention"], 3)) == (125, 0.888)
    assert (z.loc[84, "n"], round(z.loc[84, "expected_retention"], 3)) == (51, 0.529)
    own, final = z.index % config.GRID_COLS < 4, z.index % config.GRID_COLS >= 8
    assert (z[own]["n"] * z[own]["expected_retention"]).sum() / z[own]["n"].sum() > \
           (z[final]["n"] * z[final]["expected_retention"]).sum() / z[final]["n"].sum()      # the final third is harder


def test_player_press_resistance_known_answer():
    """Six players with hand-counted numbers; 12x8 grid, z = row * 12 + col, min_actions = 3.

    Baselines: zone 0 expects retention 0.5 and delta 0.02, zone 1 expects 0.9 and 0.0, every
    other zone 0.8 and 0.01.

    Alpha (Left Back, DF), six actions: four in zone 0 [+0.03, +0.01, lost, +0.05], a retained
      dribble in zone 1 (no rating), and an EXTRA-TIME action in zone 0 (+0.20, period 3).
      retention 5/6; expected (5 * 0.5 + 0.9) / 6 -> +0.2667; mean delta (0.03 + 0.01 + 0.05 + 0.20) / 4
      = 0.0725; delta above (0.01 - 0.01 + 0.03 + 0.18) / 4 = 0.0525; per 90 = (0.03 + 0.01 + 0.05) / 180 * 90
      = 0.045 -- regulation only, minutes summed over two matches.
    Beta (Right Wing Back, DF): 2/3 retained -> -0.1333; delta above 0.02. No minutes row -> per-90 NaN.
    Gamma (Goalkeeper), Delta (Right Defensive Midfield), Epsilon (Left Wing), Zeta (Center Forward).
    Eta (Left Back) has only two actions and is dropped.
    """
    from xtpress.press_resistance import player_press_resistance

    nan = np.nan
    base = pd.DataFrame({"zone": range(96), "n": 100, "expected_retention": 0.8, "expected_xt_delta": 0.01})
    base.loc[0, ["expected_retention", "expected_xt_delta"]] = [0.5, 0.02]
    base.loc[1, ["expected_retention", "expected_xt_delta"]] = [0.9, 0.0]

    #       id    name     position                    x     y      kept   delta  period
    rows = [(1.0, "Alpha", "Left Back", 5.0, 5.0, True, 0.03, 1), (1.0, "Alpha", "Left Back", 5.0, 5.0, True, 0.01, 1),
            (1.0, "Alpha", "Left Back", 5.0, 5.0, False, nan, 1), (1.0, "Alpha", "Left Back", 5.0, 5.0, True, 0.05, 2),
            (1.0, "Alpha", "Left Back", 15.0, 5.0, True, nan, 2), (1.0, "Alpha", "Left Back", 5.0, 5.0, True, 0.20, 3),
            (2.0, "Beta", "Right Wing Back", 5.0, 25.0, True, 0.02, 1), (2.0, "Beta", "Right Wing Back", 5.0, 25.0, False, nan, 1),
            (2.0, "Beta", "Right Wing Back", 5.0, 25.0, True, 0.04, 1),
            (3.0, "Gamma", "Goalkeeper", 5.0, 25.0, True, 0.01, 1), (3.0, "Gamma", "Goalkeeper", 5.0, 25.0, True, 0.01, 1),
            (3.0, "Gamma", "Goalkeeper", 5.0, 25.0, True, 0.01, 2),
            (4.0, "Delta", "Right Defensive Midfield", 5.0, 25.0, True, 0.05, 1), (4.0, "Delta", "Right Defensive Midfield", 5.0, 25.0, False, nan, 1),
            (4.0, "Delta", "Right Defensive Midfield", 5.0, 25.0, False, nan, 2),
            (5.0, "Epsilon", "Left Wing", 5.0, 25.0, True, 0.02, 1), (5.0, "Epsilon", "Left Wing", 5.0, 25.0, True, 0.0, 1),
            (5.0, "Epsilon", "Left Wing", 5.0, 25.0, False, nan, 2),
            (6.0, "Zeta", "Center Forward", 5.0, 25.0, True, 0.03, 1), (6.0, "Zeta", "Center Forward", 5.0, 25.0, True, 0.03, 1),
            (6.0, "Zeta", "Center Forward", 5.0, 25.0, True, 0.03, 2),
            (7.0, "Eta", "Left Back", 5.0, 25.0, True, 0.01, 1), (7.0, "Eta", "Left Back", 5.0, 25.0, False, nan, 1)]
    pressured = pd.DataFrame(rows[::-1], columns=["player_id", "player", "position", "start_x", "start_y", "retained", "xt_delta", "period"])
    minutes = pd.DataFrame({"match_id": [1, 2], "player_id": [1, 1], "minutes": [90.0, 90.0]})
    before = pressured.copy()

    t = player_press_resistance(pressured, base, minutes, min_actions=3).set_index("player")

    assert list(t.index) == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]          # Eta is dropped
    assert t["position_group"].to_dict() == {"Alpha": "DF", "Beta": "DF", "Gamma": "GK", "Delta": "MF", "Epsilon": "FW", "Zeta": "FW"}
    assert t["n_pressured"].tolist() == [6, 3, 3, 3, 3, 3]
    assert np.allclose(t["retention_rate"], [5 / 6, 2 / 3, 1.0, 1 / 3, 2 / 3, 1.0])
    assert np.allclose(t["retention_above_expected"], [5 / 6 - 3.4 / 6, 2 / 3 - 0.8, 0.2, 1 / 3 - 0.8, 2 / 3 - 0.8, 0.2])
    assert np.allclose(t["mean_xt_delta"], [0.0725, 0.03, 0.01, 0.05, 0.01, 0.03])
    assert np.allclose(t["xt_delta_above_expected"], [0.0525, 0.02, 0.0, 0.04, 0.0, 0.02])
    assert np.allclose(t["xt_retained_per_90"], [0.045, nan, nan, nan, nan, nan], equal_nan=True)
    assert t.loc[["Alpha", "Beta"], "retention_pct"].tolist() == [1.0, 0.5]                 # ranked within the group
    assert t.loc[["Alpha", "Beta"], "threat_pct"].tolist() == [1.0, 0.5]
    assert t.loc[["Epsilon", "Zeta"], "retention_pct"].tolist() == [0.5, 1.0]
    assert t.loc[["Gamma", "Delta"], "retention_pct"].tolist() == [1.0, 1.0]                # alone in their group
    assert list(t.columns) == ["player_id", "position", "n_pressured", "retention_rate", "mean_xt_delta",
                               "retention_above_expected", "xt_delta_above_expected", "xt_retained_per_90",
                               "position_group", "retention_pct", "threat_pct"]
    pd.testing.assert_frame_equal(pressured, before)                                        # the input is left alone

    no_minutes = player_press_resistance(pressured, base, None, min_actions=3)
    assert no_minutes["xt_retained_per_90"].isna().all() and len(no_minutes) == 6


def test_player_press_resistance_over_the_competition(resistance_table):
    t = resistance_table.set_index("player")
    assert len(t) == 179 and t["position_group"].value_counts().to_dict() == {"MF": 71, "FW": 66, "DF": 40, "GK": 2}
    assert not t["xt_retained_per_90"].isna().any()
    assert abs(t["retention_rate"].corr(t["retention_above_expected"], method="spearman") - 0.935) < 0.003
    assert round(t["retention_above_expected"].min(), 3) == -0.158 and round(t["retention_above_expected"].max(), 3) == 0.120
    foden = t.loc["Phil Foden"]
    assert (foden["position_group"], foden["n_pressured"]) == ("FW", 73)
    assert round(foden["retention_above_expected"], 3) == 0.108 and round(foden["retention_pct"], 3) == 0.985
    assert round(t.loc["Denzel Dumfries", "retention_above_expected"], 3) == -0.147
    assert t.loc["Denzel Dumfries", "position_group"] == "DF"                              # a wing back is a defender


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

def test_pressured_sample_is_the_right_size(actions, pressured):
    """A share of ACTIONS: 22.0% on the sample match, 19.6% over WC 2022.

    The original 8-12% is a share of all EVENTS (about 10% for passes and
    carries), which this frame does not contain. Keying off Ball Receipts
    instead of the action gives a few percent, far below this band.
    """
    assert 0.15 <= len(pressured) / len(actions) <= 0.25
    assert pressured["action_type"].value_counts().to_dict() == {
        "carry": 236, "pass": 100, "dribble": 39, "dispossessed": 23, "miscontrol": 7}


def test_pressured_actions_over_the_competition(full_actions):
    from xtpress.press_resistance import pressured_actions

    p = pressured_actions(full_actions)
    assert len(p) == 25270
    assert abs(len(p) / len(full_actions) - 0.196) < 0.001
    assert abs(p["successful"].mean() - 0.801) < 0.001       # 0.865 if the loss events are left out
    assert p["action_type"].value_counts().to_dict() == {
        "carry": 13700, "pass": 7908, "dribble": 1793, "dispossessed": 1431, "miscontrol": 438}


def test_pressured_actions_known_answer():
    """Ten actions; six are pressured actions. The index is not 0..9, so a reset would show.

    in:  pass, carry, dribble, dispossessed x2, miscontrol -- all under pressure
    out: pass (not pressured), carry (not pressured), shot (pressured, but not a
         decision to keep the ball), miscontrol (not pressured)
    """
    from xtpress.press_resistance import pressured_actions

    kinds = ["pass", "pass", "carry", "dribble", "shot", "dispossessed", "miscontrol", "miscontrol", "carry", "dispossessed"]
    pressed = [True, False, True, True, True, True, True, False, False, True]
    actions = pd.DataFrame({"action_type": kinds, "under_pressure": pressed, "player": list("abcdefghij")},
                           index=range(10, 20))
    before = actions.copy()

    out = pressured_actions(actions)

    assert list(out.index) == [10, 12, 13, 15, 16, 19]
    assert list(out["action_type"]) == ["pass", "carry", "dribble", "dispossessed", "miscontrol", "dispossessed"]
    assert list(out.columns) == list(actions.columns)                    # nothing added or dropped
    out["extra"] = 1
    assert "extra" not in actions.columns                                # a copy, not a view
    pd.testing.assert_frame_equal(actions, before)                       # the input is left alone


def test_retention_rate_is_plausible(pressured):
    """Measured 80.1% over WC 2022 and 81.2% on the sample match; 90.2% when not pressured.

    The original 65-80% is just under the measured value. Foul Won is missing from the
    action frame, and counting the 1,674 pressured fouls won would raise it to 81.3%.
    """
    assert 0.75 <= pressured.retained.mean() <= 0.85
    assert pressured["outcome"].value_counts().to_dict() == {
        "retained_flat": 199, "retained_progressed": 111, "lost": 76, "retained_dribble": 19}


def test_classify_outcome_over_the_competition(full_actions, model):
    from xtpress.press_resistance import classify_outcome, pressured_actions

    rated = model.rate_actions(full_actions)[["xt_start", "xt_end", "xt_delta"]]
    o = classify_outcome(pressured_actions(full_actions.join(rated)))
    assert len(o) == 25270 and abs(o["retained"].mean() - 0.801) < 0.001
    assert o["outcome"].value_counts().to_dict() == {
        "retained_flat": 12425, "retained_progressed": 6877, "lost": 5031, "retained_dribble": 937}
    assert o["outcome"].dtype == "category"


def test_classify_outcome_known_answer():
    """Ten pressured actions, one of each case. The index is not 0..9.

    pass  +0.010 -> progressed   pass  -0.002 -> flat   pass  0.000 -> flat (stayed in the zone)
    pass  failed -> lost         carry +0.005 -> progressed   carry 0.000 -> flat
    dribble ok  -> retained_dribble (never rated: its xt_delta is NaN)   dribble failed -> lost
    dispossessed, miscontrol -> lost
    """
    from xtpress.press_resistance import classify_outcome

    kinds = ["pass", "pass", "pass", "pass", "carry", "carry", "dribble", "dribble", "dispossessed", "miscontrol"]
    ok = [True, True, True, False, True, True, True, False, False, False]
    delta = [0.010, -0.002, 0.0, np.nan, 0.005, 0.0, np.nan, np.nan, np.nan, np.nan]
    pressured = pd.DataFrame({"action_type": kinds, "successful": ok, "xt_delta": delta}, index=range(100, 110))
    before = pressured.copy()

    out = classify_outcome(pressured)

    assert list(out["outcome"]) == ["retained_progressed", "retained_flat", "retained_flat", "lost", "retained_progressed",
                                    "retained_flat", "retained_dribble", "lost", "lost", "lost"]
    assert list(out["retained"]) == ok
    assert list(out["outcome"].cat.categories) == ["retained_progressed", "retained_flat", "retained_dribble", "lost"]
    assert list(out.index) == list(range(100, 110))
    pd.testing.assert_frame_equal(pressured, before)                       # the input is left alone

    with pytest.raises(ValueError):                                        # the rating was never joined
        classify_outcome(pressured.drop(columns="xt_delta"))
    unrated = pressured.copy()
    unrated.loc[100, "xt_delta"] = np.nan                                  # a successful pass with no rating
    with pytest.raises(ValueError):
        classify_outcome(unrated)


def test_zone_adjustment_actually_changes_the_ranking(resistance_table):
    """If raw and adjusted rank identically, the adjustment is not wired in."""
    raw = resistance_table.retention_rate.rank()
    adjusted = resistance_table.retention_above_expected.rank()
    assert raw.corr(adjusted, method="spearman") < 0.99

def test_press_resistance_score_known_answer():
    """Hand-built axes (means 0.05 and 0.01, stds 0.1 and 0.002) whose z-scores are -1, 0, 1 and -1, 1, 0."""
    from xtpress.press_resistance import press_resistance_score

    table = pd.DataFrame({"retention_above_expected": [-0.05, 0.05, 0.15],
                          "xt_delta_above_expected": [0.008, 0.012, 0.010]}, index=[7, 8, 9])   # means are not zero
    before = table.copy()

    even = press_resistance_score(table)
    assert isinstance(even, pd.Series) and list(even.index) == [7, 8, 9]
    assert np.allclose(even, [-1.0, 0.5, 0.5])
    assert np.allclose(press_resistance_score(table, 0.8, 0.2), [-1.0, 0.2, 0.8])
    assert np.allclose(press_resistance_score(table, 1.0, 0.0), [-1.0, 0.0, 1.0])
    pd.testing.assert_frame_equal(table, before)                           # the input is left alone

    rescaled = table.assign(xt_delta_above_expected=table.xt_delta_above_expected * 1000)
    assert np.allclose(press_resistance_score(rescaled), even)             # units must not matter


def test_press_resistance_score_blends_both_axes(resistance_table):
    """Unstandardised, retention (std 0.054) would swamp threat (std 0.0015) at 99.9%."""
    from xtpress.press_resistance import press_resistance_score

    score = press_resistance_score(resistance_table)
    assert len(score) == len(resistance_table) and score.notna().all()
    for axis in ("retention_above_expected", "xt_delta_above_expected"):
        assert 0.4 < score.corr(resistance_table[axis], method="spearman") < 0.9
