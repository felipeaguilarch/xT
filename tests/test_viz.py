"""The figures in ``xtpress.viz``.

These check geometry and behaviour, not taste: that the data lands where the
pitch says it should, that a function composes onto an axis it is given, and that
``save_as`` writes into ``config.FIGURES_DIR``. Whether a figure reads well is
judged by looking at it.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import PowerNorm

from xtpress import config, viz


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _mesh(ax):
    return next(c for c in ax.collections if isinstance(c, QuadMesh))


def test_plot_xt_surface_places_zones_where_the_pitch_does(tmp_path, monkeypatch):
    grid = np.zeros((8, 12))
    grid[0, 11] = 0.2                                   # row 0 is y = 0, the last column is the goal end

    ax = viz.plot_xt_surface(grid)
    mesh = _mesh(ax)
    assert np.asarray(mesh.get_array()).reshape(8, 12)[0, 11] == pytest.approx(0.2)
    corners = mesh.get_coordinates()                    # (rows + 1, cols + 1, 2), in pitch coordinates
    assert corners.shape == (9, 13, 2)
    assert tuple(corners[0, 0]) == (0.0, 0.0) and tuple(corners[-1, -1]) == (config.PITCH_LENGTH, config.PITCH_WIDTH)

    assert isinstance(mesh.norm, PowerNorm) and mesh.norm.gamma == 0.5     # square-root scale
    assert mesh.norm.vmin == 0 and mesh.norm.vmax == pytest.approx(0.2)

    _, own = plt.subplots()
    assert viz.plot_xt_surface(grid, ax=own) is own     # composes onto an axis it is given

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_xt_surface(grid, save_as="xt_surface.png")
    assert (tmp_path / "xt_surface.png").stat().st_size > 5_000


def test_plot_convergence_draws_a_log_line_per_fit_and_labels_it_directly(tmp_path, monkeypatch):
    fast = list(0.5 ** np.arange(1, 41))                # 40 iterations, rho exactly 0.5
    slow = list(0.9 ** np.arange(1, 121))               # 120 iterations, rho exactly 0.9
    ax = viz.plot_convergence({"fast": fast, "slow": slow})

    assert ax.get_yscale() == "log"
    drawn = {len(line.get_xdata()): line for line in ax.lines if len(line.get_xdata()) > 2}
    assert set(drawn) == {40, 120}                      # one line per fit, not the reference line
    for n, deltas in [(40, fast), (120, slow)]:
        x, y = drawn[n].get_data()
        assert x[0] == 1 and x[-1] == n                 # iterations count from 1
        assert np.allclose(y, deltas)                   # the deltas as given, not transformed

    text = "\n".join(t.get_text() for t in ax.texts)
    for needle in ["fast", "40 iterations", "rho = 0.500", "slow", "120 iterations", "rho = 0.900"]:
        assert needle in text
    assert ax.get_legend() is None                      # labelled directly

    _, own = plt.subplots()
    assert viz.plot_convergence({"fast": fast}, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_convergence({"fast": fast, "slow": slow}, save_as="convergence.png")
    assert (tmp_path / "convergence.png").stat().st_size > 5_000


def test_plot_reference_comparison_shares_one_scale_and_annotates_r_and_mae(tmp_path, monkeypatch):
    grid = np.arange(96).reshape(8, 12) / 1000          # 0 .. 0.095
    reference = grid + 0.01                             # perfectly correlated, every zone 0.01 higher

    fig = viz.plot_reference_comparison(grid, reference)
    assert isinstance(fig, matplotlib.figure.Figure)

    meshes = [c for ax in fig.axes for c in ax.collections
              if isinstance(c, QuadMesh) and np.asarray(c.get_array()).size == 96]
    assert len(meshes) == 2                             # the two surfaces (not the colour bar)
    assert meshes[0].norm is meshes[1].norm or (meshes[0].norm.vmax == meshes[1].norm.vmax)
    assert meshes[0].norm.vmax == pytest.approx(reference.max())    # ONE scale, the reference's maximum
    assert meshes[0].norm.gamma == 0.5

    scatter = next(c for ax in fig.axes for c in ax.collections if len(c.get_offsets()) == 96 and not isinstance(c, QuadMesh))
    xy = np.asarray(scatter.get_offsets())
    assert np.allclose(np.sort(xy[:, 0]), np.sort(reference.ravel()))   # x is the published value
    assert np.allclose(np.sort(xy[:, 1]), np.sort(grid.ravel()))        # y is this project's

    text = "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "r = 1.000" in text and "MAE = 0.0100" in text

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_reference_comparison(grid, reference, save_as="reference.png")
    assert (tmp_path / "reference.png").stat().st_size > 5_000


def test_progressor_table_counts_open_play_regulation_xt_per_90():
    import pandas as pd

    rated = pd.DataFrame({
        "player_id": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],                                     # float, as in the real frame
        "player": ["A", "A", "A", "A", "B", "B"],
        "team": ["X", "X", "X", "X", "Y", "Y"],
        "action_type": ["pass", "carry", "pass", "pass", "pass", "pass"],
        "xt_delta": [0.03, 0.01, 0.50, 0.90, 0.02, 0.02],
        "is_set_piece": [False, False, True, False, False, False],                       # the 0.50 is a set piece
        "period": [1, 2, 1, 3, 1, 2],                                                    # the 0.90 is in extra time
    })
    minutes = pd.DataFrame({
        "match_id": [1, 2, 1], "player_id": [1, 1, 2],
        "player": ["A", "A", "B"], "team": ["X", "X", "Y"], "minutes": [90.0, 90.0, 90.0],
    })

    t = viz.progressor_table(rated, minutes, min_minutes=100)
    assert list(t.columns) == ["player", "team", "minutes", "pass_p90", "carry_p90", "total_p90"]
    assert list(t.player) == ["A"]                                                       # B has 90 minutes, under 100
    a = t.iloc[0]
    assert a.minutes == pytest.approx(180)
    assert a.pass_p90 == pytest.approx(0.03 / 180 * 90)                                  # 0.015: set piece and extra time left out
    assert a.carry_p90 == pytest.approx(0.01 / 180 * 90)                                 # 0.005
    assert a.total_p90 == pytest.approx(0.02)

    t = viz.progressor_table(rated, minutes, min_minutes=50)
    assert list(t.player) == ["B", "A"]                                                  # sorted by total_p90, best first
    assert t.iloc[0].carry_p90 == 0                                                      # no carries: 0, not NaN
    assert t.iloc[0].total_p90 == pytest.approx(0.04)


def test_plot_top_progressors_stacks_pass_and_carry_best_player_on_top(tmp_path, monkeypatch):
    import pandas as pd

    i = np.arange(25)
    table = pd.DataFrame({
        "player": [f"P{k:02d}" for k in i], "team": "T", "minutes": 500.0 + i,
        "pass_p90": 0.01 * (i + 1), "carry_p90": 0.005 * (i + 1), "total_p90": 0.015 * (i + 1),
    }).sample(frac=1, random_state=0)                                                    # unsorted on purpose

    ax = viz.plot_top_progressors(table, n=20)
    bars = sorted(ax.patches, key=lambda p: p.get_y())
    assert len(bars) == 40                                                               # a pass and a carry segment for each of 20
    passes = [b for b in bars if b.get_x() == 0]
    carries = [b for b in bars if b.get_x() > 0]
    assert len(passes) == len(carries) == 20
    assert [round(b.get_width(), 6) for b in passes] == sorted(round(b.get_width(), 6) for b in passes)   # best at the top
    for p, c in zip(passes, carries, strict=True):
        assert c.get_x() == pytest.approx(p.get_width())                                 # the carry segment starts where the pass one ends
    assert max(c.get_x() + c.get_width() for c in carries) == pytest.approx(0.015 * 25)  # P24 is the top bar
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 20 and "P24" in labels[-1] and "P04" not in "".join(labels)     # top 20 of 25, best last (top)
    assert any("524 min" in t.get_text() for t in ax.texts)                              # minutes annotated
    assert str(config.MIN_MINUTES) in ax.get_title(loc="left")                                     # the threshold is stated

    _, own = plt.subplots()
    assert viz.plot_top_progressors(table, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_top_progressors(table, save_as="progressors.png")
    assert (tmp_path / "progressors.png").stat().st_size > 5_000


def test_plot_player_actions_draws_one_scaled_arrow_per_top_action(tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.colors import to_rgba
    from matplotlib.quiver import Quiver

    k = np.arange(12)
    mine = pd.DataFrame({
        "player": "Kylian Mbappé Lottin", "action_type": np.where(k % 2 == 0, "pass", "carry"),
        "start_x": 20.0 + k, "start_y": 10.0 + k, "end_x": 60.0 + k, "end_y": 12.0 + k,
        "xt_delta": 0.01 * (k + 1),                                                       # 0.01 .. 0.12
    })
    other = mine.assign(player="Somebody Else", xt_delta=1.0)                              # bigger, but not his
    rated = pd.concat([mine, other], ignore_index=True)

    ax = viz.plot_player_actions(rated, "Kylian Mbappé Lottin", n=8)
    arrows = [c for c in ax.collections if isinstance(c, Quiver)]
    assert len(arrows) == 8                                                                # his top 8, one call each
    widths = sorted(a.width * 72 for a in arrows)                                          # Quiver keeps inches; the width asked for was in points
    assert min(widths) >= 1.6 - 1e-9 and max(widths) == pytest.approx(3.8)                 # sqrt-scaled into 1.6 .. 3.8
    assert widths == sorted(widths) and len(set(widths)) == 8                              # more xT, wider arrow

    colours = {tuple(np.round(a.get_facecolor()[0][:3], 3)) for a in arrows}               # RGB only: the arrows are slightly transparent
    assert colours == {tuple(np.round(to_rgba(c)[:3], 3)) for c in (viz.BLUE, viz.ORANGE)}  # pass and carry
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"pass", "carry"}
    title = ax.get_title(loc="left")
    assert "Kylian Mbappé" in title and "Lottin" not in title and "8" in title             # display name, and how many

    _, own = plt.subplots()
    assert viz.plot_player_actions(rated, "Kylian Mbappé Lottin", n=8, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_player_actions(rated, "Kylian Mbappé Lottin", n=8, save_as="player.png")
    assert (tmp_path / "player.png").stat().st_size > 5_000


def test_plot_team_xt_vs_goals_has_one_point_per_team_and_both_correlations(tmp_path, monkeypatch):
    import pandas as pd

    from xtpress.report import spearman

    rng = np.random.default_rng(1)
    tm = pd.DataFrame({"match_id": np.repeat(np.arange(4), 8), "team": [f"T{i}" for i in range(8)] * 4})
    tm["xt"] = rng.normal(0.3, 0.2, len(tm))
    tm["goals"] = rng.poisson(1.3, len(tm)).astype(float)
    teams = tm.groupby("team")[["xt", "goals"]].mean()

    ax = viz.plot_team_xt_vs_goals(tm)
    points = next(c for c in ax.collections if len(c.get_offsets()) == 8)
    xy = np.asarray(points.get_offsets())
    assert np.allclose(np.sort(xy[:, 0]), np.sort(teams.xt))                             # one point per team: mean xT ...
    assert np.allclose(np.sort(xy[:, 1]), np.sort(teams.goals))                          # ... against mean goals

    text = "\n".join(t.get_text() for t in ax.texts)
    assert f"{spearman(teams.xt, teams.goals):.2f}" in text                              # across the 8 teams
    assert f"{spearman(tm.xt, tm.goals):.2f}" in text                                    # across the 32 team-matches
    assert "8 teams" in text and "32 team-matches" in text
    labelled = [t.get_text() for t in ax.texts if t.get_text() in set(teams.index)]
    assert 1 <= len(labelled) < 8                                                        # some teams named, not all (96 labels is clutter)
    zx = (teams.xt - teams.xt.mean()) / teams.xt.std()
    zy = (teams.goals - teams.goals.mean()) / teams.goals.std()
    assert teams.goals.idxmax() in labelled                                              # the most goals per match is named
    assert (zx + zy).idxmin() in labelled                                                # so is the bottom-left corner

    _, own = plt.subplots()
    assert viz.plot_team_xt_vs_goals(tm, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_team_xt_vs_goals(tm, save_as="team_xt.png")
    assert (tmp_path / "team_xt.png").stat().st_size > 5_000


def test_plot_team_pressing_inverts_ppda_and_annotates_rho_with_its_interval(tmp_path, monkeypatch):
    import pandas as pd

    from xtpress.report import bootstrap_spearman

    rng = np.random.default_rng(2)
    summary = pd.DataFrame({"team": [f"T{i:02d}" for i in range(14)],
                            "ppda": rng.uniform(6, 20, 14), "xt_per_100_pressures": rng.normal(0.0, 0.05, 14)})

    ax = viz.plot_team_pressing(summary)
    points = next(c for c in ax.collections if len(c.get_offsets()) == 14)
    xy = np.asarray(points.get_offsets())
    assert np.allclose(np.sort(xy[:, 0]), np.sort(summary.ppda))                         # x is PPDA ...
    assert np.allclose(np.sort(xy[:, 1]), np.sort(summary.xt_per_100_pressures))         # ... y is the payoff
    assert ax.get_xlim()[0] > ax.get_xlim()[1]                                           # inverted: lower PPDA (more pressing) to the right
    assert "presses harder" in ax.get_xlabel()

    b = bootstrap_spearman(summary.ppda, summary.xt_per_100_pressures)                   # the same call the README's number comes from
    text = "\n".join(t.get_text() for t in ax.texts)
    for needle in (f"{b['rho']:.2f}", f"{b['lo']:.2f}", f"{b['hi']:.2f}", "14 teams"):
        assert needle in text
    labelled = [t.get_text() for t in ax.texts if t.get_text() in set(summary.team)]
    assert 1 <= len(labelled) < 14                                                       # the outliers, not everyone

    _, own = plt.subplots()
    assert viz.plot_team_pressing(summary, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_team_pressing(summary, save_as="pressing.png")
    assert (tmp_path / "pressing.png").stat().st_size > 5_000


def test_plot_press_resistance_quadrant_scales_the_axes_marks_zero_and_draws_one_standard_error(tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.container import ErrorbarContainer

    rng = np.random.default_rng(3)
    n = 30
    table = pd.DataFrame({
        "player": [f"Player {i:02d}" for i in range(n)],
        "n_pressured": rng.integers(50, 200, n),
        "retention_above_expected": rng.normal(0, 0.05, n),
        "xt_delta_above_expected": rng.normal(0, 0.0015, n),
        "score": rng.normal(0, 1, n),
    })
    se = (0.046, 0.0012)

    ax = viz.plot_press_resistance_quadrant(table, se)
    grey = next(c for c in ax.collections if len(c.get_offsets()) == n)
    xy = np.asarray(grey.get_offsets())
    assert np.allclose(xy[:, 0], table.retention_above_expected * 100)                   # retention in percentage points
    assert np.allclose(xy[:, 1], table.xt_delta_above_expected * 1000)                   # threat times 1,000
    sizes = np.asarray(grey.get_sizes())
    assert (np.argsort(sizes, kind="stable") == np.argsort(table.n_pressured.to_numpy(), kind="stable")).all()   # bigger sample, bigger point

    assert any(list(l.get_xdata()) == [0, 0] for l in ax.lines)                          # a vertical line through zero
    assert any(list(l.get_ydata()) == [0, 0] for l in ax.lines)                          # and a horizontal one

    bars = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
    segments = [seg for lc in bars[2] for seg in lc.get_segments()]
    horizontal = [abs(s[1][0] - s[0][0]) for s in segments if s[0][1] == s[1][1]]
    vertical = [abs(s[1][1] - s[0][1]) for s in segments if s[0][0] == s[1][0]]
    assert horizontal[0] == pytest.approx(2 * se[0] * 100)                               # one standard error either side, in the plotted units
    assert vertical[0] == pytest.approx(2 * se[1] * 1000)

    named = {t.get_text() for t in ax.texts}
    assert table.loc[table.score.idxmax(), "player"] in named                            # best and worst by score are labelled
    assert table.loc[table.score.idxmin(), "player"] in named
    assert sum(name in named for name in table.player) < n                               # extremes only, not everyone
    text = "\n".join(t.get_text() for t in ax.texts)
    assert f"{table.retention_above_expected.std() / se[0]:.1f}" in text                 # the spread in standard errors, computed
    assert f"{table.xt_delta_above_expected.std() / se[1]:.1f}" in text

    _, own = plt.subplots()
    assert viz.plot_press_resistance_quadrant(table, se, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_press_resistance_quadrant(table, se, save_as="quadrant.png")
    assert (tmp_path / "quadrant.png").stat().st_size > 5_000


def test_plot_press_resistance_leaderboard_ranks_colours_by_group_and_shows_the_sample_size(tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.colors import to_rgba

    from xtpress.report import short_name

    rng = np.random.default_rng(4)
    count = 30
    table = pd.DataFrame({
        "player": [f"Player {i:02d}" for i in range(count)],
        "position_group": rng.choice(["DF", "MF", "FW"], count),
        "n_pressured": rng.integers(50, 300, count),
        "score": np.sort(rng.normal(0.5, 1, count))[::-1] + np.arange(count) * 0.0,   # distinct scores
    }).sample(frac=1, random_state=0)                                                  # unsorted on purpose

    keeper = pd.DataFrame({"player": ["A Goalkeeper"], "position_group": ["GK"], "n_pressured": [60], "score": [table.score.max() + 5]})
    table = pd.concat([table, keeper], ignore_index=True)                              # the best score of all, but a goalkeeper

    ax = viz.plot_press_resistance_leaderboard(table, n=10)
    bars = sorted(ax.patches, key=lambda p: p.get_y())
    assert len(bars) == 10
    widths = [b.get_width() for b in bars]
    assert widths == sorted(widths)                                                    # best score on the top bar
    assert widths[-1] == pytest.approx(table[table.position_group != 'GK'].score.max())

    top = table[table.position_group != "GK"].sort_values("score", ascending=False).head(10).iloc[::-1]   # goalkeepers are not ranked
    assert "A Goalkeeper" not in [t.get_text() for t in ax.get_yticklabels()]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [short_name(p) for p in top.player]
    for bar, group in zip(bars, top.position_group, strict=True):
        assert tuple(np.round(bar.get_facecolor(), 3)) == tuple(np.round(to_rgba(viz.GROUP_COLOUR[group]), 3))

    text = "\n".join(t.get_text() for t in ax.texts)
    for n_pressured in top.n_pressured:
        assert f"n = {n_pressured}" in text                                            # the sample size on every bar
    assert "not shrunk" in text                                                        # the caveat is stated
    assert any(list(l.get_xdata()) == [0, 0] for l in ax.lines)                        # a line at zero
    present = [g for g in ("DF", "MF", "FW", "GK") if g in set(top.position_group)]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == present              # legend in DF, MF, FW, GK order

    _, own = plt.subplots()
    assert viz.plot_press_resistance_leaderboard(table, n=10, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_press_resistance_leaderboard(table, n=10, save_as="leaderboard.png")
    assert (tmp_path / "leaderboard.png").stat().st_size > 5_000


def test_plot_case_study_selects_the_right_events_and_stops_at_the_first_shot(tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.quiver import Quiver

    seq = pd.Series({"sequence_id": 9, "match_id": 1, "team_id": 10, "team": "Brazil", "possession_team": "Serbia",
                     "period": 2, "start_time": 100.0, "end_time": 103.0, "won_possession_id": 7,
                     "regain_x": 99.0, "regain_y": 41.0, "seconds_to_regain": 0.8, "xt_gained": 0.12})
    pressures = pd.DataFrame({
        "match_id": [1, 1, 1, 1, 1, 2], "team_id": [10, 10, 10, 10, 20, 10], "period": [2, 2, 2, 2, 2, 2],
        "time_seconds": [100.0, 101.5, 103.0, 120.0, 101.0, 101.0],                  # the 4th is too late, the 5th and 6th are not his
        "player": ["P1", "P2", "P3", "Late", "Other team", "Other match"],
        "x": [60.0, 70.0, 80.0, 5.0, 5.0, 5.0], "y": [30.0, 40.0, 50.0, 5.0, 5.0, 5.0],
    })
    actions = pd.DataFrame({
        "match_id": [1, 1, 1, 1, 1, 1, 1], "team_id": [10, 10, 10, 10, 10, 20, 10],
        "possession": [7, 7, 7, 7, 7, 7, 8], "index": [1, 2, 3, 4, 5, 6, 7],
        "action_type": ["pass", "carry", "pass", "shot", "pass", "pass", "pass"],   # the pass after the shot, the other team's and the other possession's are out
        "start_x": [99.0, 100.0, 105.0, 110.0, 90.0, 10.0, 20.0], "start_y": [41.0, 40.0, 38.0, 36.0, 30.0, 10.0, 20.0],
        "end_x": [100.0, 105.0, 110.0, 120.0, 95.0, 15.0, 25.0], "end_y": [40.0, 38.0, 36.0, 40.0, 30.0, 10.0, 20.0],
        "player": ["A", "B", "C", "Raphinha", "Z", "Q", "R"], "xg": [np.nan, np.nan, np.nan, 0.42, np.nan, np.nan, np.nan],
        "is_goal": [False] * 7,
    })

    ax = viz.plot_case_study(seq, pressures, actions)
    labels = {t.get_text() for t in ax.texts}
    assert {"1", "2", "3"} <= labels and "4" not in labels                              # three pressures, numbered in order
    arrows = sum(len(q.U) for q in ax.collections if isinstance(q, Quiver))
    assert arrows == 3                                                                  # pass, carry, pass: nothing after the shot, nothing from the other team or possession
    offsets = [np.asarray(c.get_offsets()) for c in ax.collections if len(c.get_offsets())]
    assert any(np.allclose(o, [[99.0, 41.0]]) for o in offsets)                         # the regain, from regain_x and regain_y
    assert any(np.allclose(o, [[110.0, 36.0]]) for o in offsets)                        # the shot
    text = "\n".join(t.get_text() for t in ax.texts)
    assert "xG 0.42" in text and "Raphinha" in text
    title = ax.get_title(loc="left")
    assert "Brazil v Serbia" in title and "46'" in title                                # period 2, 100 s in: the 46th minute

    _, own = plt.subplots()
    assert viz.plot_case_study(seq, pressures, actions, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_case_study(seq, pressures, actions, save_as="case.png")
    assert (tmp_path / "case.png").stat().st_size > 5_000


def test_plot_case_study_dates_extra_time_from_the_start_of_its_period():
    """Period 4 starts at minute 105, not 135: timestamps restart at zero each period."""
    import pandas as pd

    seq = pd.Series({"sequence_id": 9, "match_id": 1, "team_id": 10, "team": "Brazil", "possession_team": "Serbia",
                     "period": 4, "start_time": 70.0, "end_time": 70.0, "won_possession_id": 7,
                     "regain_x": 99.0, "regain_y": 41.0, "seconds_to_regain": 0.8})
    pressures = pd.DataFrame({"match_id": [1], "team_id": [10], "period": [4], "time_seconds": [70.0],
                              "player": ["P1"], "x": [60.0], "y": [30.0]})
    actions = pd.DataFrame({"match_id": [1], "team_id": [10], "possession": [7], "index": [1],
                            "action_type": ["pass"], "start_x": [99.0], "start_y": [41.0], "end_x": [100.0],
                            "end_y": [40.0], "player": ["A"], "xg": [np.nan], "is_goal": [False]})

    ax = viz.plot_case_study(seq, pressures, actions)
    assert "106'" in ax.get_title(loc="left")                                           # 105 + 70 s

    with pytest.raises(ValueError, match="no pressures"):                               # a row that matches nothing
        viz.plot_case_study(seq, pressures.iloc[0:0], actions)


def test_plot_convergence_refuses_more_fits_than_it_has_colours():
    series = {f"fit {i}": list(0.5 ** np.arange(1, 11)) for i in range(5)}
    with pytest.raises(ValueError, match="at most 4"):
        viz.plot_convergence(series)


def test_build_all_writes_every_figure_the_readme_shows(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    names = viz.build_all(ctx)
    assert names == sorted(["xt_surface.png", "convergence.png", "reference_comparison.png", "top_progressors.png",
                            "player_modric.png", "player_mbappe.png", "team_xt_vs_goals.png", "team_pressing.png",
                            "case_study.png", "press_resistance_quadrant.png", "press_resistance_leaderboard.png"])
    assert all((tmp_path / name).stat().st_size > 20_000 for name in names)
    readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert all(f"figures/{name}" in readme for name in names)        # every file is embedded, no embed points nowhere
