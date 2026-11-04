"""The figures in ``xtpress.viz``.

These check geometry and behaviour, not taste: that the data lands where the
pitch says it should, that a function composes onto an axis it is given, and that
``save_as`` writes into ``config.FIGURES_DIR``. Whether a figure reads well is
judged by looking at it. Each test is skipped until the function it names exists;
delete the skip when you implement it.
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

    fig, own = plt.subplots()
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

    fig, own = plt.subplots()
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
    for p, c in zip(passes, carries):
        assert c.get_x() == pytest.approx(p.get_width())                                 # the carry segment starts where the pass one ends
    assert max(c.get_x() + c.get_width() for c in carries) == pytest.approx(0.015 * 25)  # P24 is the top bar
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 20 and "P24" in labels[-1] and "P04" not in "".join(labels)     # top 20 of 25, best last (top)
    assert any("524 min" in t.get_text() for t in ax.texts)                              # minutes annotated
    assert str(config.MIN_MINUTES) in ax.get_title(loc="left")                                     # the threshold is stated

    fig, own = plt.subplots()
    assert viz.plot_top_progressors(table, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_top_progressors(table, save_as="progressors.png")
    assert (tmp_path / "progressors.png").stat().st_size > 5_000


@pytest.mark.skip(reason="implement viz.plot_player_actions")
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

    colours = {tuple(np.round(a.get_facecolor()[0], 3)) for a in arrows}
    assert colours == {tuple(np.round(to_rgba(c), 3)) for c in (viz.BLUE, viz.ORANGE)}     # pass and carry
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"pass", "carry"}
    title = ax.get_title(loc="left")
    assert "Kylian Mbappé" in title and "Lottin" not in title and "8" in title             # display name, and how many

    fig, own = plt.subplots()
    assert viz.plot_player_actions(rated, "Kylian Mbappé Lottin", n=8, ax=own) is own

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    viz.plot_player_actions(rated, "Kylian Mbappé Lottin", n=8, save_as="player.png")
    assert (tmp_path / "player.png").stat().st_size > 5_000
