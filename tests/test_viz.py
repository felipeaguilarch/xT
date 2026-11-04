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


@pytest.mark.skip(reason="implement viz.plot_xt_surface")
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
