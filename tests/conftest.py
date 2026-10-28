"""Shared fixtures.

The Phase 01 fixtures use a single cached match so the suite stays fast and
works offline once the cache is warm. Later phases need a fitted model over a
real competition -- add those fixtures as you implement them.
"""

from __future__ import annotations

import pytest
from xtpress import config

# WC 2022, Canada v Morocco. The match every calibration figure in the plan
# was measured on, so its numbers are the ones to check against.
SAMPLE_MATCH_ID = 3857276


@pytest.fixture(scope="session")
def events():
    from xtpress import loaders

    return loaders.load_events(SAMPLE_MATCH_ID)


@pytest.fixture(scope="session")
def actions(events):
    from xtpress import loaders

    return loaders.to_actions(events)


# One match is not enough data for the headline 12x8 grid: zones 0 and 71 see
# a single successful move each, which ends where it started. That normalises
# to T[z,z] = 1.0 with s = 0, so (I - diag(m)T) has an all-zero row and xT has
# no unique solution -- assert_solvable refuses, correctly. Coarsening to 8x6
# gives every zone enough moves to escape, and the solve is well posed.
#
# Swap this to config.GRID_COLS/GRID_ROWS once you have built the real dataset
# (python -m xtpress.loaders --competition wc_2022) -- it is a one-line change.
MODEL_GRID = (config.GRID_COLS, config.GRID_ROWS)


@pytest.fixture(scope="session")
def full_actions():
    """Every action in the focus competition, from the processed parquet.

    The calibration bands and the ablation numbers were measured on exactly
    this frame (WC 2022 alone), so the fixture is pinned to
    ``config.FOCUS_COMPETITION`` rather than FIT_COMPETITIONS -- widening it
    would move the numbers the tests assert against.

    The processed parquet is gitignored, so a fresh clone does not have it.
    Skip with the command that builds it, rather than failing on a bare
    FileNotFoundError.
    """
    from xtpress import loaders

    try:
        return loaders.load_actions([config.FOCUS_COMPETITION])
    except FileNotFoundError:
        pytest.skip(
            "processed dataset missing -- run: "
            f"python -m xtpress.loaders --competition {config.FOCUS_COMPETITION}"
        )


@pytest.fixture(scope="session")
def model(full_actions):
    """The headline model: 12x8, loss modelled, fitted on the focus competition.

    One match cannot support this grid -- zones 0 and 71 see a single
    successful move each, so the structural checks ran on a coarse 8x6 stand-in
    until the dataset existed. The real fit takes ~0.1s, so there is no longer
    a reason to approximate.
    """
    from xtpress.expected_threat import ExpectedThreat

    cols, rows = MODEL_GRID
    return ExpectedThreat(cols=cols, rows=rows).fit(full_actions)

@pytest.fixture(scope="session")
def pressures(events):
    from xtpress.pressing_sequences import extract_pressures

    return extract_pressures(events)


@pytest.fixture(scope="session")
def sequences(pressures):
    from xtpress.pressing_sequences import build_press_sequences

    return build_press_sequences(pressures)


@pytest.fixture(scope="session")
def all_events(full_actions):
    """Raw events of every match ``full_actions`` covers, from the parquet cache."""
    import pandas as pd
    from xtpress import loaders

    ids = sorted(full_actions["match_id"].unique())
    return pd.concat([loaders.load_events(m) for m in ids], ignore_index=True)


@pytest.fixture(scope="session")
def ppda_table(all_events):
    """PPDA for every team in every match of the focus competition (128 rows for WC 2022)."""
    from xtpress.pressing_sequences import ppda

    return ppda(all_events)


@pytest.fixture(scope="session")
def team_summary(all_events, full_actions, model):
    """The whole pressing pipeline, one row per team (32 for WC 2022)."""
    from xtpress import pressing_sequences as ps

    sequences = ps.build_press_sequences(ps.extract_pressures(all_events))
    sequences = ps.label_sequence_outcomes(sequences, all_events)
    sequences = ps.sequence_value(sequences, model.rate_actions(full_actions), full_actions)
    return ps.team_pressing_summary(sequences, all_events)
