"""Karun Singh's grid-based Expected Threat model, fitted from scratch.

    xT(z) = s(z)*g(z) + m(z) * sum_z' T(z->z') * xT(z')

    s(z)  shot probability      shots / (shots + move attempts) from z
    m(z)  move probability      1 - s(z)
    g(z)  scoring probability   goals / shots from z  (or mean xG -- smoother)
    T     transition matrix     P(ball ends in z' | SUCCESSFUL move from z)

Note the deliberate asymmetry: m(z) counts ALL move attempts, but T is built
from successful moves only. That is why xT measures gain and never penalises a
failed action -- the gap VAEP was designed to fill. Say so in the README.

One consequence of that asymmetry, measured on WC 2022 at 12x8. Rows of T sum
to 1, so a failed move is never modelled as losing the ball and the chain ends
only in a shot. Value then mixes across the whole pitch and the surface goes
flat: r = 0.860 against the published grid, goal-end only 1.4x the own-end.
``fit(model_loss=True)``, the default, scales the continuation term by the
per-zone completion rate c(z):

    xT(z) = s(z)*g(z) + m(z) * c(z) * sum_z' T(z->z') * xT(z')

T is untouched; the mass a failed move removes is a lost ball, worth 0.
``fit(model_loss=False)`` keeps the textbook form as a baseline. The RATING of
actions still charges a failed move nothing -- see ``rate_actions``.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Karun Singh's published 12x8 surface -- the external validation target.
REFERENCE_GRID_URL = "https://karun.in/blog/data/open_xt_12x8_v1.json"
REFERENCE_GRID_CACHE = config.MODELS_DIR / "reference_xt_12x8.json"


def zone_index(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    cols: int = config.GRID_COLS,
    rows: int = config.GRID_ROWS,
) -> np.ndarray:
    """Map pitch coordinates to a flat zone index in [0, cols*rows).

    Convention, and keep it EVERYWHERE: ``z = row * cols + col``. That is the
    layout ``ndarray.reshape(rows, cols)`` assumes, so ``ExpectedThreat.grid``
    depends on it; flip it and every heatmap comes out transposed, silently.

    Clip the INDEX, not the coordinate (trap #7). A shot on the goal line sits at
    exactly x=120, which scales to column ``cols`` -- one past the last bin.
    Clipping to ``cols - 1`` states the intent and also rescues a coordinate that
    arrives slightly off-pitch, which shaving an epsilon off the coordinate would
    not.

    Returns -1 for rows with missing or non-finite coordinates, so the caller
    drops them explicitly rather than silently binning them to zone 0. Callers
    must filter ``z >= 0`` before using the result: ``np.bincount`` raises on
    negatives, and as an array index -1 silently reads the last zone.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    col = (np.where(valid, x, 0.0) / config.PITCH_LENGTH * cols).astype(int)
    row = (np.where(valid, y, 0.0) / config.PITCH_WIDTH * rows).astype(int)

    col = np.clip(col, 0, cols - 1)
    row = np.clip(row, 0, rows - 1)

    return np.where(valid, row * cols + col, -1)


def zone_centres(cols: int = config.GRID_COLS, rows: int = config.GRID_ROWS) -> np.ndarray:
    """Return an (n_zones, 2) array of zone centre coordinates, for plotting."""
    z = np.arange(cols * rows)

    col = z % cols
    row = z // cols

    centre_x = (col + 0.5) * config.PITCH_LENGTH / cols
    centre_y = (row + 0.5) * config.PITCH_WIDTH / rows

    return np.column_stack([centre_x, centre_y])


def _is_shot(actions: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the shots the model counts: attempts, minus penalties.

    Shared deliberately. shot_and_move_counts and scoring_probability must
    agree on what a shot is, because trap #8 is only harmless while they do:
    the zones where g falls back to 0 are exactly the zones where s is 0, so
    the fallback is multiplied away. Let the two definitions drift and that
    guess starts leaking into the fitted grid with nothing failing.

    Penalties are excluded because every one is struck from the same spot and
    would burn a false hotspot into g (trap #5).
    """
    return (actions.action_type.to_numpy() == "shot") & ~actions.is_penalty.to_numpy()


def _is_move(actions: pd.DataFrame) -> np.ndarray:
    """Boolean mask of move ATTEMPTS -- passes and carries, won or lost."""
    return np.isin(actions.action_type.to_numpy(), config.MOVE_TYPES)


def shot_and_move_counts(
    actions: pd.DataFrame,
    cols: int = config.GRID_COLS,
    rows: int = config.GRID_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Count shots and move ATTEMPTS per zone.

    Returns ``(shots, moves)``, each shape (n_zones,).

    Moves are all passes and carries attempted from the zone, successful or
    not -- a pass that gets cut out was still a decision to move rather than
    shoot. Exclude penalties from the shot count (trap #5).

    Both counts key off the START zone: a shot belongs to where it was struck,
    a move to where the player set off from.

    Set-piece passes ARE counted as moves. A corner is arguably not "a decision
    to move rather than shoot", so this is a judgement call rather than a fact
    -- it is recorded in the README with the other model decisions.
    """
    z = zone_index(actions.start_x, actions.start_y, cols, rows)

    # Create masks for negative pitch coords
    #                  only moves
    #                  shots that are non-penalty
    on_pitch = z >= 0

    is_move = on_pitch & _is_move(actions)
    is_shot = on_pitch & _is_shot(actions)

    # minlength pins the output at n_zones. Without it the length would be
    # max(z) + 1, so a grid whose last zone never sees an action would return a
    # short array and misalign silently against every other per-zone vector.
    n_zones = cols * rows
    shots = np.bincount(z[is_shot], minlength=n_zones)
    moves = np.bincount(z[is_move], minlength=n_zones)

    return shots, moves


def action_probabilities(
    shots: np.ndarray, moves: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(s, m)``, the per-zone shot and move probabilities.

    Must satisfy ``s + m == 1`` everywhere; ``m`` is derived as ``1 - s`` so the
    identity holds exactly rather than to floating-point luck.

    A zone with no events resolves to ``s = 0, m = 1``: there is no evidence anyone
    shoots from there, but value can still pass through. Left as 0/0 it would be
    NaN, and because every zone feeds every other through ``T @ xt``, one NaN
    blanks the whole grid within two iterations.
    """
    total = shots + moves

    s = np.divide(shots, total, out=np.zeros(len(total), dtype=float), where=total > 0)
    m = 1.0 - s

    return s, m


def scoring_probability(
    actions: pd.DataFrame,
    use_xg: bool = True,
    cols: int = config.GRID_COLS,
    rows: int = config.GRID_ROWS,
) -> np.ndarray:
    """Return ``g``, the per-zone probability that a shot becomes a goal.

    Two routes, and the choice belongs in your model metadata:
      * ``use_xg=True``  -- mean ``shot_statsbomb_xg`` per zone. Smoother, and
        defensible. Recommended default.
      * ``use_xg=False`` -- raw goals/shots. A corner zone with three shots and
        one goal gives 0.33, which is nonsense.

    Either way penalties are excluded first (trap #5), through ``_is_shot``.

    A zone with no shots resolves to 0 (trap #8). That is harmless rather than
    merely convenient: ``s`` is also 0 there, because ``shot_and_move_counts`` uses
    the same ``_is_shot`` mask, and the solve only ever uses the product ``s * g``,
    so the fallback is multiplied by zero. Measured on WC 2022 at 12x8, 68 of 96
    zones have no shots. If the two definitions of "a shot" ever drift apart, this
    stops being true silently.
    """

    n_zones = cols * rows
    z = zone_index(actions.start_x, actions.start_y, cols, rows)

    is_shot = _is_shot(actions) & (z >= 0)

    shots = np.bincount(z[is_shot], minlength=n_zones)

    weights = actions.xg.to_numpy() if use_xg else actions.is_goal.to_numpy()
    numerator = np.bincount(z[is_shot], weights=weights[is_shot], minlength=n_zones)

    return np.divide(numerator, shots, out=np.zeros(n_zones), where=shots > 0)


def transition_matrix(
    actions: pd.DataFrame,
    cols: int = config.GRID_COLS,
    rows: int = config.GRID_ROWS,
) -> np.ndarray:
    """Return the row-stochastic (n_zones, n_zones) matrix T.

    SUCCESSFUL moves only -- an intercepted pass does not put the ball in the
    target zone. Count start->end zone pairs, then normalise each row to sum to 1.

    A row with no observed moves falls back to UNIFORM, never to a self-transition:
    ``T[z, z] = 1`` in a zone where the ball cannot be lost or shot would make that
    row of ``(I - diag(m c) T)`` all zeros and the solve singular (see
    ``assert_solvable``). No zone needs the fallback on WC 2022 at 12x8.

    Because every row sums to 1, a failed move is NOT modelled here. That is what
    ``completion_rates`` and ``fit(model_loss=True)`` add back.

    Check: ``np.allclose(T.sum(axis=1), 1.0)``.
    """
    n_zones = cols * rows

    z_start = zone_index(actions.start_x, actions.start_y, cols, rows)
    z_end = zone_index(actions.end_x, actions.end_y, cols, rows)

    # SUCCESSFUL moves only
    ok = (
        _is_move(actions)
        & actions.successful.to_numpy()
        & (z_start >= 0)
        & (z_end >= 0)
    )

    # Flatten each (start, end) pair to one integer the same way a zone index
    # flattens (row, col), then bincount it and fold back to a square.
    pair = z_start[ok] * n_zones + z_end[ok]
    counts = np.bincount(pair, minlength=n_zones * n_zones).reshape(n_zones, n_zones)

    # A row with no observed moves falls back to UNIFORM, not to a
    # self-transition. T[z,z]=1 in a zone with m=1 makes row z of
    # (I - diag(m)T) all zeros and the solve singular -- see assert_solvable.
    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        row_totals,
        out=np.full((n_zones, n_zones), 1.0 / n_zones),
        where=row_totals > 0,
    )


def completion_rates(
    actions: pd.DataFrame,
    cols: int = config.GRID_COLS,
    rows: int = config.GRID_ROWS,
) -> np.ndarray:
    """Return ``c``, the per-zone share of move attempts that arrive.

    ``c(z) = successful moves from z / all move attempts from z``, keyed to the
    START zone like every other per-zone count. Shots, dribbles and the loss
    events (Dispossessed, Miscontrol) are not move attempts and are excluded --
    this is the denominator of ``m(z)``, so it must be the same set of actions.

    This is the quantity the textbook recursion drops. T is built from
    completions only, so its rows sum to 1 and a failed move vanishes from the
    chain; ``m * c`` puts the loss back (see the module docstring).

    A zone with no attempts gets the OVERALL completion rate rather than 1.0.
    Assuming a free pass there would manufacture a zone where the ball is
    never lost, which is the bug this factor exists to remove.
    """
    n_zones = cols * rows
    z = zone_index(actions.start_x, actions.start_y, cols, rows)

    attempted = _is_move(actions) & (z >= 0)
    completed = attempted & actions.successful.to_numpy()

    n_attempted = np.bincount(z[attempted], minlength=n_zones)
    n_completed = np.bincount(z[completed], minlength=n_zones)

    overall = n_completed.sum() / max(n_attempted.sum(), 1)
    return np.divide(
        n_completed,
        n_attempted,
        out=np.full(n_zones, overall),
        where=n_attempted > 0,
    )


def unreachable_zones(m: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Zones from which the chain can never be left, in any number of moves.

    ``m`` is the probability the ball is still in play after one step: the move
    probability, times the completion rate when loss is modelled. A zone is a
    way out of the chain when ``m < 1`` -- a shot is taken there, or (with loss
    modelled) a move can fail. A zone that can reach no such zone has no
    well-defined xT: its equation reduces to ``xt = xt``, satisfied by every
    number, and ``(I - diag(m)T)`` is singular. ``np.linalg.solve`` reports
    this as a bare "Singular matrix".

    Catches two shapes of the same fault:
      * one self-absorbing zone -- ``m=1`` and ``T[z,z]=1``, an all-zero row
      * a CLOSED GROUP of zones that only ever pass among themselves, where no
        single row is all-zero and a row-wise check sees nothing wrong

    Nearly always a sample-size artefact. One successful move out of a zone
    normalises to ``T[z,z] = 1.0``, which asserts the ball never leaves.
    """
    can_exit = (1.0 - m) > 0              # a shot, or a failable move, here
    edges = T > 0

    # Breadth-first closure: keep adding zones that can reach the set in one
    # move. Bounded by n_zones iterations; converges long before that.
    for _ in range(len(m)):
        grown = can_exit | (edges @ can_exit)
        if np.array_equal(grown, can_exit):
            break
        can_exit = grown

    return np.flatnonzero(~can_exit)


def assert_solvable(m: np.ndarray, T: np.ndarray) -> None:
    """Raise with a diagnosis if the xT system has no unique solution.

    Pass the CONTINUATION probability ``m * c``, not ``m`` alone, so a zone
    where moves can fail counts as a way out of the chain.
    """
    dead = unreachable_zones(m, T)
    if len(dead):
        raise ValueError(
            f"xT has no unique solution: no way out of the chain (a shot, or a "
            f"lost ball) is reachable from zone(s) {dead.tolist()}, so "
            f"(I - diag(m)T) is singular. The ball enters and never leaves, and "
            f"nobody shoots from there. Almost always too "
            f"little data per zone -- fit on more matches, or coarsen the grid. "
            f"Check that transition_matrix does not use a self-transition as its "
            f"empty-row fallback, which manufactures exactly this."
        )


class ExpectedThreat:
    """A fitted xT surface.

    Attributes after ``fit``:
        s, m, g   (n_zones,)  per-zone probabilities
        c         (n_zones,)  per-zone move completion rate (all 1.0 when
                              fitted with model_loss=False)
        T         (n_zones, n_zones) transition matrix
        xt        (n_zones,)  the fitted values
    """

    def __init__(self, cols: int = config.GRID_COLS, rows: int = config.GRID_ROWS):
        self.cols = cols
        self.rows = rows
        self.n_zones = cols * rows
        self.s = self.m = self.g = self.c = self.T = self.xt = None
        self.metadata: dict = {}

    # -- fitting ----------------------------------------------------------
    def fit(
        self,
        actions: pd.DataFrame,
        use_xg: bool = True,
        model_loss: bool = True,
    ) -> "ExpectedThreat":
        """Estimate s, m, g, c and T from an action frame, then solve.

        ``model_loss=True`` (default) scales the continuation term by the
        per-zone completion rate, so a failed move ends the chain with value 0.
        ``False`` is the textbook recursion, kept as the BASELINE: its rows of
        T sum to 1, failures vanish, and the surface comes out near-flat. Both
        are recorded in the metadata -- the choice belongs there.

        The guard runs on ``m * c``, after T is built and before either solver.
        Both fail uselessly on a singular system -- the closed form raises a
        bare LinAlgError, and the iteration silently fails to converge --
        whereas the guard names the offending zones.
        """
        # Pass the instance grid explicitly. These helpers default to
        # config.GRID_*, so omitting them silently fits a 12x8 model inside
        # an ExpectedThreat(16, 12), and the sensitivity check would report
        # numbers identical to the headline model.
        grid = dict(cols=self.cols, rows=self.rows)
        self.s, self.m = action_probabilities(*shot_and_move_counts(actions, **grid))
        self.g = scoring_probability(actions, use_xg=use_xg, **grid)
        self.T = transition_matrix(actions, **grid)
        self.c = (
            completion_rates(actions, **grid) if model_loss else np.ones(self.n_zones)
        )
        assert_solvable(self.m * self.c, self.T)

        self.xt = self.solve_closed_form()

        self.metadata = {
            "cols": self.cols,
            "rows": self.rows,
            "n_zones": self.n_zones,
            "n_actions": len(actions),
            "n_matches": actions.match_id.nunique(),
            "competitions": sorted(actions.competition.dropna().unique().tolist()),
            "use_xg": use_xg,
            "model_loss": model_loss,
            "penalties_excluded": True,  # always: see _is_shot
            "fitted_on": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        return self

    def solve_iterative(self, tol: float = 1e-12, max_iter: int = 2000):
        """Value iteration. Returns ``(xt, deltas)``.

            xt <- s*g + m*c * (T @ xt),  starting from zeros

        Iteration k has a real meaning: P(score within the next k actions).
        Keep the per-iteration max|delta| -- it is your convergence figure.

        ``tol`` bounds the STEP between successive iterates, NOT the distance to the
        answer. When you stop, the error still remaining is roughly
        ``step / (1 - rho)``, where rho is the spectral radius of ``diag(m*c) T``.
        Measured on WC 2022 at 12x8, rho = 0.880 with loss modelled (the default), so
        that factor is about 8x; without it rho = 0.964 and the factor is about 28x.
        Stopping at a 1e-10 step in the second case leaves roughly 3e-09 of error,
        which overshoots the 1e-9 the agreement test asks for. Choose tol a couple of
        orders tighter than the accuracy you need.

        Convergence takes about ``log(tol) / log(rho)`` iterations: 173 to a 1e-12 step
        with loss modelled, 617 without. The count is large because rho is near 1,
        which is because shots are rare (about 1% of all actions) and value leaks out
        of the chain slowly. Modelling loss lets failed moves leak value too, so rho
        is smaller.

        Raises if max_iter runs out. A truncated solve returns a plausible vector that
        is simply wrong, and silently disagrees with the closed form by orders of
        magnitude.
        """
        # Local, not self.xt. Starting from the already-solved values would
        # converge in one iteration with delta 0, which makes the agreement
        # test with solve_closed_form vacuous and leaves plot_convergence a
        # single point instead of a line. This method must stay an INDEPENDENT
        # check on the closed form, so it reads nothing the closed form wrote.
        xt = np.zeros(self.n_zones)
        deltas = []

        for _ in range(max_iter):
            xt_next = self.s * self.g + (self.m * self.c) * (self.T @ xt)
            delta = float(np.max(np.abs(xt_next - xt)))
            deltas.append(delta)
            xt = xt_next

            if delta < tol:
                break
        else:
            # for-else: reached only when the loop was never broken out of.
            rho = deltas[-1] / deltas[-2] if len(deltas) > 1 else float("nan")
            raise RuntimeError(
                f"value iteration did not converge in {max_iter} iterations "
                f"(last step {delta:.2e}, tol {tol:.0e}). Observed contraction "
                f"rate ~{rho:.4f}, so roughly "
                f"{np.log(tol) / np.log(rho):.0f} iterations are needed. Raise "
                f"max_iter, or use solve_closed_form."
            )

        return xt, deltas

    def solve_closed_form(self) -> np.ndarray:
        """Exact solve of ``(I - diag(m*c) T) xT = s*g``.

        ``keep[:, None] * T`` is ``diag(keep) @ T`` without materialising the
        diagonal. This must agree with ``solve_iterative`` to 1e-9 -- each is
        a free correctness check on the other.
        """
        keep = self.m * self.c
        A = np.eye(self.n_zones) - keep[:, None] * self.T
        return np.linalg.solve(A, self.s * self.g)

    # -- use --------------------------------------------------------------
    def grid(self) -> np.ndarray:
        """Return xt reshaped to (rows, cols), for plotting and comparison."""
        return self.xt.reshape(self.rows, self.cols)

    def rate_actions(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Add ``xt_start``, ``xt_end``, ``xt_delta`` and ``xt_delta_clipped``.

        Returns ONLY the rated rows -- successful passes and carries -- as a new frame
        with the original index kept, so it can be joined back. Failed moves, shots and
        losses are absent, not NaN: the model has no term for them, so they are
        excluded rather than charged ``-xt_start``.

        ``xt_delta`` is net progression: backward moves are negative and kept.
        ``xt_delta_clipped`` floors it at zero, for gross threat. Set pieces are left
        in and flagged by ``is_set_piece``; excluding them is a decision for the
        headline table, not this function.

        Needs only ``xt``, so it works on a loaded model.
        """
        if self.xt is None:
            raise RuntimeError("There are no values to rate against: fit or load a model first.")

        mask = _is_move(actions) & actions.successful.to_numpy()
        rated = actions[mask].copy()

        grid = dict(cols=self.cols, rows=self.rows)
        z_start = zone_index(rated.start_x, rated.start_y, **grid)
        z_end = zone_index(rated.end_x, rated.end_y, **grid)
        assert (z_start >= 0).all() and (z_end >= 0).all(), "a rated action maps to zone -1: missing coordinates survived the filter"

        rated["xt_start"] = self.xt[z_start]
        rated["xt_end"] = self.xt[z_end]
        rated["xt_delta"] = rated.xt_end - rated.xt_start
        rated["xt_delta_clipped"] = rated.xt_delta.clip(lower=0)

        return rated

    # -- persistence ------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        """Write ``xt_grid_{cols}x{rows}.npy`` plus a JSON metadata sidecar.

        The sidecar must record: grid dims, which competitions were fitted on,
        whether penalties were excluded, and whether g came from xG or the raw
        goal rate. Future you will not remember.
        """
        if path is None:
            # The variants must not share a filename, or fitting the baseline
            # after the headline model silently overwrites it.
            variant = "" if self.metadata.get("model_loss", True) else "_baseline"
            path = config.MODELS_DIR / f"xt_grid_{self.cols}x{self.rows}{variant}.npy"

        sidecar = path.with_suffix(".json")

        np.save(path, self.grid())
        sidecar.write_text(json.dumps(self.metadata, indent=2))

        return path

    @classmethod
    def load(cls, path: Path) -> "ExpectedThreat":
        """Load a saved grid and its metadata.

        Returns a PARTIAL model: ``xt`` (flat, n_zones), ``metadata``, and the grid
        dimensions read from the sidecar. ``grid()`` and ``rate_actions`` work.

        ``s``, ``m``, ``g``, ``c`` and ``T`` are None. They are estimated from an
        actions frame, and that frame is not saved, so they cannot be reconstructed
        from the file alone. The file keeps the fitted surface, not the working that
        produced it.

        Both solvers therefore raise on a loaded model. ``load`` is for reading a
        surface, not resuming a fit.

        Raises ValueError when the array and the sidecar disagree about the grid size,
        which means the two files are a mismatched pair.
        """

        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())

        model = cls(cols=metadata["cols"], rows=metadata["rows"])
        model.xt = np.load(path).ravel()
        model.metadata = metadata

        if model.xt.size != model.n_zones:
            raise ValueError(
                f"mismatched model files: {path.name} holds {model.xt.size} values "
                f"but {sidecar.name} describes a {model.cols}x{model.rows} grid "
                f"({model.n_zones} zones)"
            )

        return model


def load_reference_grid(refresh: bool = False) -> np.ndarray:
    """Fetch Karun Singh's published 12x8 grid, shape (8, 12).

    Source: https://karun.in/blog/data/open_xt_12x8_v1.json

    Your single highest-value validation. Expect Pearson r > 0.97 and MAE < 0.01;
    his grid was fitted on different leagues, so the SHAPE agreeing is the goal,
    not the exact values.

    The published grid is exactly mirror-symmetric top to bottom (the flipped grid
    is identical), so the y-orientation cannot change r and no flip is needed. A
    grid fitted from raw events will not be symmetric: 4.0% mean asymmetry on
    WC 2022 at 12x8.

    Cached in ``config.MODELS_DIR``, which git tracks, so a fresh clone has it and
    the test suite stays offline. ``refresh=True`` downloads it again.
    """
    if REFERENCE_GRID_CACHE.exists() and not refresh:
        data = json.loads(REFERENCE_GRID_CACHE.read_text())
    else:
        with urllib.request.urlopen(REFERENCE_GRID_URL, timeout=30) as response:
            data = json.loads(response.read())
        REFERENCE_GRID_CACHE.write_text(json.dumps(data))

    grid = np.array(data, dtype=float)
    if grid.shape != (8, 12):
        raise ValueError(
            f"reference grid should be (8, 12) to match ExpectedThreat.grid(), "
            f"but got {grid.shape} -- the source format may have changed"
        )
    return grid
