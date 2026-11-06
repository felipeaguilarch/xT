"""Every number the README quotes, recomputed from the data.

The README states results; this module is where they come from. Each public
function returns the values behind one table or one block of figures in the
README, as plain floats, dicts or DataFrames, and the README's "Note." under each
table names the function. ``tests/test_report.py`` pins the README's values, so
the text cannot drift from the code unnoticed.

Run from the command line (about a minute to build the context, mostly the model
fit and the pressing pipeline):

    python -m xtpress.report                 # every section
    python -m xtpress.report table_1 gk      # chosen sections
    python -m xtpress.report --list          # the section names

Everything is computed from ``data/processed`` (the action frame), ``data/raw``
(the raw events) and ``data/models/reference_xt_12x8.json`` (Karun Singh's grid).
Build the first two with ``python -m xtpress.loaders --competition wc_2022``.

Conventions that matter when you compare with the README:

* Bootstraps are percentile intervals with ``np.random.default_rng(0)``.
* "Held out" means cross-fitting by match inside the focus competition.
* Spearman is the Pearson correlation of average ranks (no scipy needed).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys

import numpy as np
import pandas as pd

from . import config, loaders
from . import press_resistance as pr
from . import pressing_sequences as ps
from .expected_threat import ExpectedThreat, load_reference_grid, zone_index

BOOTSTRAP_REPS = 4000
BOOTSTRAP_SEED = 0

# Display names for the README tables and the figures: StatsBomb gives the full legal name.
SHORT_NAMES = {
    "Hirving Rodrigo Lozano Bahena": "Hirving Lozano",
    "Aziz Eraltay Behich": "Aziz Behich",
    "Rafael Alexandre Conceição Leão": "Rafael Leão",
    "Lionel Andrés Messi Cuccittini": "Lionel Messi",
    "Richarlison de Andrade": "Richarlison",
    "Gue-Sung Cho": "Gue-Sung Cho",
    "Unai Simón Mendibil": "Unai Simón",
    "Damián Emiliano Martínez": "Emiliano Martínez",
    "Diogo Meireles Costa": "Diogo Costa",
    "Kylian Mbappé Lottin": "Kylian Mbappé",
    "Raphael Dias Belloli": "Raphinha",
    "Achraf Hakimi Mouh": "Achraf Hakimi",
    "Cody Mathès Gakpo": "Cody Gakpo",
    "Dayotchanculle Upamecano": "Dayot Upamecano",
    "Theo Bernard François Hernández": "Theo Hernández",
    "Abdelkarim Hassan Al Haj Fadlalla": "Abdelkarim Hassan",
    "Jean-Eric Maxim Choupo-Moting": "Eric Choupo-Moting",
    "Neymar da Silva Santos Junior": "Neymar",
    "Salem Mohammed Al Dawsari": "Salem Al-Dawsari",
    "Sergio Busquets i Burgos": "Sergio Busquets",
    "Alejandro Darío Gómez": "Alejandro Gómez",
    "Ernesto Alexis Vega Rojas": "Alexis Vega",
}


def short_name(name: str) -> str:
    """The README's display name for a StatsBomb player name (unchanged if unmapped)."""
    return SHORT_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Context:
    """Everything the README numbers are derived from, built once."""

    competition: str
    actions: pd.DataFrame
    model: ExpectedThreat
    rated: pd.DataFrame
    events: pd.DataFrame
    minutes: pd.DataFrame
    pressures: pd.DataFrame
    sequences: pd.DataFrame
    summary: pd.DataFrame
    pressured: pd.DataFrame
    baselines: pd.DataFrame
    players: pd.DataFrame


def _sequences(events, actions, rated, pressures, gap=config.PRESS_SEQUENCE_GAP_SECONDS,
               window=config.PRESS_REGAIN_WINDOW_SECONDS, high_x=config.PRESS_REGAIN_MIN_X):
    seq = ps.build_press_sequences(pressures, gap_seconds=gap)
    seq = ps.label_sequence_outcomes(seq, events, window=window, high_regain_min_x=high_x)
    return ps.sequence_value(seq, rated, actions)


@functools.lru_cache(maxsize=2)
def build_context(competition: str = config.FOCUS_COMPETITION) -> Context:
    """Fit the headline model and run the pressing and press-resistance pipelines."""
    actions = loaders.load_actions([competition])
    model = ExpectedThreat(cols=config.GRID_COLS, rows=config.GRID_ROWS).fit(actions)
    rated = model.rate_actions(actions)
    events = pd.concat([loaders.load_events(int(m)) for m in sorted(actions.match_id.unique())],
                       ignore_index=True)
    minutes = loaders.minutes_played(events)
    pressures = ps.extract_pressures(events)
    sequences = _sequences(events, actions, rated, pressures)
    summary = ps.team_pressing_summary(sequences, events)
    pressured = pr.classify_outcome(pr.pressured_actions(
        actions.join(rated[["xt_start", "xt_end", "xt_delta"]])))
    baselines = pr.zone_baselines(pressured)
    players = pr.player_press_resistance(pressured, baselines, minutes)
    players["score"] = pr.press_resistance_score(players)
    return Context(competition, actions, model, rated, events, minutes, pressures,
                   sequences, summary, pressured, baselines, players)


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def spearman(a, b) -> float:
    """Spearman correlation: Pearson on average ranks."""
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    return float(a.rank().corr(b.rank()))


def bootstrap_spearman(a, b, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Spearman rho with a 95% percentile bootstrap interval over the rows."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        i = rng.integers(0, len(a), len(a))
        draws.append(spearman(a[i], b[i]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"rho": spearman(a, b), "lo": float(lo), "hi": float(hi), "reps": reps, "seed": seed}


def _xy(events: pd.DataFrame) -> pd.DataFrame:
    xy = pd.DataFrame(events["location"].map(loaders._xy).tolist(), index=events.index, columns=["x", "y"])
    return events.join(xy)


# ---------------------------------------------------------------------------
# Part I: the Expected Threat surface
# ---------------------------------------------------------------------------

def expected_threat_checks(ctx: Context) -> dict:
    """Validation rows for the surface, and the figures quoted in the model decisions."""
    m, a = ctx.model, ctx.actions
    grid, ref = m.grid(), load_reference_grid()
    xt_iter, deltas = m.solve_iterative()
    baseline = ExpectedThreat(cols=config.GRID_COLS, rows=config.GRID_ROWS).fit(a, model_loss=False)

    moves = a[a.action_type.isin(config.MOVE_TYPES)]
    passes = a[a.action_type == "pass"]
    shots = a[(a.action_type == "shot") & ~a.is_penalty]
    shot_zones = np.unique(zone_index(shots.start_x, shots.start_y))
    return {
        "iterative_vs_closed_max_abs": float(np.abs(xt_iter - m.xt).max()),
        "iterations": len(deltas),
        "pearson_r_vs_published": float(np.corrcoef(grid.ravel(), ref.ravel())[0, 1]),
        "mae_vs_published": float(np.abs(grid - ref).mean()),
        "baseline_pearson_r": float(np.corrcoef(baseline.grid().ravel(), ref.ravel())[0, 1]),
        "goal_end_over_own_end": float(grid[:, -1].mean() / grid[:, 0].mean()),
        "baseline_goal_end_over_own_end": float(baseline.grid()[:, -1].mean() / baseline.grid()[:, 0].mean()),
        "published_goal_end_over_own_end": float(ref[:, -1].mean() / ref[:, 0].mean()),
        "transition_rows_max_deviation": float(np.abs(m.T.sum(axis=1) - 1).max()),
        "pass_completion": float(passes.successful.mean()),
        "n_actions": len(a), "n_matches": int(a.match_id.nunique()),
        "moves_arriving": float(moves.successful.mean()),
        "completion_zone_min": float(m.c.min()), "completion_zone_max": float(m.c.max()),
        "set_piece_share_of_move_attempts": float(moves.is_set_piece.mean()),
        "penalties_excluded": int(((a.action_type == "shot") & a.is_penalty).sum()),
        "zones_without_shots": int(config.N_ZONES - len(shot_zones[shot_zones >= 0])),
        "published_grid_asymmetry": float(np.abs(ref - ref[::-1]).mean() / ref.mean()),
        "asymmetry_top_bottom": float(np.abs(grid - grid[::-1]).mean() / grid.mean()),
        "grid_max": float(grid.max()), "grid_min": float(grid.min()),
    }


def cross_fitted_team_matches(ctx: Context) -> pd.DataFrame:
    """One row per (match, team): net xT, goals and xG, with the grid cross-fitted by match.

    Matches are sorted by id and alternate between two folds; each fold is rated
    by a grid fitted on the other, so no match is rated by a grid that saw it.
    Open play, regulation time. Goals are shots flagged as goals, penalties
    included, shootouts and extra time excluded. Columns: ``match_id, team, xt,
    goals, xg``. This is what ``viz.plot_team_xt_vs_goals`` draws.
    """
    a = ctx.actions
    ids = sorted(a.match_id.unique())
    fold = a.match_id.map({m: i % 2 for i, m in enumerate(ids)})
    rated = pd.concat([ExpectedThreat().fit(a[fold != f]).rate_actions(a[fold == f]) for f in (0, 1)])
    rated = rated[~rated.is_set_piece & rated.period.isin(config.REGULATION_PERIODS)]
    tm = rated.groupby(["match_id", "team"]).xt_delta.sum().rename("xt").reset_index()

    reg = a[a.period.isin(config.REGULATION_PERIODS) & (a.action_type == "shot")]
    goals = reg[reg.is_goal].groupby(["match_id", "team"]).size().rename("goals").reset_index()
    xg = reg.groupby(["match_id", "team"]).xg.sum().rename("xg").reset_index()
    return tm.merge(goals, how="left").merge(xg, how="left").fillna({"goals": 0, "xg": 0})


def team_xt_vs_goals(ctx: Context, reps: int = BOOTSTRAP_REPS) -> dict:
    """Net xT per match against goals and xG, with the grid cross-fitted by match.

    The per-match table is ``cross_fitted_team_matches``; this adds the team means,
    the in-sample comparison and the bootstrap intervals.
    """
    tm = cross_fitted_team_matches(ctx)
    tt = tm.groupby("team")[["xt", "goals", "xg"]].mean()

    insample = ctx.rated[~ctx.rated.is_set_piece & ctx.rated.period.isin(config.REGULATION_PERIODS)]
    insample = (insample.groupby(["match_id", "team"]).xt_delta.sum().rename("xt").reset_index()
                .merge(tm[["match_id", "team", "goals"]], how="left")
                .fillna({"goals": 0}))

    out = {"n_teams": len(tt), "n_team_matches": len(tm), "goals": int(tm.goals.sum()),
           "goalless_team_matches": int((tm.goals == 0).sum()),
           "in_sample_team_match_rho": spearman(insample.xt, insample.goals),
           "goals_vs_xg_team_match_rho": spearman(tm.goals, tm.xg),
           "goals_vs_xg_team_rho": spearman(tt.goals, tt.xg)}
    for level, df in (("team", tt), ("team_match", tm)):
        for target in ("goals", "xg"):
            b = bootstrap_spearman(df.xt, df[target], reps=reps)
            out[f"{level}_xt_vs_{target}"] = {k: b[k] for k in ("rho", "lo", "hi")}
    return out


def figure_facts(ctx: Context, case_study_sequence: int = 772) -> dict:
    """The figures quoted in the sentences under the figures in the README.

    ``surface``, ``convergence`` and ``reference`` are the values drawn in Figures 1 to 3;
    ``progressors`` and ``player_actions`` those of Figures 4 and 5; ``team_xt`` the
    teams labelled in Figure 6; ``case_study`` the sequence of Figure 8 (picked by its
    ``sequence_id`` column) and ``leaderboard`` the position mix of Figure 10. Figures 7
    and 9 quote numbers that ``pressing``, ``press_resistance`` and
    ``press_resistance_standard_errors`` already give.
    """
    from . import viz  # drawn from here, so imported late: viz imports this module

    g, ref = ctx.model.grid(), load_reference_grid()
    baseline = ExpectedThreat(cols=config.GRID_COLS, rows=config.GRID_ROWS).fit(ctx.actions, model_loss=False)
    steps = {"default": ctx.model.solve_iterative()[1], "baseline": baseline.solve_iterative()[1]}

    table = viz.progressor_table(ctx.rated, ctx.minutes)
    top = table.head(20)
    actions = {}
    for name in ("Luka Modrić", "Kylian Mbappé Lottin"):
        best = ctx.rated[ctx.rated.player == name].nlargest(30, "xt_delta")
        actions[short_name(name)] = best.action_type.value_counts().to_dict()
    mbappe_carries = ctx.rated[ctx.rated.player == "Kylian Mbappé Lottin"].nlargest(30, "xt_delta")
    mbappe_carries = mbappe_carries[mbappe_carries.action_type == "carry"]
    gap = np.abs(g - ref)

    tm = cross_fitted_team_matches(ctx)
    teams = tm.groupby("team")[["xt", "goals"]].mean()

    outfield = ctx.players[ctx.players.position_group != "GK"]
    best20 = outfield.sort_values("score", ascending=False).head(20)

    seq = ctx.sequences[ctx.sequences.sequence_id == case_study_sequence].iloc[0]
    won = ctx.sequences[ctx.sequences.won_possession]
    press = ctx.pressures[(ctx.pressures.match_id == seq.match_id) & (ctx.pressures.team_id == seq.team_id)
                          & (ctx.pressures.period == seq.period)
                          & ctx.pressures.time_seconds.between(seq.start_time, seq.end_time)]
    poss = ctx.actions[(ctx.actions.match_id == seq.match_id) & (ctx.actions.possession == seq.won_possession_id)
                       & (ctx.actions.team_id == seq.team_id)].sort_values("index")
    shot = poss[poss.action_type == "shot"].iloc[0]
    before_shot = poss[poss["index"] < shot["index"]]
    carry = before_shot.iloc[-1]
    return {
        "surface": {"goal_mouth_zone_min": float(min(g[3, 11], g[4, 11])), "penalty_spot_zone": float(g[4, 10]),
                    "halfway_zone": float(g[3, 6]), "own_third_zone": float(g[3, 1]),
                    "published_max": float(ref.max()), "zones_above_0_05": int((g > 0.05).sum())},
        "convergence": {"iterations": {k: len(v) for k, v in steps.items()},
                        "rho": {k: float(v[-1] / v[-2]) for k, v in steps.items()}},
        "reference": {"largest_gap": float(gap.max()), "zones_with_largest_gaps": int((gap > 0.07).sum()),
                      "zones_off_by_more_than_0_01": int((gap > 0.01).sum()),
                      "share_of_abs_error_in_largest_two": float(np.sort(gap.ravel())[-2:].sum() / gap.sum())},
        "progressors": {"players": len(table), "teams": int(table.team.nunique()),
                        "top3": [(short_name(p), float(v)) for p, v in zip(table.player[:3], table.total_p90[:3], strict=True)],
                        "pass_share_of_top20": float(top.pass_p90.sum() / top.total_p90.sum()),
                        "carry_led_in_top20": int((top.carry_p90 > top.pass_p90).sum())},
        "player_actions": actions,
        "mbappe_top_carries_from_left_third": int((mbappe_carries.start_y < config.PITCH_WIDTH / 3).sum()),
        "team_xt": {t: {"xt": float(teams.xt[t]), "goals": float(teams.goals[t])}
                    for t in ("England", "Germany", "Costa Rica")},
        "leaderboard": {"top20_by_group": best20.position_group.value_counts().to_dict(),
                        "mean_score_by_group": {k: float(v) for k, v in
                                                outfield.groupby("position_group").score.mean().items()},
                        "forward_share_of_outfield": float((outfield.position_group == "FW").mean()),
                        "top20_n_pressured_min": int(best20.n_pressured.min()),
                        "top20_n_pressured_max": int(best20.n_pressured.max())},
        "case_study": {"team": seq.team, "opponent": seq.possession_team, "pressures": len(press),
                       "duration_s": float(seq.end_time - seq.start_time),
                       "seconds_to_regain": float(seq.seconds_to_regain), "regain_x": float(seq.regain_x),
                       "last_presser": short_name(press.sort_values("time_seconds").player.iloc[-1]),
                       "shooter": short_name(shot.player), "shot_xg": float(shot.xg),
                       "carry_before_shot_yards": float(np.hypot(carry.end_x - carry.start_x, carry.end_y - carry.start_y)),
                       "xt_gained": float(seq.xt_gained), "mean_xt_per_won_sequence": float(won.xt_gained.mean()),
                       "share_of_sequences_with_3_plus_pressures": float((ctx.sequences.n_pressures >= 3).mean())},
    }


# ---------------------------------------------------------------------------
# Part II: pressing
# ---------------------------------------------------------------------------

def table_1(ctx: Context, n: int = 3) -> pd.DataFrame:
    """Table 1: the ``n`` best and ``n`` worst teams by xT won per 100 pressures."""
    s = ctx.summary.copy()
    s["matches"] = s.team.map(ctx.events.groupby("team").match_id.nunique())
    s = s.sort_values("xt_per_100_pressures", ascending=False)
    s = pd.concat([s.head(n), s.tail(n)])
    return s[["team", "matches", "ppda", "regain_rate", "xt_per_100_pressures", "shots_from_presses"]].reset_index(drop=True)


def pressing_overview(ctx: Context) -> dict:
    """The pressing figures quoted in the Finding, Pressing results and Validation."""
    seq, ev, s = ctx.sequences, ctx.events, ctx.summary
    won = seq[seq.won_possession]
    pm = ps.ppda(ev)
    touch = ps.pressures_per_opponent_half_touch(ev)
    pm_touch = pm.merge(touch, on=["match_id", "team_id"])
    xy_press = ctx.pressures
    per_tm = seq.groupby(["match_id", "team_id"]).size()
    boot = bootstrap_spearman(s.ppda, s.xt_per_100_pressures)
    high = won[won.won_high]
    many = s[s.team.map(ev.groupby("team").match_id.nunique()) >= 4]       # teams with four or more matches
    return {
        "n_pressures": len(ctx.pressures), "n_sequences": len(seq),
        "won": int(seq.won_possession.sum()), "regain_rate": float(seq.won_possession.mean()),
        "won_high": int(seq.won_high.sum()), "high_rate": float(seq.won_high.mean()),
        "high_share_of_wins": float(won.won_high.mean()),
        "teams": len(s), "median_ppda_team": float(s.ppda.median()),
        "ppda_team_min": float(s.ppda.min()), "ppda_team_max": float(s.ppda.max()),
        "teams_ppda_inside_6_15": int(s.ppda.between(6, 15).sum()),
        "team_matches": len(pm), "median_ppda_team_match": float(pm.ppda.median()),
        "ppda_team_match_inside_6_15": float(pm.ppda.between(6, 15).mean()),
        "ppda_team_match_min": float(pm.ppda.min()), "ppda_team_match_max": float(pm.ppda.max()),
        "median_payoff": float(s.xt_per_100_pressures.median()),
        "ppda_payoff": boot,
        "shots_per_team_min": int(s.shots_from_presses.min()), "shots_per_team_max": int(s.shots_from_presses.max()),
        "median_seconds_to_regain": float(won.seconds_to_regain.median()),
        "share_regain_within_2s": float((won.seconds_to_regain <= 2).mean()),
        "p90_seconds_to_regain": float(won.seconds_to_regain.quantile(0.9)),
        "median_regain_x": float(won.regain_x.median()),
        "regain_own_third": float((won.regain_x < 40).mean()),
        "regain_final_third": float((won.regain_x >= 80).mean()),
        "xt_per_won_sequence": float(won.xt_gained.mean()),
        "won_reaching_shot": float(won.reached_shot.mean()),
        "zone_xt_high_regain": float(high.start_zone_xt.mean()),
        "zone_xt_other_wins": float(won[~won.won_high].start_zone_xt.mean()),
        "high_regains_reaching_shot": float(high.reached_shot.mean()),
        "sequences_per_team_match_median": float(per_tm.median()),
        "sequences_per_team_match_min": int(per_tm.min()), "sequences_per_team_match_max": int(per_tm.max()),
        "single_pressure_share": float((seq.n_pressures == 1).mean()),
        "median_sequence_duration_s": float((seq.end_time - seq.start_time).median()),
        "pressures_per_sequence": float(seq.n_pressures.sum() / len(seq)),
        "touch_ratio_median": float(touch.pressures_per_opp_half_touch.median()),
        "touch_ratio_min": float(touch.pressures_per_opp_half_touch.min()),
        "touch_ratio_max": float(touch.pressures_per_opp_half_touch.max()),
        "pressures_in_opponent_half": float((xy_press.x >= config.HALFWAY_X).mean()),
        "ppda_vs_touch_team_match_rho": spearman(pm_touch.ppda, pm_touch.pressures_per_opp_half_touch),
        "ppda_vs_touch_team_rho": spearman(s.ppda, s.pressures_per_opp_half_touch),
        "teams_with_4_plus_matches": len(many),
        "ppda_payoff_rho_4_plus_matches": spearman(many.ppda, many.xt_per_100_pressures),
    }


def sensitivity(ctx: Context) -> dict:
    """How the pressing results move with the window, the gap and the high-regain line."""
    ev, a, rated, p = ctx.events, ctx.actions, ctx.rated, ctx.pressures
    base = ctx.summary.set_index("team").xt_per_100_pressures
    own = ctx.sequences.possession_team == ctx.sequences.team

    windows = {}
    for w in (2, 4, 6, 8, 10, 12, 15):
        seq = ctx.sequences if w == config.PRESS_REGAIN_WINDOW_SECONDS else _sequences(ev, a, rated, p, window=w)
        summ = ps.team_pressing_summary(seq, ev).set_index("team")
        windows[w] = {
            "regain_rate": float(seq.won_possession.mean()),
            "regain_rate_excluding_own_possession": float(seq.won_possession[seq.possession_team != seq.team].mean()),
            "ppda_payoff_rho": spearman(summ.ppda, summ.xt_per_100_pressures),
            "payoff_rank_rho_vs_default": spearman(summ.xt_per_100_pressures, base.reindex(summ.index)),
        }
    gaps = {}
    for g in (2, 5, 12):
        seq = ctx.sequences if g == config.PRESS_SEQUENCE_GAP_SECONDS else _sequences(ev, a, rated, p, gap=g)
        summ = ps.team_pressing_summary(seq, ev).set_index("team")
        gaps[g] = {"n_sequences": len(seq), "regain_rate": float(seq.won_possession.mean()),
                   "payoff_max_abs_diff_vs_default": float((summ.xt_per_100_pressures - base.reindex(summ.index)).abs().max())}
    seq = ctx.sequences
    won = seq.won_possession
    high_line = {x: {"share_of_sequences": float((won & (seq.regain_x >= x)).mean())} for x in (50, 60, 70, 80)}
    return {"windows": windows, "gaps": gaps, "high_regain_line": high_line,
            "own_possession_sequences": int(own.sum()), "own_possession_share": float(own.mean())}


def pressing_decisions(ctx: Context) -> dict:
    """Linking, counterpress, credit-once and the touch definition: the Pressing decisions."""
    ev, pressures, seq = ctx.events, ctx.pressures, ctx.sequences
    linked = ps.link_pressures_to_actions(pressures, ev)
    method = linked.link_method.fillna("none").value_counts(normalize=True)

    related = linked[linked.link_method == "related"][["id", "action_id", "action_player_id"]]
    sub = pressures[pressures.id.isin(related.id)]
    agree = {}
    for direction in ("nearest", "forward"):
        w = _window_link(sub, ev, direction).merge(related, on="id", suffixes=("_window", "_related"))
        same_event = w.action_id_window == w.action_id_related
        agree[direction] = {
            "matched_an_action": float(len(w) / len(sub)),
            # two denominators: of the pressures the fallback matched, and of ALL pressures with a true link
            "same_event_of_matched": float(same_event.mean()),
            "same_event_of_all_linked": float(same_event.sum() / len(sub)),
            "same_player_of_matched": float((w.action_player_id_window == w.action_player_id_related).mean()),
        }

    opp = seq[seq.possession_team != seq.team]
    cp, no = opp[opp.any_counterpress], opp[~opp.any_counterpress]
    won = seq[seq.won_possession]
    credited = won.sort_values("seconds_to_regain").drop_duplicates(["match_id", "team_id", "won_possession_id"])
    duplicates = len(won) - len(credited)
    extra_shots = int(won.reached_shot.sum() - credited.reached_shot.sum())

    return {
        "linked_by_related_events": float(method.get("related", 0)),
        "linked_by_window": float(method.get("window", 0)),
        "unlinked": float(method.get("none", 0)),
        "fallback_vs_true_link": agree,
        "counterpress_share_of_pressures": float(pressures.counterpress.mean()),
        "counterpress_share_of_sequences": float(seq.any_counterpress.mean()),
        "counterpress_regain_rate_vs_opponent_possession": float(cp.won_possession.mean()),
        "other_regain_rate_vs_opponent_possession": float(no.won_possession.mean()),
        "counterpress_high_rate": float(cp.won_high.mean()),
        "other_high_rate": float(no.won_high.mean()),
        "counterpress_high_share_of_wins": float(cp[cp.won_possession].won_high.mean()),
        "other_high_share_of_wins": float(no[no.won_possession].won_high.mean()),
        "counterpress_xt_per_win": float(cp[cp.won_possession].xt_gained.mean()),
        "other_xt_per_win": float(no[no.won_possession].xt_gained.mean()),
        "won_sequences": len(won),
        "won_sequences_in_shared_possessions": int(won.duplicated(["match_id", "team_id", "won_possession_id"], keep=False).sum()),
        "share_in_shared_possessions": float(
            won.duplicated(["match_id", "team_id", "won_possession_id"], keep=False).mean()),
        "surplus_sequences_dropped_by_credit_once": duplicates,
        "xt_inflation_if_double_counted": float(won.xt_gained.sum() / credited.xt_gained.sum() - 1),
        "extra_shots_if_double_counted": extra_shots,
    }


def _window_link(pressures: pd.DataFrame, events: pd.DataFrame, direction: str = "nearest") -> pd.DataFrame:
    """The two-sided time-window fallback of ``link_pressures_to_actions``, with a direction.

    ``direction="nearest"`` is the package's behaviour; ``"forward"`` looks only
    for opponent actions at or after the pressure, which is what a one-sided
    join would do (used to show why the window is two-sided).
    """
    ev = events.assign(time_seconds=loaders._timestamp_seconds(events["timestamp"]))
    teams = ev[["match_id", "team_id"]].dropna().drop_duplicates()
    opp = pressures[["id", "match_id", "team_id"]].merge(teams, on="match_id", suffixes=("", "_opp"))
    opp = opp[opp["team_id"] != opp["team_id_opp"]][["id", "team_id_opp"]]
    left = pressures.merge(opp, on="id")[["id", "match_id", "period", "time_seconds", "team_id_opp"]].sort_values("time_seconds")
    cand = ev[loaders._flag(ev, "under_pressure")][["id", "match_id", "period", "time_seconds", "team_id", "player_id"]]
    cand = cand.rename(columns={"id": "action_id", "player_id": "action_player_id"}).sort_values("time_seconds")
    out = pd.merge_asof(left, cand, on="time_seconds", left_by=["match_id", "period", "team_id_opp"],
                        right_by=["match_id", "period", "team_id"], direction=direction,
                        tolerance=config.PRESS_MATCH_WINDOW_SECONDS)
    return out.dropna(subset=["action_id"])[["id", "action_id", "action_player_id"]]


def ppda_alternatives(ctx: Context) -> dict:
    """PPDA under other definitions: set-piece passes out, and the wrong zone threshold."""
    ev = ctx.events
    base = _ppda_variant(ev)
    no_sp = _ppda_variant(ev, drop_set_piece_passes=True)
    both = base.merge(no_sp, on=["match_id", "team_id"], suffixes=("", "_ex"))

    e = _xy(ev)
    zone = e[(e.type == "Pass") & (e.x <= config.PPDA_OPPONENT_PASS_MAX_X)]
    sp = zone.pass_type.isin(loaders._SET_PIECE_PASS_TYPES)
    wrong = {thr: float(_ppda_variant(ev, pass_max_x=thr).ppda.median()) for thr in (48.0, 60.0, 72.0, 96.0)}
    return {
        "passes_in_zone": len(zone), "set_piece_passes_in_zone": int(sp.sum()),
        "set_piece_share": float(sp.mean()),
        "goal_kicks_in_zone": int((zone.pass_type == "Goal Kick").sum()),
        "throw_ins_in_zone": int((zone.pass_type == "Throw-in").sum()),
        "median_ppda": float(both.ppda.median()), "median_ppda_without_set_pieces": float(both.ppda_ex.median()),
        "rank_rho_team_match": spearman(both.ppda, both.ppda_ex),
        "median_ppda_by_opponent_pass_max_x": wrong,
    }


def _ppda_variant(events, pass_max_x=config.PPDA_OPPONENT_PASS_MAX_X,
                  def_min_x=config.PPDA_DEFENSIVE_ACTION_MIN_X, drop_set_piece_passes=False):
    """``ppda`` with the two zone thresholds and the set-piece choice as arguments.

    With the package defaults it equals ``pressing_sequences.ppda`` (a test checks).
    """
    e = _xy(events)
    teams = e.groupby(["match_id", "team_id"])["team"].first()
    passes = e[(e.type == "Pass") & (e.x <= pass_max_x)]
    if drop_set_piece_passes:
        passes = passes[~passes.pass_type.isin(loaders._SET_PIECE_PASS_TYPES)]
    own = passes.groupby(["match_id", "team_id"]).size().reindex(teams.index, fill_value=0)
    opp = own.groupby(level="match_id").transform("sum") - own
    is_def = e.type.str.lower().isin(config.PPDA_DEFENSIVE_ACTIONS)
    defs = e[is_def & (e.x >= def_min_x)].groupby(["match_id", "team_id"]).size().reindex(teams.index, fill_value=0)
    out = pd.DataFrame({"team": teams, "opp_passes": opp, "def_actions": defs}).reset_index()
    out["ppda"] = out.opp_passes / out.def_actions
    return out


def touch_definitions(ctx: Context) -> dict:
    """Spearman of the team ranking under other definitions of a touch, against the default."""
    e = _xy(ctx.events)
    teams = e.groupby(["match_id", "team_id"])["team"].first()
    pressures = e[(e.type == "Pressure") & (e.x >= config.HALFWAY_X)].groupby(["match_id", "team_id"]).size().reindex(teams.index, fill_value=0)

    def ratio(types):
        t = e[e.type.str.lower().isin(types) & (e.x <= config.HALFWAY_X)]
        own = t.groupby(["match_id", "team_id"]).size().reindex(teams.index, fill_value=0)
        opp = own.groupby(level="match_id").transform("sum") - own
        d = pd.DataFrame({"team": teams, "p": pressures, "t": opp}).groupby("team")[["p", "t"]].sum()
        return d.p / d.t

    default = ratio(config.TOUCH_EVENT_TYPES)
    variants = {
        "passes only": ("pass",),
        "receipts only": ("ball receipt*",),
        "on-ball events without receipts": tuple(t for t in config.TOUCH_EVENT_TYPES if t != "ball receipt*"),
    }
    return {name: spearman(ratio(types), default) for name, types in variants.items()}


# ---------------------------------------------------------------------------
# Part III: press resistance
# ---------------------------------------------------------------------------

def press_resistance_overview(ctx: Context) -> dict:
    """The league-wide figures at the top of Press resistance results, and the Validation rows."""
    a, p, players = ctx.actions, ctx.pressured, ctx.players
    candidates = config.MOVE_TYPES + ("dribble",) + config.LOSS_EVENT_TYPES
    up = a.under_pressure.astype(bool)
    pool = a[a.action_type.isin(candidates)]
    unpressured = pool[~up[pool.index]]
    passes = a[a.action_type == "pass"]
    dist = p.outcome.value_counts(normalize=True)
    no_loss_events = p[~p.action_type.isin(config.LOSS_EVENT_TYPES)]
    base = ctx.baselines
    raw_vs_adj = spearman(players.retention_rate, players.retention_above_expected)
    split = split_half_reliability(ctx)
    return {
        "actions": len(a), "pressured": len(p), "pressured_share": float(len(p) / len(a)),
        "retention_pressured": float(p.retained.mean()),
        "retention_unpressured_same_events": float(unpressured.successful.mean()),
        "retention_passes_and_carries_unpressured": float(
            a[a.action_type.isin(config.MOVE_TYPES) & ~up].successful.mean()),
        "retention_without_loss_events": float(no_loss_events.retained.mean()),
        "pass_completion_pressured": float(passes[up[passes.index]].successful.mean()),
        "pass_completion_unpressured": float(passes[~up[passes.index]].successful.mean()),
        "outcome_shares": {k: float(v) for k, v in dist.items()},
        "loss_events": int(p.action_type.isin(config.LOSS_EVENT_TYPES).sum()),
        "failed_dribbles": int(((p.action_type == "dribble") & ~p.retained).sum()),
        "dribble_success": float(p[p.action_type == "dribble"].retained.mean()),
        "players_ever_pressured": int(p.player_id.nunique()),
        "players_kept": len(players),
        "median_pressured_actions_kept": float(players.n_pressured.median()),
        "retention_vs_expected_min": float(players.retention_above_expected.min()),
        "retention_vs_expected_max": float(players.retention_above_expected.max()),
        "retention_vs_expected_sd": float(players.retention_above_expected.std()),
        "threat_vs_expected_sd": float(players.xt_delta_above_expected.std()),
        "retention_vs_expected_mean": float(players.retention_above_expected.mean()),
        "binomial_se_at_median": float((p.retained.mean() * (1 - p.retained.mean())
                                        / players.n_pressured.median()) ** 0.5),
        "axes_rho": spearman(players.retention_above_expected, players.xt_delta_above_expected),
        "raw_vs_adjusted_retention_rho": raw_vs_adj,
        "zone_baseline_min_n": int(base.n.min()), "zone_baseline_median_n": float(base.n.median()),
        "zones_under_50": int((base.n < 50).sum()),
        "split_half": split,
    }


def press_resistance_standard_errors(ctx: Context) -> dict:
    """Sampling noise on one player's two press-resistance axes, at the median player.

    ``retention``: the binomial standard error of a retention rate over the median
    number of pressured actions (as a proportion, so 0.046 is 4.6 points).
    ``threat``: the spread of xT per retained move around its zone average, divided
    by the square root of the number of retained moves a median player makes. Both
    are what ``viz.plot_press_resistance_quadrant`` draws as its reference cross.
    """
    p, players = ctx.pressured, ctx.players
    median_actions = players.n_pressured.median()
    rate = p.retained.mean()
    retention = (rate * (1 - rate) / median_actions) ** 0.5

    moves = p[p.outcome.isin(["retained_progressed", "retained_flat"])]
    zone = zone_index(moves.start_x, moves.start_y)
    resid = moves.xt_delta - moves.assign(z=zone).groupby("z").xt_delta.transform("mean")
    moves_per_player = len(moves) / len(p) * median_actions
    return {"retention": float(retention), "threat": float(resid.std() / moves_per_player ** 0.5)}


def split_half_reliability(ctx: Context, min_actions: int = 25) -> dict:
    """Rank correlation of each adjusted axis between odd- and even-numbered matches.

    Baselines come from the whole competition; each half needs ``min_actions``
    pressured actions per player.
    """
    order = {m: i for i, m in enumerate(sorted(ctx.pressured.match_id.unique()))}
    odd = ctx.pressured.match_id.map(order) % 2 == 0
    halves = [pr.player_press_resistance(ctx.pressured[f], ctx.baselines, ctx.minutes, min_actions=min_actions)
              .set_index("player_id") for f in (odd, ~odd)]
    j = halves[0].join(halves[1], lsuffix="_1", rsuffix="_2", how="inner")
    return {"players": len(j),
            "retention_rho": spearman(j.retention_above_expected_1, j.retention_above_expected_2),
            "threat_rho": spearman(j.xt_delta_above_expected_1, j.xt_delta_above_expected_2)}


def press_resistance_decisions(ctx: Context, sample_match: int = 3857276) -> dict:
    """The figures behind the Press resistance decisions bullets."""
    p, a, ev = ctx.pressured, ctx.actions, ctx.events
    foul = ev[(ev.type == "Foul Won") & loaders._flag(ev, "under_pressure")]
    kept = p.retained.sum()
    sm = ev[ev.match_id == sample_match]
    sm_pressured = sm[loaders._flag(sm, "under_pressure")]
    loo_rho, loo_max_diff = _leave_one_out(ctx)

    retained_moves = p[p.outcome.isin(["retained_progressed", "retained_flat"])]
    zone = zone_index(p.start_x, p.start_y)
    zone_moves = zone_index(retained_moves.start_x, retained_moves.start_y)

    def eta_squared(values, groups):
        v = pd.Series(np.asarray(values, dtype=float))
        g = v.groupby(np.asarray(groups))
        return float(((g.mean() - v.mean()) ** 2 * g.size()).sum() / ((v - v.mean()) ** 2).sum())

    return {
        "pressured_foul_wons": len(foul), "foul_won_share_of_sample": float(len(foul) / len(p)),
        "retention_if_foul_won_counted": float((kept + len(foul)) / (len(p) + len(foul))),
        "retained_dribbles": int((p.outcome == "retained_dribble").sum()),
        "pressured_shots_excluded": int(((a.action_type == "shot") & a.under_pressure.astype(bool)).sum()),
        "extra_time_share_of_pressured": float((~p.period.isin(config.REGULATION_PERIODS)).mean()),
        "position_labels": int(ctx.players.position.nunique()),
        "zone_variance_explained_retention": eta_squared(p.retained.astype(float), zone),
        "zone_variance_explained_xt_delta": eta_squared(retained_moves.xt_delta, zone_moves),
        "retained_moves_exactly_zero_delta": float((retained_moves.xt_delta == 0).mean()),
        "retained_moves_negative_delta": float((retained_moves.xt_delta < 0).mean()),
        "leave_one_out_rho": loo_rho, "leave_one_out_max_abs_diff": loo_max_diff,
        "sample_match_pressured_carries": int((sm_pressured.type == "Carry").sum()),
        "sample_match_pressured_passes": int((sm_pressured.type == "Pass").sum()),
        "sample_match_pressured_ball_receipts": int((sm_pressured.type == "Ball Receipt*").sum()),
        "pressured_move_share_of_passes": float(a[a.action_type == "pass"].under_pressure.astype(bool).mean()),
        "pressured_move_share_of_carries": float(a[a.action_type == "carry"].under_pressure.astype(bool).mean()),
    }


def _leave_one_out(ctx: Context) -> tuple[float, float]:
    """Retention above expectation with each player's own actions left out of the baseline.

    Returns (Spearman against the standard figure, largest absolute difference), over
    the players kept in the table.
    """
    p = ctx.pressured.assign(zone=zone_index(ctx.pressured.start_x, ctx.pressured.start_y))
    p["pid"] = p.player_id.astype(int)
    zone_sum, zone_n = p.groupby("zone").retained.sum(), p.groupby("zone").size()
    own = p.groupby(["pid", "zone"]).retained.agg(["sum", "size"]).reset_index()
    own["expected_loo"] = (own.zone.map(zone_sum) - own["sum"]) / (own.zone.map(zone_n) - own["size"])
    p = p.merge(own[["pid", "zone", "expected_loo"]], on=["pid", "zone"])
    loo = p.groupby("pid").retained.mean() - p.groupby("pid").expected_loo.mean()
    std = ctx.players.set_index("player_id").retention_above_expected
    loo = loo.reindex(std.index)
    return spearman(loo, std), float((loo - std).abs().max())


def table_2(ctx: Context, n_top: int = 5, n_bottom: int = 3) -> pd.DataFrame:
    """Table 2: the best and worst players by the 50/50 score, with their two axes."""
    t = ctx.players.sort_values("score", ascending=False)
    t = pd.concat([t.head(n_top), t.tail(n_bottom).sort_values("score", ascending=False)])
    out = t[["player", "position_group", "n_pressured", "retention_rate",
             "retention_above_expected", "xt_delta_above_expected", "score"]].copy()
    out["player"] = out.player.map(short_name)
    return out.reset_index(drop=True)


def score_weights(ctx: Context) -> dict:
    """How much the top 10 changes with the weights, and what each end of the score tracks."""
    t = ctx.players
    even = pr.press_resistance_score(t)
    top = set(even.nlargest(10).index)
    out = {"even_vs_retention_rho": spearman(even, t.retention_above_expected),
           "even_vs_threat_rho": spearman(even, t.xt_delta_above_expected),
           "unstandardised_blend_vs_retention_rho": spearman(
               0.5 * t.retention_above_expected + 0.5 * t.xt_delta_above_expected, t.retention_above_expected),
           "retention_sd_over_threat_sd": float(t.retention_above_expected.std() / t.xt_delta_above_expected.std())}
    for wr, wt in ((0.7, 0.3), (0.3, 0.7)):
        s = pr.press_resistance_score(t, wr, wt)
        out[f"top10_overlap_{int(wr * 100)}_{int(wt * 100)}"] = len(top & set(s.nlargest(10).index))
        out[f"rho_vs_even_{int(wr * 100)}_{int(wt * 100)}"] = spearman(s, even)
    best_ret = t.sort_values("retention_above_expected", ascending=False).head(3)
    out["best_by_retention"] = [short_name(p) for p in best_ret.player]
    out["best_by_retention_in_top10"] = [bool(i in top) for i in best_ret.index]
    return out


def table_3(ctx: Context) -> pd.DataFrame:
    """Table 3: outfield position groups. Mean over players, as in the README."""
    t = ctx.players[ctx.players.position_group != "GK"]
    return (t.groupby("position_group")
             .agg(players=("player_id", "size"), retention=("retention_rate", "mean"),
                  retention_vs_expected=("retention_above_expected", "mean"))
             .reindex(["DF", "MF", "FW"]))


def position_group_tests(ctx: Context) -> dict:
    """Why forwards come out lowest: the differences, and the dribble and loss shares."""
    t = ctx.players
    p = ctx.pressured.assign(_pid=ctx.pressured.player_id.astype(int))
    p = p.merge(t[["player_id", "position_group"]], left_on="_pid", right_on="player_id", suffixes=("", "_t"))
    p = p[p.position_group != "GK"]
    p["dribble"] = p.action_type == "dribble"
    p["loss_event"] = p.action_type.isin(config.LOSS_EVENT_TYPES)
    g = p.groupby("position_group")
    a = "retention_above_expected"
    diffs = {}
    for name, hi, lo in (("MF_minus_FW", "MF", "FW"), ("DF_minus_FW", "DF", "FW"), ("MF_minus_DF", "MF", "DF")):
        x, y = t[t.position_group == hi][a], t[t.position_group == lo][a]
        se = (x.var() / len(x) + y.var() / len(y)) ** 0.5
        diffs[name] = {"points": float((x.mean() - y.mean()) * 100), "standard_errors": float((x.mean() - y.mean()) / se)}
    nd = p[~p.dribble]
    passes = p[p.action_type == "pass"]
    return {
        "differences": diffs,
        "dribble_share": g.dribble.mean().round(4).to_dict(),
        "loss_event_share": g.loss_event.mean().round(4).to_dict(),
        "retention_without_dribbles": nd.groupby("position_group").retained.mean().round(4).to_dict(),
        "pass_retention": passes.groupby("position_group").retained.mean().round(4).to_dict(),
    }


def table_4(ctx: Context, min_actions: int = 25) -> pd.DataFrame:
    """Table 4: goalkeepers with at least ``min_actions`` pressured actions (descriptive only)."""
    t = pr.player_press_resistance(ctx.pressured, ctx.baselines, ctx.minutes, min_actions=min_actions)
    gk = t[t.position_group == "GK"].sort_values("n_pressured", ascending=False)
    out = gk[["player", "n_pressured", "retention_rate", "retention_above_expected", "xt_delta_above_expected"]].copy()
    out["player"] = out.player.map(short_name)
    return out.reset_index(drop=True)


def goalkeeper_note(ctx: Context, min_actions: int = 25) -> dict:
    """The figures quoted around Table 4."""
    gk = table_4(ctx, min_actions)
    p = ctx.pressured
    keepers = p[p.position == "Goalkeeper"]
    outfield = p[p.position != "Goalkeeper"]
    zone = zone_index(p.start_x, p.start_y)
    keeper_zones = set(zone[(p.position == "Goalkeeper").to_numpy()])
    in_zones = p[np.isin(zone, list(keeper_zones))]
    near_goal = p[p.start_x < 18]
    n_med = gk.n_pressured.median()
    se = (keepers.retained.mean() * (1 - keepers.retained.mean()) / n_med) ** 0.5
    wide = pr.player_press_resistance(ctx.pressured, ctx.baselines, ctx.minutes, min_actions=min_actions)
    return {
        "keepers_at_threshold": len(gk), "keepers_at_50": int((ctx.players.position_group == "GK").sum()),
        "keepers_pressured_at_all": int(keepers.player_id.nunique()),
        "pressured_actions": len(keepers), "pooled_retention": float(keepers.retained.mean()),
        "carry_share": float((keepers.action_type == "carry").mean()),
        "median_actions": float(n_med), "standard_error": float(se),
        "sd_vs_expected": float(gk.retention_above_expected.std()),
        "sd_over_se": float(gk.retention_above_expected.std() / se),
        "mean_vs_expected": float(gk.retention_above_expected.mean()),
        "start_within_18_yards": float((keepers.start_x < 18).mean()),
        "keeper_share_of_actions_within_18_yards": float((near_goal.position == "Goalkeeper").mean()),
        "keeper_retention_in_keeper_zones": float(in_zones[in_zones.position == "Goalkeeper"].retained.mean()),
        "outfield_retention_in_keeper_zones": float(in_zones[in_zones.position != "Goalkeeper"].retained.mean()),
        "threat_vs_expected_min": float(gk.xt_delta_above_expected.min()),
        "threat_vs_expected_max": float(gk.xt_delta_above_expected.max()),
        "outfield_threat_sd": float(wide[wide.position_group != "GK"].xt_delta_above_expected.std()),
        "outfield_pooled_retention": float(outfield.retained.mean()),
    }


# ---------------------------------------------------------------------------
# The Validation table, row by row
# ---------------------------------------------------------------------------

def validation(ctx: Context) -> dict:
    """The values in Table 5 (Validation), gathered from the sections above."""
    et = expected_threat_checks(ctx)
    ov = pressing_overview(ctx)
    sens = sensitivity(ctx)
    pro = press_resistance_overview(ctx)
    xg = team_xt_vs_goals(ctx)
    return {"expected_threat": et, "pressing": ov, "sensitivity": sens,
            "press_resistance": pro, "team_xt_vs_goals": xg}


SECTIONS = {
    "expected_threat": expected_threat_checks,
    "table_1": table_1, "pressing": pressing_overview, "sensitivity": sensitivity,
    "pressing_decisions": pressing_decisions, "ppda": ppda_alternatives, "touch": touch_definitions,
    "press_resistance": press_resistance_overview, "pr_decisions": press_resistance_decisions,
    "table_2": table_2, "weights": score_weights,
    "table_3": table_3, "groups": position_group_tests, "table_4": table_4, "gk": goalkeeper_note,
    "xt_vs_goals": team_xt_vs_goals,
    "figures": figure_facts,
}


def _plain(x):
    if isinstance(x, pd.DataFrame):
        return x.round(4).to_dict("records")
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Recompute the numbers the README quotes.")
    parser.add_argument("sections", nargs="*", help="section names (default: all); see --list")
    parser.add_argument("--list", action="store_true", help="print the section names and exit")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(SECTIONS))
        return
    chosen = args.sections or list(SECTIONS)
    unknown = [s for s in chosen if s not in SECTIONS]
    if unknown:
        sys.exit(f"unknown section(s): {unknown}; see --list")
    ctx = build_context()
    for name in chosen:
        print(f"\n== {name} ==")
        print(json.dumps(_plain(SECTIONS[name](ctx)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
