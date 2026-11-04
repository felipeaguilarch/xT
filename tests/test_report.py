"""The README's numbers, pinned to the code that produces them.

Each test recomputes one table or block of figures through ``xtpress.report`` and
asserts the value the README prints, to the precision the README prints it. If the
pipeline or the data changes underneath the README, the matching test fails and
names the figure.

Tables 1 to 4 are also parsed out of the README itself and compared cell by cell
with what the code produces, so a hand edit to those tables is caught too. The
figures quoted in the running text are pinned as literals in the tests below; a
hand edit to a sentence is not caught.

Everything here needs the full WC 2022 dataset (``python -m xtpress.loaders
--competition wc_2022``); the ``ctx`` fixture skips itself without it. The
expensive sections (the sensitivity runs, the cross-fit) are computed once.
"""

import numpy as np
import pandas as pd
import pytest

from xtpress import config, loaders, report
from xtpress import pressing_sequences as ps

approx = pytest.approx


@pytest.fixture(scope="module")
def et(ctx):
    return report.expected_threat_checks(ctx)


@pytest.fixture(scope="module")
def pressing(ctx):
    return report.pressing_overview(ctx)


@pytest.fixture(scope="module")
def sens(ctx):
    return report.sensitivity(ctx)


@pytest.fixture(scope="module")
def decisions(ctx):
    return report.pressing_decisions(ctx)


@pytest.fixture(scope="module")
def resistance(ctx):
    return report.press_resistance_overview(ctx)


@pytest.fixture(scope="module")
def xt_goals(ctx):
    return report.team_xt_vs_goals(ctx)


# -- The README's own tables ----------------------------------------------------

def readme_table(label):
    """Rows (lists of cells) of the table captioned ``**label**`` in the README, header and rule dropped."""
    text = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.split(f"**{label}**", 1)[1].splitlines()
    rows = []
    for line in lines:
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]


def minus(text):
    return text.replace("-", "\N{MINUS SIGN}")


def pct(x):
    return f"{x * 100:.1f}%"


def pts(x):
    return minus(f"{x * 100:+.1f} pts")


def test_readme_tables_match_the_code(ctx):
    t1 = [[r.team, str(r.matches), f"{r.ppda:.1f}", pct(r.regain_rate), f"{r.xt_per_100_pressures:.3f}", str(r.shots_from_presses)]
          for r in report.table_1(ctx).itertuples()]
    t2 = [[r.player, r.position_group, str(r.n_pressured), pct(r.retention_rate), pts(r.retention_above_expected),
           minus(f"{r.xt_delta_above_expected:+.4f}"), minus(f"{r.score:.2f}")] for r in report.table_2(ctx).itertuples()]
    t3 = [[g, str(int(r.players)), pct(r.retention), pts(r.retention_vs_expected)] for g, r in report.table_3(ctx).iterrows()]
    t4 = [[r.player, str(r.n_pressured), pct(r.retention_rate), pts(r.retention_above_expected)] for r in report.table_4(ctx).itertuples()]
    for label, computed in (("Table 1", t1), ("Table 2", t2), ("Table 3", t3), ("Table 4", t4)):
        assert readme_table(label) == computed, f"{label} in the README differs from report.{label.lower().replace(' ', '_')}"


# -- Table 1 -----------------------------------------------------------------

def test_table_1_pressing_results(ctx):
    t = report.table_1(ctx).set_index("team")
    expected = {  # team: (matches, PPDA, regain rate, xT per 100 pressures, shots)
        "Canada": (3, 13.6, 0.164, 0.325, 10), "Brazil": (5, 10.4, 0.212, 0.324, 13),
        "Germany": (3, 10.2, 0.200, 0.276, 11), "Mexico": (3, 7.4, 0.159, 0.073, 2),
        "South Korea": (4, 11.0, 0.142, 0.069, 2), "Costa Rica": (3, 19.5, 0.144, 0.045, 0),
    }
    assert list(t.index) == list(expected)
    for team, (matches, p, regain, payoff, shots) in expected.items():
        r = t.loc[team]
        assert r.matches == matches and r.shots_from_presses == shots
        assert r.ppda == approx(p, abs=0.05)
        assert r.regain_rate == approx(regain, abs=0.0005)
        assert r.xt_per_100_pressures == approx(payoff, abs=0.0005)


# -- Table 5, the Expected Threat rows and the model decisions ------------------

def test_surface_validation_rows(et):
    assert et["iterative_vs_closed_max_abs"] < 1e-9 and et["iterations"] == 173
    assert et["pearson_r_vs_published"] == approx(0.996, abs=0.0005)
    assert et["mae_vs_published"] == approx(0.0077, abs=0.00005)
    assert et["baseline_pearson_r"] == approx(0.860, abs=0.0005)
    assert et["baseline_goal_end_over_own_end"] == approx(1.4, abs=0.05)
    assert et["published_goal_end_over_own_end"] == approx(12.6, abs=0.05)
    assert et["transition_rows_max_deviation"] < 1e-9
    assert et["pass_completion"] == approx(0.822, abs=0.0005)
    assert (et["n_actions"], et["n_matches"]) == (128_711, 64)


def test_expected_threat_decisions(et):
    assert et["moves_arriving"] == approx(0.90, abs=0.005)
    assert et["completion_zone_min"] == approx(0.55, abs=0.005) and et["completion_zone_max"] == approx(0.95, abs=0.005)
    assert et["set_piece_share_of_move_attempts"] == approx(0.051, abs=0.0005)
    assert et["penalties_excluded"] == 23 and et["zones_without_shots"] == 68
    assert et["asymmetry_top_bottom"] == approx(0.040, abs=0.0005)
    assert et["published_grid_asymmetry"] == approx(0.0, abs=1e-12)


# -- The finding, Pressing results, Why it matters --------------------------------

def test_headline_pressing_figures(pressing):
    assert pressing["ppda_payoff"]["rho"] == approx(-0.09, abs=0.005)
    assert pressing["ppda_payoff"]["lo"] == approx(-0.44, abs=0.005)
    assert pressing["ppda_payoff"]["hi"] == approx(0.28, abs=0.005)
    assert (pressing["ppda_payoff"]["reps"], pressing["ppda_payoff"]["seed"]) == (4000, 0)


def test_pressing_results_prose(pressing):
    assert pressing["median_ppda_team"] == approx(11.3, abs=0.05)
    assert pressing["median_payoff"] == approx(0.159, abs=0.0005)
    assert (pressing["shots_per_team_min"], pressing["shots_per_team_max"]) == (0, 15)
    assert pressing["median_seconds_to_regain"] == approx(2.5, abs=0.05)
    assert pressing["share_regain_within_2s"] == approx(0.41, abs=0.005)
    assert pressing["p90_seconds_to_regain"] == approx(6.8, abs=0.05)
    assert pressing["median_regain_x"] == approx(37, abs=0.5)
    assert pressing["regain_own_third"] == approx(0.54, abs=0.005)
    assert pressing["regain_final_third"] == approx(0.05, abs=0.005)


def test_why_it_matters(pressing):
    assert pressing["zone_xt_high_regain"] == approx(0.0142, abs=0.00005)
    assert pressing["zone_xt_other_wins"] == approx(0.0066, abs=0.00005)
    assert pressing["high_regains_reaching_shot"] == approx(0.159, abs=0.0005)
    assert pressing["won_reaching_shot"] == approx(0.104, abs=0.0005)


def test_pressing_validation_rows(pressing):
    assert pressing["ppda_team_min"] == approx(7.4, abs=0.05) and pressing["ppda_team_max"] == approx(20.1, abs=0.05)
    assert pressing["teams_ppda_inside_6_15"] == 30 and pressing["teams"] == 32
    assert pressing["ppda_team_match_inside_6_15"] == approx(0.77, abs=0.005)
    assert pressing["regain_rate"] == approx(0.163, abs=0.0005)
    assert (pressing["n_sequences"], pressing["won"], pressing["won_high"]) == (12_424, 2_024, 435)
    assert pressing["high_rate"] == approx(0.035, abs=0.0005)
    assert pressing["high_share_of_wins"] == approx(0.215, abs=0.0005)
    assert pressing["xt_per_won_sequence"] == approx(0.0139, abs=0.00005)
    assert pressing["touch_ratio_median"] == approx(0.075, abs=0.0005)
    assert pressing["touch_ratio_min"] == approx(0.027, abs=0.0005) and pressing["touch_ratio_max"] == approx(0.159, abs=0.0005)
    assert pressing["pressures_in_opponent_half"] == approx(0.45, abs=0.005)


def test_limitations_and_checks_figures(pressing):
    assert pressing["teams_with_4_plus_matches"] == 16
    assert pressing["ppda_payoff_rho_4_plus_matches"] == approx(-0.30, abs=0.005)
    assert pressing["single_pressure_share"] == approx(0.76, abs=0.005)
    assert pressing["median_sequence_duration_s"] == 0
    assert pressing["ppda_vs_touch_team_rho"] == approx(-0.63, abs=0.005)
    assert pressing["sequences_per_team_match_median"] == 94
    assert (pressing["sequences_per_team_match_min"], pressing["sequences_per_team_match_max"]) == (61, 163)
    assert pressing["pressures_per_sequence"] == approx(1.33, abs=0.005)


# -- Pressing decisions ------------------------------------------------------------

def test_sensitivity_to_the_regain_window(sens):
    w = sens["windows"]
    assert w[2]["regain_rate"] == approx(0.067, abs=0.0005)
    assert w[6]["regain_rate"] == approx(0.137, abs=0.0006)
    assert w[8]["regain_rate"] == approx(0.163, abs=0.0005)
    assert w[12]["regain_rate"] == approx(0.206, abs=0.0005)
    assert w[15]["regain_rate"] == approx(0.233, abs=0.0005)
    assert w[8]["regain_rate_excluding_own_possession"] == approx(0.191, abs=0.0005)
    rho = [v["ppda_payoff_rho"] for v in w.values()]
    assert -0.24 < min(rho) and max(rho) < -0.05          # "between -0.05 and -0.23 at every window"
    assert w[2]["payoff_rank_rho_vs_default"] == approx(0.59, abs=0.005)
    assert min(w[k]["payoff_rank_rho_vs_default"] for k in (4, 6, 10, 12, 15)) > 0.91


def test_sensitivity_to_the_gap_and_the_high_line(sens):
    g = sens["gaps"]
    assert (g[2]["n_sequences"], g[12]["n_sequences"]) == (14_588, 10_394)
    assert g[2]["regain_rate"] == approx(0.157, abs=0.0005) and g[12]["regain_rate"] == approx(0.189, abs=0.0005)
    assert all(v["payoff_max_abs_diff_vs_default"] < 1e-12 for v in g.values())    # the payoff does not depend on the gap
    h = sens["high_regain_line"]
    assert [round(h[x]["share_of_sequences"], 3) for x in (50, 60, 70, 80)] == [0.056, 0.035, 0.019, 0.009]
    assert sens["own_possession_sequences"] == 1_840 and sens["own_possession_share"] == approx(0.15, abs=0.005)


def test_linking_counterpress_and_credit_once(decisions):
    d = decisions
    assert d["linked_by_related_events"] == approx(0.905, abs=0.0005)
    assert d["linked_by_window"] == approx(0.050, abs=0.0005) and d["unlinked"] == approx(0.044, abs=0.0005)
    near, fwd = d["fallback_vs_true_link"]["nearest"], d["fallback_vs_true_link"]["forward"]
    assert near["matched_an_action"] == approx(0.96, abs=0.005)
    assert near["same_event_of_all_linked"] == approx(0.58, abs=0.005)      # README: "agrees ... only 58%"
    assert fwd["same_event_of_all_linked"] == approx(0.37, abs=0.005)       # README: "forward only agrees 37%"
    assert near["same_event_of_matched"] == approx(0.599, abs=0.0005)       # the docstring's figure, a different denominator
    assert near["same_player_of_matched"] == approx(0.993, abs=0.0005)
    assert d["counterpress_share_of_pressures"] == approx(0.201, abs=0.0005)
    assert d["counterpress_share_of_sequences"] == approx(0.215, abs=0.0005)
    assert d["counterpress_regain_rate_vs_opponent_possession"] == approx(0.137, abs=0.0005)
    assert d["other_regain_rate_vs_opponent_possession"] == approx(0.202, abs=0.0005)
    assert d["counterpress_high_rate"] == approx(0.057, abs=0.0005) and d["other_high_rate"] == approx(0.038, abs=0.0005)
    assert d["counterpress_xt_per_win"] == approx(0.0120, abs=0.00005) and d["other_xt_per_win"] == approx(0.0142, abs=0.00005)
    assert d["won_sequences_in_shared_possessions"] == 122 and d["won_sequences"] == 2_024
    assert d["share_in_shared_possessions"] == approx(0.06, abs=0.001)
    assert d["xt_inflation_if_double_counted"] == approx(0.049, abs=0.0005) and d["extra_shots_if_double_counted"] == 9


def test_ppda_definition_choices(ctx):
    a = report.ppda_alternatives(ctx)
    assert (a["set_piece_passes_in_zone"], a["passes_in_zone"]) == (4_130, 47_484)
    assert (a["goal_kicks_in_zone"], a["throw_ins_in_zone"]) == (1_002, 1_437)
    assert a["set_piece_share"] == approx(0.087, abs=0.0005)
    assert a["median_ppda"] == approx(11.7, abs=0.05) and a["median_ppda_without_set_pieces"] == approx(10.5, abs=0.05)
    assert a["rank_rho_team_match"] == approx(0.998, abs=0.0005)
    assert a["median_ppda_by_opponent_pass_max_x"][48.0] == approx(6.7, abs=0.05)   # the wrong threshold in "What the checks turned up"
    assert pressing_ppda_equals_package(ctx)


def pressing_ppda_equals_package(ctx):
    mine = report._ppda_variant(ctx.events).set_index(["match_id", "team_id"]).ppda
    theirs = ps.ppda(ctx.events).set_index(["match_id", "team_id"]).ppda
    return bool(np.allclose(mine.sort_index(), theirs.sort_index(), equal_nan=True))


def test_touch_definitions_agree(ctx):
    assert min(report.touch_definitions(ctx).values()) >= 0.988


# -- Press resistance --------------------------------------------------------------------

def test_press_resistance_overview(resistance):
    r = resistance
    assert (r["actions"], r["pressured"]) == (128_711, 25_270) and r["pressured_share"] == approx(0.196, abs=0.0005)
    assert r["retention_pressured"] == approx(0.801, abs=0.0005)
    assert r["retention_unpressured_same_events"] == approx(0.890, abs=0.0005)
    assert r["retention_passes_and_carries_unpressured"] == approx(0.902, abs=0.0005)
    assert r["retention_without_loss_events"] == approx(0.865, abs=0.0005)
    assert r["pass_completion_pressured"] == approx(0.708, abs=0.0005)
    assert r["pass_completion_unpressured"] == approx(0.837, abs=0.0005)
    o = r["outcome_shares"]
    assert (round(o["retained_progressed"], 3), round(o["retained_flat"], 3),
            round(o["retained_dribble"], 3), round(o["lost"], 3)) == (0.272, 0.492, 0.037, 0.199)
    assert (r["loss_events"], r["failed_dribbles"]) == (1_869, 856)
    assert r["dribble_success"] == approx(0.52, abs=0.005)
    assert (r["players_ever_pressured"], r["players_kept"]) == (663, 179)


def test_press_resistance_noise_figures(resistance):
    r = resistance
    assert r["median_pressured_actions_kept"] == 74
    assert r["retention_vs_expected_min"] == approx(-0.158, abs=0.0005) and r["retention_vs_expected_max"] == approx(0.120, abs=0.0005)
    assert r["retention_vs_expected_sd"] == approx(0.054, abs=0.0005)
    assert r["threat_vs_expected_sd"] == approx(0.0015, abs=0.00005)
    assert round(r["binomial_se_at_median"] * 100, 1) == 4.6               # 4.6 points
    assert r["axes_rho"] == approx(-0.18, abs=0.005)
    assert r["raw_vs_adjusted_retention_rho"] == approx(0.935, abs=0.0005)
    assert (r["zone_baseline_min_n"], r["zone_baseline_median_n"], r["zones_under_50"]) == (28, 251, 2)
    assert r["split_half"]["players"] == 65
    assert r["split_half"]["retention_rho"] == approx(0.26, abs=0.005) and r["split_half"]["threat_rho"] == approx(0.32, abs=0.005)


def test_table_2_leaderboard(ctx):
    t = report.table_2(ctx).set_index("player")
    expected = {  # group, n, retention, vs expected, xT vs expected, score
        "Hirving Lozano": ("FW", 72, 0.792, 0.026, 0.0051, 1.82), "Aziz Behich": ("DF", 78, 0.795, 0.024, 0.0049, 1.72),
        "Rafael Leão": ("FW", 50, 0.800, 0.025, 0.0047, 1.66), "Phil Foden": ("FW", 73, 0.904, 0.108, 0.0017, 1.46),
        "Lionel Messi": ("FW", 352, 0.855, 0.063, 0.0020, 1.16), "Mathew Leckie": ("MF", 111, 0.667, -0.116, -0.0005, -1.33),
        "Richarlison": ("FW", 71, 0.732, -0.034, -0.0031, -1.40), "Gue-Sung Cho": ("FW", 62, 0.661, -0.106, -0.0049, -2.65),
    }
    assert list(t.index) == list(expected)
    for name, (group, n, ret, vs, thr, score) in expected.items():
        r = t.loc[name]
        assert (r.position_group, r.n_pressured) == (group, n)
        assert r.retention_rate == approx(ret, abs=0.0005) and r.retention_above_expected == approx(vs, abs=0.0005)
        assert r.xt_delta_above_expected == approx(thr, abs=0.00005) and r.score == approx(score, abs=0.005)


def test_score_weights(ctx):
    w = report.score_weights(ctx)
    assert w["even_vs_retention_rho"] == approx(0.65, abs=0.005) and w["even_vs_threat_rho"] == approx(0.55, abs=0.005)
    assert w["unstandardised_blend_vs_retention_rho"] == approx(0.999, abs=0.0005)
    assert (w["top10_overlap_70_30"], w["top10_overlap_30_70"]) == (5, 7)
    assert w["best_by_retention"] == ["Steven Bergwijn", "Salis Abdul Samed", "Unai Simón"]
    assert not any(w["best_by_retention_in_top10"])


def test_table_3_position_groups(ctx):
    t = report.table_3(ctx)
    assert list(t.index) == ["DF", "MF", "FW"] and list(t.players) == [40, 71, 66]
    assert list(t.retention.round(3)) == [0.817, 0.830, 0.776]
    assert list(t.retention_vs_expected.round(3)) == [0.012, 0.021, -0.009]


def test_position_group_text(ctx):
    g = report.position_group_tests(ctx)
    d = g["differences"]
    assert [(round(d[k]["points"], 1), round(d[k]["standard_errors"], 1)) for k in ("MF_minus_FW", "DF_minus_FW", "MF_minus_DF")] == [
        (2.9, 3.3), (2.1, 1.8), (0.9, 0.9)]
    assert g["dribble_share"] == approx({"DF": 0.056, "FW": 0.106, "MF": 0.069}, abs=0.0005)
    assert g["loss_event_share"]["FW"] == approx(0.103, abs=0.001)
    assert g["loss_event_share"]["DF"] == approx(0.058, abs=0.0005) and g["loss_event_share"]["MF"] == approx(0.063, abs=0.0005)
    assert g["retention_without_dribbles"]["FW"] == approx(0.810, abs=0.0005) and g["retention_without_dribbles"]["MF"] == approx(0.857, abs=0.0005)
    assert g["pass_retention"]["FW"] == approx(0.706, abs=0.0005) and g["pass_retention"]["MF"] == approx(0.775, abs=0.0005)


def test_table_4_goalkeepers(ctx):
    t = report.table_4(ctx).set_index("player")
    expected = {"Yassine Bounou": (59, 0.847, 0.010), "Unai Simón": (55, 0.964, 0.117), "Emiliano Martínez": (49, 0.776, -0.055),
                "Mathew Ryan": (47, 0.830, -0.002), "Diogo Costa": (39, 0.744, -0.102), "Dominik Livaković": (38, 0.763, -0.081),
                "Jordan Pickford": (29, 0.759, -0.065), "Yann Sommer": (29, 0.931, 0.083), "Andries Noppert": (27, 0.667, -0.173),
                "Hugo Lloris": (26, 0.654, -0.183)}
    assert set(t.index) == set(expected)
    for name, (n, ret, vs) in expected.items():
        assert t.loc[name].n_pressured == n
        assert t.loc[name].retention_rate == approx(ret, abs=0.0005) and t.loc[name].retention_above_expected == approx(vs, abs=0.0005)


def test_goalkeeper_note(ctx):
    n = report.goalkeeper_note(ctx)
    assert (n["keepers_at_threshold"], n["keepers_at_50"], n["keepers_pressured_at_all"]) == (10, 2, 40)
    assert n["pressured_actions"] == 760 and n["pooled_retention"] == approx(0.813, abs=0.0005)
    assert n["carry_share"] == approx(0.56, abs=0.005)
    assert n["median_actions"] == 38.5 and n["standard_error"] == approx(0.063, abs=0.0005)
    assert n["sd_vs_expected"] == approx(0.099, abs=0.0005) and n["sd_over_se"] == approx(1.6, abs=0.05)
    assert n["mean_vs_expected"] == approx(-0.045, abs=0.0005)
    assert n["start_within_18_yards"] == approx(0.93, abs=0.005)
    assert n["keeper_share_of_actions_within_18_yards"] == approx(0.41, abs=0.005)
    assert n["keeper_retention_in_keeper_zones"] == approx(0.813, abs=0.0005)
    assert n["outfield_retention_in_keeper_zones"] == approx(0.843, abs=0.0005)
    assert n["threat_vs_expected_min"] == approx(-0.0002, abs=0.00005) and n["threat_vs_expected_max"] == approx(0.0006, abs=0.00005)
    assert n["outfield_threat_sd"] == approx(0.0017, abs=0.00005)


def test_press_resistance_decisions(ctx):
    d = report.press_resistance_decisions(ctx)
    assert d["pressured_foul_wons"] == 1_674 and round(d["foul_won_share_of_sample"], 3) == 0.066
    assert round(d["retention_if_foul_won_counted"], 3) == 0.813
    assert (d["retained_dribbles"], d["pressured_shots_excluded"], d["position_labels"]) == (937, 238, 23)
    assert round(d["extra_time_share_of_pressured"], 3) == 0.022
    assert round(d["zone_variance_explained_retention"], 3) == 0.025 and round(d["zone_variance_explained_xt_delta"], 3) == 0.086
    assert round(d["retained_moves_exactly_zero_delta"], 2) == 0.36 and round(d["retained_moves_negative_delta"], 2) == 0.28
    assert round(d["leave_one_out_rho"], 4) == 0.9999 and round(d["leave_one_out_max_abs_diff"], 3) == 0.011
    assert (d["sample_match_pressured_carries"], d["sample_match_pressured_passes"],
            d["sample_match_pressured_ball_receipts"]) == (236, 100, 47)
    assert round(d["pressured_move_share_of_passes"], 3) == 0.115 and round(d["pressured_move_share_of_carries"], 3) == 0.255


# -- Held-out check against goals ---------------------------------------------------------

def test_team_xt_against_goals(xt_goals):
    x = xt_goals
    assert (x["n_teams"], x["n_team_matches"], x["goals"], x["goalless_team_matches"]) == (32, 128, 165, 44)
    t, m = x["team_xt_vs_goals"], x["team_match_xt_vs_goals"]
    assert (round(t["rho"], 2), round(t["lo"], 2), round(t["hi"], 2)) == (0.30, -0.09, 0.62)
    assert (round(m["rho"], 2), round(m["lo"], 2), round(m["hi"], 2)) == (0.25, 0.07, 0.42)
    assert x["in_sample_team_match_rho"] == approx(0.25, abs=0.005)
    assert round(x["team_xt_vs_xg"]["rho"], 2) == 0.54 and round(x["team_match_xt_vs_xg"]["rho"], 2) == 0.56
    assert (round(x["team_match_xt_vs_xg"]["lo"], 2), round(x["team_match_xt_vs_xg"]["hi"], 2)) == (0.41, 0.68)
    assert (round(x["team_xt_vs_xg"]["lo"], 2), round(x["team_xt_vs_xg"]["hi"], 2)) == (0.20, 0.78)
    assert x["goals_vs_xg_team_match_rho"] == approx(0.54, abs=0.005)


# -- The module's own helpers -------------------------------------------------------------

def test_window_link_matches_the_package(ctx):
    sample = ctx.pressures.head(3000)
    mine = report._window_link(sample, ctx.events, "nearest").set_index("id").action_id
    theirs = ps._link_by_time_window(sample, ctx.events).set_index("id").action_id
    assert mine.sort_index().equals(theirs.sort_index())


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    a, b = report.bootstrap_spearman(x, y, reps=300), report.bootstrap_spearman(x, y, reps=300)
    assert a == b and a["lo"] < a["rho"] < a["hi"]
    assert report.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == approx(1.0)
    assert report.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == approx(-1.0)


def test_short_names():
    assert report.short_name("Lionel Andrés Messi Cuccittini") == "Lionel Messi"
    assert report.short_name("Someone Unmapped") == "Someone Unmapped"
