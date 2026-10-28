# xT & Pressing

Expected Threat and pressing analysis built from scratch on StatsBomb open data.

> **Status:** work in progress. Part I (Expected Threat, action rating, minutes played)
> and Part II (pressing sequences, PPDA, the xT value of what a press wins) are complete
> and validated; Part III (player-level press resistance) is in progress.

## Contents

1. [The finding](#the-finding)
2. [Why it matters](#why-it-matters)
3. [How it is measured](#how-it-is-measured)
   - [Expected Threat](#expected-threat)
   - [Pressing](#pressing)
   - [Press resistance](#press-resistance)
4. [Pressing results](#pressing-results)
5. [Supporting evidence](#supporting-evidence)
6. [Validation](#validation)
7. [Model decisions](#model-decisions)
   - [Expected Threat decisions](#expected-threat-decisions)
   - [Pressing decisions](#pressing-decisions)
8. [Limitations](#limitations)
9. [What the checks turned up](#what-the-checks-turned-up)
10. [Reproducing it](#reproducing-it)
11. [Data](#data)
12. [Repo](#repo)

---

## The finding

How hard a team presses says little about what the pressing earns. Across the 32 teams
at WC 2022, PPDA and the expected threat won per 100 pressures show no detectable
relationship (Spearman ρ = −0.09, 95% bootstrap interval −0.44 to 0.30): Mexico pressed
hardest (PPDA 7.4) and earned 0.073, while Brazil pressed at a middling 10.4 and earned
0.324.

*Provisional: the headline will be rewritten once press resistance (Part III) is in.
The figure for it, `figures/press_resistance.png`, is not generated yet.*

## Why it matters

PPDA quantifies how much a team presses the opponents, more specifically how many passes are allowed before engaging defensively. The measure here prices what the press wins, in the same units as every pass and carry in the model. Where the ball is won matters: the zone where it is won is worth 0.0142 xT for a high regain against 0.0066 for the rest, and 15.9% of high regains reach a shot against 10.4% of all wins. So the useful question for a coach or an analyst is not how much a team presses, but what it does once the ball is won, and where.

## How it is measured

### Expected Threat

The pitch is split into a 12×8 grid. Each zone's value is the probability that a possession starting there ends in a goal, found by treating play as a Markov chain: from each zone the ball is shot, moved to another zone, or lost, using the frequencies observed in WC 2022. The xT of a pass or carry is the value of where it ends minus where it started. See [`xtpress/expected_threat.py`](xtpress/expected_threat.py).

### Pressing

Pressure events are grouped into sequences: within a team and match, a new sequence starts after a 5 s gap, a change of possession, or a new period. A sequence succeeds if the next possession belongs to the pressing team and starts within 8 s of the last pressure, and it is "high" if the ball is won in the opponent's half. The xT of everything that possession went on to produce is added up from the same rated actions as Part I, so pressing is priced in the same units as every pass and carry, and each won possession is credited once. Intensity is reported two ways: PPDA, and pressures per opponent-half touch. See[`xtpress/pressing_sequences.py`](xtpress/pressing_sequences.py).

### Press resistance

*In progress.* The aim is a player-level measure of how often a player keeps the ball when pressured, adjusted for the zone in which they received it. See[`xtpress/press_resistance.py`](xtpress/press_resistance.py).

## Pressing results

WC 2022, 32 teams, pooled over every match each team played. A selection from the ends of the payoff ranking:

| Team        | Matches | PPDA | Regain rate | xT per 100 pressures | Shots from presses |
| ----------- | ------- | ---- | ----------- | -------------------- | ------------------ |
| Canada      | 3       | 13.6 | 16.4%       | 0.325                | 10                 |
| Brazil      | 5       | 10.4 | 21.2%       | 0.324                | 13                 |
| Germany     | 3       | 10.2 | 20.0%       | 0.276                | 11                 |
| Mexico      | 3       | 7.4  | 15.9%       | 0.073                | 2                  |
| South Korea | 4       | 11.0 | 14.2%       | 0.069                | 2                  |
| Costa Rica  | 3       | 19.5 | 14.4%       | 0.045                | 0                  |

Across all 32 teams the median PPDA is 11.3 and the median payoff is 0.159 xT per 100 pressures. Most teams played only three matches, so a team's total is built from very few shots (0 to 15): read the ends of the table as examples, not as rankings.

When a press wins the ball it does so quickly and deep: the median time from the last pressure to the regain is 2.5 s (41% within 2 s, 90% within 6.8 s), and the median regain is at x = 37, with 54% in the team's own third and only 5% in the final third. The PPDA–payoff correlation stays between −0.05 and −0.23 for every regain window from 2 to 15 s.

## Supporting evidence

| Figure                         | What it shows                                         |
| ------------------------------ | ----------------------------------------------------- |
| ![](figures/xt_surface.png)    | The fitted value surface.                             |
| ![](figures/team_pressing.png) | Pressing intensity against pressing payoff.           |
| ![](figures/case_study.png)    | One sequence: the press, the turnover, what followed. |

*The figures are not generated yet (`xtpress/viz.py`); the links will resolve once they are. Add one line of interpretation under each when they are.*

## Validation

| Check                                               | Target                    | Result                                                                                                                       |
| --------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Iterative vs closed-form solve                      | agree to 1e-9             | 7.0e-12 (173 iterations)                                                                                                     |
| Correlation with Karun Singh's published 12×8 grid | r > 0.97                  | 0.996 (MAE 0.0077)                                                                                                           |
| Same, without the completion factor (baseline)      | r > 0.97                  | 0.860 - fails; see below                                                                                                     |
| Transition matrix rows sum to 1                     | within 1e-9               | 2.2e-16                                                                                                                      |
| Pass completion rate                                | 0.78–0.84                | 0.822                                                                                                                        |
| PPDA across teams                                   | 6–15                     | 7.4–20.1 across 32 teams (30 inside, median 11.3); 77% of 128 team-matches inside 6–15                                     |
| Press-to-regain rate                                | 13–21% (6–12 s windows) | 16.3% at the default 8 s window (13.7% at 6 s, 20.6% at 12 s); 19.1% at 8 s excluding sequences in the team's own possession |
| High regains (ball won at x ≥ 60)                  | none                      | 3.5% of sequences (5.6% at x ≥ 50, 1.9% at x ≥ 70)                                                                         |
| xT earned per won possession                        | 0.01–0.03                | 0.0139                                                                                                                       |
| Won possessions that reach a shot                   | 10–20%                   | 10.4%                                                                                                                        |
| Pressures per opponent-half touch                   | none                      | median 0.075 per team-match (0.027–0.159); 45% of all pressures are in the opponent's half                                  |
| Sequences never span a possession change            | all                       | all (tested)                                                                                                                 |
| Retention under pressure, league-wide               | 65–80%                   | — (Part III)                                                                                                                |
| Team xT vs goals, held out                          | Spearman ρ 0.45–0.65    | —                                                                                                                           |

Measured on WC 2022 (64 matches, 128,711 actions), 12×8 grid, fitted and checked on the same competition. The published grid was fitted on other leagues, so this tests agreement of shape, not accuracy. It is also exactly symmetric top to bottom; this grid is not (4.0% mean asymmetry, and no symmetrisation is applied).

The completion factor was found by checking against the published grid, not derived in advance. The textbook recursion, where the rows of the transition matrix sum to 1 and a failed move simply vanishes, gave r=0.860 and a nearly flat surface (goal-end 1.4x the own-end, against 12.6x in the published grid). The factor has no free parameters - it is each zone's observed completion rate - but I chose it after the mismatch. The baseline is kept (`fit(..., model_loss=False)`) so the comparison can be rerun.

The PPDA band of 6–15 was first measured on three matches. Over 64 it does not hold for every team: 23% of team-matches fall outside it, at both ends (Japan 26.3 and Morocco 25.2 don't press as high, Ecuador 4.2 was the one that pressed the most).

The press-to-regain reference of about 15–20% was set without a window. It holds only for windows of roughly 8 to 12 s (16.3% to 20.6%); at 6 s it is 13.7%. The target is therefore stated as 13–21% across 6–12 s windows, and the result is always quoted with its window.

## Model decisions

### Expected Threat decisions

- **Loss is modelled in the surface, not in the rating.** A zone's value is scaled by its observed completion rate (about 90% of moves arrive, from 55% to 95% by zone), so a lost ball ends the chain with value 0. The textbook form is kept as a baseline and reported in Validation.
- Failed actions are excluded from the rating rather than charged `-xt_start`, so the rating measures gain only.
- Negative deltas are kept, so the figure is *net* progression. A clipped-at-zero column is provided alongside.
- Set pieces are flagged and excluded from the headline tables. They are 5.1% of move attempts and still count as moves in the shoot-or-move split, which is a judgement call: a corner is arguably not a decision to move rather than shoot.
- Penalties are excluded from shots (23 in this sample). Every one is struck from the same spot and would make a false hotspot in `g(z)`.
- Scoring probability `g(z)` comes from mean StatsBomb xG per zone, not the raw goal rate. 68 of 96 zones have no shots and get 0, which is harmless because shot probability there is also 0.
- The grid is 12×8 and is not symmetrised.

### Pressing decisions

- **Frames.** StatsBomb records every event in the frame of the team that performed it, so a Pressure and the possession team's action are mirror images. Pressure coordinates are mirrored before they are compared with the possession team's, and `mean_x` of a sequence is kept raw because it answers how high the team pressed.
- **Linking pressures to actions.** 90.5% of pressures name the action they affected in `related_events`. The rest are matched to the nearest action in time (within 2 s, same match, period and opposing team), which agrees with the true link 58% of the time; matching forward only agrees 37%.
- **Sequence gap.** 5 s; a change of possession or period also starts a new sequence. The gap changes the sequence count a lot (2 s gives 14,588 sequences, 12 s gives 10,394) and the regain rate only from 15.7% to 18.9%, and it does not change the team payoff at all: the payoff divides by pressures and credits won possessions, so it does not depend on how pressures are grouped.
- **Success.** The next possession (id + 1) is the pressing team's and starts within 8 s of the last pressure. The recovery event itself is not used, because it is often logged in the old possession. The 8 s window is a real choice: the regain rate is 6.7% at 2 s, 16.3% at 8 s and 23.3% at 15 s. The team ranking by payoff is stable from 4 s upward (rank correlation 0.92 or more against 8 s) but not at 2 s (0.59), where there are too few wins to rank on.
- **High regain.** The ball is won at x ≥ 60, the opponent's half, in the pressing team's own frame. That is 3.5% of sequences (21.5% of wins); if I move the line: 5.6% at x ≥ 50, 1.9% at x ≥ 70 and 0.9% at x ≥ 80.
- **Presses in the team's own possession.** 15% of sequences (1,840 of 12,424) are logged during the pressing team's own possession. They cannot win the ball, so they are never counted as successes, but they stay in the denominators.
- **Counterpress.** 20.1% of pressures are flagged as counterpress, and 21.5% of sequences contain one. Against the opponent's possession they win the ball less often (13.7% against 20.2%) but more often high (5.7% against 3.8%), and what they win is worth less (0.0120 xT against 0.0142).
- **PPDA.** Numerator is opponent passes with x ≤ 72 over the denominator, team's own Duels, Interceptions and Fouls Committed with         x ≥ 48; thats 60% of the pitch from the team with possession perspective. Aerial duels are included (half of all duels). Pooled across matches, not averaged.
- **Touches.** Pressures per opponent-half touch counts eight on-ball event types, including receipts. The ranking barely changes between definitions (Spearman ≥ 0.988).
- **Credit.** Each won possession is credited once. 6% of won sequences share a possession with another won sequence, and counting both would add 4.9% to the xT and nine shots.

## Limitations

- The surface models loss but the rating does not. A failed pass lowers the value of the zone it started from, but an individual failed action still scores zero, so xT never charges a player for losing the ball and rewards volume progression. Closing that needs an action-value model such as VAEP.
- Pressure events are a human annotation against a five-yard convention, not ground truth.
- The sample is a single tournament (WC 2022: 64 matches, 128,711 actions); player-level figures carry real uncertainty.
- Most teams played three matches (maximum seven), so team-level pressing figures rest on very few sequences and shots. Among the 16 teams with four or more matches the PPDA-payoff correlation is −0.30, against −0.09 for all 32, which shows how unstable it is at this size.
- Sequences are mostly single presses: 76% contain one pressure and the median lasts 0 s, so "sequence" overstates how much tactical structure there is.
- The payoff is not causal. It is everything the won possession produced, which depends on who won it as well as on the press, so good attacking sides will rate higher.
- PPDA counts the opponent's passes, so a side that plays long over the press looks less pressed. Pressures per opponent-half touch avoids that, but the two agree only partly (Spearman −0.63).
- Press resistance is confounded by who a player is asked to receive against; the zone adjustment reduces this but does not remove it.
- Per-zone completion rates rest on as few as 50 attempts in the quietest zones.
- The PPDA–payoff correlation is small at every regain window from 2 to 15 s (−0.05 to −0.23), so the finding is not an artefact of the 8 s choice. The larger uncertainty is the sample: the 95% interval over the 32 teams is −0.44 to 0.30.
- Counterpress sequences are pooled with the rest although they behave differently (lower regain rate, higher share won high), so a team that counterpresses a lot is not directly comparable with one that does not.

## What the checks turned up

The main risk in this project was a wrong answer that looks right, so every function has a
hand-counted test, and each test was checked by breaking the code on purpose and confirming
that it fails. A few of the things that came out of it:

- **PPDA is not confined to 6–15.** The band came from three matches; over 64, 23% of team-matches fall outside it and they are real teams.
- **A team makes about 94 pressing sequences per match** (median, range 61–163), not the 40–80 I had been told to expect.
- **Plausible wrong answers are easy.** Using the wrong zone threshold for the opponent's passes gives a median PPDA of 6.7, inside the usual band and wrong. Dividing the payoff by sequences instead of pressures inflates it by a factor of 1.33.
- **Presses overlap.** 6% of won sequences share a won possession with another one, and 15% of all sequences are logged during the pressing team's own possession.
- **The fallback for linking pressures is weaker than it looks.** It matches an action 96% of the time but agrees with the true link only 58% of the time.
- **The window and the gap do not matter equally.** Moving the gap from 2 s to 12 s changes the sequence count by 29% and the team payoff by nothing; moving the regain window from 2 s to 15 s takes the regain rate from 6.7% to 23.3%.

## Reproducing it

Requires Python 3.10+. Developed on 3.14.

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

```bash
python -m xtpress.loaders --competition wc_2022
```

Run the tests:

```bash
python -m pytest -q
```

Tests that need the full competition skip themselves until the data above has been built.

The pressing pipeline end to end, for the whole competition (about a minute, mostly fitting the model):

```python
import pandas as pd
from xtpress import config, loaders
from xtpress.expected_threat import ExpectedThreat
from xtpress.pressing_sequences import *

actions = loaders.load_actions([config.FOCUS_COMPETITION])
rated = ExpectedThreat(cols=config.GRID_COLS, rows=config.GRID_ROWS).fit(actions).rate_actions(actions)
events = pd.concat([loaders.load_events(m) for m in sorted(actions.match_id.unique())], ignore_index=True)

sequences = build_press_sequences(extract_pressures(events))
sequences = label_sequence_outcomes(sequences, events)
sequences = sequence_value(sequences, rated, actions)
summary = team_pressing_summary(sequences, events)
```

## Data

Data provided by [StatsBomb](https://statsbomb.com/what-we-do/soccer-data/) under their
free open data licence, available at
[statsbomb/open-data](https://github.com/statsbomb/open-data).

<!-- Add the StatsBomb logo here. Attribution is a condition of the licence. -->

## Repo

```
xtpress/
  config.py              constants: pitch, grid, paths, thresholds
  loaders.py             StatsBomb fetch, cache, tidy action frame
  expected_threat.py     grid, transition matrix, xT solve and scoring
  pressing_sequences.py  press detection, PPDA, team pressing value
  press_resistance.py    player-level metric with zone adjustment
  viz.py                 every figure in this README
notebooks/               exploration; kept deliberately, it shows process
data/                    gitignored cache and processed parquet; fitted models are tracked
figures/                 exported PNGs used above
tests/                   the acceptance checks from the validation table
docs/                    the workbook and the project brief
```
