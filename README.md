# xT & Pressing

Expected Threat and pressing analysis built from scratch on StatsBomb open data.

> **Status:** work in progress. Part I (the Expected Threat model, action rating
> and minutes played) is complete and validated; pressing and press resistance
> are in progress.

---

## The finding

<!-- TWO SENTENCES. The sharpest thing you found. Then the figure, immediately.
     Not "this project implements xT" -- that is a method, not a finding.
     Aim for something like: "Team X presses harder than anyone at the
     tournament but converts it into the least threat, and the gap traces to
     two specific players who lose the ball under pressure at well below the
     rate their positions suggest." -->

![](figures/press_resistance.png)

## Why it matters

<!-- One short paragraph, football language, no formulas. What would a coach
     or an analyst do differently knowing this? -->

## How it is measured

**Expected Threat.** <!-- One paragraph. The grid, the Markov chain, what a
value means. Link to xtpress/expected_threat.py -- do not paste code here. -->

**Press resistance.**

## Supporting evidence

| Figure                         | What it shows                                         |
| ------------------------------ | ----------------------------------------------------- |
| ![](figures/xt_surface.png)    | The fitted value surface.                             |
| ![](figures/team_pressing.png) | Pressing intensity against pressing payoff.           |
| ![](figures/case_study.png)    | One sequence: the press, the turnover, what followed. |

<!-- One line of interpretation under each. -->

## Validation

<!-- Fill in with real numbers as you go. -->

| Check                                               | Target                 | Result                   |
| --------------------------------------------------- | ---------------------- | ------------------------ |
| Iterative vs closed-form solve                      | agree to 1e-9          | 7.0e-12 (173 iterations) |
| Correlation with Karun Singh's published 12×8 grid | r > 0.97               | 0.996 (MAE 0.0077)       |
| Same, without the completion factor (baseline)      | r > 0.97               | 0.860 - fails; see below |
| Transition matrix rows sum to 1                     | within 1e-9            | 2.2e-16                  |
| Pass completion rate                                | 0.78–0.84             | 0.822                    |
| PPDA across teams                                   | 6–15                  | —                       |
| Press-to-regain rate                                | ~15–20%               | —                       |
| Retention under pressure, league-wide               | 65–80%                | —                       |
| Team xT vs goals, held out                          | Spearman ρ 0.45–0.65 | —                       |

Measured on WC 2022 (64 matches, 128,711 actions), 12×8 grid, fitted and checked on the same competition. The published grid was fitted on other leagues, so this tests agreement of shape, not accuracy. It is also exactly symmetric top to bottom; this grid is not (4.0% mean asymmetry, and no symmetrisation is applied).

The completion factor was found by checking against the published grid, not derived in advance. The textbook recursion, where the rows of the transition matrix sum to 1 and a failed move simply vanishes, gave r=0.860 and a nearly flat surface (goal-end 1.4x the own-end, against 12.6x in the published grid). The factor has no free parameters - it is each zone's observed completion rate - but I chose it after the mismatch. The baseline is kept (`fit(..., model_loss=False)`) so the comparison can be rerun.

## Model decisions

- **Loss is modelled in the surface, not in the rating.** A zone's value is scaled by its observed completion rate (about 90% of moves arrive, from 55% to 95% by zone), so a lost ball ends the chain with value 0. The textbook form is kept as a baseline and reported in Validation.
- Failed actions are excluded from the rating rather than charged `-xt_start`, so the rating measures gain only.
- Negative deltas are kept, so the figure is *net* progression. A clipped-at-zero column is provided alongside.
- Set pieces are flagged and excluded from the headline tables. They are 5.1% of move attempts and still count as moves in the shoot-or-move split, which is a judgement call: a corner is arguably not a decision to move rather than shoot.
- Penalties are excluded from shots (23 in this sample). Every one is struck from the same spot and would make a false hotspot in `g(z)`.
- Scoring probability `g(z)` comes from mean StatsBomb xG per zone, not the raw goal rate. 68 of 96 zones have no shots and get 0, which is harmless because shot probability there is also 0.
- The grid is 12×8 and is not symmetrised.

## Limitations

- The surface models loss but the rating does not. A failed pass lowers the value of the zone it started from, but an individual failed action still scores zero, so xT never charges a player for losing the ball and rewards volume progression. Closing that needs an action-value model such as VAEP.
- Pressure events are a human annotation against a five-yard convention, not ground truth.
- The sample is a single tournament (WC 2022: 64 matches, 128,711 actions); player-level figures carry real uncertainty.
- Press resistance is confounded by who a player is asked to receive against; the zone adjustment reduces this but does not remove it.
- Per-zone completion rates rest on as few as 50 attempts in the quietest zones.

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
data/                    gitignored cache and processed parquet
figures/                 exported PNGs used above
tests/                   the acceptance checks from the validation table
```
