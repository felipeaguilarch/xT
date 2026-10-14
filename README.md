# xT & Pressing

Expected Threat and pressing analysis built from scratch on StatsBomb open data.

> **Status:** scaffold. Nothing below the line is written yet.

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

**Press resistance.** <!-- One paragraph. The two axes, and crucially the
zone adjustment: actual minus expected given where the player received the
ball. State the weights used in the composite score. -->

Model decisions worth stating plainly:

- Failed actions score zero, not negative — xT measures gain only.
- Negative deltas are kept, so the figure is *net* progression.
- Set pieces are flagged and excluded from the headline tables.
- Scoring probability `g(z)` comes from mean StatsBomb xG per zone, not the raw goal rate.

## Supporting evidence

| Figure | What it shows |
| --- | --- |
| ![](figures/xt_surface.png) | The fitted value surface. |
| ![](figures/team_pressing.png) | Pressing intensity against pressing payoff. |
| ![](figures/case_study.png) | One sequence: the press, the turnover, what followed. |

<!-- One line of interpretation under each. -->

## Validation

<!-- Fill in with real numbers as you go. -->

| Check | Target | Result |
| --- | --- | --- |
| Iterative vs closed-form solve | agree to 1e-9 | — |
| Correlation with Karun Singh's published 12×8 grid | r > 0.97 | — |
| Transition matrix rows sum to 1 | within 1e-9 | — |
| Pass completion rate | 0.78–0.84 | — |
| PPDA across teams | 6–15 | — |
| Press-to-regain rate | ~15–20% | — |
| Retention under pressure, league-wide | 65–80% | — |
| Team xT vs goals, held out | Spearman ρ 0.45–0.65 | — |

## Limitations

<!-- Write this section properly. It is the cheapest credibility in the whole
     project and the thing that separates someone who ran a model from someone
     who understands one. At minimum: -->

- xT credits gains and never penalises losses, so it rewards volume progression.
- Pressure events are a human annotation against a five-yard convention, not ground truth.
- The sample is a small number of tournaments; player-level figures carry real uncertainty.
- Press resistance is confounded by who a player is asked to receive against; the zone adjustment reduces this but does not remove it.

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
