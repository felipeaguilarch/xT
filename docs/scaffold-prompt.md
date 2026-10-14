# Reusable prompt: "learning scaffold" project generator

Paste everything below the line into a **fresh Claude Code session in an empty
directory**. Fill in the BRIEF block first. Everything else is fixed machinery.

---

## BRIEF — fill these in, delete the examples

- **PROJECT NAME / PACKAGE NAME:** `<e.g. xtpress>`
- **ONE-LINE DESCRIPTION:** `<e.g. Expected Threat and pressing analysis on StatsBomb open data>`
- **DOMAIN / SUBJECT:** `<e.g. football analytics; or credit risk, seismology, RNA-seq, options pricing>`
- **THE DELIVERABLE (what a reader should walk away believing):** `<e.g. a validated value surface over the pitch plus a player-level metric derived from it>`
- **DATA SOURCE:** `<e.g. StatsBomb open data via statsbombpy; or a Kaggle CSV, an API, a synthetic generator>`
- **LANGUAGE / STACK:** `<e.g. Python 3.10+, pandas, numpy, matplotlib>`
- **MY LEVEL:** `<e.g. comfortable with pandas, new to Markov models>`
- **WHAT I ALREADY KNOW HOW TO DO (implement this for me, do not stub it):** `<e.g. data fetching and cleaning>`
- **WHAT I WANT TO LEARN BY WRITING IT MYSELF (stub this):** `<e.g. the model, the metric, the figures>`
- **ROUGH SIZE:** `<e.g. 5-6 modules, ~35 stub functions, 4 parts>`

---

## YOUR TASK

Build me a **learning scaffold**: a real, runnable project skeleton where the
foundations work, the interesting parts are unimplemented stubs, and a printed
workbook teaches me how to fill them in without ever handing me the answer.

Work in the current directory. Do not implement anything I said I want to learn.

### Non-negotiable rules

1. **Measured, not remembered.** Every calibration number, threshold, band and
   expected value that appears anywhere — config constants, test assertions,
   workbook VERIFY blocks — must come from code you actually ran against real
   data in this session. If you cannot measure it, do not state it. Never invent
   a plausible-looking number.
2. **No solutions.** Stub bodies are `raise NotImplementedError`. The workbook
   explains *what*, *why*, and *which idiom* — it may show a one- or two-line
   library idiom in isolation (e.g. how `np.bincount` takes weights), never an
   assembled function body.
3. **Traps are specific.** A trap is a failure mode that produces *plausible
   but wrong* output — the silent kind. "Remember to handle nulls" is not a
   trap. "A completed pass has no outcome field, so testing for success by
   looking for a success value gives you a 5% completion rate" is a trap. Find
   the real ones in this domain by inspecting the actual data.
4. **Every claim is checkable.** Each stub gets a verification I can run. Each
   verification that can be automated becomes a test.

### Phase 0 — Reconnaissance (do this before writing any project file)

Fetch or generate a real sample of the data. Inspect it in a scratch script.
Measure the base rates, distributions, coordinate conventions, encoding quirks,
column semantics and edge cases. Find at least 10 domain traps by *observing*
them, not by recalling them. Keep a numbered trap list — the code and workbook
will reference traps by number.

Report the recon findings to me before continuing.

### Phase 1 — Plan

Decompose the work into **parts** (major chunks, e.g. "the model", "the
pipeline", "the metric", "the figures") and, within them, **chapters in strict
dependency order** — one chapter per stub function or tight group of them. Each
chapter must be completable in one sitting. Show me the outline.

### Phase 2 — Scaffold

Create the directory tree, packaging and hygiene files:

- `<package>/` with `__init__.py` carrying the docstring and `__version__`
- `pyproject.toml` (build backend, name, version, requires-python, test paths)
- `requirements.txt` grouped by purpose with an inline comment per line saying
  why it is there
- `.gitignore` that ignores caches and derived data but **tracks fitted
  models**, with comments explaining each decision
- `data/raw/`, `data/processed/`, `data/models/`, `figures/`, `notebooks/`,
  `tests/`, `docs/` — each with a `.gitkeep` where empty
- `notebooks/exploration.ipynb` — a starter notebook, kept deliberately because
  it shows process

### Phase 3 — The config module

`<package>/config.py`: pure constants, no heavy imports, so everything else can
import it for free. Paths, geometry/units, taxonomy, thresholds, source ids,
reporting minimums. **Every non-obvious constant carries a comment saying why
that value and not another**, with trap references. Constants that encode a
relationship to each other must be checkable, and get a test.

### Phase 4 — The working foundation

Fully implement the modules I listed under "what I already know how to do".
These must actually run end to end against the real data source, with caching,
a CLI entry point (`python -m <package>.<module> --...`), and a tidy, documented
output schema declared as a module-level column list. This module must stay
ignorant of everything downstream — it answers exactly one question.

### Phase 5 — The stubs

Every remaining module is stubs. **Docstring anatomy — every stub gets:**

- what it returns, with exact shape / columns / dtype
- the convention it must obey, stated so it can be obeyed everywhere else
- *why* the design is this way, when the design is arguable
- the specific trap, by number
- the check that proves it right, with the measured band
- for classes: attributes present after fitting, and the method split

Type-hint everything. Group modules with section banners. Where two routes are
defensible, name both, recommend one, and say the choice belongs in the model
metadata.

### Phase 6 — Tests as the definition of done

`tests/conftest.py` with session-scoped fixtures over one cached real sample, so
the suite is fast and works offline once warm.

`tests/test_invariants.py`: the acceptance checks, grouped by part, in the same
order as the chapters. Tests for already-implemented code run and pass now.
Tests for stubs are `@pytest.mark.skip(reason="implement <module>.<function>")`
— the reason names the function. Each test asserts a **measured** band and its
docstring says which silent failure it catches. The file header explains the
loop: implement, delete the skip mark, run the suite.

### Phase 7 — The README as a portfolio artifact

Structure it finding-first, not method-first, with HTML comments where I write
prose. Sections: **The finding** (two sentences plus the headline figure),
**Why it matters** (one paragraph, domain language, no formulas), **How it is
measured** (one paragraph per component, linking to source files, never pasting
code), a bulleted list of *model decisions worth stating plainly*, **Supporting
evidence** (figure table), **Validation** (check / target / result table, results
blank for me to fill), **Limitations** (write this one properly — it is the
cheapest credibility in the project), **Reproducing it** (exact commands),
**Data** (licence and attribution), **Repo** (annotated tree).

Mark it `> **Status:** scaffold.` at the top.

### Phase 8 — The workbook PDF

`docs/<name>-workbook.pdf`. This is the centrepiece. Typeset it properly —
title page, table of contents with page numbers, part title pages, running
headers, page numbers. Aim for roughly one page per chapter, 35-50 pages.

**Chapter 0, "How to use this book":** what is already implemented and therefore
out of scope; the deal (no finished functions, here is what you get instead);
the loop (pick chapter, read the why, write it, delete the skip mark, run the
suite); how many tests are skipped and why they matter; a table of module vs
stub count vs what it produces; any edit I must make to the *implemented* code
before a later part, with the exact snippet and the reasoning; and the notation
used throughout.

**Every chapter thereafter:** chapter number, title, the function name as a
subtitle, then in this order —

- what the thing *is*, in prose, in one short paragraph
- **WHY IT WORKS THIS WAY** — the design argument, where one exists. This is
  where you explain the deliberate asymmetries, the things that look like bugs
  and are not, and the model's defining limitation. Say explicitly when a
  section is the answer to the hardest question anyone will ask me about the
  project.
- comparison tables when there are competing routes: route / how / cost, then a
  recommendation and the disclosure it obliges
- **the idiom** — the one- or two-line library trick that makes it three lines
  instead of thirty, with a sentence on *why it generalises beyond this project*
- **TRAP** — set off visually. The silent failure, what wrong output it produces,
  and how to avoid it. Say which single line is the most dangerous in the part.
- **VERIFY** — the measured band, and what a violation implies about which
  mistake was made

**Appendices:** a trap index; a table of every skipped test — name, what it
catches, which chapter; a single table of every measured calibration target with
its band and where it was measured; and further reading with real sources.

Voice: direct, second person, unhedged. Short sentences. No filler, no
motivational padding, no emoji. Assume I am smart and impatient. Tell me when to
stop struggling and ask for the answer.

### Phase 9 — Handoff report

Finish by telling me, in the terminal:

- the tree, annotated
- what runs today vs what raises `NotImplementedError`, with counts per module
- the exact commands to set up the environment and to verify the scaffold works
- which tests pass now and which are skipped
- the recommended reading and implementation order
- every calibration number you measured, and on what sample
- anything you could not measure and therefore left out
