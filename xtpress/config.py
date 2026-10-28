"""Project-wide constants: pitch geometry, grid resolution, paths, thresholds.

Nothing in here should import pandas or numpy -- it is pure configuration so
that every other module can import it without cost.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # one parquet per match, as fetched
PROCESSED_DIR = DATA_DIR / "processed"  # tidy action frames per competition
MODELS_DIR = DATA_DIR / "models"        # fitted xT grids + metadata sidecars
FIGURES_DIR = PROJECT_ROOT / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Pitch geometry (StatsBomb convention)
# --------------------------------------------------------------------------
# Origin is TOP-LEFT. x runs 0 -> 120 toward the goal the ACTING team attacks.
# y runs 0 -> 80 downward. See trap #2: defensive events are mirrored relative
# to the possession team's frame.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# --------------------------------------------------------------------------
# xT grid
# --------------------------------------------------------------------------
# Start at 12x8. At 16x12 the 192x192 transition matrix has roughly nine
# observations per cell on ~180 matches -- treat larger grids as a sensitivity
# check, not the headline model.
GRID_COLS = 12
GRID_ROWS = 8
N_ZONES = GRID_COLS * GRID_ROWS

# --------------------------------------------------------------------------
# Competitions -- (competition_id, season_id), verified against competitions.json
# --------------------------------------------------------------------------
COMPETITIONS = {
    "wc_2022": (43, 106),        # 64 matches, has 360 data -- headline competition
    "euro_2024": (55, 282),      # 51 matches
    "wwc_2023": (72, 107),       # 64 matches
    "weuro_2025": (53, 315),     # 31 matches
    "laliga_2020_21": (11, 90),  # 35 matches
    "bundesliga_2023_24": (9, 281),
}

# Pool these to fit the grid; report on FOCUS_COMPETITION.
FIT_COMPETITIONS = ["wc_2022", "euro_2024", "wwc_2023"]
FOCUS_COMPETITION = "wc_2022"

# --------------------------------------------------------------------------
# Action taxonomy
# --------------------------------------------------------------------------
MOVE_TYPES = ("pass", "carry")

# action_type values (lowercase, as loaders.to_actions emits them) for the
# events that mean the player lost the ball while pressured.
LOSS_EVENT_TYPES = ("dispossessed", "miscontrol")

# Counted as defensive actions for PPDA (Trainor's definition).
PPDA_DEFENSIVE_ACTIONS = ("duel", "interception", "foul committed")

# --------------------------------------------------------------------------
# PPDA zone
# --------------------------------------------------------------------------
# The physical region is the opponent's own defensive 60% of the pitch.
# Because coordinates are in the ACTING team's frame, the two sides of the
# ratio use different thresholds for the same strip of grass:
PPDA_OPPONENT_PASS_MAX_X = 72.0   # opponent passes, in the opponent's frame
PPDA_DEFENSIVE_ACTION_MIN_X = 48.0  # your defensive actions, in your frame

# --------------------------------------------------------------------------
# Pressures per opponent-half touch
# --------------------------------------------------------------------------
# Your pressures in the opponent's half (x >= HALFWAY_X, your frame) set against
# the opponent's touches in their own half (x <= HALFWAY_X, their frame). The
# same strip of grass again, but 120 - 60 = 60, so for the halfway line the two
# thresholds happen to be one number.
HALFWAY_X = PITCH_LENGTH / 2

# What counts as a touch (lowercased, like PPDA_DEFENSIVE_ACTIONS). Receipts are
# in so that a side that plays long over the press is still seen on the ball.
TOUCH_EVENT_TYPES = (
    "pass", "ball receipt*", "carry", "dribble", "shot",
    "miscontrol", "dispossessed", "clearance",
)

# --------------------------------------------------------------------------
# Pressing thresholds
# --------------------------------------------------------------------------
PRESS_SEQUENCE_GAP_SECONDS = 5.0  # gap that starts a new pressing sequence
PRESS_REGAIN_WINDOW_SECONDS = 8.0  # window in which a turnover counts as won
PRESS_MATCH_WINDOW_SECONDS = 2.0  # two-sided fallback join (trap #15)

# A high regain is one won in the opponent's half. The turnover event is
# performed BY the pressing team, so it is already in their attacking frame
# (trap #2) and the opponent's half is simply x >= 60 -- no mirroring needed.
# Keep this as a flag on the sequence rather than a hard filter, so you can
# report both "all regains" and "high regains" without refitting.
PRESS_REGAIN_MIN_X = 60.0

# --------------------------------------------------------------------------
# Reporting thresholds
# --------------------------------------------------------------------------
MIN_MINUTES = 450            # minimum minutes for the progressor leaderboard
MIN_PRESSURED_ACTIONS = 50   # minimum sample for a press-resistance rating

# Periods 1 and 2 only for minutes played; period 5 is the shootout (trap #6).
REGULATION_PERIODS = (1, 2)
EXCLUDED_PERIODS = (5,)


