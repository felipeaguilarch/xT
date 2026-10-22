"""StatsBomb data access: fetch, cache, and reshape events into tidy actions.

This module is the one you will carry into later projects, so it must not know
anything about xT, grids, or pressing. It answers exactly one question:
"give me clean actions for this competition".

Run as a script to build a dataset:

    python -m xtpress.loaders --competition wc_2022
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Columns every tidy action frame must have, in this order.
ACTION_COLUMNS = [
    "competition", "match_id", "index", "period", "minute", "second", "time_seconds",
    "possession", "possession_team",
    "team", "team_id", "player", "player_id", "position",
    "action_type", "start_x", "start_y", "end_x", "end_y",
    "successful", "is_goal", "xg",
    "is_set_piece", "is_penalty", "under_pressure", "counterpress",
    "play_pattern", "body_part",
]

# StatsBomb event types that carry the ball.
_KEEP_TYPES = {"Pass": "pass", 
               "Carry": "carry", 
               "Shot": "shot",
               "Dribble": "dribble",
               "Dispossessed": "dispossessed",
               "Miscontrol": "miscontrol"}

# pass_type / shot_type values that mean a dead-ball restart. Note that
# pass_type also contains "Recovery" and "Interception", which are NOT
# set pieces -- they describe how open play resumed.
_SET_PIECE_PASS_TYPES = {"Kick Off", "Throw-in", "Goal Kick", "Free Kick", "Corner"}
_SET_PIECE_SHOT_TYPES = {"Free Kick", "Corner", "Penalty", "Kick Off"}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def load_competitions() -> pd.DataFrame:
    """Return the StatsBomb competition/season index.

    Use this rather than hardcoding ids -- season ids are NOT chronological
    (trap #10). ``config.COMPETITIONS`` holds the handful already looked up.
    """
    from statsbombpy import sb

    return sb.competitions()


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return the match list for one competition-season."""
    from statsbombpy import sb

    return sb.matches(competition_id=competition_id, season_id=season_id)


def load_events(match_id: int, refresh: bool = False) -> pd.DataFrame:
    """Return the raw flat event frame for one match, cached to parquet.

    statsbombpy emits a NoAuthWarning on every call because you have no
    credentials. That is correct -- it means you are hitting the free open
    data. Do not try to silence it by supplying credentials.
    """
    path = config.RAW_DIR / f"{match_id}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    from statsbombpy import sb

    events = sb.events(match_id=match_id)
    events.to_parquet(path)
    return events


def load_competition_events(
    competition_id: int, season_id: int, refresh: bool = False
) -> pd.DataFrame:
    """Concatenate raw events for every match in a competition-season."""
    from tqdm import tqdm

    matches = load_matches(competition_id, season_id)
    frames = [
        load_events(int(mid), refresh=refresh)
        for mid in tqdm(matches.match_id, desc=f"{competition_id}/{season_id}")
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------

def _xy(value) -> tuple[float, float]:
    """Return (x, y) from a StatsBomb location, whatever container it is in.

    A fresh ``sb.events()`` gives Python lists; the same frame read back from
    the parquet cache gives numpy arrays. Testing ``isinstance(v, list)``
    therefore works on the first run and silently nulls every coordinate on
    the second. Test for length instead.

    Slicing to two values also handles ``shot_end_location``, which carries a
    third element -- the height at the goal line (trap #4).
    """
    if value is None or not hasattr(value, "__len__") or len(value) < 2:
        return (np.nan, np.nan)
    return (float(value[0]), float(value[1]))


def _timestamp_seconds(timestamp: pd.Series) -> pd.Series:
    """Convert "MM:SS.mmm" style timestamps to seconds elapsed in the period."""
    parts = timestamp.str.split(":", expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _flag(events: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a StatsBomb boolean flag to a real bool column.

    ``under_pressure`` and ``counterpress`` are object dtype holding True or
    NaN -- they are never False. Left alone they poison any filter or groupby.
    """
    if column not in events.columns:
        return pd.Series(False, index=events.index, dtype=bool)
    return events[column].fillna(False).astype(bool)


def _column(events: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    """Return a column, or a default-filled Series when the match lacks it.

    sb.events() only emits columns that have at least one value, so a match
    with no penalties has no ``shot_type`` at all.
    """
    if column not in events.columns:
        return pd.Series(default, index=events.index)
    return events[column]


def unpack_locations(events: pd.DataFrame) -> pd.DataFrame:
    """Split StatsBomb's list-valued location columns into float columns.

    ``location`` -> start_x, start_y
    ``pass_end_location`` / ``carry_end_location`` -> end_x, end_y

    Shots keep NaN end coordinates; xT is not defined for them.
    """
    out = events.copy()

    starts = _column(out, "location").map(_xy)
    out[["start_x", "start_y"]] = pd.DataFrame(starts.tolist(), index=out.index)

    ends = _column(out, "pass_end_location").fillna(
        _column(out, "carry_end_location")
    )
    out[["end_x", "end_y"]] = pd.DataFrame(ends.map(_xy).tolist(), index=out.index)

    return out


def to_actions(events: pd.DataFrame, competition: str | None = None) -> pd.DataFrame:
    """Reduce ~35 StatsBomb event types to the three that carry the ball.

    Returns a frame with exactly ``ACTION_COLUMNS``, one row per action.

    Sanity targets: pass completion 0.78-0.84, 10-30 shots per match across
    both teams.

    ``competition`` labels every row with the key from ``config.COMPETITIONS``.
    The raw events carry match_id but no competition, so once frames from
    several competitions are concatenated that provenance is unrecoverable --
    and ExpectedThreat.save has to record which competitions it was fitted on.
    Stamp it here, at the only point that still knows.
    """
    df = unpack_locations(events)
    df = df[df["type"].isin(_KEEP_TYPES)].copy()

    # Trap #6: period 5 is the penalty shootout, not football.
    df = df[~df["period"].isin(config.EXCLUDED_PERIODS)]

    # Trap #11: Half Start, Tactical Shift and friends have no player. They
    # cannot reach here given the type filter, but a null player_id would
    # still poison every groupby downstream.
    df = df[df["player_id"].notna()]

    action_type = df["type"].map(_KEEP_TYPES)
    pass_outcome = _column(df, "pass_outcome")
    pass_type = _column(df, "pass_type")
    shot_type = _column(df, "shot_type")

    # Default True covers Carry and Shot, which have no success/failure of
    # their own -- a carry simply happened. Every type that DOES carry an
    # outcome has to override it below, or the default quietly scores it a win.
    successful = pd.Series(True, index=df.index, dtype=bool)

    # Trap #1: a COMPLETED pass has NO outcome. Inverting this gives a ~17%
    # completion rate and a model that points the wrong way.
    is_pass = action_type == "pass"
    successful[is_pass] = pass_outcome[is_pass].isna()

    # A take-on has its own outcome column, and it is close to a coin flip --
    # roughly half of all Dribbles are Incomplete.
    is_dribble = action_type == "dribble"
    successful[is_dribble] = _column(df, "dribble_outcome")[is_dribble].eq("Complete")

    # Losing the ball is never a successful action. These two types ARE the
    # loss, so they have no outcome column to read -- state it directly.
    successful[action_type.isin(("dispossessed", "miscontrol"))] = False

    is_set_piece = (
        pass_type.isin(_SET_PIECE_PASS_TYPES) | shot_type.isin(_SET_PIECE_SHOT_TYPES)
    ).fillna(False)

    # Trap #5: every penalty is struck from the same spot, so leaving them in
    # burns a false hotspot into g(z). is_set_piece cannot do this job -- it
    # also covers free kicks and corners, which are real shots from real zones.
    is_penalty = shot_type.eq("Penalty").fillna(False)

    # There is no generic body_part column -- each event type has its own.
    body_part = _column(df, "pass_body_part").fillna(_column(df, "shot_body_part"))

    out = pd.DataFrame(
        {
            "competition": competition,
            "match_id": df["match_id"],
            "index": df["index"],
            "period": df["period"],
            "minute": df["minute"],
            "second": df["second"],
            "time_seconds": _timestamp_seconds(df["timestamp"]),
            "possession": df["possession"],
            "possession_team": df["possession_team"],
            "team": df["team"],
            "team_id": df["team_id"],
            "player": df["player"],
            "player_id": df["player_id"],
            "position": _column(df, "position"),
            "action_type": action_type,
            "start_x": df["start_x"],
            "start_y": df["start_y"],
            "end_x": df["end_x"],
            "end_y": df["end_y"],
            "successful": successful,
            "is_goal": _column(df, "shot_outcome").eq("Goal").fillna(False),
            "xg": _column(df, "shot_statsbomb_xg"),
            "is_set_piece": is_set_piece,
            "is_penalty": is_penalty,
            "under_pressure": _flag(df, "under_pressure"),
            "counterpress": _flag(df, "counterpress"),
            "play_pattern": df["play_pattern"],
            "body_part": body_part,
        }
    )

    return (
        out[ACTION_COLUMNS]
        .sort_values(["match_id", "period", "index"])
        .reset_index(drop=True)
    )


def minutes_played(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (match_id, player_id) with minutes played.

    NOT IMPLEMENTED -- this is Phase 04 work, and it is the most under-rated
    task in the project. There is no ready-made column. The recipe:

      1. Starters come from the ``Starting XI`` event's ``tactics`` lineup.
      2. ``Substitution`` events give the exit minute and the replacement.
      3. Period length is the timestamp of the LAST event in that period --
         not 45, because of stoppage time.
      4. minutes = (exit or period end) - (entry or period start), summed over
         ``config.REGULATION_PERIODS``.

    Note this needs the RAW events, not the action frame -- Starting XI and
    Substitution are filtered out by ``to_actions``.
    """
    # 0. regulation periods only, timestamps as seconds
    ev = events[events["period"].isin(config.REGULATION_PERIODS)].copy()
    ev["t"] = _timestamp_seconds(ev["timestamp"])

    # 1. ONE clock for the whole match
    p_len = ev.groupby(["match_id", "period"])["t"].max().unstack("period")
    ev["clock"] = ev["t"] + ev["match_id"].map(p_len[1]).where(ev["period"] == 2, 0.0)
    match_end = p_len[1] + p_len[2]

    # 2. starters: eleven per team, on at time zero
    xi = ev[ev["type"] == "Starting XI"]
    starters = pd.DataFrame(
        [(r.match_id, r.team, p["player"]["id"], p["player"]["name"])
         for r in xi.itertuples() for p in r.tactics["lineup"]],
        columns=["match_id", "team", "player_id", "player"])
    starters["on"] = 0.0

    # 3. substitutions: each event is BOTH an exit and an entrance
    subs = ev[ev["type"] == "Substitution"]
    arriving = pd.DataFrame({
        "match_id": subs["match_id"], "team": subs["team"],
        "player_id": subs["substitution_replacement_id"].astype(int),
        "player": subs["substitution_replacement"],
        "on": subs["clock"]})
    leaving = pd.DataFrame({
        "match_id": subs["match_id"],
        "player_id": subs["player_id"].astype(int),
        "off": subs["clock"]})

    # 4. a red card is another way to leave
    card = pd.concat([_column(ev, "foul_committed_card"),
                      _column(ev, "bad_behaviour_card")], axis=1)
    sent_off = ev[card.isin(["Red Card", "Second Yellow"]).any(axis=1)]
    red = pd.DataFrame({
        "match_id": sent_off["match_id"],
        "player_id": sent_off["player_id"].astype(int),
        "off": sent_off["clock"]})

    # 5. assemble
    offs = pd.concat([leaving, red]).groupby(["match_id", "player_id"],
                                             as_index=False)["off"].min()
    out = pd.concat([starters, arriving], ignore_index=True).merge(
        offs, on=["match_id", "player_id"], how="left")
    out["off"] = out["off"].fillna(out["match_id"].map(match_end))
    out["minutes"] = (out["off"] - out["on"]) / 60
    return out[["match_id", "player_id", "player", "team", "minutes"]]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _processed_path(competition_key: str):
    comp_id, season_id = config.COMPETITIONS[competition_key]
    return config.PROCESSED_DIR / f"actions_{comp_id}_{season_id}.parquet"


def build_dataset(competition_key: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch, reshape and persist one competition to processed parquet."""
    comp_id, season_id = config.COMPETITIONS[competition_key]
    events = load_competition_events(comp_id, season_id, refresh=refresh)
    actions = to_actions(events, competition=competition_key)
    actions.to_parquet(_processed_path(competition_key), index=False)
    return actions


def load_actions(competition_keys: list[str] | None = None) -> pd.DataFrame:
    """Read processed action frames back, concatenated.

    Defaults to ``config.FIT_COMPETITIONS``. This is the entry point every
    downstream module should use.
    """
    keys = competition_keys or config.FIT_COMPETITIONS
    frames = []
    for key in keys:
        path = _processed_path(key)
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} missing -- run: python -m xtpress.loaders --competition {key}"
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a processed action dataset.")
    parser.add_argument(
        "--competition", default=config.FOCUS_COMPETITION,
        choices=sorted(config.COMPETITIONS), help="competition key from config.COMPETITIONS",
    )
    parser.add_argument("--refresh", action="store_true", help="ignore the raw cache")
    args = parser.parse_args()

    df = build_dataset(args.competition, refresh=args.refresh)
    passes = df[df.action_type == "pass"]
    print(f"{args.competition}: {len(df):,} actions from {df.match_id.nunique()} matches")
    print(f"  pass completion {passes.successful.mean():.3f}")
    print(f"  shots {(df.action_type == 'shot').sum():,}")
