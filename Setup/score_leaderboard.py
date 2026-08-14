"""Score the submissions students drag into the shared Drive folder.

Reads a flat folder of submission files, scores each against the unblinded
test labels, and writes the leaderboards to a Google Sheet.

    python Setup/score_leaderboard.py --submissions "/path/to/Submissions to Score" \\
                                      --truth ~/private/test_labels.csv \\
                                      --dry-run

    python Setup/score_leaderboard.py --submissions ... --truth ... \\
                                      --sheet-id 1AbC...xyz

--dry-run scores everything and prints what it would publish, touching no
sheet and needing no credentials. Point it at the folder the day before.

Rules, all deliberate:

  * a submission is REJECTED outright if it is malformed, has duplicate or
    missing molecules, or has any NaN prediction. NaN would otherwise be
    silently excluded from the metric, letting a team be scored on the subset
    they felt confident about.
  * a rejected file does not consume a budget slot -- it was never scored.
  * of a pair's accepted files, the FIRST TEN by prepared_at are eligible.
    Later ones are recorded and shown, but not ranked.
  * accuracy ranks a pair by their single best MA-RAE.
  * calibration ranks by their smallest |expected - actual|, chosen
    independently -- it may point at a different submission than accuracy.
    A pair with no expected score on any submission is not ranked at all.

The ledger (a CSV beside the sheet) makes re-runs incremental: a file already
scored at the same mtime is skipped. Delete the ledger to force a full rescore.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


BUDGET = 10

#: one row per submission file we have looked at
LEDGER_COLS = ["file", "mtime", "pair", "model", "split", "prepared_at",
               "expected", "actual", "cal_error", "status"] \
    + [f"RAE {e}" for e in common.ENDPOINTS] \
    + [f"MAE {e}" for e in common.ENDPOINTS]


# --------------------------------------------------------------------- truth

def load_truth(path: str) -> pd.DataFrame:
    """The unblinded test labels. Accepts raw assay units or log scale."""
    truth = pd.read_csv(path)
    if common.ID_COL not in truth.columns:
        raise SystemExit(f"{path} has no '{common.ID_COL}' column")

    # "LogD" is both a raw assay name and a log-scale name, so it cannot be
    # used to tell the two apart. Look for a column that only raw files have.
    raw_only = set(common.RAW_ENDPOINTS) - set(common.ENDPOINTS)
    if raw_only & set(truth.columns):
        truth = common.to_log_scale(truth)      # raw units -> what we model
    missing = [e for e in common.ENDPOINTS if e not in truth.columns]
    if missing:
        raise SystemExit(f"{path} is missing endpoint columns: {missing}")
    return truth


# ---------------------------------------------------------------- one file

def parse_submission(path: str):
    """(predictions, metadata, error). error is None when the file is readable."""
    try:
        preds, meta = common.read_submission(path)
    except Exception as exc:
        return None, {}, f"unreadable ({type(exc).__name__})"

    # a hand-edited file may have lost its header; the filename still has the pair
    if not meta.get("pair"):
        stem = os.path.basename(path).split("__")[0]
        meta["pair"] = stem or "(unknown)"
        meta.setdefault("model", "(no header)")
    return preds, meta, None


def validate(preds: pd.DataFrame, truth: pd.DataFrame) -> str | None:
    """Why this submission cannot be scored, or None if it can."""
    if common.ID_COL not in preds.columns:
        return f"no '{common.ID_COL}' column"
    missing_cols = [e for e in common.ENDPOINTS if e not in preds.columns]
    if missing_cols:
        return f"missing endpoint columns: {', '.join(missing_cols)}"
    if preds[common.ID_COL].duplicated().any():
        n = int(preds[common.ID_COL].duplicated().sum())
        return f"{n} duplicate molecules"

    gap = set(truth[common.ID_COL]) - set(preds[common.ID_COL])
    if gap:
        return f"{len(gap)} test molecules missing"

    values = preds[common.ENDPOINTS].apply(pd.to_numeric, errors="coerce")
    n_nan = int(values.isna().sum().sum())
    if n_nan:
        # NaNs are dropped by evaluate() rather than penalised, so a partial
        # submission would be scored on a favourable subset. Reject instead.
        return f"{n_nan} missing or non-numeric predictions"
    return None


def score_one(preds: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Per-endpoint RAE and MAE, plus the MA-RAE the leaderboard ranks on."""
    metrics = common.evaluate(truth, preds)
    out = {"actual": float(metrics["RAE"].mean())}
    for endpoint in common.ENDPOINTS:
        out[f"RAE {endpoint}"] = float(metrics.loc[endpoint, "RAE"])
        out[f"MAE {endpoint}"] = float(metrics.loc[endpoint, "MAE"])
    return out


# ----------------------------------------------------------------- the folder

def scan(folder: str) -> list[tuple[str, float]]:
    if not os.path.isdir(folder):
        raise SystemExit(f"No such folder: {folder}")
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".csv") and not name.startswith("."):
            path = os.path.join(folder, name)
            out.append((path, round(os.path.getmtime(path), 3)))
    return out


def score_folder(folder: str, truth: pd.DataFrame,
                 ledger_path: str | None = None) -> pd.DataFrame:
    """Score everything new in `folder`, returning the full ledger."""
    ledger = pd.DataFrame(columns=LEDGER_COLS)
    if ledger_path and os.path.exists(ledger_path):
        ledger = pd.read_csv(ledger_path)

    seen = {(r.file, r.mtime) for r in ledger.itertuples()}
    rows, n_new = [], 0

    for path, mtime in scan(folder):
        name = os.path.basename(path)
        if (name, mtime) in seen:
            continue
        n_new += 1

        row = {"file": name, "mtime": mtime}
        preds, meta, error = parse_submission(path)
        row["pair"] = meta.get("pair", "(unknown)")
        row["model"] = meta.get("model", "")
        row["split"] = meta.get("split", "")
        row["prepared_at"] = meta.get("prepared_at", "")
        try:
            row["expected"] = float(meta.get("expected_ma_rae"))
        except (TypeError, ValueError):
            row["expected"] = np.nan

        error = error or validate(preds, truth)
        if error:
            row["status"] = f"rejected: {error}"
        else:
            row.update(score_one(preds, truth))
            row["status"] = "scored"
            if not np.isnan(row["expected"]):
                row["cal_error"] = abs(row["expected"] - row["actual"])
        rows.append(row)
        print(f"  {name:44s} {row['pair']:16s} {row['status']}")

    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
    ledger = ledger.reindex(columns=LEDGER_COLS)
    # a file re-dragged after an edit appears twice; keep the newer mtime
    ledger = (ledger.sort_values("mtime")
                    .drop_duplicates("file", keep="last")
                    .reset_index(drop=True))
    print(f"{n_new} new file(s), {len(ledger)} in the ledger")

    if ledger_path:
        os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
        ledger.to_csv(ledger_path, index=False)
    return ledger


# ------------------------------------------------------------------- tables

def apply_budget(ledger: pd.DataFrame, budget: int = BUDGET) -> pd.DataFrame:
    """Mark each pair's first `budget` SCORED files eligible for ranking.

    Rejected files never consume a slot, so a malformed file is a free retry.
    Recomputed every run rather than stored, so changing the budget is just a
    flag rather than a ledger migration.
    """
    out = ledger.copy()
    out["eligible"] = False
    scored = out["status"] == "scored"
    order = out[scored].sort_values(["pair", "prepared_at", "file"])
    keep = order.groupby("pair").head(budget).index
    out.loc[keep, "eligible"] = True
    out.loc[scored & ~out["eligible"], "status"] = "over budget"
    return out


def build_tables(ledger: pd.DataFrame, budget: int = BUDGET) -> dict:
    """The sheet, one DataFrame per tab."""
    marked = apply_budget(ledger, budget)
    ranked = marked[marked["eligible"]]
    tables = {}

    # --- every submission, all teams, always visible -----------------------
    cols = (["pair", "model", "split", "prepared_at", "expected", "actual",
             "cal_error", "status"]
            + [f"RAE {e}" for e in common.ENDPOINTS] + ["file"])
    tables["submissions"] = (marked.reindex(columns=cols)
                                   .sort_values(["pair", "prepared_at"],
                                                na_position="last"))

    # --- accuracy: each pair's single best MA-RAE ---------------------------
    if ranked.empty:
        tables["accuracy"] = pd.DataFrame(columns=["rank", "pair", "MA-RAE"])
    else:
        best = ranked.loc[ranked.groupby("pair")["actual"].idxmin()].copy()
        used = ranked.groupby("pair").size()
        best["submissions used"] = best["pair"].map(used).astype(str) + f"/{budget}"
        best = best.sort_values("actual").reset_index(drop=True)
        best.insert(0, "rank", best.index + 1)
        tables["accuracy"] = best.reindex(columns=(
            ["rank", "pair", "actual", "model", "split", "submissions used"]
            + [f"RAE {e}" for e in common.ENDPOINTS])
        ).rename(columns={"actual": "MA-RAE"})

    # --- calibration: each pair's smallest |expected - actual| -------------
    cal = ranked.dropna(subset=["cal_error"])
    if cal.empty:
        tables["calibration"] = pd.DataFrame(
            columns=["rank", "pair", "cal_error"])
    else:
        best_cal = cal.loc[cal.groupby("pair")["cal_error"].idxmin()].copy()
        best_cal = best_cal.sort_values("cal_error").reset_index(drop=True)
        best_cal.insert(0, "rank", best_cal.index + 1)
        tables["calibration"] = best_cal.reindex(columns=[
            "rank", "pair", "cal_error", "expected", "actual", "model", "split",
            "prepared_at"])

    # --- one tab per endpoint, ranked by lowest MAE ------------------------
    for endpoint in common.ENDPOINTS:
        col = f"MAE {endpoint}"
        have = ranked.dropna(subset=[col])
        if have.empty:
            tables[endpoint] = pd.DataFrame(columns=["rank", "pair", "MAE"])
            continue
        best_ep = have.loc[have.groupby("pair")[col].idxmin()].copy()
        best_ep = best_ep.sort_values(col).reset_index(drop=True)
        best_ep.insert(0, "rank", best_ep.index + 1)
        tables[endpoint] = (best_ep.reindex(columns=[
            "rank", "pair", col, f"RAE {endpoint}", "model", "split"])
            .rename(columns={col: "MAE", f"RAE {endpoint}": "RAE"}))

    return tables


# -------------------------------------------------------------------- sheets

def _as_grid(df: pd.DataFrame) -> list[list]:
    """A DataFrame as the 2D list gspread wants, with NaN blanked."""
    body = df.where(pd.notna(df), "")
    rows = [list(map(str, df.columns))]
    for record in body.itertuples(index=False):
        rows.append([round(v, 4) if isinstance(v, float) else v for v in record])
    return rows


def write_sheet(client, sheet_id: str, tables: dict) -> None:
    """Rewrite every tab from the tables. Three-ish API calls, idempotent."""
    book = client.open_by_key(sheet_id)
    existing = {ws.title: ws for ws in book.worksheets()}

    for title, df in tables.items():
        grid = _as_grid(df)
        rows, cols = max(len(grid), 2), max(len(grid[0]), 1)
        if title in existing:
            ws = existing[title]
            ws.clear()
            ws.resize(rows=rows, cols=cols)
        else:
            ws = book.add_worksheet(title=title, rows=rows, cols=cols)
        ws.update(grid, "A1")
        ws.freeze(rows=1)
        print(f"  wrote {title:22s} {len(df):4d} row(s)")

    # a fresh Sheet arrives with an empty "Sheet1" -- get rid of it
    if "Sheet1" in existing and "Sheet1" not in tables:
        book.del_worksheet(existing["Sheet1"])


# ---------------------------------------------------------------------- main

def run(submissions: str, truth_path: str, sheet_id: str | None = None,
        ledger_path: str | None = None, budget: int = BUDGET,
        client=None) -> dict:
    """Score the folder and (unless dry) publish. Returns the tables."""
    truth = load_truth(truth_path)
    print(f"truth: {len(truth)} molecules\nscoring {submissions}")

    ledger = score_folder(submissions, truth, ledger_path)
    tables = build_tables(ledger, budget)

    if sheet_id is None:
        print("\n-- dry run, nothing published --")
        for title, df in tables.items():
            if title in ("submissions", "accuracy", "calibration"):
                print(f"\n### {title} ({len(df)} rows)")
                print(df.head(15).to_string(index=False) if len(df) else "  (empty)")
        rejected = ledger[ledger["status"].str.startswith("rejected", na=False)]
        if len(rejected):
            print(f"\n### {len(rejected)} rejected")
            print(rejected[["pair", "file", "status"]].to_string(index=False))
        return tables

    if client is None:
        import gspread
        client = gspread.service_account()      # needs a key; Colab passes a client
    print("\npublishing:")
    write_sheet(client, sheet_id, tables)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submissions", required=True,
                    help='the flat "Submissions to Score" folder')
    ap.add_argument("--truth", required=True,
                    help="CSV of unblinded test labels -- keep this private")
    ap.add_argument("--sheet-id", default=None,
                    help="Google Sheet id from its URL. Omit for a dry run.")
    ap.add_argument("--ledger", default=None,
                    help="where to remember what has been scored "
                         "(default: alongside --truth)")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--dry-run", action="store_true",
                    help="score and print, publish nothing")
    a = ap.parse_args()

    ledger = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.truth)),
                                      "leaderboard_ledger.csv")
    run(a.submissions, a.truth,
        sheet_id=None if a.dry_run else a.sheet_id,
        ledger_path=ledger, budget=a.budget)
