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
  * nor does a DUPLICATE: predictions a pair already submitted, with the same
    expected score, are recorded once. Re-dragging a file, or renaming a model
    and resubmitting it, is free. A different expected score over the same
    predictions is a new submission -- it is a different bet.
  * nor does a pair's FIRST CONSTANT baseline, where every endpoint predicts
    one value for every molecule. The train mean is a reference point, not a
    model, so it is free -- but it IS ranked. Only the first: a ranked freebie
    is a run at the calibration board, and one is a reference point where
    unlimited would be a strategy.
  * of a pair's accepted files, the FIRST TEN by prepared_at are eligible.
    Later ones are recorded and shown, but not ranked.
  * accuracy ranks a pair by their single best MA-RAE.
  * calibration ranks by their smallest |expected - actual|, chosen
    independently -- it may point at a different submission than accuracy.
    A pair with no expected score on any submission is not ranked at all.
    Constant baselines are excluded here: flat predictions calibrate
    near-perfectly by construction, so they would own the tab.

The ledger (a CSV beside the sheet) makes re-runs incremental: a file already
scored at the same mtime is skipped. Delete the ledger to force a full rescore.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


BUDGET = 10

#: predictions are compared for equality at this many decimal places. Six is
#: far below the noise floor of every assay here, so two files that agree to
#: six places are the same model, not two attempts.
PRED_PRECISION = 6

#: one row per submission file we have looked at
LEDGER_COLS = ["file", "mtime", "pair", "model", "split", "prepared_at",
               "expected", "actual", "cal_error", "pred_hash", "constant",
               "status"] \
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


def pred_hash(preds: pd.DataFrame) -> str:
    """A fingerprint of the predicted numbers, ignoring row order and header.

    Two files with the same fingerprint are the same set of predictions, so a
    team that re-drags a file, or submits a model they already submitted under
    a new name, is not spending a second budget slot on it.
    """
    canon = (preds[[common.ID_COL] + common.ENDPOINTS]
             .sort_values(common.ID_COL)
             .reset_index(drop=True))
    for e in common.ENDPOINTS:
        canon[e] = pd.to_numeric(canon[e], errors="coerce").round(PRED_PRECISION)
    return hashlib.md5(canon.to_csv(index=False).encode()).hexdigest()[:16]


def is_constant(preds: pd.DataFrame) -> bool:
    """True if every endpoint predicts one value for every molecule.

    That is the train-mean baseline, or any other constant. It is a useful
    reference point and students should feel free to submit it, but it is not a
    model, so it does not spend a budget slot.

    All nine endpoints have to be flat. A file that models one endpoint and
    falls back to the mean for the rest is still a real attempt at that one.
    With 2282 molecules a genuine model will not hit this by accident.
    """
    for e in common.ENDPOINTS:
        v = pd.to_numeric(preds[e], errors="coerce").round(PRED_PRECISION)
        if v.nunique(dropna=True) > 1:
            return False
    return True


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
            row["pred_hash"] = pred_hash(preds)
            row["constant"] = is_constant(preds)
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

def mark_duplicates(ledger: pd.DataFrame) -> pd.DataFrame:
    """Flag a pair's repeat submissions of predictions they already sent.

    Keyed on (pair, predictions, expected score), so re-sending the same
    numbers with the same bet is a no-op. Changing the expected score IS a new
    submission even with identical predictions -- it is a different bet, and
    the calibration board ranks the bet rather than the model.

    Rows from a ledger written before hashing existed have no fingerprint and
    are left alone; delete the ledger to rescore them if you care.
    """
    out = ledger.copy()
    if "pred_hash" not in out.columns:
        return out
    scored = (out["status"] == "scored") & out["pred_hash"].notna()
    if not scored.any():
        return out

    # dropna=False so two files that both logged no expected score still group
    order = out[scored].sort_values(["prepared_at", "file"])
    first = order.groupby(["pair", "pred_hash", "expected"],
                         dropna=False)["file"].transform("first")
    for idx in order.index[order["file"] != first]:
        out.loc[idx, "status"] = f"duplicate of {first.loc[idx]}"
    return out


def mark_constant(ledger: pd.DataFrame) -> pd.DataFrame:
    """Give each pair ONE free flat-prediction baseline: ranked, no slot spent.

    Run AFTER mark_duplicates, or re-dragging the same mean file would look
    like a second distinct baseline and start costing slots.

    One rather than unlimited because a ranked freebie is a run at the
    calibration board -- predict the mean, bet an MA-RAE of 1.0, be nearly
    perfectly calibrated. One is a fair reference point; unlimited is a
    strategy. A pair's second flat file is an ordinary submission and pays for
    itself.
    """
    out = ledger.copy()
    out["free"] = False
    if "constant" not in out.columns:
        return out
    flat = (out["status"] == "scored") & (out["constant"] == True)  # noqa: E712
    if not flat.any():
        return out

    order = out[flat].sort_values(["prepared_at", "file"])
    firsts = order.groupby("pair").head(1).index
    out.loc[firsts, "free"] = True
    out.loc[firsts, "status"] = "constant baseline (free)"
    return out


def apply_budget(ledger: pd.DataFrame, budget: int = BUDGET) -> pd.DataFrame:
    """Mark each pair's first `budget` SCORED files eligible for ranking.

    Rejected files never consume a slot, so a malformed file is a free retry.
    Neither do duplicates, nor a pair's first constant baseline -- though that
    baseline is still ranked, which is why `free` and `eligible` are separate.

    Recomputed every run rather than stored, so changing the budget is just a
    flag rather than a ledger migration.
    """
    out = mark_constant(mark_duplicates(ledger))
    out["eligible"] = False

    # the free baseline is ranked without spending anything
    out.loc[out["free"], "eligible"] = True

    paying = out["status"] == "scored"
    order = out[paying].sort_values(["pair", "prepared_at", "file"])
    keep = order.groupby("pair").head(budget).index
    out.loc[keep, "eligible"] = True
    out.loc[paying & ~out["eligible"], "status"] = "over budget"
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
        # a free baseline is ranked but spends nothing, so it must not show up
        # in the count -- a pair can be on the board having used 0 slots
        paid = marked[marked["eligible"] & ~marked["free"]]
        used = paid.groupby("pair").size()
        best["submissions used"] = (best["pair"].map(used).fillna(0)
                                    .astype(int).astype(str) + f"/{budget}")
        best = best.sort_values("actual").reset_index(drop=True)
        best.insert(0, "rank", best.index + 1)
        tables["accuracy"] = best.reindex(columns=(
            ["rank", "pair", "actual", "model", "split", "submissions used"]
            + [f"RAE {e}" for e in common.ENDPOINTS])
        ).rename(columns={"actual": "MA-RAE"})

    # --- calibration: each pair's smallest |expected - actual| -------------
    # Constant baselines are excluded. A flat prediction scores an MA-RAE of
    # about 1.0 by construction, and the expected value comes out of the same
    # arithmetic on the training set, so it calibrates near-perfectly without
    # any modelling skill -- left in, it wins this tab and nobody can beat it.
    # Excluded whether or not it was free: a paid second constant is just as
    # trivial to predict. They stay on accuracy and the endpoint tabs.
    cal = ranked
    if "constant" in cal.columns:
        cal = cal[cal["constant"] != True]  # noqa: E712  -- keeps NaN
    cal = cal.dropna(subset=["cal_error"])
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


# --------------------------------------------------------------------- xlsx

def write_xlsx(path: str, tables: dict) -> None:
    """Write every tab to one .xlsx, for when the Sheets API is unavailable.

    A Workspace admin can block Colab's OAuth consent ("your institution's
    admin needs to review third-party authored notebook code"), which kills
    write_sheet but not the Drive mount. Dropping the workbook straight into
    the shared folder needs no credentials at all: Drive previews it in the
    Sheets viewer, so students click one file and see the same tabs.

    Written to a temporary file and moved into place, so a half-written
    workbook is never what Drive picks up to sync.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise SystemExit("write_xlsx needs openpyxl: pip install openpyxl")

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    # keep the .xlsx extension on the temp name -- pandas picks its writer
    # from the extension and rejects anything else
    stem, ext = os.path.splitext(os.path.basename(path))
    tmp = os.path.join(folder, f".{stem}.tmp{ext or '.xlsx'}")

    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for title, df in tables.items():
            # Excel caps tab names at 31 characters; ours are shorter, but a
            # renamed endpoint should degrade rather than raise.
            body = df.copy()
            for col in body.columns:
                if body[col].dtype.kind == "f":
                    body[col] = body[col].round(4)
            body.to_excel(writer, sheet_name=title[:31], index=False)

            ws = writer.sheets[title[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(body.columns, start=1):
                seen = body[col].astype(str)
                width = max([len(str(col))] + [len(v) for v in seen]) + 2
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(width, 40)
            print(f"  wrote {title[:31]:22s} {len(df):4d} row(s)")

    try:
        os.replace(tmp, path)
    except OSError:
        # some Drive FUSE mounts refuse rename; fall back to a direct copy
        import shutil
        shutil.copyfile(tmp, path)
        os.remove(tmp)
    print(f"workbook: {path}")


# ---------------------------------------------------------------------- main

def run(submissions: str, truth_path: str, sheet_id: str | None = None,
        ledger_path: str | None = None, budget: int = BUDGET,
        client=None, xlsx_path: str | None = None) -> dict:
    """Score the folder and (unless dry) publish. Returns the tables.

    Publishing to a Sheet (sheet_id) and to a workbook (xlsx_path) are
    independent: pass either, both, or neither.
    """
    truth = load_truth(truth_path)
    print(f"truth: {len(truth)} molecules\nscoring {submissions}")

    ledger = score_folder(submissions, truth, ledger_path)
    tables = build_tables(ledger, budget)

    if xlsx_path:
        print("\nwriting workbook:")
        write_xlsx(xlsx_path, tables)

    if sheet_id is None:
        print("\n-- no sheet id, nothing published to Sheets --")
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


def watch(submissions: str, truth_path: str, every_minutes: float = 10.0,
          passes: int | None = None, **kwargs) -> dict:
    """Re-run `run` every `every_minutes` until you interrupt the cell.

    Any keyword `run` takes is passed straight through. Each pass is
    incremental, so a quiet pass costs one directory listing.

    A transient API error is printed and swallowed rather than ending the
    event -- an expired token or a Sheets hiccup at 14:00 should not mean the
    leaderboard is dead until someone notices. Interrupt (the stop button in
    Colab) to stop; `passes` caps the count instead, which is what the tests
    use.
    """
    tables: dict = {}
    n = 0
    print(f"auto-refresh every {every_minutes:g} min -- interrupt to stop")
    while passes is None or n < passes:
        n += 1
        print(f"\n===== pass {n} at {datetime.now().strftime('%H:%M:%S')} =====")
        ok = False
        try:
            tables = run(submissions, truth_path, **kwargs)
            ok = True
        except KeyboardInterrupt:
            print("\nstopped after the run.")
            break
        except Exception:
            traceback.print_exc()
            print("!! that pass failed -- carrying on to the next one")

        # only summarise a pass that actually ran, or a failure would report
        # the previous pass's numbers as if they were fresh
        accuracy = tables.get("accuracy") if ok else None
        if accuracy is not None and len(accuracy):
            print(f"    {len(accuracy)} pair(s) ranked, "
                  f"best MA-RAE {accuracy['MA-RAE'].min():.4f}")

        if passes is not None and n >= passes:
            break
        try:
            time.sleep(every_minutes * 60)
        except KeyboardInterrupt:
            print("\nstopped while waiting.")
            break
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
    ap.add_argument("--xlsx", default=None,
                    help="also write every tab to this .xlsx. Needs no "
                         "credentials -- drop it in the shared folder and "
                         "Drive previews it in the Sheets viewer.")
    ap.add_argument("--ledger", default=None,
                    help="where to remember what has been scored "
                         "(default: alongside --truth)")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--watch", type=float, default=None, metavar="MINUTES",
                    help="keep re-scoring and republishing this often, "
                         "until interrupted")
    ap.add_argument("--dry-run", action="store_true",
                    help="score and print, publish nothing")
    a = ap.parse_args()

    ledger = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.truth)),
                                      "leaderboard_ledger.csv")
    opts = dict(sheet_id=None if a.dry_run else a.sheet_id,
                ledger_path=ledger, budget=a.budget,
                xlsx_path=None if a.dry_run else a.xlsx)
    if a.watch:
        watch(a.submissions, a.truth, every_minutes=a.watch, **opts)
    else:
        run(a.submissions, a.truth, **opts)
