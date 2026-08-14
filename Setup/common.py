"""
common.py - shared utilities for the OpenADMET / ExpansionRx hackathon.

Every notebook starts with:

    !wget -q -O common.py <URL-to-this-file>
    import common
    common.setup(pair="your-pair-name")

Nothing in this module requires you to have run any other notebook.
Anything a notebook needs that is expensive to compute is downloaded
pre-built, so you can start from any notebook in any order.

--------------------------------------------------------------------------
INSTRUCTOR: three things to configure before the event. Search for SETUP.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import io
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

__version__ = "1.0"

# =========================================================================
# SETUP 1 of 3 -- data sources.
#
# Primary source is a pair of CSVs checked into this repo under Data/, or
# staged in the shared Drive folder by instructor/fetch_data.py. No network,
# no HuggingFace, no surprises on the day. The Hub is only a fallback if
# those files are missing.
# =========================================================================

TRAIN_CSV = "expansion_data_train.csv"
TEST_CSV = "expansion_data_test_blinded.csv"

# This file lives in <repo>/Setup/common.py, next to <repo>/Data and
# <repo>/Notebooks. Anchor on __file__ so the cwd doesn't matter.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DATA_DIR = os.path.join(REPO_ROOT, "Data")

TRAIN_REPO = "openadmet/openadmet-expansionrx-challenge-train-data"
TEST_REPO = "openadmet/openadmet-expansionrx-challenge-test-data-blinded"

# Fallback repo names tried in order if the primary is unreachable.
TRAIN_REPO_FALLBACKS = ["openadmet/openadmet-challenge-train-data"]
TEST_REPO_FALLBACKS = ["openadmet/openadmet-challenge-test-data-blinded"]

# =========================================================================
# SETUP 2 of 3 -- submission target.
#
# Either point LEADERBOARD_URL at a scoring endpoint, or leave it blank and
# let every pair write into SHARED_DRIVE_NAME, a Google Drive *shared drive*
# that all participants are members of (Contributor role is enough).
# The instructor script Setup/score_leaderboard.py reads SHARED_DIR.
#
# In Colab a shared drive is mounted at
#     /content/drive/Shareddrives/<name of the shared drive>
# ADMET_SHARED_DIR overrides all of this with an explicit path.
# =========================================================================

LEADERBOARD_URL = os.environ.get("ADMET_LEADERBOARD_URL", "")

#: name of the shared drive (or of a folder inside one) holding team folders
SHARED_DRIVE_NAME = "OpenADMET_TeamFolders"

SHARED_DIR = os.environ.get(
    "ADMET_SHARED_DIR", f"/content/drive/Shareddrives/{SHARED_DRIVE_NAME}"
)

# SETUP 3 of 3 -- how many leaderboard submissions each pair gets for the day.
SUBMISSION_BUDGET = 10

# =========================================================================
# Endpoints
# =========================================================================

#: raw assay column -> (needs log transform, unit multiplier, log-scale name)
CONVERSION = {
    "LogD":                          (False, 1.0,  "LogD"),
    "KSOL":                          (True,  1e-6, "LogS"),
    "HLM CLint":                     (True,  1.0,  "Log_HLM_CLint"),
    "MLM CLint":                     (True,  1.0,  "Log_MLM_CLint"),
    "Caco-2 Permeability Papp A>B":  (True,  1e-6, "Log_Caco_Papp_AB"),
    "Caco-2 Permeability Efflux":    (True,  1.0,  "Log_Caco_ER"),
    "MPPB":                          (True,  1.0,  "Log_Mouse_PPB"),
    "MBPB":                          (True,  1.0,  "Log_Mouse_BPB"),
    "MGMB":                          (True,  1.0,  "Log_Mouse_MPB"),
}

RAW_ENDPOINTS = list(CONVERSION.keys())
ENDPOINTS = [v[2] for v in CONVERSION.values()]  # log-scale names; model these
ID_COL = "Molecule Name"
SMILES_COL = "SMILES"

#: rough grouping by what the assay physically measures. Used as a starting
#: point for task-affinity grouping in card_multitask -- not gospel.
ENDPOINT_FAMILIES = {
    "physchem":       ["LogD", "LogS"],
    "metabolism":     ["Log_HLM_CLint", "Log_MLM_CLint"],
    "permeability":   ["Log_Caco_Papp_AB", "Log_Caco_ER"],
    "protein_binding": ["Log_Mouse_PPB", "Log_Mouse_BPB", "Log_Mouse_MPB"],
}


# =========================================================================
# Workspace
# =========================================================================

_STATE = {"pair": None, "workdir": None, "split": None}


def in_colab() -> bool:
    return "google.colab" in sys.modules


def setup(pair: str, mount_drive: bool = True, quiet: bool = False) -> str:
    """Create (or reattach to) this pair's folder in the shared Drive.

    Run this once at the top of every notebook. ONE folder holds everything
    you produce -- your split, your predictions, your submission files --
    which is how notebooks share results without needing to be run in any
    particular order. The repo you cloned is read-only and disappears when
    the runtime does; this folder is what persists.

    Returns the folder path.
    """
    pair = re.sub(r"[^A-Za-z0-9_-]+", "-", pair.strip().lower()).strip("-")
    if not pair:
        raise ValueError("Pick a pair name, e.g. setup(pair='beetroot')")

    if mount_drive and in_colab():
        try:
            from google.colab import drive  # type: ignore
            if not os.path.isdir("/content/drive/MyDrive"):
                drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Drive is best-effort
            warnings.warn(f"Could not mount Drive ({exc}).")

    base = _workspace_base()
    workdir = os.path.join(base, pair)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "predictions"), exist_ok=True)
    _STATE["pair"] = pair
    _STATE["workdir"] = workdir

    if not quiet:
        print(f"pair   : {pair}")
        print(f"folder : {workdir}")
        n = len(submissions_prepared())
        print(f"submission files on hand: {n}")
    return workdir


def _shared_dir_candidates() -> list[str]:
    """Every place SHARED_DRIVE_NAME could plausibly show up once Drive mounts.

    Colab mounts shared drives under /content/drive/Shareddrives/<drive name>.
    We also look one level in (in case it is a folder inside a bigger shared
    drive) and in MyDrive (in case someone made a shortcut instead).
    """
    import glob

    cands = []
    env = os.environ.get("ADMET_SHARED_DIR", "").strip()
    if env:
        cands.append(env)
    if SHARED_DIR:
        cands.append(SHARED_DIR)
    for root in ("/content/drive/Shareddrives", "/content/drive/Shared drives"):
        cands.append(os.path.join(root, SHARED_DRIVE_NAME))
        cands.extend(sorted(glob.glob(os.path.join(root, "*", SHARED_DRIVE_NAME))))
    cands.append(os.path.join("/content/drive/MyDrive", SHARED_DRIVE_NAME))

    out, seen = [], set()
    for c in cands:
        c = c.rstrip("/")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def shared_dir() -> str | None:
    """The shared drive folder holding every team folder, or None if unseen."""
    for cand in _shared_dir_candidates():
        if os.path.isdir(cand):
            return cand
    return None


def _workspace_base() -> str:
    """The shared Drive folder, or a local fallback if it is not reachable."""
    found = shared_dir()
    if found:
        return found
    local = os.path.abspath(os.environ.get("ADMET_HOME", "./admet_hackathon"))
    listing = ""
    root = "/content/drive/Shareddrives"
    if os.path.isdir(root):
        names = sorted(os.listdir(root))
        listing = ("   Shared drives visible to you right now: "
                   + (", ".join(names) if names else "(none)") + "\n")
    print(f"!! shared drive '{SHARED_DRIVE_NAME}' is not visible.\n"
          f"{listing}"
          f"   Falling back to {local}, which is LOST when this runtime\n"
          "   disconnects. Fix it in three steps, then re-run this cell:\n"
          "     1. open drive.google.com and click 'Shared drives' in the left\n"
          f"        sidebar -- you should see '{SHARED_DRIVE_NAME}' listed. If you\n"
          "        do not, ask an instructor to add your Google account to it as\n"
          "        a Contributor (a shared link is not enough).\n"
          "     2. in Colab, click the folder icon in the left sidebar and\n"
          "        'Mount Drive', and allow access when prompted.\n"
          "     3. re-run this cell.")
    return local


def submissions_dir(create: bool = True) -> str:
    """Where submission files go -- the same folder as everything else."""
    return workdir()


def workdir() -> str:
    if _STATE["workdir"] is None:
        raise RuntimeError("Call common.setup(pair='your-pair-name') first.")
    return _STATE["workdir"]


def pair_name() -> str:
    if _STATE["pair"] is None:
        raise RuntimeError("Call common.setup(pair='your-pair-name') first.")
    return _STATE["pair"]


def download_workspace(name: str = "workspace.zip") -> None:
    """Zip the workspace and download it (escape hatch if Drive failed)."""
    import shutil
    out = shutil.make_archive(name.replace(".zip", ""), "zip", workdir())
    if in_colab():
        from google.colab import files  # type: ignore
        files.download(out)
    print(f"wrote {out}")


# =========================================================================
# Data loading
# =========================================================================

_CACHE: dict = {}


def _resolve_csv(repo: str, prefer: str = "") -> str:
    """Find a CSV inside a HuggingFace dataset repo without hardcoding names."""
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV found in {repo}")
    if prefer:
        for f in files:
            if prefer.lower() in f.lower():
                return f
    return sorted(files, key=len)[0]


def _read_repo_csv(repos: list[str], prefer: str) -> pd.DataFrame:
    errors = []
    for repo in repos:
        try:
            fname = _resolve_csv(repo, prefer)
            return pd.read_csv(f"hf://datasets/{repo}/{fname}")
        except Exception as exc:
            errors.append(f"  {repo}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not load data from any known repo:\n" + "\n".join(errors) +
        "\n\nIf the dataset moved, edit TRAIN_REPO / TEST_REPO at the top of common.py."
    )


def data_dirs() -> list[str]:
    """Directories searched for the challenge CSVs, in priority order."""
    dirs = []
    override = os.environ.get("ADMET_DATA_DIR", "")
    if override:
        dirs.append(override)
    dirs += [
        REPO_DATA_DIR,                            # <repo>/Data  (this layout)
        os.path.join(os.getcwd(), "Data"),        # Data/ under the cwd
        os.getcwd(),                              # next to the notebook
    ]
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def data_path(filename: str) -> str:
    """Full path to a staged CSV.

    Returns the first location that actually holds the file; if none do,
    returns the preferred location (<repo>/Data/<filename>) so the caller
    can report a sensible path in its error message.
    """
    for d in data_dirs():
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(REPO_DATA_DIR, filename)


def _load_raw(kind: str) -> pd.DataFrame:
    """Local staged CSV first, HuggingFace Hub only as a fallback."""
    filename, repos, hint = (
        (TRAIN_CSV, [TRAIN_REPO] + TRAIN_REPO_FALLBACKS, "train")
        if kind == "train" else
        (TEST_CSV, [TEST_REPO] + TEST_REPO_FALLBACKS, "test"))

    local = data_path(filename)
    if os.path.exists(local):
        return pd.read_csv(local)
    print(f"{filename} not found in {', '.join(data_dirs())}; "
          "falling back to HuggingFace...")
    return _read_repo_csv(repos, hint)


def load_train(log_scale: bool = True) -> pd.DataFrame:
    """Training set. log_scale=True returns the transformed columns you model."""
    key = ("train", log_scale)
    if key not in _CACHE:
        raw = _load_raw("train")
        _CACHE[("train", False)] = raw
        _CACHE[("train", True)] = to_log_scale(raw)
    return _CACHE[key].copy()


def load_test(log_scale: bool = True) -> pd.DataFrame:
    """Blinded test set: molecules only, endpoint columns are empty."""
    key = ("test", log_scale)
    if key not in _CACHE:
        raw = _load_raw("test")
        for c in RAW_ENDPOINTS:
            if c not in raw.columns:
                raw[c] = np.nan
        _CACHE[("test", False)] = raw
        _CACHE[("test", True)] = to_log_scale(raw)
    return _CACHE[key].copy()


# =========================================================================
# Log-scale conversion
#
# NOTE: this reproduces the transform used in the official OpenADMET
# tutorial, including the "+1 before scaling" zero-guard. That guard is
# defensible but arbitrary, and it is NOT applied consistently across
# endpoints with different units. See 01_eda.ipynb.
# =========================================================================

def to_log_scale(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in (ID_COL, SMILES_COL) if c in df.columns]].copy()
    for raw, (needs_log, mult, name) in CONVERSION.items():
        if raw not in df.columns:
            out[name] = np.nan
            continue
        v = pd.to_numeric(df[raw], errors="coerce").astype(float)
        if needs_log:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log10((v + 1.0) * mult)
        out[name] = v
    return out


def from_log_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Invert to_log_scale, back to raw assay units."""
    out = df[[c for c in (ID_COL, SMILES_COL) if c in df.columns]].copy()
    for raw, (needs_log, mult, name) in CONVERSION.items():
        if name not in df.columns:
            continue
        v = pd.to_numeric(df[name], errors="coerce").astype(float)
        out[raw] = (10.0 ** v) / mult - 1.0 if needs_log else v
    return out


# =========================================================================
# Metrics
# =========================================================================

def rae(y_true, y_pred) -> float:
    """Relative absolute error: total absolute error divided by the total
    absolute error of a model that always predicts the mean OF y_true.

    RAE = 1.0  means "no better than guessing the average of the scored set"
    RAE = 0.5  means "half the error of guessing" (roughly the winning entry)
    RAE > 1.0  means "worse than guessing"

    Note the denominator uses the mean of the set being scored. A model that
    predicts the *training* mean therefore scores exactly 1.0 only in-sample,
    or on a holdout drawn from the same distribution. On a temporal holdout it
    scores above 1.0, and the excess measures how far the data drifted -- see
    00_start_here.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan
    y_true, y_pred = y_true[m], y_pred[m]
    denom = np.abs(y_true - y_true.mean()).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / denom)


def _safe_corr(y_true, y_pred, kind: str) -> float:
    from scipy import stats
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    fn = stats.spearmanr if kind == "spearman" else stats.kendalltau
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(y_true[m], y_pred[m])[0])


def evaluate(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame,
             endpoints: list[str] | None = None) -> pd.DataFrame:
    """Per-endpoint MAE / RAE / R2 / Spearman / Kendall.

    Both frames need a 'Molecule Name' column; they are aligned on it, so
    row order does not matter. Missing truth values are skipped per endpoint.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    endpoints = endpoints or [e for e in ENDPOINTS if e in y_true_df.columns]
    t = y_true_df.set_index(ID_COL)
    p = y_pred_df.set_index(ID_COL)
    shared = t.index.intersection(p.index)
    if len(shared) == 0:
        raise ValueError("No overlapping molecules between truth and predictions.")
    t, p = t.loc[shared], p.loc[shared]

    rows = []
    for e in endpoints:
        if e not in p.columns:
            rows.append({"endpoint": e, "n": 0, "MAE": np.nan, "RAE": np.nan,
                         "R2": np.nan, "Spearman": np.nan, "Kendall": np.nan})
            continue
        yt, yp = t[e].to_numpy(float), p[e].to_numpy(float)
        m = np.isfinite(yt) & np.isfinite(yp)
        if m.sum() < 2:
            rows.append({"endpoint": e, "n": int(m.sum()), "MAE": np.nan, "RAE": np.nan,
                         "R2": np.nan, "Spearman": np.nan, "Kendall": np.nan})
            continue
        rows.append({
            "endpoint": e,
            "n": int(m.sum()),
            "MAE": float(mean_absolute_error(yt[m], yp[m])),
            "RAE": rae(yt[m], yp[m]),
            "R2": float(r2_score(yt[m], yp[m])),
            "Spearman": _safe_corr(yt[m], yp[m], "spearman"),
            "Kendall": _safe_corr(yt[m], yp[m], "kendall"),
        })
    out = pd.DataFrame(rows).set_index("endpoint")
    return out


def ma_rae(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
    """The leaderboard metric: RAE averaged equally over all nine endpoints.

    Averaging RAE rather than MAE is what stops LogD and KSOL -- which have
    the most data and the widest ranges -- from dominating the score.
    """
    return float(evaluate(y_true_df, y_pred_df)["RAE"].mean(skipna=True))


# =========================================================================
# Splits
# =========================================================================

def random_split(df: pd.DataFrame, frac_val: float = 0.2, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    pick = rng.random(len(df)) < frac_val
    return pd.Series(np.where(pick, "val", "train"), index=df.index)

SPLITTERS = {
    "random": random_split
}

DEFAULT_SPLIT_SEED = int(os.environ.get("ADMET_DEFAULT_SPLIT_SEED", "0"))
DEFAULT_FRAC_VAL = 0.2


def make_split(df: pd.DataFrame, method: str, frac_val: float = DEFAULT_FRAC_VAL,
               seed: int = DEFAULT_SPLIT_SEED) -> pd.Series:
    """Run a named splitter, passing only the kwargs it actually accepts."""
    try:
        fn = SPLITTERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown split method {method!r}. "
            f"Pick one of: {', '.join(sorted(SPLITTERS))}"
        ) from None
    kwargs = {"frac_val": frac_val}
    if "seed" in inspect.signature(fn).parameters:
        kwargs["seed"] = seed
    return fn(df, **kwargs)


def check_split(fold, df: pd.DataFrame) -> pd.Series:
    """Validate a split before you spend five minutes training on it.

    Catches the three things that actually go wrong: wrong length, an index
    that does not line up with df, and labels that are not "train"/"val".
    Returns the split as a Series so you can chain it.

    "unused" is a legal third label, for splits that hold some molecules out
    of both sides -- rolling-origin cross-validation, where a fold must not
    train on compounds registered after its validation block.
    """
    fold = pd.Series(fold) if not isinstance(fold, pd.Series) else fold
    if len(fold) != len(df):
        raise ValueError(
            f"split has {len(fold)} labels but df has {len(df)} rows")
    if not fold.index.equals(df.index):
        raise ValueError(
            "split index does not match df -- return a Series with "
            "index=df.index (a fresh index from .reset_index() will not align)")
    bad = sorted(set(fold.dropna().unique()) - {"train", "val", "unused"})
    if bad:
        raise ValueError(
            f'labels must be "train", "val" or "unused"; found {bad}')
    if fold.isna().any():
        raise ValueError(f"{int(fold.isna().sum())} rows have no label")
    n_train = int((fold == "train").sum())
    n_val = int((fold == "val").sum())
    if n_train == 0 or n_val == 0:
        raise ValueError(
            f"a split needs both sides; got {n_train} train / {n_val} val")
    return fold


def splits_dir(create: bool = True) -> str:
    """Your library of saved splits: <folder>/splits."""
    d = os.path.join(workdir(), "splits")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _split_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name).strip().lower()).strip("-")
    if not slug:
        raise ValueError("Give the split a name, e.g. name='top-logD-decile'")
    return slug


def save_split(split: pd.Series, df: pd.DataFrame, method: str,
               name: str = "") -> str:
    """Add a split to your library under a name you choose.

    Saving does not make it "the" split -- there is no current split. Every
    time you train, you pick one by name with load_split(name=...). Save as
    many as you like and compare them.

    We save the ASSIGNMENT (molecule -> fold), not the recipe. That way a
    split you invented yourself ports just as well as a built-in one.

    IMPORTANT: folds are assigned to every training molecule, including ones
    with no label for a given endpoint. Each notebook filters down to the
    molecules it can actually use. Doing it the other way round -- splitting
    per endpoint -- silently validates single-task and multitask models on
    different molecules and makes their scores incomparable.
    """
    split = check_split(split, df)
    slug = _split_slug(name or method)
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), "fold": np.asarray(split)})
    path = os.path.join(splits_dir(), slug + ".csv")
    out.to_csv(path, index=False)
    meta = {"name": slug, "method": method,
            "n_train": int((out["fold"] == "train").sum()),
            "n_val": int((out["fold"] == "val").sum()),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path.replace(".csv", ".json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved split '{slug}': {meta['n_train']} train / {meta['n_val']} val")
    return path


def list_splits() -> pd.DataFrame:
    """Every split you have saved today, newest last."""
    cols = ["name", "method", "n_train", "n_val", "saved_at"]
    d = os.path.join(_STATE["workdir"] or ".", "splits")
    if not os.path.isdir(d):
        return pd.DataFrame(columns=cols)
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        meta = {"name": f[:-4]}
        try:
            with open(os.path.join(d, f.replace(".csv", ".json"))) as fh:
                meta.update(json.load(fh))
        except Exception:
            pass
        rows.append({c: meta.get(c) for c in cols})
    return pd.DataFrame(rows, columns=cols).sort_values("saved_at").reset_index(drop=True)


def saved_split_names() -> list[str]:
    return list_splits()["name"].tolist()


def split_metadata() -> dict:
    """Metadata for the split you most recently loaded this session."""
    return dict(_STATE["split"] or {})


def load_split(df: pd.DataFrame | None = None, name: str | None = None,
               verbose: bool = True,
               frac_val: float = DEFAULT_FRAC_VAL,
               seed: int | None = None):
    """Pick the split to train against, by name. There is no default.

    `name` is either one of your saved splits (see list_splits()) or one of
    the few built-in recipes in SPLITTERS, computed on the spot. Saved splits
    win if the names collide. There is deliberately no default: which split
    you validate against is the decision this notebook is about.

    Returns (fold_series_aligned_to_df, metadata_dict).
    """
    if name is None:
        raise ValueError(
            "Say which split to use, e.g. load_split(train, name='random').\n"
            f"  saved     : {', '.join(saved_split_names()) or '(none yet)'}\n"
            f"  built-ins : {', '.join(sorted(SPLITTERS))}")

    slug = _split_slug(name)
    path = os.path.join(splits_dir(create=False), slug + ".csv")
    if not os.path.exists(path) and name not in SPLITTERS:
        raise ValueError(
            f"No split called {name!r}.\n"
            f"  saved     : {', '.join(saved_split_names()) or '(none yet)'}\n"
            f"  built-ins : {', '.join(sorted(SPLITTERS))}")

    df = load_train() if df is None else df

    if os.path.exists(path):
        saved = pd.read_csv(path).drop_duplicates(ID_COL).set_index(ID_COL)["fold"]
        fold = df[ID_COL].map(saved)
        missing = int(fold.isna().sum())
        if missing:
            warnings.warn(f"{missing} molecules are not in saved split "
                          f"'{slug}'; treating them as train.")
            fold = fold.fillna("train")
        meta = {"name": slug, "method": slug}
        try:
            with open(path.replace(".csv", ".json")) as fh:
                meta.update(json.load(fh))
        except Exception:
            pass
        if verbose:
            print(f"using your saved split '{slug}' "
                  f"({(fold == 'val').sum()} val molecules)")
    else:
        seed = DEFAULT_SPLIT_SEED if seed is None else seed
        fold = make_split(df, name, frac_val=frac_val, seed=seed)
        meta = {"name": name, "method": name, "frac_val": frac_val}
        if "seed" in inspect.signature(SPLITTERS[name]).parameters:
            meta["seed"] = seed
        if verbose:
            print(f"using the built-in {name} split "
                  f"({(fold == 'val').sum()} val molecules) -- "
                  "save_split() it if you want to keep it.")

    _STATE["split"] = meta
    return fold.reset_index(drop=True), meta


# =========================================================================
# Predictions
#
# There is one way to save a prediction vector: prepare_submission(). It writes
# a single self-describing file, and read_submission() reads it back, which is
# how a later notebook picks up work from an earlier one.
# =========================================================================

def check_predictions(pred: pd.DataFrame, against: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate shape before you burn a submission on a malformed file."""
    if ID_COL not in pred.columns:
        raise ValueError(f"predictions must have a '{ID_COL}' column")
    missing = [e for e in ENDPOINTS if e not in pred.columns]
    if missing:
        raise ValueError(f"missing endpoint columns: {missing}")
    if pred[ID_COL].duplicated().any():
        raise ValueError("duplicate molecules in predictions")
    if against is not None:
        gap = set(against[ID_COL]) - set(pred[ID_COL])
        if gap:
            raise ValueError(f"{len(gap)} test molecules have no prediction, "
                             f"e.g. {sorted(gap)[:3]}")
    n_nan = int(pred[ENDPOINTS].isna().sum().sum())
    if n_nan:
        warnings.warn(f"{n_nan} predictions are NaN and will score as a miss.")
    return pred[[ID_COL] + ENDPOINTS]


# =========================================================================
# Scores -- one row per (model, split), accumulated across the whole day
# =========================================================================

#: how many splits to score a single model against. More than this and the
#: table stops being something you can read and reason about at 4pm.
EVAL_SPLIT_CAP = 4

SCORE_COLS = ["model", "split", "ma_rae", "note", "logged_at"]


def known_split_names() -> list[str]:
    """Everything load_split() will accept: your library plus the built-ins."""
    return saved_split_names() + [s for s in SPLITTERS
                                  if s not in saved_split_names()]


def scored_split_names() -> list[str]:
    """Anything you have logged a score against, including CV scheme names."""
    scores = list_scores()
    if scores.empty:
        return []
    return sorted(scores["split"].dropna().astype(str).unique().tolist())


def check_submission_split(split: str) -> str:
    """Validate the ONE split a submission is betting on.

    Looser than check_eval_splits on purpose: a submission may cite a CV
    scheme ("cv-cluster") that is not itself a loadable split, as long as you
    logged a score against it. It only has to be a thing you can point at.
    """
    split = str(split).strip()
    if not split:
        raise ValueError(
            "Name the split your estimate came from -- that is the hypothesis "
            "you are betting on.")
    known = known_split_names() + scored_split_names()
    if split not in known:
        raise ValueError(
            f"Nothing called {split!r} in your splits or your score table.\n"
            f"  splits : {', '.join(known_split_names()) or '(none yet)'}\n"
            f"  scored : {', '.join(scored_split_names()) or '(none yet)'}")
    return split


def check_eval_splits(names) -> list[str]:
    """Validate the handful of splits you are scoring a model against.

    Deliberately capped: comparing one model on four splits tells you
    something, on twelve it tells you nothing you will actually read.

    These names have to be loadable, since you are about to train against
    them. A submission is checked more loosely -- see check_submission_split.
    """
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("Pick at least one split to score against.")
    if len(names) > EVAL_SPLIT_CAP:
        raise ValueError(
            f"{len(names)} splits is too many -- pick at most {EVAL_SPLIT_CAP}. "
            "Keep the one you trust most and drop the rest; you can always "
            "come back and score against another.")
    known = known_split_names()
    unknown = [n for n in names if _split_slug(n) not in known and n not in SPLITTERS]
    if unknown:
        raise ValueError(
            f"No split called {unknown}. You have: "
            f"{', '.join(known) or '(none yet)'}")
    return names


def log_score(model: str, split: str, ma_rae: float, note: str = "") -> str:
    """Record what one model scored on one split.

    Re-logging the same (model, split) overwrites the old row, so re-running a
    cell does not litter the table.
    """
    scores = list_scores()
    keep = scores[~((scores["model"] == model) & (scores["split"] == split))]
    row = {"model": model, "split": split, "ma_rae": float(ma_rae),
           "note": note,
           "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = pd.concat([keep, pd.DataFrame([row])], ignore_index=True)
    path = os.path.join(workdir(), "scores.csv")
    out.to_csv(path, index=False)
    print(f"logged {model} on '{split}': MA-RAE = {ma_rae:.3f}")
    return path


def score(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame,
          model: str | None = None, split: str | None = None,
          endpoints: list[str] | None = None, note: str = "") -> pd.DataFrame:
    """Per-endpoint MAE / RAE / R2 / Spearman / Kendall -- and log the MA-RAE.

    Name the model and the split and it records one row in your score table,
    which is what builds up common.score_matrix() over the day:

        metrics = common.score(truth, preds, "lgbm-rdkit", SPLIT)

    Leave them out and it just computes, for a one-off look at numbers you do
    not want in the table:

        common.score(train_truth, train_preds)
    """
    if isinstance(model, (list, tuple, set)):
        raise TypeError(
            "score()'s third argument is the MODEL NAME, not the endpoints. "
            f"You passed {model!r}.\n"
            "  to score a subset : common.score(truth, preds, endpoints=[...])\n"
            "  to log a result   : common.score(truth, preds, \"my-model\", SPLIT)")

    metrics = evaluate(y_true_df, y_pred_df, endpoints)
    if model and split:
        scored = list(endpoints) if endpoints else ENDPOINTS
        if len(scored) < len(ENDPOINTS):
            # MA-RAE is the average over ALL nine endpoints. A model scored on
            # fewer has an average over a different, easier or harder set --
            # the two numbers are not comparable, and the leaderboard cannot
            # rank a partial model against a complete one.
            note = (note + f" [partial: {len(scored)}/{len(ENDPOINTS)} endpoints]").strip()
            warnings.warn(
                f"{model!r} was scored on {len(scored)} of {len(ENDPOINTS)} "
                "endpoints. Its MA-RAE averages over those only, so it is not "
                "comparable with a full model and will NOT count toward the "
                "MA-RAE leaderboard. Useful for the per-endpoint boards, and "
                "for comparing against another partial model on the same "
                "endpoints.")
        log_score(model, split, metrics["RAE"].mean(), note=note)
    return metrics


def list_scores() -> pd.DataFrame:
    """Every (model, split) score you have logged today."""
    path = os.path.join(_STATE["workdir"] or ".", "scores.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=SCORE_COLS)
    return pd.read_csv(path)


def score_matrix() -> pd.DataFrame:
    """Models down the side, splits across the top. The 4pm view."""
    s = list_scores()
    if s.empty:
        return pd.DataFrame()
    return s.pivot_table(index="model", columns="split", values="ma_rae")


# =========================================================================
# Submission
# =========================================================================

def submissions_prepared() -> pd.DataFrame:
    """Every submission file you have written, newest last.

    Preparing is free -- this is not a count of what you have spent. What you
    spend is what you drag into the Scored folder.
    """
    cols = ["model", "file", "prepared_at", "expected_ma_rae", "split", "why"]
    d = _STATE["workdir"] or "."
    rows = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not (f.startswith(f"{_STATE['pair']}__") and f.endswith(".csv")):
            continue
        path = os.path.join(d, f)
        _, meta = read_submission(path)
        meta["file"] = f
        row = {c: meta.get(c) for c in cols}
        row["path"] = path          # read_submission(path) to get the numbers back
        rows.append(row)
    return pd.DataFrame(rows, columns=cols + ["path"])


#: metadata lines at the top of a submission start with this.
META_PREFIX = "#"


def write_submission(path: str, pred: pd.DataFrame, record: dict) -> str:
    """Write ONE file: metadata as comment lines, then the predictions.

        # pair: beetroot
        # submitted_at: 20260804T221531Z
        ...
        Molecule Name,LogD,LogS,...

    Every metadata line starts with '#', so the whole thing still reads as a
    normal CSV: pd.read_csv(path, comment='#').
    """
    with open(path, "w") as fh:
        fh.write(_submission_text(pred, record))
    return path


def _submission_text(pred: pd.DataFrame, record: dict) -> str:
    # newlines in a value would break the one-line-per-key format
    header = "".join(f"{META_PREFIX} {k}: {str(v).replace(chr(10), ' ')}\n"
                     for k, v in record.items())
    buf = io.StringIO()
    pred.to_csv(buf, index=False)
    return header + buf.getvalue()


def read_submission(path: str) -> tuple[pd.DataFrame, dict]:
    """Inverse of write_submission: (predictions, metadata)."""
    record = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(META_PREFIX):
                break
            key, _, value = line[len(META_PREFIX):].strip().partition(":")
            record[key.strip()] = value.strip()
    return pd.read_csv(path, comment=META_PREFIX), record


def prepare_submission(pred: pd.DataFrame, model: str, split: str,
                       why: str | None = None) -> str:
    """Write one submission file into your folder in the shared Drive.

    Returns the path. Writing the file is ALL this does, and it is free --
    prepare as many as you like. A submission only counts once you drag the
    file into the Scored folder, and you get SUBMISSION_BUDGET of those for
    the whole day. With a budget that small you cannot use the leaderboard as
    a validation set, so your internal validation has to be honest.

    model : the name you logged this model under with score_and_log.
    split : which split you are betting on. You may have scored this model
            against four; a submission commits to one. There is no default --
            naming it is the point, it is the hypothesis you are betting on.
    why   : optional one line on what is different about this model. Defaults
            to the note you attached when you logged the score, and may be
            left empty.

    The predicted MA-RAE is not an argument: it is read from your score table,
    so you cannot submit a number you never measured.
    """
    split = check_submission_split(split)
    pred = check_predictions(pred, against=load_test())

    scores = list_scores()
    row = scores[(scores["model"] == model) & (scores["split"] == split)]
    if row.empty:
        mine = scores[scores["model"] == model]
        if mine.empty:
            raise ValueError(
                f"No score logged for {model!r}. Run score_and_log() first -- "
                "the number you submit has to be one you measured.\n"
                f"You have logged: {', '.join(sorted(scores['model'].unique())) or '(nothing yet)'}")
        have = "\n".join(f"  {r.split:<14s} {r.ma_rae:.3f}"
                          for r in mine.itertuples())
        raise ValueError(
            f"{model!r} has no score on {split!r}. It does have:\n{have}")

    expected_ma_rae = float(row["ma_rae"].iloc[0])
    if why is None:
        note = row["note"].iloc[0]
        why = "" if pd.isna(note) else str(note)   # an empty note reads back as NaN

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"pair": pair_name(), "model": model,
              "prepared_at": stamp,
              "expected_ma_rae": expected_ma_rae, "why": str(why).strip(),
              "split": split, "n": int(len(pred))}

    stem = f"{pair_name()}__{stamp}"
    path = os.path.join(submissions_dir(), stem + ".csv")
    n = 2                       # two prepares in the same second must not collide
    while os.path.exists(path):
        path = os.path.join(submissions_dir(), f"{stem}-{n}.csv")
        n += 1
    write_submission(path, pred, record)
    print(f"wrote your submission file:\n  {path}")
    print(f"you predicted MA-RAE = {expected_ma_rae:.3f} (from '{split}').")
    print(f"Nothing has been submitted yet -- drag this file into the Scored "
          f"folder to spend one of your {SUBMISSION_BUDGET} submissions.")
    return path


def submit(pred: pd.DataFrame, model: str, split: str,
           why: str | None = None) -> str:
    """prepare_submission(), plus a POST if a leaderboard endpoint is configured.

    With no LEADERBOARD_URL set -- the drag-it-across workflow -- this is
    exactly prepare_submission().
    """
    path = prepare_submission(pred, model, split, why)
    if LEADERBOARD_URL:
        try:
            import requests
            with open(path) as fh:
                body = fh.read()          # same single file, metadata included
            r = requests.post(LEADERBOARD_URL,
                              files={"predictions": (os.path.basename(path), body)},
                              timeout=60)
            r.raise_for_status()
            print(r.text[:500])
        except Exception as exc:
            warnings.warn(
                f"Leaderboard POST failed ({exc}). Your file is still in your "
                f"shared-drive folder: {path}")
    return path


def download_submission(path: str) -> None:
    """Download a submission file to your laptop (Colab only)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        from google.colab import files  # type: ignore
        files.download(path)
    except ImportError:
        print(f"Not in Colab -- your file is already on this machine:\n  {path}")


# =========================================================================
# Chemistry helpers
# =========================================================================

def fingerprints_to_array(fps) -> np.ndarray:
    from rdkit import DataStructs
    n_bits = len(next(f for f in fps if f is not None))
    arr = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def nearest_neighbour_similarity(query_fps, reference_fps,
                                 exclude_self: bool = False) -> np.ndarray:
    """For each query molecule, the Tanimoto similarity to its closest
    neighbour in the reference set.
    """
    from rdkit import DataStructs
    ref = [f for f in reference_fps if f is not None]
    out = np.full(len(query_fps), np.nan)
    for i, fp in enumerate(query_fps):
        if fp is None or not ref:
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref))
        if exclude_self:
            j = int(np.argmax(sims))
            if sims[j] >= 0.999:
                sims = np.delete(sims, j)
        out[i] = sims.max() if len(sims) else np.nan
    return out


MORGAN2_NOISE_FLOOR = 0.27  # Landrum 2021: below this, a "neighbour" is noise


def rdkit_descriptors(smiles) -> pd.DataFrame:
    """~200 fast RDKit physicochemical descriptors. Seconds, not minutes."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    names = [n for n, _ in Descriptors.descList]
    calc = Descriptors.CalcMolDescriptors
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rows.append({n: np.nan for n in names})
        else:
            try:
                rows.append(calc(mol))
            except Exception:
                rows.append({n: np.nan for n in names})
    out = pd.DataFrame(rows, columns=names).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def clean_features(X_train: pd.DataFrame, *others: pd.DataFrame):
    """Drop all-NaN / constant columns, median-impute the rest.

    Fit on train only -- imputing with statistics from your validation set is
    a quiet way to leak information and flatter your score.
    """
    keep = X_train.columns[(X_train.notna().any()) & (X_train.nunique(dropna=True) > 1)]
    medians = X_train[keep].median()
    out = [X_train[keep].fillna(medians)]
    for o in others:
        out.append(o.reindex(columns=keep).fillna(medians))
    return out[0] if not others else tuple(out)


def predict_mean_baseline(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """The null model: predict each endpoint's training mean for every molecule.

    Scores exactly 1.0 in-sample and ~1.0 on a random holdout. On a temporal
    holdout it scores ABOVE 1.0, because the RAE denominator uses the held-out
    set's own mean -- the excess is a direct measurement of distribution drift
    between early and late compounds. See 00_start_here.

    Whatever it scores is the bar. Anything you build should beat it; if it
    does not, something is broken, and knowing that early is cheap.
    """
    pred = pd.DataFrame({ID_COL: target[ID_COL].to_numpy()})
    for endpoint in ENDPOINTS:
        pred[endpoint] = float(train[endpoint].mean(skipna=True))
    return pred
