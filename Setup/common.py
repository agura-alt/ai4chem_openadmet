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

# Pre-built artifacts (descriptors, CheMeleon embeddings, harmonized external
# data). Normally these sit in an `artifacts/` subfolder of the shared,
# read-only Drive folder that also holds the notebooks -- no web host needed.
# ARTIFACT_BASE is an optional HTTP fallback. See instructor/SETUP.md.
MATERIALS_DIR = os.environ.get(
    "ADMET_MATERIALS_DIR", "/content/drive/MyDrive/AI4Chem_ADMET_Hackathon")
ARTIFACT_BASE = os.environ.get("ADMET_ARTIFACT_BASE", "")

# =========================================================================
# SETUP 2 of 3 -- submission target.
#
# Either point LEADERBOARD_URL at a scoring endpoint, or leave it blank and
# set SHARED_DIR to a Google Drive folder shared with all participants.
# The instructor script instructor/score_leaderboard.py reads SHARED_DIR.
# =========================================================================

LEADERBOARD_URL = os.environ.get("ADMET_LEADERBOARD_URL", "")
SHARED_DIR = os.environ.get("ADMET_SHARED_DIR", "/content/drive/MyDrive/admet_hackathon_submissions")

# SETUP 3 of 3 -- how many leaderboard submissions each pair gets for the day.
SUBMISSION_BUDGET = 6

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

_STATE = {"pair": None, "workdir": None}


def in_colab() -> bool:
    return "google.colab" in sys.modules


def setup(pair: str, mount_drive: bool = True, quiet: bool = False) -> str:
    """Create (or reattach to) this pair's persistent workspace.

    Run this once at the top of every notebook. Everything you produce --
    your split, your cached features, your predictions -- lands in this one
    folder, which is how notebooks share results without needing to be run
    in any particular order.

    Returns the workspace path.
    """
    pair = re.sub(r"[^A-Za-z0-9_-]+", "-", pair.strip().lower()).strip("-")
    if not pair:
        raise ValueError("Pick a pair name, e.g. setup(pair='beetroot')")

    if mount_drive and in_colab():
        try:
            from google.colab import drive  # type: ignore
            if not os.path.isdir("/content/drive/MyDrive"):
                drive.mount("/content/drive")
            base = "/content/drive/MyDrive/admet_hackathon"
        except Exception as exc:  # pragma: no cover - Drive is best-effort
            warnings.warn(
                f"Could not mount Drive ({exc}). Falling back to local storage, "
                "which is LOST when this runtime disconnects. Use "
                "common.download_workspace() before you close the tab."
            )
            base = "/content/admet_hackathon"
    else:
        base = os.path.abspath(os.environ.get("ADMET_HOME", "./admet_hackathon"))

    workdir = os.path.join(base, pair)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "predictions"), exist_ok=True)
    _STATE["pair"] = pair
    _STATE["workdir"] = workdir
    if not quiet:
        print(f"pair      : {pair}")
        print(f"workspace : {workdir}")
        n = len(list_predictions())
        print(f"prediction files on hand: {n}")
    return workdir


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
        os.path.join(MATERIALS_DIR, "data"),      # shared Drive folder
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


def _read_table(src: str) -> pd.DataFrame:
    return pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)


def load_artifact(name: str) -> pd.DataFrame:
    """Load a pre-built artifact (descriptors, embeddings, external data).

    Looks in this repo and the shared Drive folder first, then an optional
    HTTP fallback. These are precomputed so that a dead Colab runtime never
    costs you an hour.
    """
    looked = []
    for d in data_dirs():
        for candidate in (os.path.join(d, "artifacts", name),
                          os.path.join(d, name)):
            looked.append(candidate)
            if os.path.exists(candidate):
                return _read_table(candidate)
    if ARTIFACT_BASE:
        return _read_table(ARTIFACT_BASE.rstrip("/") + "/" + name)
    raise RuntimeError(
        f"Artifact '{name}' not found.\n"
        "Looked in:\n  " + "\n  ".join(looked) + "\n"
        "Ask an instructor, or compute it yourself with the (slower) code in "
        "the notebook."
    )


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

def _registration_number(names: pd.Series) -> pd.Series:
    """Compound IDs look like 'E-0001321'. The number increases over time as
    compounds are registered, so it is a usable proxy for synthesis date --
    which is what lets us build a temporal split without a date column.
    """
    return names.astype(str).str.extract(r"(\d+)", expand=False).astype(float)


def temporal_split(df: pd.DataFrame, frac_val: float = 0.2) -> pd.Series:
    order = _registration_number(df[ID_COL]).rank(method="first", na_option="bottom")
    cutoff = order.quantile(1.0 - frac_val)
    return pd.Series(np.where(order > cutoff, "val", "train"), index=df.index)


def random_split(df: pd.DataFrame, frac_val: float = 0.2, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    pick = rng.random(len(df)) < frac_val
    return pd.Series(np.where(pick, "val", "train"), index=df.index)


def scaffold_split(df: pd.DataFrame, frac_val: float = 0.2, seed: int = 0) -> pd.Series:
    """Bemis-Murcko scaffold split: whole scaffold groups go to one side, so
    the validation set contains chemical series the model has never seen.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(df[SMILES_COL]):
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smi), includeChirality=False)
        except Exception:
            s = ""
        scaffolds.setdefault(s or f"__unparsed_{i}", []).append(i)

    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)  # break ties between equally sized scaffolds
    groups = sorted(groups, key=len, reverse=True)

    target = int(round(frac_val * len(df)))
    labels = np.array(["train"] * len(df), dtype=object)
    n_val = 0
    for g in groups:
        if n_val + len(g) <= target:
            labels[g] = "val"
            n_val += len(g)
    return pd.Series(labels, index=df.index)


def similarity_split(df: pd.DataFrame, frac_val: float = 0.2,
                     radius: int = 2, n_bits: int = 2048) -> pd.Series:
    """Hold out the molecules that are LEAST similar to everything else.

    A deliberately pessimistic split: it asks how the model behaves on the
    edge of its own chemical space, which is the situation you are actually
    in when you score a newly designed compound.
    """
    fps = morgan_fingerprints(df[SMILES_COL], radius=radius, n_bits=n_bits)
    sim = nearest_neighbour_similarity(fps, fps, exclude_self=True)
    cutoff = np.nanquantile(sim, frac_val)
    return pd.Series(np.where(sim <= cutoff, "val", "train"), index=df.index)


SPLITTERS = {
    "temporal": temporal_split,
    "random": random_split,
    "scaffold": scaffold_split,
    "similarity": similarity_split,
}


def save_split(split: pd.Series, df: pd.DataFrame, method: str,
               rationale: str = "", val_score: float | None = None) -> str:
    """Freeze a train/val assignment so every other notebook can reuse it.

    We save the ASSIGNMENT (molecule -> fold), not the recipe. That way a
    split you invented yourself ports just as well as a built-in one.

    IMPORTANT: folds are assigned to every training molecule, including ones
    with no label for a given endpoint. Each notebook filters down to the
    molecules it can actually use. Doing it the other way round -- splitting
    per endpoint -- silently validates single-task and multitask models on
    different molecules and makes their scores incomparable.
    """
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), "fold": np.asarray(split)})
    path = os.path.join(workdir(), "split.csv")
    out.to_csv(path, index=False)
    meta = {"method": method, "rationale": rationale, "val_score": val_score,
            "n_train": int((out["fold"] == "train").sum()),
            "n_val": int((out["fold"] == "val").sum()),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(workdir(), "split_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved split '{method}': {meta['n_train']} train / {meta['n_val']} val")
    return path


def split_metadata() -> dict:
    """Just the metadata for this pair's split -- no data download needed."""
    try:
        with open(os.path.join(workdir(), "split_meta.json")) as fh:
            return json.load(fh)
    except Exception:
        return {"method": "temporal (default)", "rationale": "", "val_score": None}


def load_split(df: pd.DataFrame | None = None, verbose: bool = True):
    """Load this pair's saved split, or fall back to the default temporal one.

    Never raises just because you skipped 02_validation.
    Returns (fold_series_aligned_to_df, metadata_dict).
    """
    df = load_train() if df is None else df
    path = os.path.join(workdir(), "split.csv")
    if os.path.exists(path):
        saved = pd.read_csv(path).drop_duplicates(ID_COL).set_index(ID_COL)["fold"]
        fold = df[ID_COL].map(saved).fillna("train")
        try:
            with open(os.path.join(workdir(), "split_meta.json")) as fh:
                meta = json.load(fh)
        except Exception:
            meta = {"method": "saved"}
        if verbose:
            print(f"using YOUR split: {meta.get('method')} "
                  f"({(fold == 'val').sum()} val molecules)")
        return fold.reset_index(drop=True), meta
    fold = temporal_split(df)
    if verbose:
        print("no saved split found -- using the default temporal split. "
              "Run 02_validation.ipynb to choose your own.")
    return fold, {"method": "temporal (default)", "rationale": "", "val_score": None}


# =========================================================================
# Predictions
# =========================================================================

def blank_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy()})
    for e in ENDPOINTS:
        out[e] = np.nan
    return out


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


def save_predictions(pred: pd.DataFrame, name: str, note: str = "") -> str:
    """Save predictions into your workspace under a name you choose.

    Every notebook writes this same format, which is the only reason the
    ensemble card can pick up work from notebooks you ran hours earlier.
    """
    pred = check_predictions(pred)
    meta = split_metadata()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip().lower()).strip("-")
    path = os.path.join(workdir(), "predictions", f"{slug}.csv")
    pred.to_csv(path, index=False)
    with open(path.replace(".csv", ".json"), "w") as fh:
        json.dump({"name": slug, "note": note, "split": meta.get("method"),
                   "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  fh, indent=2)
    print(f"saved '{slug}' ({len(pred)} molecules)")
    return path


def list_predictions() -> pd.DataFrame:
    d = os.path.join(_STATE["workdir"] or ".", "predictions")
    if not os.path.isdir(d):
        return pd.DataFrame(columns=["name", "note", "split", "saved_at"])
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        meta = {"name": f[:-4], "note": "", "split": "?", "saved_at": ""}
        try:
            with open(os.path.join(d, f.replace(".csv", ".json"))) as fh:
                meta.update(json.load(fh))
        except Exception:
            pass
        rows.append(meta)
    return pd.DataFrame(rows)


def load_predictions(name: str) -> pd.DataFrame:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip().lower()).strip("-")
    path = os.path.join(workdir(), "predictions", f"{slug}.csv")
    if not os.path.exists(path):
        have = ", ".join(list_predictions()["name"].tolist()) or "(none yet)"
        raise FileNotFoundError(f"No predictions named '{slug}'. You have: {have}")
    return pd.read_csv(path)


# =========================================================================
# Submission
# =========================================================================

def submissions_used() -> int:
    log = os.path.join(_STATE["workdir"] or ".", "submissions.jsonl")
    if not os.path.exists(log):
        return 0
    with open(log) as fh:
        return sum(1 for line in fh if line.strip())


def submit(pred: pd.DataFrame, expected_ma_rae: float, why: str,
           warmup: bool = False) -> None:
    """Send predictions to the leaderboard.

    You get SUBMISSION_BUDGET submissions for the whole day. That is
    deliberate: with a budget this small you cannot use the leaderboard as a
    validation set, so your internal validation has to be honest.

    expected_ma_rae : what you think you will score, BEFORE you find out.
                      This feeds the calibration leaderboard, which ranks
                      pairs on how close that guess was -- not on the score.
    why             : one sentence on what changed since your last submission.
    warmup          : the null-model submission from 00_start_here. Free --
                      it does not spend any of your budget.
    """
    used = submissions_used()
    if not warmup and used >= SUBMISSION_BUDGET:
        raise RuntimeError(
            f"You have used all {SUBMISSION_BUDGET} submissions. Your best "
            "already-submitted entry still counts -- go write your report.")
    if not why.strip():
        raise ValueError("Say what changed. Future-you writing slide 2 will thank you.")
    if not np.isfinite(expected_ma_rae):
        raise ValueError("Commit to a number for expected_ma_rae before submitting.")

    test = load_test()
    pred = check_predictions(pred, against=test)
    meta = split_metadata()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"pair": pair_name(), "submitted_at": stamp,
              "expected_ma_rae": float(expected_ma_rae), "why": why.strip(),
              "split": meta.get("method"), "n": int(len(pred)),
              "attempt": 0 if warmup else used + 1, "warmup": bool(warmup)}

    delivered = False
    if LEADERBOARD_URL:
        try:
            import requests
            buf = io.StringIO()
            pred.to_csv(buf, index=False)
            r = requests.post(LEADERBOARD_URL,
                              files={"predictions": (f"{pair_name()}.csv", buf.getvalue())},
                              data={"meta": json.dumps(record)}, timeout=60)
            r.raise_for_status()
            delivered = True
            print(r.text[:500])
        except Exception as exc:
            warnings.warn(f"Leaderboard POST failed ({exc}); falling back to shared folder.")
    if not delivered:
        os.makedirs(SHARED_DIR, exist_ok=True)
        tag = "warmup" if warmup else f"attempt{used + 1}"
        stem = f"{pair_name()}__{stamp}__{tag}"
        pred.to_csv(os.path.join(SHARED_DIR, stem + ".csv"), index=False)
        with open(os.path.join(SHARED_DIR, stem + ".json"), "w") as fh:
            json.dump(record, fh, indent=2)
        print(f"submitted to the shared folder as {stem}")

    if warmup:
        print("warm-up submission -- this one is free, your budget is untouched.")
        return
    with open(os.path.join(workdir(), "submissions.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")
    left = SUBMISSION_BUDGET - (used + 1)
    print(f"submission {used + 1} of {SUBMISSION_BUDGET}. {left} left.")
    print(f"you predicted MA-RAE = {expected_ma_rae:.3f}. Write that on slide 3.")


# =========================================================================
# Chemistry helpers
# =========================================================================

def morgan_fingerprints(smiles, radius: int = 2, n_bits: int = 2048):
    """Morgan (ECFP-like) bit fingerprints as a list of RDKit bit vectors."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        out.append(gen.GetFingerprint(mol) if mol is not None else None)
    return out


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

    This is the number that tells you whether a prediction is interpolation
    or extrapolation. For 2048-bit Morgan-2 fingerprints, similarities below
    about 0.27 are indistinguishable from comparing two random molecules --
    below that threshold, "nearest neighbour" carries no information.
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
    med = X_train[keep].median()
    out = [X_train[keep].fillna(med)]
    for o in others:
        out.append(o.reindex(columns=keep).fillna(med))
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
    pred = blank_predictions(target)
    for e in ENDPOINTS:
        pred[e] = float(train[e].mean(skipna=True))
    return pred
