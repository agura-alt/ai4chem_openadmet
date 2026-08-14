"""Precompute the descriptor sets that are too slow to build in a notebook.

Run once, before the event, and commit the output. Everything lands in
<repo>/Data/artifacts/ so a cloned repo has it and no notebook pays the cost.

    pip install mordredcommunity
    python Setup/precompute_descriptors.py             # everything
    python Setup/precompute_descriptors.py mordred     # just one
    python Setup/precompute_descriptors.py --list      # what is available

Each artifact is a parquet keyed on the molecule ID, covering train AND test
in one file, so a notebook can reindex it against either frame.

Only genuinely expensive things belong here. Mordred takes ~2.5 min and lands
at 33 MB, which every student clones -- that is worth it once. MACCS keys and
bigger Morgan fingerprints take seconds, so the notebook computes those live
rather than making everyone download them.

Timings on 7608 molecules (5326 train + 2282 test), 8-core laptop:

    mordred     ~140 s       1373 descriptors after pruning, 33 MB

Adding a new one: write a function that takes a list of SMILES and returns a
DataFrame with one row per SMILES, then add it to ARTIFACTS. Ask first whether
it is slow enough to be worth the clone -- under a minute, it is not.

Parquet needs pyarrow:  pip install pyarrow
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


OUT_DIR = os.path.join(common.REPO_DATA_DIR, "artifacts")


# ---------------------------------------------------------------- descriptors

def mordred_descriptors(smiles) -> pd.DataFrame:
    """~1600 2D descriptors. The slow one -- this is why the script exists."""
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        raise SystemExit(
            "mordred is not installed. Try:  pip install mordredcommunity\n"
            "(the original 'mordred' package is unmaintained and breaks on "
            "modern numpy)")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]

    df = calc.pandas([mols[i] for i in keep], quiet=False)
    # reindex rather than assign into a preallocated frame: mordred returns a
    # mix of float/bool/error-object columns and positional assignment fights
    # with the dtypes. Unparseable molecules come back as all-NaN rows, so row
    # i still lines up with smiles[i].
    df.index = keep
    out = df.reindex(range(len(mols)))
    return out.apply(pd.to_numeric, errors="coerce")   # errors -> NaN


ARTIFACTS = {
    "mordred": (mordred_descriptors, "mordred_descriptors.parquet"),
}


# ---------------------------------------------------------------------- main

def build(name: str, force: bool = False) -> str:
    fn, filename = ARTIFACTS[name]
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path) and not force:
        print(f"{name}: already at {path} (use --force to rebuild)")
        return path

    train, test = common.load_train(), common.load_test()
    both = pd.concat([train[[common.ID_COL, common.SMILES_COL]],
                      test[[common.ID_COL, common.SMILES_COL]]],
                     ignore_index=True)
    print(f"{name}: computing for {len(both)} molecules...")

    t0 = time.time()
    X = fn(both[common.SMILES_COL].tolist())
    X = _shrink(X, name)
    X.insert(0, common.ID_COL, both[common.ID_COL].to_numpy())

    os.makedirs(OUT_DIR, exist_ok=True)
    X.to_parquet(path, index=False, compression="zstd")
    mb = os.path.getsize(path) / 1e6
    print(f"{name}: {X.shape[1] - 1} columns, {time.time() - t0:.0f}s, "
          f"{mb:.1f} MB -> {path}")
    return path


def _shrink(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """These files get committed and cloned by every student on every Colab
    start, so size is not cosmetic. float64 -> float32 halves it, and mordred
    in particular ships hundreds of columns that are all-NaN or constant and
    carry no signal. 59.5 MB -> ~24 MB for mordred.
    """
    before = X.shape[1]
    useful = X.columns[(X.notna().any()) & (X.nunique(dropna=True) > 1)]
    X = X[useful]
    floats = X.select_dtypes("float64").columns
    X = X.astype({c: "float32" for c in floats})
    if X.shape[1] < before:
        print(f"{name}: dropped {before - X.shape[1]} all-NaN/constant columns")
    return X


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}

    if "--list" in flags:
        for k, (_, f) in ARTIFACTS.items():
            here = os.path.exists(os.path.join(OUT_DIR, f))
            print(f"  {k:12s} {'built' if here else 'not built':10s} {f}")
        raise SystemExit

    wanted = args or list(ARTIFACTS)
    unknown = [w for w in wanted if w not in ARTIFACTS]
    if unknown:
        raise SystemExit(f"Unknown artifact {unknown}. "
                         f"Pick from: {', '.join(ARTIFACTS)}")
    for name in wanted:
        build(name, force="--force" in flags)
