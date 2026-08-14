# External data

Five ADME datasets from [Therapeutics Data Commons](https://tdcommons.ai/single_pred_tasks/adme/),
committed here as `.tab` files so nothing in this repo touches the network on
the day. They are byte-for-byte what Harvard Dataverse serves — no cleaning,
no unit conversion, no column renaming. `External_Data.ipynb` reads them
directly with `pd.read_csv(..., sep="\t")`.

| file | TDC dataset | rows | units as shipped |
|---|---|---|---|
| `caco2_wang.tab` | `Caco2_Wang` | 910 | cm/s, log10 |
| `solubility_aqsoldb.tab` | `Solubility_AqSolDB` | 9,982 | log mol/L |
| `ppbr_az.tab` | `PPBR_AZ` | 2,828 | percent bound |
| `clearance_hepatocyte_az.tab` | `Clearance_Hepatocyte_AZ` | 1,213 | µL/min/10⁶ cells |
| `lipophilicity_astrazeneca.tab` | `Lipophilicity_AstraZeneca` | 4,200 | logD at pH 7.4 |

Two things about these files are load-bearing for the notebook's exercise, so
do not "tidy" them:

- **They disagree about column names.** Four use `Drug_ID` / `Drug` / `Y`;
  `clearance_hepatocyte_az.tab` uses `ID` / `X` / `Y`.
- **`ppbr_az.tab` has a fourth column, `Species`**, and it is not all human:
  1,614 *Homo sapiens*, 717 *Rattus norvegicus*, 244 *Canis lupus familiaris*,
  162 *Mus musculus*, 91 *Cavia porcellus*. PyTDC's ADME loader returns only
  `Drug` and `Y`, so anyone going through it never sees this.

## Refetching

The Dataverse file ids are permanent. If a file is lost or corrupted:

```bash
curl -L -o Data/artifacts/caco2_wang.tab                https://dataverse.harvard.edu/api/access/datafile/4259569
curl -L -o Data/artifacts/solubility_aqsoldb.tab        https://dataverse.harvard.edu/api/access/datafile/4259610
curl -L -o Data/artifacts/ppbr_az.tab                   https://dataverse.harvard.edu/api/access/datafile/6413140
curl -L -o Data/artifacts/clearance_hepatocyte_az.tab   https://dataverse.harvard.edu/api/access/datafile/4266187
curl -L -o Data/artifacts/lipophilicity_astrazeneca.tab https://dataverse.harvard.edu/api/access/datafile/4259595
```

Check what you get before committing it. Dataverse goes into maintenance
without warning and serves an HTML error page with a success code when it
does, which lands on disk looking like a data file — open one and confirm it
starts with a header row, not `<!DOCTYPE html>`.

## Other files here

`mordred_descriptors.parquet` is a precomputed descriptor cache for
`Descriptors.ipynb`, built by `Setup/precompute_descriptors.py`. Unrelated to
the above.
