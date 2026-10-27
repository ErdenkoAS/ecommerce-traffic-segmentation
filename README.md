# ecommerce-traffic-segmentation

Code for the paper *Unsupervised segmentation of e-commerce web traffic for anti-fraud monitoring*
(A. S. Erdenko, S. N. Bruskin; submitted to MDPI Inventions).

The repository contains the `trafseg` package: preprocessing and sessionization of
HTTP server logs, session-level feature extraction, the segmentation pipeline, the
synthetic benchmark generator, and the evaluation scripts that produce every table
and figure in the paper.

## Layout

```
config/          URI typing rules, static-resource patterns, bot signatures, feature parameters
src/trafseg/     package
  io_eclog.py    reader for the EClog dataset
  preprocess.py  filtering, URI typing, sessionization
  features/      session feature families
  generator/     synthetic benchmark with ground truth
  evaluation/    metrics and reports
scripts/         entry points; everything in the paper is produced here, with fixed seeds
notebooks/       exploration only, not used for reported results
data/            not tracked; see data/README.md
results/         generated tables and figures
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

Real-world validation uses the EClog dataset (Chodak, Suchacka & Chawla, Computer Networks 2020),
Harvard Dataverse, doi:10.7910/DVN/Z834IK, licence CC0. It is not redistributed
here; see `data/README.md` for the expected location.

## Reproducing the paper

```
python scripts/run_eclog.py data/raw/eclog_1day.csv            # sessions
python scripts/build_features.py data/processed/eclog_1day_requests.parquet   # feature matrix
```

Further scripts are added as the corresponding sections of the paper are completed.

## Licence

MIT.
