# ecommerce-traffic-segmentation

Code for the paper *Unsupervised segmentation of e-commerce web traffic for anti-fraud monitoring*
(A. S. Erdenko, S. N. Bruskin; submitted to MDPI Inventions).

The repository contains the `trafseg` package: preprocessing and sessionization of
HTTP server logs, session-level feature extraction, the segmentation pipeline, the
synthetic benchmark generator, and the evaluation scripts that produce every table
and figure in the paper.

## Layout

```
config/          URI typing rules, static patterns, bot signatures, feature and generator parameters
src/trafseg/     package
  io_eclog.py    reader for the EClog dataset
  preprocess.py  filtering, URI typing, sessionization
  features/      session feature families
  generator/     synthetic benchmark in the EClog format, with ground truth
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
# real data (EClog)
python scripts/run_eclog.py data/raw/eclog_1day.csv
python scripts/build_features.py data/processed/eclog_1day_requests.parquet

# synthetic benchmark: same preprocessing and features, plus true labels
python scripts/run_generator.py --requests 100000 --seed 0
python scripts/run_eclog.py data/synthetic/synth_100k_s0.csv --out data/synthetic
python scripts/build_features.py data/synthetic/synth_100k_s0_requests.parquet \
    --clients data/synthetic/synth_100k_s0_clients.parquet
```

Further scripts are added as the corresponding sections of the paper are completed.

## Licence

MIT.
