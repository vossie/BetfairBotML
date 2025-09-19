Temporal-split training with two heads in one run:
  1) Value head: p(win) with optional isotonic calibration + PnL-style backtest.
  2) Price-move head: short-horizon label (pm_up) built from the stream.

Split logic (dates are inclusive):
  - Validation: [ASOF-1, ASOF]
  - Training: the previous `TRAIN_DAYS` ending at (ASOF-2).

Example: ASOF=2025-09-18, TRAIN_DAYS=13 →
  Train: 2025-09-04 .. 2025-09-16
  Valid: 2025-09-17 .. 2025-09-18

Run:
  python train_edge_temporal.py \
    --curated s3://betfair-curated \
    --sport horse-racing \
    --asof 2025-09-18 \
    --train-days 13 \
    --preoff-mins 30 \
    --downsample-secs 5 \
    --commission 0.02 \
    --edge-thresh 0.02 \
    --pm-horizon-secs 60 \
    --pm-tick-threshold 1
