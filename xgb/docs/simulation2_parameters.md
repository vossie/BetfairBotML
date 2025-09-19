# Simulation Parameters

This document explains the key parameters used when running the **streaming betting simulator** and what the simulation represents.

## Command Arguments

```bash
BASE_ARGS=(
  --stake-cap-market 50
  --stake-cap-day 2000
  --max-exposure-day 5000
  --days-before 0
  --curated /mnt/nvme/betfair-curated
  --sport horse-racing
  --date "$DATE_ARG"
  --preoff-mins 180
  --min-edge 0.02
  --kelly 0.25
  --commission 0.05
  --top-n-per-market 1
  --side auto
  --bets-out ./output/bets.csv
)
```

### Parameter Details

- **`--stake-cap-market 50`**  
  Maximum total stake allowed in a single market (e.g. one race). Ensures no over-exposure in an individual event.

- **`--stake-cap-day 2000`**  
  Maximum total stake across all markets for a single simulation day.

- **`--max-exposure-day 5000`**  
  Global exposure throttle for the day. This mimics a bankroll constraint so that outstanding bets cannot exceed this amount.

- **`--days-before 0`**  
  Defines the date range. `0` means *only the specified `--date`*.  
  Example: `--date 2025-09-17 --days-before 0` → runs only for 2025-09-17.

- **`--curated /mnt/nvme/betfair-curated`**  
  Path to curated input data (market definitions, orderbook snapshots, results).

- **`--sport horse-racing`**  
  Sport to simulate on (currently horse racing data).

- **`--date "$DATE_ARG"`**  
  End date of the simulation window.

- **`--preoff-mins 180`**  
  Time horizon (in minutes) before scheduled off-time to include markets.  
  Here: capture data from up to 3 hours before the start.

- **`--min-edge 0.02`**  
  Minimum edge (expected value margin) required for a bet to be considered. Bets with edge below 2% are ignored.

- **`--kelly 0.25`**  
  Fractional Kelly multiplier. Determines stake size as a fraction of the Kelly criterion.  
  `0.25` = 25% Kelly, more conservative than full Kelly.

- **`--commission 0.05`**  
  Commission rate applied on winnings (Betfair default = 5%).

- **`--top-n-per-market 1`**  
  Limit on number of selections per market. Here we only bet the single best candidate.

- **`--side auto`**  
  Let the simulator choose whether to **back** or **lay** based on where the edge is higher.  
  Alternatives: `back` (always back), `lay` (always lay).

- **`--bets-out ./output/bets.csv`**  
  Path for saving detailed bet-by-bet output.

---

## What the Simulation Does

- Replays curated exchange data (market books, prices, results) **as if in real-time**.  
- Models latency, cooldowns, bankroll limits, and exposure caps.  
- Scores each snapshot with the trained model(s), producing probabilities (`p_hat`).  
- Compares model probabilities with market odds to find **edges**.  
- Applies a **fractional Kelly staking strategy** to size bets.  
- Caps bets per market/day to respect bankroll constraints.  
- Settles bets when the race result is known, applying commission.  
- Outputs:
  - `bets.csv`: all individual bets placed, with stakes, odds, side, and PnL.
  - Aggregations: PnL per market, PnL binned by time-to-off.

In short, this sim answers:  
👉 *“Given my model and bankroll settings, what bets would I have placed on this date, and what would my profit/loss have been?”*
