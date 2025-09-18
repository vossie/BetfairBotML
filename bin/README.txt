DROP-IN PACKAGE
===============

This zip includes:

- ml/sim2.py
  Streaming simulator with:
    * --min-stake, --tick-snap, --slip-ticks
    * Level-1 depth fill (cap by best quote size)
    * Latency, cooldowns, exposure caps
    * Uses odds_exec for PnL when present

- bin/run-simulator2-0-30.sh
  Runner that focuses on 0–30 minutes pre-off with execution realism flags enabled.

INSTALL
-------
1) Backup your existing files:
   cp ml/sim2.py ml/sim2.py.bak  # if present
   cp bin/run-simulator2-0-30.sh bin/run-simulator2-0-30.sh.bak  # if present

2) Copy these over your repo (from the zip root):
   cp ml/sim2.py /opt/BetfairBotML/ml/sim2.py
   cp bin/run-simulator2-0-30.sh /opt/BetfairBotML/bin/run-simulator2-0-30.sh
   chmod +x /opt/BetfairBotML/bin/run-simulator2-0-30.sh

3) Test the CLI shows the new flags:
   /opt/BetfairBotML/.venv/bin/python -m ml.sim2 --help | egrep 'min-stake|tick-snap|slip-ticks|stream-bucket-secs'

4) Run:
   /opt/BetfairBotML/bin/run-simulator2-0-30.sh 2025-09-17 --model
