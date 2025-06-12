#!/bin/bash

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Experiments for Figure 11"

python3 util/run_trh.py

echo "[INFO] Generating Figure 11"
python3 $HAMMER_ROOT/plot_scripts/plot_fig11.py

echo "[INFO] Done. Figure 11 is stored as 'results/fig11/fig11.pdf'"
