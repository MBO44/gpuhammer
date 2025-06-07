#!/bin/bash

mkdir -p $HAMMER_ROOT/results/fig2

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Experiments for Figure 2"

bash $HAMMER_ROOT/data_scripts/fig2/execute_fig2.sh

echo "[INFO] Generating Figure 2"

python3 $HAMMER_ROOT/plot_scripts/plot_fig2.py

echo "[INFO] Done. Figure 2 is stored as 'results/fig2/fig2.pdf'"
