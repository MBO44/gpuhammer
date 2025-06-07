#!/bin/bash

mkdir -p $HAMMER_ROOT/results/fig5_6

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Experiments for Figure 5 and 6"

bash $HAMMER_ROOT/data_scripts/fig5_6/execute_fig5_6.sh

echo "[INFO] Generating Figure 5"
python3 $HAMMER_ROOT/plot_scripts/plot_fig5a.py
python3 $HAMMER_ROOT/plot_scripts/plot_fig5b.py

echo "[INFO] Generating Figure 6"
python3 $HAMMER_ROOT/plot_scripts/plot_fig6.py

echo "[INFO] Done. Figure 5a, 5b, and 6 is stored in 'results/fig5_6/' as 'fig*.pdf'"
