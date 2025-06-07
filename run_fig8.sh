#!/bin/bash

mkdir -p $HAMMER_ROOT/results/fig8

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Experiments for Figure 8"

bash $HAMMER_ROOT/data_scripts/fig8/run_delay_st.sh
bash $HAMMER_ROOT/data_scripts/fig8/run_delay_mt.sh
bash $HAMMER_ROOT/data_scripts/fig8/run_delay_mw.sh

echo "[INFO] Generating Figure 8"
python3 $HAMMER_ROOT/plot_scripts/plot_fig8.py

echo "[INFO] Done. Figure 8 is stored as 'results/fig8/fig8.pdf'"
