#!/bin/bash

mkdir -p $HAMMER_ROOT/results/fig10

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Experiments for Figure 10"

bash $HAMMER_ROOT/data_scripts/fig10/run_delay_8w.sh
sleep 3s
bash $HAMMER_ROOT/data_scripts/fig10/run_delay_12w.sh
sleep 3s
bash $HAMMER_ROOT/data_scripts/fig10/run_delay_16w.sh
sleep 3s
bash $HAMMER_ROOT/data_scripts/fig10/run_delay_24w.sh
sleep 3s
bash $HAMMER_ROOT/data_scripts/fig10/run_delay_6w_2t.sh
sleep 3s
bash $HAMMER_ROOT/data_scripts/fig10/run_delay_8w_2t.sh
sleep 3s
bash $HAMMER_ROOT/data_scripts/fig10/run_delay_8w_3t.sh
sleep 3s

echo "[INFO] Generating Figure 10"
python3 $HAMMER_ROOT/plot_scripts/plot_fig10a.py
python3 $HAMMER_ROOT/plot_scripts/plot_fig10b.py

echo "[INFO] Done. Figure 10a and 10b is stored in 'results/fig10' as 'fig*.pdf'"
