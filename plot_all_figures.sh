#!/bin/bash

cd $HAMMER_ROOT/plot_scripts

echo "[INFO] Generating Figure 2"
python3 plot_fig2.py

echo "[INFO] Generating Figure 5"
python3 plot_fig5a.py $HAMMER_ROOT/results/fig5_6/CONFLICT_TIMING.txt $HAMMER_ROOT/results/row_sets/CONF_SET_0.txt
python3 plot_fig5b.py $HAMMER_ROOT/results/fig5_6/CONFLICT_TIMING.txt $HAMMER_ROOT/results/fig5_6/BASE_TIMING.txt $HAMMER_ROOT/results/row_sets/CONF_SET_0.txt

echo "[INFO] Generating Figure 6"
python3 plot_fig6.py $HAMMER_ROOT/results/fig5_6/BASE_TIMING.txt

echo "[INFO] Generating Figure 8"
python3 plot_fig8.py $HAMMER_ROOT/results/fig8/st_delays.txt $HAMMER_ROOT/results/fig8/mt_delays.txt $HAMMER_ROOT/results/fig8/mw_delays.txt 10000

echo "[INFO] Generating Figure 10"
python3 plot_fig10a.py $HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_12w.txt $HAMMER_ROOT/results/fig10/delay_16w.txt $HAMMER_ROOT/results/fig10/delay_24w.txt 10000
python3 plot_fig10b.py $HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_6w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_3t.txt 10000

echo "[INFO] Generating Figure 11"
python3 plot_fig11.py

echo "[INFO] Generating Table 4"
bash plot_t4.sh

echo "[INFO] Generating Figure 12"
python3 plot_fig12.py

echo "[INFO] Generating Figure 14"
python3 plot_fig14.py $HAMMER_ROOT/results/row_sets/ROW_SET_0.txt

echo "[INFO] Generating Figure 15"
python3 plot_fig15.py

cd ../