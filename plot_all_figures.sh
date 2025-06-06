#!/bin/bash

cd $HAMMER_ROOT/plot_scripts

echo "[INFO] Generating Figure 2"
python3 plot_fig2.py

echo "[INFO] Generating Figure 5"
python3 plot_fig5a.py
python3 plot_fig5b.py

echo "[INFO] Generating Figure 6"
python3 plot_fig6.py

echo "[INFO] Generating Figure 8"
python3 plot_fig8.py

echo "[INFO] Generating Figure 10"
python3 plot_fig10a.py
python3 plot_fig10b.py

echo "[INFO] Generating Figure 11"
python3 plot_fig11.py

echo "[INFO] Generating Table 4"
bash plot_t4.sh

echo "[INFO] Generating Figure 12"
python3 plot_fig12.py

echo "[INFO] Generating Figure 13"
python3 plot_fig13.py

echo "[INFO] Generating Figure 15"
python3 plot_fig15.py

echo "[INFO] Generating Figure 16"
python3 plot_fig16.py

cd ../