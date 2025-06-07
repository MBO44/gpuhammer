#!/bin/bash

cd $HAMMER_ROOT/results/fig13_t4

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Experiments for Figure 13 and Table 4"

rm -rf val
mkdir val
tar -xvf $HAMMER_ROOT/ILSVRC2012_img_val.tar -C $HAMMER_ROOT/results/fig13_t4/val
python3 $HAMMER_ROOT/util/filter_validation_set.py

bash $HAMMER_ROOT/data_scripts/fig13_t4/run_hammer_manual_B1.sh
bash $HAMMER_ROOT/data_scripts/fig13_t4/run_hammer_manual_B2.sh
bash $HAMMER_ROOT/data_scripts/fig13_t4/run_hammer_manual_D1.sh
bash $HAMMER_ROOT/data_scripts/fig13_t4/run_hammer_manual_D3.sh
rm exploit_control.txt memory_control.txt model_control.txt

echo "[INFO] Generating Figure 13"
python3 $HAMMER_ROOT/plot_scripts/plot_fig13.py

echo "[INFO] Generating Table 4"
bash $HAMMER_ROOT/plot_scripts/plot_t4.sh

echo "[INFO] Done. Figure 13 is stored as 'results/fig13_t4/fig13.pdf' and Table 4 is stored as 'results/fig13_t4/t4.txt"
