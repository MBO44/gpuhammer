#!/bin/bash

echo "-------------------------------------------"
echo ""
echo "###########################################"
echo "[INFO] 1. Setup Anaconda and Dependencies"
echo "###########################################"
bash run_setup.sh

echo "-------------------------------------------"
echo ""
echo "###########################################"
echo "[INFO] 2. Building GPUHammer"
echo "###########################################"

conda init
source activate base
conda activate rmm_dev
cmake -S $HAMMER_ROOT/src -B $HAMMER_ROOT/src/out/build
cd $HAMMER_ROOT/src/out/build
make
cd $HAMMER_ROOT

echo "-------------------------------------------"
echo ""
echo "###########################################"
echo "[INFO] 3. Running Artifacts"
echo "###########################################"

bash run_row_sets.sh
bash run_fig2.sh
bash run_fig5_6.sh
bash run_fig8.sh
bash run_fig10.sh
bash run_t1_t3.sh
bash run_fig11.sh
bash run_fig12.sh
bash run_fig13_t4.sh
