#!/bin/bash

echo ""
echo "-------------------------------------------"
echo ""
echo "[INFO] Starting Row Hammer Campagin for Table 1 and 3"
echo "[INFO] This will run for ~1 day."

python3  $HAMMER_ROOT/util/run_campaign.py --bank_ids A B C D

echo "[INFO] Generating Table 1 and Table 3"
python3 $HAMMER_ROOT/plot_scripts/plot_t1_t3.py --banks A B C D

echo "[INFO] Done. Figure 13 is stored as 'results/campaign/t1.txt' and Table 4 is stored as 'results/campaign/t3.txt"
