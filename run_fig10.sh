mkdir -p $HAMMER_ROOT/results/fig10
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

python3 ~/gpuhammer/plot_scripts/plot_fig10a.py $HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_12w.txt $HAMMER_ROOT/results/fig10/delay_16w.txt $HAMMER_ROOT/results/fig10/delay_24w.txt 10000
python3 ~/gpuhammer/plot_scripts/plot_fig10b.py $HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_6w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_3t.txt 10000