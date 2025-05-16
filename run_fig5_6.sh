mkdir -p $HAMMER_ROOT/results/fig5_6
bash $HAMMER_ROOT/data_scripts/fig5_6/execute_fig5_6.sh
python3 $HAMMER_ROOT/plot_scripts/plot_fig5a.py $HAMMER_ROOT/results/fig5_6/CONFLICT_TIMING.txt $HAMMER_ROOT/results/row_sets/CONF_SET_0.txt
python3 $HAMMER_ROOT/plot_scripts/plot_fig5b.py $HAMMER_ROOT/results/fig5_6/CONFLICT_TIMING.txt $HAMMER_ROOT/results/fig5_6/BASE_TIMING.txt $HAMMER_ROOT/results/row_sets/CONF_SET_0.txt
python3 $HAMMER_ROOT/plot_scripts/plot_fig6.py $HAMMER_ROOT/results/fig5_6/BASE_TIMING.txt