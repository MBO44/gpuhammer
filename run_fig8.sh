mkdir -p $HAMMER_ROOT/results/fig8
bash $HAMMER_ROOT/data_scripts/fig8/run_delay_st.sh
bash $HAMMER_ROOT/data_scripts/fig8/run_delay_mt.sh
bash $HAMMER_ROOT/data_scripts/fig8/run_delay_mw.sh
python3 $HAMMER_ROOT/plot_scripts/plot_fig8.py $HAMMER_ROOT/results/fig8/st_delays.txt $HAMMER_ROOT/results/fig8/mt_delays.txt $HAMMER_ROOT/results/fig8/mw_delays.txt 10000
