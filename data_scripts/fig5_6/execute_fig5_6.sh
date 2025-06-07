
python3 $HAMMER_ROOT/util/run_timing_task.py gt --range $((47 * (2 ** 30))) --size $((47 * (2 ** 30))) --it 10 --step 256 --file $HAMMER_ROOT/results/fig5_6/CONFLICT_TIMING.txt
sleep 3s
python3 $HAMMER_ROOT/util/run_timing_task.py gt --same --range $((47 * (2 ** 30))) --size $((47 * (2 ** 30))) --it 10 --step 256 --file $HAMMER_ROOT/results/fig5_6/BASE_TIMING.txt
sleep 3s
