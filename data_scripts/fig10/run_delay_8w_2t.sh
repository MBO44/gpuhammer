# Run the synchronization test

# Variables
bank_id=0

num_agg=16          # Number of aggressors
num_warp=8          # Number of warps
num_thread=2        # Number of threads per warp
round=1             # No. of round per tREFI, each round hammers <num_agg> rows

min_delay=0         # Minimum delay to test
max_delay=200       # Maximum delay to test

num_rows=64169      # Number of rows in the row_set (line number - 1)
rowid=100           # Id of a row to test the delays, can be arbitrary
iterations=10000

# Memory Properties
addr_step=256           # Set to be the <step> parameter used in finding conf_set/row_set
mem_size=50465865728    # Bytes of memory allocated for hammering (recommend: size of memory - 1GB)

# File paths
rowset_file="$HAMMER_ROOT/results/row_sets/ROW_SET_${bank_id}.txt"
time_file="$HAMMER_ROOT/results/fig10/delay_8w_2t.txt"
log_file="$HAMMER_ROOT/results/fig10/log.txt"

> $log_file
> $time_file


# Running the test
nvidia-smi -q > $log_file
echo "Start hammering ..."

$HAMMER_ROOT/src/out/build/sync_delay $rowset_file $((num_agg - 1)) $addr_step $iterations $rowid $mem_size $time_file $num_warp $num_thread $round $min_delay $max_delay $num_rows > $log_file

echo "Hammering done."
