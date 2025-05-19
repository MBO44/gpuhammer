flag_reuse=false

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --reuse) flag_reuse=true ;;
    *) echo "Unknown option: $1" ;;
  esac
  shift
done

if ! $flag_reuse; then
  echo "Force Re-running of Required Data"
  for val in 0 256 2048 5120 6400; do
    python3 $HAMMER_ROOT/util/run_timing_task.py conf_set --range $((47 * (1 << 30))) --size $((47 * (1<<30))) --it 15 --step 256 --threshold 27 --file $HAMMER_ROOT/results/row_sets/CONF_SET_$val.txt --trgtBankOfs $val
    sleep 3s
    python3 $HAMMER_ROOT/util/run_timing_task.py row_set --size $((47 * (1<<30))) --it 15 --threshold 27 --trgtBankOfs $val --outputFile $HAMMER_ROOT/results/row_sets/ROW_SET_$val.txt $HAMMER_ROOT/results/row_sets/CONF_SET_$val.txt
    sleep 3s
  done
else
  files=($HAMMER_ROOT/results/row_sets/CONF_SET_0.txt $HAMMER_ROOT/results/row_sets/CONF_SET_256.txt $HAMMER_ROOT/results/row_sets/CONF_SET_2048.txt
        $HAMMER_ROOT/results/row_sets/CONF_SET_5120.txt $HAMMER_ROOT/results/row_sets/CONF_SET_6400.txt \
        $HAMMER_ROOT/results/row_sets/ROW_SET_0.txt $HAMMER_ROOT/results/row_sets/ROW_SET_256.txt $HAMMER_ROOT/results/row_sets/ROW_SET_2048.txt
        $HAMMER_ROOT/results/row_sets/ROW_SET_5120.txt $HAMMER_ROOT/results/row_sets/ROW_SET_6400.txt)

  for file in "${files[@]}"; do
  if [ ! -e "$file" ]; then
    echo "Required Data DNE, Exiting..."
    exit
  fi
  done
fi
