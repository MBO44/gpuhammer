mkdir -p $HAMMER_ROOT/results/fig8

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
  bash $HAMMER_ROOT/data_scripts/fig8/run_delay_st.sh
  bash $HAMMER_ROOT/data_scripts/fig8/run_delay_mt.sh
  bash $HAMMER_ROOT/data_scripts/fig8/run_delay_mw.sh
else
  files=($HAMMER_ROOT/results/fig8/st_delays.txt $HAMMER_ROOT/results/fig8/mt_delays.txt $HAMMER_ROOT/results/fig8/mw_delays.txt)

  for file in "${files[@]}"; do
  if [ ! -e "$file" ]; then
    echo "Required Data DNE, Exiting..."
    exit
  fi
  done
fi

python3 $HAMMER_ROOT/plot_scripts/plot_fig8.py $HAMMER_ROOT/results/fig8/st_delays.txt $HAMMER_ROOT/results/fig8/mt_delays.txt $HAMMER_ROOT/results/fig8/mw_delays.txt 10000
