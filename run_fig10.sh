mkdir -p $HAMMER_ROOT/results/fig10

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
else
  files=($HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_12w.txt $HAMMER_ROOT/results/fig10/delay_16w.txt $HAMMER_ROOT/results/fig10/delay_24w.txt \
          $HAMMER_ROOT/results/fig10/delay_6w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_3t.txt)

  for file in "${files[@]}"; do
  if [ ! -e "$file" ]; then
    echo "Required Data DNE, Exiting..."
    exit
  fi
  done
fi

python3 ~/gpuhammer/plot_scripts/plot_fig10a.py $HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_12w.txt $HAMMER_ROOT/results/fig10/delay_16w.txt $HAMMER_ROOT/results/fig10/delay_24w.txt 10000
python3 ~/gpuhammer/plot_scripts/plot_fig10b.py $HAMMER_ROOT/results/fig10/delay_8w.txt $HAMMER_ROOT/results/fig10/delay_6w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_2t.txt $HAMMER_ROOT/results/fig10/delay_8w_3t.txt 10000