mkdir -p $HAMMER_ROOT/results/fig2

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
  bash $HAMMER_ROOT/data_scripts/fig2/execute_fig2.sh
else
  files=($HAMMER_ROOT/results/fig2/LOAD_TIMING.txt)

  for file in "${files[@]}"; do
  if [ ! -e "$file" ]; then
    echo "Required Data DNE, Exiting..."
    exit
  fi
  done
fi

python3 $HAMMER_ROOT/plot_scripts/plot_fig2.py