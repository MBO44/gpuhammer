mkdir -p $HAMMER_ROOT/results/fig5_6

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
  bash $HAMMER_ROOT/data_scripts/fig5_6/execute_fig5_6.sh
else
  files=($HAMMER_ROOT/results/fig5_6/CONFLICT_TIMING.txt $HAMMER_ROOT/results/fig5_6/BASE_TIMING.txt)

  for file in "${files[@]}"; do
  if [ ! -e "$file" ]; then
    echo "Required Data DNE, Exiting..."
    exit
  fi
  done
fi
