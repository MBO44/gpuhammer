flag_reuse=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --reuse) flag_reuse="--reuse" ;;
    *) echo "Unknown option: $1" ;;
  esac
  shift
done

bash run_row_sets.sh $flag_reuse
bash run_fig2.sh $flag_reuse
bash run_fig5_6.sh $flag_reuse
bash run_fig8.sh $flag_reuse
bash run_fig10.sh $flag_reuse
bash run_fig12_t4.sh $flag_reuse
bash run_fig14.sh