flag_reuse=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --reuse) flag_reuse="--reuse" ;;
    *) echo "Unknown option: $1" ;;
  esac
  shift
done

echo "---------------------------"
echo ""
echo "#####################"
echo "[INFO] 1. Setup Anaconda and Dependencies"
echo "#####################"

bash run_setup.sh

echo "---------------------------"
echo ""
echo "#####################"
echo "[INFO] 2. Building GPUHammer"
echo "#####################"

conda init
source activate base
conda activate rmm_dev
cmake -S $HAMMER_ROOT/src -B $HAMMER_ROOT/src/out/build
cd $HAMMER_ROOT/src/out/build
make
cd $HAMMER_ROOT

echo "---------------------------"
echo ""
echo "#####################"
echo "[INFO] 3. Running Artifacts"
echo "#####################"

bash run_row_sets.sh $flag_reuse
bash run_fig2.sh $flag_reuse
bash run_fig5_6.sh $flag_reuse
bash run_fig8.sh $flag_reuse
bash run_fig10.sh $flag_reuse
bash run_fig11.sh
bash run_fig12_t4.sh $flag_reuse
bash run_fig14.sh
bash run_fig15.sh
