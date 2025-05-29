cd $HAMMER_ROOT/results/fig12_t4

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
  rm -rf val
  mkdir val
  ***REMOVED***
  tar -xvf ILSVRC2012_img_val.tar -C val
  rm ILSVRC2012_img_val.tar
  python3 $HAMMER_ROOT/util/filter_validation_set.py

  bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_B1.sh
  bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_B2.sh
  bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_D1.sh
  bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_D3.sh
  rm exploit_control.txt memory_control.txt model_control.txt
else
  files=($HAMMER_ROOT/results/fig12_t4/B1/alexnet.txt $HAMMER_ROOT/results/fig12_t4/B1/dense.txt $HAMMER_ROOT/results/fig12_t4/B1/inception.txt $HAMMER_ROOT/results/fig12_t4/B1/resnet.txt $HAMMER_ROOT/results/fig12_t4/B1/vgg.txt \
          $HAMMER_ROOT/results/fig12_t4/B2/alexnet.txt $HAMMER_ROOT/results/fig12_t4/B2/dense.txt $HAMMER_ROOT/results/fig12_t4/B2/inception.txt $HAMMER_ROOT/results/fig12_t4/B2/resnet.txt $HAMMER_ROOT/results/fig12_t4/B2/vgg.txt \
          $HAMMER_ROOT/results/fig12_t4/D1/alexnet.txt $HAMMER_ROOT/results/fig12_t4/D1/dense.txt $HAMMER_ROOT/results/fig12_t4/D1/inception.txt $HAMMER_ROOT/results/fig12_t4/D1/resnet.txt $HAMMER_ROOT/results/fig12_t4/D1/vgg.txt \
          $HAMMER_ROOT/results/fig12_t4/D3/alexnet.txt $HAMMER_ROOT/results/fig12_t4/D3/dense.txt $HAMMER_ROOT/results/fig12_t4/D3/inception.txt $HAMMER_ROOT/results/fig12_t4/D3/resnet.txt $HAMMER_ROOT/results/fig12_t4/D3/vgg.txt)

  for file in "${files[@]}"; do
    if [ ! -e "$file" ]; then
      echo "Required Data DNE, Exiting..."
      exit
    fi
  done
fi
