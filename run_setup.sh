if [ "$1" = "clean" ]; then
    echo "***********************************************************"
    echo "Erasing RMM Conda Environment"
    bash $HAMMER_ROOT/util/env_purge.sh
else
    echo "***********************************************************"
    echo "Setting up RMM Conda Environment"
    bash $HAMMER_ROOT/util/env_setup.sh
fi