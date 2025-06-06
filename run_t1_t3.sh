#!/bin/bash

python3 util/run_campaign.py --bank_ids A B C D

python3 plot_scripts/plot_t1_t3.py --bank_ids A B C D