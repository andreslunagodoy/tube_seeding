#!/bin/bash
# stdout and stderr are printed to terminal and logged to run_cma_es.log and run_cma_es.err
/usr/bin/time -v optimize-ordering-NN --save-path ./maxhigh --s-max 12 --n-indices 11 --n-ibp-operators 18 --target-integral 1,1,1,1,1,1,1,1,0,0,-12 --top-sector 1,1,1,1,1,1,1,1,0,0,0 --ibp-file IBP_LI --cut 1,2,3,4,5,6,7,8 --variables 'd->79,m1->31,m2->43,m3->11,m4->59' --max-eval 1000 --optimize-seeds-only --mutation-size 0.05 --freeze-variable-ordering  > >(tee run_cma_es.log) 2> >(tee run_cma_es.err >&2)

echo
echo "Summary of seeds (1,...,1, -x, -y, -z) used"
python summarize_maxcut_seeds.py

# Plot the seeds found after optimization
python plot_seeds.py --seed-op-file maxhigh/resorted_seed_op_list.txt --export-figure maxhigh/seeds.png --title "Maximal-cut seeds for reducing (1,...,1,0,0,-12)"
