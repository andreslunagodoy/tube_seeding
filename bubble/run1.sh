#!/bin/bash
mkdir -p run1
/usr/bin/time -v optimize-ordering-NN --ibp-file IBP --trivial-sector-file trivialsector --n-indices 2 --n-ibp-operators 2 --save-path run1 --top-sector 1,1 --s-max 0 --r-max 12 --d-max 12 --target-integral 6,6 --variables 'm0->37,d->13' --max-evals 9000 --mutation-size 0.025 --learning-rate 0.02 > run1.log
