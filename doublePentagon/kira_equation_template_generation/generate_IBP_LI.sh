#!/bin/bash
# Generate IBP_LI by concatenating IBP and LI sectormappings, then
# rename Mandelstam invariants: s23→m1, s34→m2, s45→m3, s51→m4

set -e

cd "$(dirname "$0")"

cat sectormappings/doublePentagon/IBP sectormappings/doublePentagon/LI > IBP_LI

sed -i \
    -e 's/s23/m1/g' \
    -e 's/s34/m2/g' \
    -e 's/s45/m3/g' \
    -e 's/s51/m4/g' \
    IBP_LI

rm -rf kira.log results/ tmp/ sectormappings/

echo "Written and substituted: IBP_LI"
