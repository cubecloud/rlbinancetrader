#!/bin/bash

cd $1

files=$(ls *.env)

set -o allexport
for i in $files
do
	source $i
	echo "$i - exported"
done
set +o allexport

echo "Run test_rllaboratory_maskable_ppo_mlp.py"
python $1test_rllaboratory_maskable_ppo_mlp.py