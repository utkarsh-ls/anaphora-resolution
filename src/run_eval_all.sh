#!/bin/bash
mkdir -p out

for file in `ls ../all_configs`;
    do
    cp ../all_configs/$file configs.py;
    (python eval.py --eval_all_file  > ./out/${file//config.py/.out}) || exit 1;
done
