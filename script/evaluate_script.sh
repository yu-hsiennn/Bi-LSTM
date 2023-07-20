#!/bin/bash
conda activate torch
cd ..
echo "Bash version ${BASH_VERSION}..."
prefix="missing_rate"
time="0926"
python evaluate.py -p $time > ./log/${prefix}_0%.txt
$date >> ./log/${prefix}_0%.txt
for i in {5..80..5}
do
  python evaluate.py -p $time -td ${prefix}_${i}% > ./log/${prefix}_${i}%.txt
  $date >> ./log/${prefix}_${i}%.txt
done
cd script