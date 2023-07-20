
prefix="10"
time="1011"
dataset="ChoreoMaster_Normal"
testset="ChoreoMaster_Normal"
path="./log/choreo/"
for i in {1..2..1}
do
  python evaluate.py -p $time -ts $testset -s $dataset -o $prefix > $path${prefix}_${i}.txt
done
prefix="30"
for i in {1..2..1}
do
  python evaluate.py -p $time -ts $testset -s $dataset -o $prefix > $path${prefix}_${i}.txt
done
prefix="60"
for i in {1..2..1}
do
  python evaluate.py -p $time -ts $testset -s $dataset -o $prefix > $path${prefix}_${i}.txt
done