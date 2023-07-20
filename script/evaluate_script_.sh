cd ..
# prefix="10"
time="sample2std"
dataset="ChoreoMaster"
testset="Choreomaster"
# for i in {1..5..1}
# do
#   python evaluate.py -p $time -ts $dataset -s $dataset -o $prefix > ./log/sample_ChoreoMaster/${prefix}_${i}.txt
# done
# prefix="30"
# for i in {1..5..1}
# do
#   python evaluate.py -p $time -ts $dataset -s $dataset -o $prefix > ./log/sample_ChoreoMaster/${prefix}_${i}.txt
# done
prefix="60"
for i in {1..5..1}
do
  python evaluate.py -p $time -ts $dataset -s $dataset -o $prefix > ./log/sample_ChoreoMaster/${prefix}_${i}.txt
done