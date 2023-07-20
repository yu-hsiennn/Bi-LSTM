cd ..
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2010 -d Exp/missing_point -f S09_03_01.pkl -o result/exp/missing_point/S09_03_01_0926_V3_Human36M_train_angle_01_2010  -s
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2030 -d Exp/missing_point -f S09_03_01.pkl -o result/exp/missing_point/S09_03_01_0926_V3_Human36M_train_angle_01_2030  -s
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2060 -d Exp/missing_point -f S09_03_01.pkl -o result/exp/missing_point/S09_03_01_0926_V3_Human36M_train_angle_01_2060  -s
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2010 -d Human3.6M/missing_rate_25% -f s_09_act_03_subact_01_ca_01.pickle -o result/exp/missing_point/25%  -s
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2010 -d Human3.6M/test_angle -f s_09_act_03_subact_01_ca_01.pickle -o result/exp/missing_point/0%  -s -v