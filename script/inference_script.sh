# Human36
## V1

python inference.py -t infill -m 1212_V1_Human3.6M_train_angle_01_2010 -d Exp/missing_point -f S09_03_01.pkl -o /home/156785978/refactor/BiLSTM-VAE/result/for_thesis/infilling/v1_motion/S09_03_01_V1  -s


## V3
### 20

python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2010 -d Exp/missing_point -f S09_03_01.pkl -o result/exp/missing_point/S09_03_01_0926_V3_Human36M_train_angle_01_2010  -s
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2030 -d Exp/missing_point -f S09_03_01.pkl -o result/exp/missing_point/S09_03_01_0926_V3_Human36M_train_angle_01_2030  -s
python inference.py -t infill -m 0926_V3_Human3.6M_train_angle_01_2060 -d Exp/missing_point -f S09_03_01.pkl -o result/exp/missing_point/S09_03_01_0926_V3_Human36M_train_angle_01_2060  -s

## 4

## 2

# ChoreoMaster
#20
python inference.py -t infill -m 1011_V3_ChoreoMaster_Normal_train_angle_01_2010 -d  -f  -o  -v -s

# 4
python inference.py -t infill -m 0308_V3_ChoreoMaster_Normal_train_angle_01_0402 -d  -f  -o  -v -s
python inference.py -t infill -m 0215_V3_ChoreoMaster_Normal_train_angle_01_0410 -d  -f  -o  -v -s
# 2
python inference.py -t infill -m 0308_V3_ChoreoMaster_Normal_train_angle_01_0201 -d  -f  -o  -v -s
python inference.py -t infill -m 0308_V3_ChoreoMaster_Normal_train_angle_01_0201 -d Exp/others -f skeleton_0_380.pkl -o result/exp/others/tmp -v -s

python inference.py -t infill -m 1011_V3_ChoreoMaster_Normal_train_angle_01_2010 -d Human3.6M/test_angle -f s_09_act_03_subact_01_ca_01.pickle -o result/nthu_report/0328/S09_03_01_broken -v -s
python inference.py -t infill -m 1011_V3_ChoreoMaster_Normal_train_angle_01_2010 -d ChoreoMaster_Normal/test_angle -f d_act_19_ca_01.pkl -o result/nthu_report/0328/d_act_19_ca_01_2010 -v -s
python inference.py -t infill -m 0215_V3_ChoreoMaster_Normal_train_angle_01_0210 -d ChoreoMaster_Normal/test_angle -f d_act_19_ca_01.pkl -o result/nthu_report/0328/d_act_19_ca_01_0210 -v -s




python inference.py -t infill -m 0609_V3_ChoreoHM36_train_angle_01_2010 -d Human3.6M/test_angle -f s_09_act_03_subact_01_ca_01.pickle -o result/ohmygod/S09_03_01_broken -v -s