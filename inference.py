import os
import pathlib
import numpy as np
import pickle
import argparse    
import torch
from processing import get_single_data, calculate_position
from visualize import AnimePlot

class Inference:
    def __init__(self, joint_def, model, data_dir) -> None:
        with open("Tpose/T_pose_normalize_x_inverse.pkl", 'rb')as fpick:
            self.TPose = pickle.load(fpick)[0]
        self.joint_def = joint_def
        self.model = model
        self.inp_len = int(model.split('_')[-1][:2])
        self.out_len = int(model.split('_')[-1][-2:])
        self.dataset = data_dir.split('/')[0]
        self.dir = data_dir.split('/')[1]
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.__load_models()

    def __load_models(self):
        for part in self.joint_def.part_list:
            self.models[part] = self.load_model(part)

    def add_noise(self, motion):
        noise = np.array([0.04 * np.random.normal(0, 1, len(motion)) for _ in range(45)])    
        return motion + noise.T

    def mpjpe(self, y, out):
        sqerr = (out -  y)**2
        distance = np.zeros((sqerr.shape[0], int(sqerr.shape[1]/3))) # distance shape : batch_size, 15
        for i, k in enumerate(np.arange(0, sqerr.shape[1],3)): 
            distance[:, i] = np.sqrt(np.sum(sqerr[:,k:k+3],axis=1))  # batch_size
        return np.mean(distance)

    def load_model(self, part):
        path = os.path.join("ckpt", self.model, part)
        model = torch.load(path + "/best.pth",map_location = self.DEVICE).to(self.DEVICE)
        model.eval()
        return model
            
    def load_data(self, file):
        print(">>> Data loaded -->", file)
        data = get_single_data(self.dataset, self.dir, file)
        print(len(data))
        data = torch.tensor(data.astype("float32"))
        return data

    def interpolation(self, data):
        data = list(data.numpy())
        ran = int(self.inp_len/2)
        result = data[:ran]
        for j in range(0, len(data)-(self.inp_len+self.out_len), (ran+self.out_len)):
            last_frame = result[-1]
            for frame in range(self.out_len):
                result.append(last_frame+frame*(data[j+ran+self.out_len]-last_frame)/(self.out_len+1))
            result.extend(data[j+ran+self.out_len:j+ran+self.out_len+ran])
        tail = len(data) - len(result)
        result.extend(data[-tail:])
        return np.array(result)

    def demo(self, dim, model, data):
        motion = data.to(self.DEVICE)
        motion = motion.view((1, -1, dim))
        inp = motion
        out, _, _ = model(inp, 50, 50)
        result = out
        result = result.view((-1,dim))
        return result

    def concatenation(self, dim, model, data):
        motion = data.to(self.DEVICE)
        motion = motion.view((1, -1, dim))
        result = motion[:, :int(self.inp_len/2), :]
        ran = int(self.inp_len/2)
        for j in range(0, len(data)-ran, ran):
            missing_data = torch.ones_like(motion[:, 0:self.out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, j+ ran : j + ran * 2, :]), 1)
            out, _, _ = model(inp, self.inp_len+self.out_len, self.inp_len+self.out_len)
            result = torch.cat((result, out[:, ran:2 * ran + self.out_len, :]), 1)
            
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, motion[:, -tail:, :]), 1)  
        result = result.view((-1,dim))
        return result

    def infilling(self, dim, model, data):
        motion = data.to(self.DEVICE)
        motion = motion.view((1, -1, dim))
        result = motion[:, :int(self.inp_len/2), :]
        ran = int(self.inp_len/2)
        for j in range(0, len(data)-(ran*2+self.out_len), ran+self.out_len):
            missing_data = torch.ones_like(motion[:, 0:self.out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, j + self.out_len +ran: j + self.out_len + ran*2, :]), 1)
            out, _, _ = model(inp, self.inp_len+self.out_len, self.inp_len+self.out_len)
            result = torch.cat((result, out[:, ran:2 * ran + self.out_len, :]), 1)
            
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, motion[:, -tail:, :]), 1)  
        result = result.view((-1,dim))
        return result

    def smooth(self, dim, model, data):
        test = data.to(self.DEVICE)
        test = test.view((1,-1,dim))
        ran = int(self.inp_len/2)
        result = test[:, :ran, :]
        for j in range(0, len(data)-(self.inp_len+self.out_len), self.out_len):
            missing_data = torch.ones_like(test[:, 0:self.out_len, :])
            inp = torch.cat((result[:, j:j+ran, :], missing_data, test[:, j+ran+self.out_len:j+self.inp_len+self.out_len, :]), 1)
            out, _, _ = model(inp, self.inp_len+self.out_len, self.inp_len+self.out_len)                 
            result = torch.cat((result, out[:, ran:ran+self.out_len, :]), 1)  
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, test[:, -tail:, :]), 1)
        result = result.view((-1,dim))
        return result

    def get_result(self, data, model, part):
        dim = self.joint_def.n_joints_part[part]
        if self.type == 'concat':
            result = self.concatenation(dim, model, data)
        elif self.type == 'infill':
            result = self.infilling(dim, model, data)
        elif self.type == 'smooth':
            result = self.smooth(dim, model, data)
        elif self.type == 'inter':
            result = self.interpolation(data)
            return result
        elif self.type == 'demo':
            result = self.demo(dim, model, data)
        else:
            print('No this type!!')
        return result.detach().cpu().numpy()

    def main(self, file, type, partial = True):
        data = self.load_data(file)
        self.type = type
        if partial:
            partDatas = {}
            for part in self.joint_def.part_list:
                model = self.models[part]
                part_data = self.joint_def.cat_torch(part, data)
                partDatas[part] = self.get_result(part_data, model, part)
            pred = self.joint_def.combine_numpy(partDatas)
        else:
            part = 'entire'
            model = self.load_model(part)
            pred = self.get_result(data, model, part)
        pred = calculate_position(pred, self.TPose)
        gt = calculate_position(data, self.TPose)
        if type == 'concat':
            result = np.zeros_like(pred)
            ran = int(self.inp_len/2)
            for j in range(0, len(gt)-ran+1, ran):
                step = int(j / ran)
                result[(ran + self.out_len) * step: (ran + self.out_len) * step + ran] = gt[j: j + ran]
            gt = result
        assert(len(gt) == len(pred))
        return gt, pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, help="Evaluation type", required=True) # infill or concat or smooth
    parser.add_argument("-m", "--model", type=str, help="Model name", required=True) # e.g. Human3.6M_train_angle_01_1_1010
    parser.add_argument("-d", "--dataset", type=str, help="Dataset Dir", required=True) # e.g. Human3.6M/test_angle
    parser.add_argument("-f", "--file", type=str, help="File name")                          
    parser.add_argument("-o", "--out", type=str, help="Output Dir", default='result/demo.pkl')
    parser.add_argument("-v", "--visual", help="Visualize", action="store_true")
    parser.add_argument("-p","--partial", help = "Partial model", action="store_true")
    parser.add_argument("-s","--save", help="save or not", action="store_true")
    args = parser.parse_args()

    joint_ver = "JointDef" + args.model.split('_')[1]
    mod = __import__('model_joints', fromlist=joint_ver)
    joint_def = getattr(mod, joint_ver)
    joint_def = joint_def()
    inference = Inference(joint_def, args.model, args.dataset)
    gt, pred = inference.main(args.file, args.type)
    
    path = args.out.split('.')
    dir_idx = args.out.rfind('/')
    dir = args.out[:dir_idx]
    pathlib.Path(str(dir)).mkdir(parents=True, exist_ok=True)
    if args.save:
        with open(f'{path[0]}.pkl', 'wb') as fpick:
            pickle.dump(pred, fpick)
        with open(f'{path[0]}_ori.pkl', 'wb') as fpick:
            pickle.dump(gt, fpick)
    if args.visual:
        figure = AnimePlot()
        labels = ['Predicted', 'Ground Truth']
        figure.set_fig(labels, path[0])
        figure.set_data([pred, gt], len(pred))
        figure.animate()