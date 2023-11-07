from model.model_joints import *
from utils.processing import *
from utils.arguments import get_args
from utils.data_utils import load_Tpos, load_pkl_file, load_xlsx_file, save_pkl_file
from utils.visualization import AnimePlot
import os, sys
import numpy as np
import torch

class Inference:
    def __init__(self, args) -> None:
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Tpos = load_Tpos(args.tpose)
        self.model_path = args.model
        self.inp_len = int(os.path.basename(args.model).split('_')[-2])
        self.out_len = int(os.path.basename(args.model).split('_')[-1])
        self.type = args.type
        self.vis = args.visual
        self.save_path = args.save
        self.joint_def = getattr(sys.modules[__name__], f"JointDef{os.path.basename(args.model).split('_')[1]}")()
        self.joint_info = Joints_info(os.path.basename(args.model).split('_')[1])
        self.models = self.load_model()
        self.results = []

    def load_model(self):
        models = {}
        for part in self.joint_def.part_list:
            model = torch.load(f"{os.path.join(self.model_path, part)}/best.pth", self.DEVICE).to(self.DEVICE)
            model.eval()
            models[part] = model       
        return models   
    
    def concat2seq(self, data1, data2, model, dim):
        """
        Concatenate two sequences of data using a model.

        Args:
        data1 (Tensor): The first sequence of data.
        data2 (Tensor): The second sequence of data.
        model: The model used for concatenation.
        dim (int): The dimension of the data.

        Returns:
        Tensor: The concatenated sequence of data.

        Note: This function takes two sequences of data (data1 and data2), a model, and the dimension of the data.
        It concatenates the two sequences using the model and returns the resulting concatenated sequence.
        """
        prefix_size = self.inp_len // 2
        generate_size = self.inp_len + self.out_len
        
        # Move data1 and data2 to the same device as missing_data
        data1 = data1.to(self.DEVICE)
        data2 = data2.to(self.DEVICE)

        # Reshape data1 and data2
        data1 = data1.view(1, -1, dim)
        data2 = data2.view(1, -1, dim)
        missing_data = torch.ones(1, self.out_len, dim).to(self.DEVICE)
        
        inp = torch.cat((data1, missing_data, data2), 1)
        out, _, _ = model(inp, generate_size, generate_size)
        result = torch.cat((data1, out[:, prefix_size : generate_size - prefix_size, :], data2), 1)
        
        return result.view((-1,dim)).detach().cpu().numpy()

    def concatenation(self, data):
        """
        Perform concatenation of multiple data sequences.

        Args:
        data (list of Tensors): A list of data sequences to concatenate.

        Returns:
        list: A list of concatenated data sequences.
        GT: Ground Truth.

        Note: This function performs concatenation of multiple data sequences. It calculates angles and uses a crossfading
        technique to combine the sequences. The function returns a list of concatenated data sequences.
        """
        joints_connect = self.joint_info.get_joints_connect()
        total_data = len(data)
        prefix_size = self.inp_len // 2
        
        self.results.extend(calculate_position(data[0], self.Tpos, joints_connect))
    
        for idx in range(0, total_data - 1):
            first_data = calculate_angle(self.results[-prefix_size : ], joints_connect)
            second_data = data[idx + 1][ : prefix_size]
            self.results = self.results[:-prefix_size]

            part_results = {}
            for part in self.joint_def.part_list:
                dim = self.joint_def.n_joints_part[part]
                model = self.models[part]
                first_part_data = self.joint_def.cat_torch(part, torch.tensor(first_data.astype("float32")))
                second_part_data = self.joint_def.cat_torch(part, torch.tensor(second_data.astype("float32")))
                part_results[part] = self.concat2seq(first_part_data, second_part_data, model, dim)
            
            result = calculate_position(self.joint_def.combine_numpy(part_results), self.Tpos, joints_connect)
            
            first_result, second_result = result[:prefix_size], result[prefix_size:]
            result = crossfading(first_result, second_result, 1)
            first_result, second_result = result[:-prefix_size], result[-prefix_size:]
            self.results.extend(crossfading(first_result, second_result, 1))
            self.results.extend(calculate_position(data[idx + 1][prefix_size : ], self.Tpos, joints_connect))
            
        GT = np.zeros_like(np.array(self.results))
        GT_index = 0
        for idx in range(0, total_data):
            end = GT_index + len(data[idx])
            GT[GT_index : end] = calculate_position(data[idx], self.Tpos, joints_connect)
            GT_index += (len(data[idx]) + self.out_len)
        
        return  np.array(self.results), GT

    def infilling(self, data):
        """
        Fill in missing motion data and generate a completed motion sequence using a set of models.

        Args:
        data (numpy.ndarray): Input motion data with missing values to be infilled.

        Returns:
        numpy.ndarray: Completed motion sequence with missing values filled in.
        GT: Ground Truth.

        Note: This method takes motion data with missing values, fills them in using a set of models,
        and generates a completed motion sequence. It operates on individual parts of the data and combines the results.
        """
        total_len = len(data)
        generate_size = self.inp_len + self.out_len
        prefix_size = self.inp_len // 2
        step = self.out_len + prefix_size
        
        part_results = {}
        for part in self.joint_def.part_list:
            dim = self.joint_def.n_joints_part[part]
            model = self.models[part]
            part_data = self.joint_def.cat_torch(part, torch.tensor(data.astype("float32")))
            
            part_data = part_data.to(self.DEVICE).view((1,-1,dim))
            part_results[part] = part_data[:, : prefix_size, :]
            for start in range(0, total_len - generate_size, step):
                missing_data = torch.ones_like(part_data[:, 0 : self.out_len, :])
                input_data = torch.cat((
                    part_results[part][:, (start) : (start + prefix_size), :],
                    missing_data,
                    part_data[:, (start + prefix_size + self.out_len) : (start + generate_size)]
                ), 1)
                out, _, _ = model(input_data, generate_size, generate_size)
                part_results[part] = torch.cat((part_results[part], out[:, prefix_size : generate_size, :]), 1)
            reduant = total_len - len(part_results[part].view((-1,dim)))
            if reduant > 0:
                part_results[part] = torch.cat((part_results[part], part_data[:, -reduant : , :]), 1).view((-1,dim)).detach().cpu().numpy()
    
        self.results = calculate_position(self.joint_def.combine_numpy(part_results), self.Tpos, self.joint_info.get_joints_connect())
        
        return np.array(self.results), calculate_position(data, self.Tpos, self.joint_info.get_joints_connect())

    def smooth(self, data):
        """
        Smooth and complete motion data using a set of models.

        Args:
        data (numpy.ndarray): Input motion data to be smoothed and completed.

        Returns:
        numpy.ndarray: Smoothed and completed motion data.
        GT: Ground Truth.

        Note: This method takes motion data, fills in missing values, and smoothes it using a set of models.
        It operates on individual parts of the data and combines the results to generate the final smoothed data.
        """
        total_len = len(data)
        generate_size = self.inp_len + self.out_len
        prefix_size = self.inp_len // 2
        step = self.out_len
        
        part_results = {}
        for part in self.joint_def.part_list:
            dim = self.joint_def.n_joints_part[part]
            model = self.models[part]
            part_data = self.joint_def.cat_torch(part, torch.tensor(data.astype("float32")))
            
            part_data = part_data.to(self.DEVICE).view((1,-1,dim))
            part_results[part] = part_data[:, : prefix_size, :]
            for start in range(0, total_len - generate_size, step):
                missing_data = torch.ones_like(part_data[:, 0 : self.out_len, :])
                input_data = torch.cat((
                    part_results[part][:, (start) : (start + prefix_size), :],
                    missing_data,
                    part_data[:, (start + prefix_size + self.out_len) : (start + generate_size)]
                ), 1)
                out, _, _ = model(input_data, generate_size, generate_size)
                part_results[part] = torch.cat((part_results[part], out[:, prefix_size : prefix_size + self.out_len, :]), 1)
            reduant = total_len - len(part_results[part].view((-1,dim)))
            if reduant > 0:
                part_results[part] = torch.cat((part_results[part], part_data[:, -reduant : , :]), 1).view((-1,dim)).detach().cpu().numpy()
        
        self.results = calculate_position(self.joint_def.combine_numpy(part_results), self.Tpos, self.joint_info.get_joints_connect())

        return np.array(self.results), calculate_position(data, self.Tpos, self.joint_info.get_joints_connect())

    def get_result(self, data):
        if self.type == 'concat':
            result, GT = self.concatenation(data)
        elif self.type == 'infill':
            result, GT = self.infilling(data)
        elif self.type == 'smooth':
            result, GT = self.smooth(data)
        else:
            print('No this type!!')
            exit()
        return result, GT

    def visualization(self, res, gt, type, draw = True):
        if draw:
            figure = AnimePlot(self.joint_info)
            labels = ['Predicted', 'Ground Truth'] if type in ["infill", "smooth"] else ['concat',  'origin']
            figure.set_fig(labels, self.save_path)
            figure.set_data([res, gt], len(res))
            figure.animate()

    def main(self):
        if args.file.split('.')[-1] in ["pkl", "pickle"]:
            data = load_pkl_file(args.file)
            data = calculate_angle(normalize(data, self.joint_info.get_joint()), self.joint_info.get_joints_connect())
        elif args.file.split('.')[-1] == "xlsx":
            data = [calculate_angle(normalize(data, self.joint_info.get_joint()), self.joint_info.get_joints_connect()) for data in load_xlsx_file(args.file)]
        else:
            print("Data file must be pickle/xml")
        
        result, GT = self.get_result(data)
        self.visualization(result, GT, self.type, self.vis)
        
        if not os.path.exists(self.save_path[:len(os.path.basename(self.save_path))]):
            os.system(f'mkdir "{self.save_path[:len(os.path.basename(self.save_path))]}"')
            
        save_pkl_file(f"{self.save_path}_pred.pkl", result)
        

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    inference = Inference(args)
    inference.main()