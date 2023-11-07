from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm
from utils.arguments import get_args
from utils.data_utils import *
from model.model_joints import *
import numpy as np
import utils.loss as loss
import model.bilstm_vae as Model
import os, sys
import pickle, json
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train():
    def __init__(self, joint_def, dataset, train_dir, train_ca, inp_len, out_len, hyperparams) -> None:
        self.joint_def = joint_def
        self.dataset = dataset
        self.train_dir = train_dir
        self.train_ca = train_ca
        self.inp_len = inp_len
        self.out_len = out_len
        self.hyperparams = hyperparams
        
    def total_loss(self, x, output, y, out_len, mean, log_var, epoch):
        weight = 1
        return loss.motion_loss(x, output, y, out_len, weight_scale=1), loss.velocity_loss(x, output, y, 
                                    out_len = out_len , weight_scale=5), loss.KL_loss(mean, log_var, weight) 

    def save_result(self, save_path, result):
        with open(save_path+ "/result.txt" , "a") as f:
            f.write(result)
            f.write("\n")

    def load_data(self):
        validation_split = .2
        shuffle_dataset = True
        random_seed= 42
        # Get data
        train_data = get_train_data(self.dataset, self.train_dir, ca=self.train_ca, inp_len=self.inp_len, out_len=self.out_len, randomInput=False)   

        dataset_size = len(train_data['x'])
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = train_sampler)
        test_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = test_sampler)

        return train_, test_, train_sampler, test_sampler

    def divide_data(self, train_sampler, test_sampler, part):
        train_data = get_part_train_data(self.dataset, self.train_dir, ca=self.train_ca, part=part,
                                inp_len=self.inp_len, out_len=self.out_len, randomInput=False, joint_def=joint_def)
        train_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = train_sampler)
        test_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = test_sampler)
        return train_, test_
        
    def load_model(self, part):
        dim = self.joint_def.n_joints_part[part]
        # Get model
        E_L = Model.Encoder_LSTM(inp=dim)
        D_L = Model.Decoder_LSTM(inp=dim)

        E_l = Model.Encoder_latent(inp = 512)
        D_l = Model.Decoder_latent()            
        model = Model.MTGVAE(E_L, D_L, E_l, D_l).to(DEVICE)
        model = model.to(DEVICE)

        return model

    def train_epoch(self, model, train_, optimizer, epoch):
        model.train()
        losses = loss.AverageMeter()

        for i, (x, y) in enumerate(tqdm(train_)):
            x_np = x.numpy()
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss_dict = {'angle': loss.AverageMeter(), 'vel': loss.AverageMeter(), 'kl': loss.AverageMeter()}
            out, mean, log_var = model(x, self.inp_len + self.out_len, self.inp_len + self.out_len)
            loss_angle, loss_vel, loss_kl = self.total_loss(x_np, out, y.to(DEVICE), self.inp_len + self.out_len, mean, log_var, epoch)
            loss_ = loss_angle + loss_vel + loss_kl

            loss_dict['angle'].update(loss_angle.item(), x.size(0))
            loss_dict['vel'].update(loss_vel.item(), x.size(0))
            loss_dict['kl'].update(loss_kl.item(), x.size(0))

            losses.update(loss_.item(), x.size(0))
            loss_.backward()
            optimizer.step()

        train_loss = losses.avg
        epoch_loss = (loss_dict['angle'].avg, loss_dict['vel'].avg, loss_dict['kl'].avg)

        return train_loss, epoch_loss

    def eval_epoch(self, model, test_, epoch):
        model.eval()
        losses = loss.AverageMeter()

        for i, (x, y) in enumerate(tqdm(test_)):
            x_np = x.numpy()
            x = x.to(DEVICE)
            loss_dict = {'angle': loss.AverageMeter(), 'vel': loss.AverageMeter(), 'kl': loss.AverageMeter()}
            out, mean, log_var = model(x, self.inp_len + self.out_len, self.inp_len + self.out_len)
            loss_angle, loss_vel, loss_kl = self.total_loss(x_np, out, y.to(DEVICE), self.inp_len + self.out_len, mean, log_var, epoch)
            loss_ = loss_angle + loss_vel + loss_kl

            loss_dict['angle'].update(loss_angle.item(), x.size(0))
            loss_dict['vel'].update(loss_vel.item(), x.size(0))
            loss_dict['kl'].update(loss_kl.item(), x.size(0))

            losses.update(loss_.item(), x.size(0))

        return losses.avg

    def save_best_model(self, model, best_loss, model_path, test_loss):
        if test_loss < best_loss:
            torch.save(model, os.path.join(model_path, "best.pth"))
            best_loss = test_loss
        else:
            torch.save(model, os.path.join(model_path, "last.pth"))

        return best_loss

    def print_and_save_result(self, result, model_path):
        print('Using device:', DEVICE)
        print(result)
        self.save_result(model_path, result)

    def save_loss_info(self, model_path, loss_list, loss_detail):
        with open(os.path.join(model_path, 'loss.pkl'), 'wb') as fpick:
            pickle.dump(loss_list, fpick)
        with open(os.path.join(model_path, 'loss_detail.pkl'), 'wb') as fpick:
            pickle.dump(loss_detail, fpick)
            
    def train(self, model, part, train_, test_, save_path):
        best_loss = 1000
        optimizer = Adam(model.parameters(), lr=self.hyperparams['lr'])
        loss_list = []
        loss_detail = []
        model_path = os.path.join(save_path, part)

        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        for epoch in range(self.hyperparams['epochs']):
            train_loss, epoch_loss = self.train_epoch(model, train_, optimizer, epoch)
            test_loss = self.eval_epoch(model, test_, epoch)
            best_loss = self.save_best_model(model, best_loss, model_path, test_loss)

            result = f"Part = {part}, Epoch = {epoch + 1}/{self.hyperparams['epochs']}, train_loss = {train_loss:.7f}, test_loss = {test_loss:.7f}"
            loss_list.append([train_loss, test_loss])
            loss_detail.append(epoch_loss)
            self.print_and_save_result(result, model_path)

        self.save_loss_info(model_path, loss_list, loss_detail)
        print("Training done!")

if __name__=='__main__':
    # Get command line arguments
    args = get_args()

    # Define the save path for model checkpoints
    save_path = os.path.join("ckpt", f"{args.prefix}_{args.version}_{os.path.basename(args.dataset)}_{args.train_dir}_{args.train_ca}_{args.inp_len}_{args.out_len}")

    # Define hyperparameters for training
    hyperparams = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr
    }

    # Create the save path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.system(f'mkdir "{save_path}"')
        # Store training options in a JSON file for reference
        opt = {
            "dataset": os.path.basename(args.dataset),
            "train_dir": args.train_dir,
            "train_ca": args.train_ca,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "inp_len": args.inp_len,
            "out_len": args.out_len
        }
        with open(os.path.join(save_path, 'opt.json'), 'w') as fp:
            json.dump(opt, fp)
    
    # Determine the joint definition based on the version specified
    joint_def = getattr(sys.modules[__name__], f"JointDef{args.version}")()
    
    train = Train(joint_def, args.dataset, args.train_dir, args.train_ca, args.inp_len, args.out_len, hyperparams)
    
    # Load training and testing data and their samplers
    train_, test_, train_sampler, test_sampler = train.load_data()
    
    # Train models for each part defined in the joint definition
    for part in joint_def.part_list:
        model = train.load_model(part)
        train_, test_ = train.divide_data(train_sampler, test_sampler, part)
        train.train(model, part, train_, test_, save_path)