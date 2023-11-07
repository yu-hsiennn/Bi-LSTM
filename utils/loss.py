import torch
import torch.nn as nn

MSE = nn.MSELoss()

def motion_loss(x, output, y, out_len=30, weight_scale=1):
    return MSE(output, y)*weight_scale


def velocity_loss(x, output, y, out_len=30, weight_scale=1):
    velocity_gt = y[:, 1:, :]-y[:, :-1, :]
    velocity_output = output[:, 1:, :]-output[:, :-1, :]
    
    return MSE(velocity_output, velocity_gt)*weight_scale

def KL_loss(mean, log_var, weight_scale=1):
    return (- 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())) * weight_scale

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=="__main__":
    x = torch.randn(1, 50, 45)
    x[:, 10:-10, :] = 1
    y = torch.randn(1, 50, 45)
    output = torch.randn(1, 50, 45)
    mt_loss = motion_loss(x, output, y, 50, weight_scale=1)
    print(mt_loss)
    vel_loss = velocity_loss(x, output, y, 50, weight_scale=1)
    print(vel_loss)