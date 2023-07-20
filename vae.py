import torch.nn as nn
import torch
# from train import DEVICE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder_LSTM(nn.Module):
    def __init__(self, inp=45):
        super(Encoder_LSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size = inp,
            hidden_size= 1024, #192
            num_layers = 1,
            batch_first = True,
            bidirectional=True
        )
        self.fc = nn.Linear(1024*2, 512) #192*2, 96
        
    def forward(self, x):
        r_out, state = self.bilstm(x) # r_out[:,-1,:]
        out = self.fc(r_out[:,-1,:])
        return out, state

class Decoder_LSTM(nn.Module):
    def __init__(self, inp=45):
        super(Decoder_LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size = 512, #96
            hidden_size= 1024, #192
            num_layers = 1,
            batch_first = True,
            bidirectional=True
        )
        self.w1 = nn.Linear(1024*2, inp) #192*2

    def forward(self, x, hidden=None, out_len=30):
        x = [x.view(-1,1,512) for i in range(out_len)] #96
        x = torch.cat(x, 1)
        y, _  = self.rnn(x, hidden)
        y = self.w1(y)
        return y

# 512 -> 1024 -> 1024 -> 1024 -> 512
class Encoder_latent(nn.Module):
    def __init__(self, inp =512):
        super(Encoder_latent, self).__init__()
        self.input_dim = inp
        self.hidden_dim = 1024
        self.latent_dim = 512
        self.FC_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.FC_hidden2= nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_hidden3= nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.FC_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.FC_var = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x): # x shape = 128 * 1024
        x = self.FC_hidden(x)
        residual = x
        x = self.relu(x)
        x = self.relu(self.FC_hidden2(x))
        x = self.relu(self.FC_hidden3(x))
        x = x + residual
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        std = torch.exp(0.5*log_var)
        z = self.reparameterization(mean, std) # 512
        return z, mean, log_var

    def reparameterization(self, mean, std):
        epsilon = torch.rand_like(std).to(DEVICE)        # sampling epsilon # 給定隨機數值
        #epsilon = torch.FloatTensor(std.size()).normal_().to(DEVICE)
        scale = 2
        z = mean + std * epsilon * scale                          # re-parameterization trick
        return z
    
# 512 -> 1024 -> 1024 -> 1024 -> 512
class Decoder_latent(nn.Module):
    def __init__(self):
        super(Decoder_latent, self).__init__()
        self.output_dim = 512 #1024
        self.latent_dim = 512
        self.hidden_dim = 1024
        self.cat_dim = 1024
        self.FC_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.FC_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_hidden3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(1024, self.output_dim) # 1024 -> 512 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, eA):
        x = self.FC_hidden(x)
        residual = x
        x = self.relu(x)
        x = self.relu(self.FC_hidden2(x))
        x = self.relu(self.FC_hidden3(x))
        x = x + residual
        x = self.out(x) # 1024 -> 512 
        y = torch.cat((eA, x), 1) # 512 + 512
        eB_hat = self.out(y)
        return eB_hat

# class MTVAE(nn.Module):# 10 10
    # def __init__(self, Encoder_LSTM, Decoder_LSTM, Encoder_latent, Decoder_latent):
    #     super(MTVAE, self).__init__()
    #     self.Encoder_LSTM_x = Encoder_LSTM
    #     self.Encoder_LSTM_y = Encoder_LSTM
    #     self.Decoder_LSTM = Decoder_LSTM
    #     self.Encoder_latent = Encoder_latent
    #     self.Decoder_latent = Decoder_latent
    #     self.FC_mean = nn.Linear(1024, 512)
    #     self.FC_state = nn.Linear(1024*2, 1024)

    # def forward(self, x, inp_len, out_len):
    #     y = x[:,int(inp_len/2):,:]
    #     x = x[:,:int(inp_len/2),:]
    #     eA, stateA = self.Encoder_LSTM_x(x) # 512gggg////
    #     eB, stateB = self.Encoder_LSTM_y(y) # 512
    #     e = self.FC_mean(torch.cat((eA, eB), 1))
    #     # e = torch.cat((eA, eB), 1) # 1024
    #     state = (self.FC_state(torch.cat((stateA[0], stateB[0]), 2)),
    #             self.FC_state(torch.cat((stateA[1], stateB[1]), 2)))
    #     z, mean, log_var = self.Encoder_latent(e) # 1024
    #     eB_hat           = self.Decoder_latent(z, e)
    #     y = self.Decoder_LSTM(eB_hat, state, out_len)
    #     return y, mean, log_var

class MTGVAE(nn.Module): #final
    def __init__(self, Encoder_LSTM, Decoder_LSTM, Encoder_latent, Decoder_latent):
        super(MTGVAE, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM
        self.Encoder_latent = Encoder_latent
        self.Decoder_latent = Decoder_latent
    
    def forward(self, x, inp_len, out_len):
        eA, state = self.Encoder_LSTM(x)
        z, mean, log_var = self.Encoder_latent(eA)
        eB_hat           = self.Decoder_latent(z, eA)
        y = self.Decoder_LSTM(eB_hat, state, out_len)
        return y, mean, log_var

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    E_L = Encoder_LSTM(inp=45).to(DEVICE)
    D_L = Decoder_LSTM(inp=45)
    E_l = Encoder_latent(inp = 512)
    D_l = Decoder_latent()
    model = MTGVAE(E_L, D_L, E_l, D_l).to(DEVICE)

    inp = torch.randn(1, 30, 45).to(DEVICE)
    out, mean, log_var = model(inp, 30, 60)
    print(out.shape)

# import torch.nn as nn
# import torch

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Encoder_LSTM(nn.Module):
#     def __init__(self, inp=45):
#         super(Encoder_LSTM, self).__init__()
#         self.bilstm = nn.LSTM(
#             input_size=inp,
#             hidden_size=1024,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.fc = nn.Linear(1024 * 2, 512)
        
#     def forward(self, x):
#         r_out, state = self.bilstm(x)
#         out = self.fc(r_out[:, -1, :])
#         return out, state

# class Decoder_LSTM(nn.Module):
#     def __init__(self, inp=45):
#         super(Decoder_LSTM, self).__init__()
#         self.rnn = nn.LSTM(
#             input_size=512,
#             hidden_size=1024,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.w1 = nn.Linear(1024 * 2, inp)

#     def forward(self, x, hidden=None, out_len=30):
#         x = [x.view(-1, 1, 512) for _ in range(out_len)]
#         x = torch.cat(x, 1)
#         y, _ = self.rnn(x, hidden)
#         y = self.w1(y)
#         return y

# class Encoder_latent(nn.Module):
#     def __init__(self, inp=1024, cond_dim=10):
#         super(Encoder_latent, self).__init__()
#         self.input_dim = inp
#         self.hidden_dim = 1024
#         self.latent_dim = 512
#         self.cond_dim = cond_dim
#         self.FC_hidden = nn.Linear(self.input_dim + self.cond_dim, self.hidden_dim)
#         self.FC_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.FC_hidden3 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.FC_mean = nn.Linear(self.hidden_dim, self.latent_dim)
#         self.FC_var = nn.Linear(self.hidden_dim, self.latent_dim)

#     def forward(self, x, c):
#         x = torch.cat((x, c), dim=1)
#         x = self.FC_hidden(x)
#         residual = x
#         x = self.relu(x)
#         x = self.relu(self.FC_hidden2(x))
#         x = self.relu(self.FC_hidden3(x))
#         x = x + residual
#         mean = self.FC_mean(x)
#         log_var = self.FC_var(x)
#         var = torch.exp(0.5 * log_var)
#         z = self.reparameterization(mean, var)
#         return z, mean, log_var

#     def reparameterization(self, mean, var):
#         epsilon = torch.randn_like(var).to(DEVICE)
#         z = mean + var * epsilon
#         return z

# class Decoder_latent(nn.Module):
#     def __init__(self, cond_dim=10):
#         super(Decoder_latent, self).__init__()
#         self.output_dim = 512
#         self.latent_dim = 512
#         self.cond_dim = cond_dim
#         self.FC_hidden = nn.Linear(self.latent_dim + self.cond_dim, self.latent_dim)
#         self.out = nn.Linear(self.latent_dim, self.output_dim)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, z, c):
#         x = torch.cat((z, c), dim=1)
#         x = self.relu(self.FC_hidden(x))
#         x = self.relu(self.FC_hidden(x))
#         x = self.relu(self.FC_hidden(x))
#         y = self.out(x)
#         return y

# class CVAE(nn.Module):
#     def __init__(self, Encoder_LSTM, Decoder_LSTM, Encoder_latent, Decoder_latent, cond_dim=10):
#         super(CVAE, self).__init__()
#         self.Encoder_LSTM = Encoder_LSTM
#         self.Decoder_LSTM = Decoder_LSTM
#         self.Encoder_latent = Encoder_latent
#         self.Decoder_latent = Decoder_latent
#         self.cond_dim = cond_dim

#     def forward(self, x, c, inp_len, out_len):
#         e, state = self.Encoder_LSTM(x)
#         z, mean, log_var = self.Encoder_latent(e, c)
#         e_hat = self.Decoder_latent(z, c)
#         y = self.Decoder_LSTM(e_hat, state, out_len)
#         return y, mean, log_var

# if __name__ == "__main__":
#     E_L = Encoder_LSTM(inp=45).to(DEVICE)
#     D_L = Decoder_LSTM(inp=45)
#     E_l = Encoder_latent(inp=1024, cond_dim=10)
#     D_l = Decoder_latent(cond_dim=10)
#     model = CVAE(E_L, D_L, E_l, D_l, cond_dim=10).to(DEVICE)

#     inp = torch.randn(1, 30, 45).to(DEVICE)
#     cond = torch.randn(1, 10).to(DEVICE)
#     out, mean, log_var = model(inp, cond, 30, 60)
#     print(out.shape)


