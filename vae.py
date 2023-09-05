import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
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
            # add dropout
            dropout = 0.1,
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
            # add dropout
            dropout = 0.1,
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
        # self.hidden_dim = 1024
        self.hidden_dim = 256
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
        # epsilon = torch.FloatTensor(std.size()).normal_().to(DEVICE)
        scale = 1
        z = mean + std * epsilon * scale                          # re-parameterization trick
        return z
    
# 512 -> 1024 -> 1024 -> 1024 -> 512
class Decoder_latent(nn.Module):
    def __init__(self):
        super(Decoder_latent, self).__init__()
        self.output_dim = 512 #1024
        self.latent_dim = 512
        # self.hidden_dim = 1024
        self.hidden_dim = 256
        self.cat_dim = 1024
        self.FC_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.FC_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_hidden3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.out = nn.Linear(1024, self.output_dim) # 1024 -> 512 
        # self._out = nn.Linear(1024, self.output_dim)
        self.out = nn.Linear(256, self.output_dim) # 256 -> 512 
        self._out = nn.Linear(1024, self.output_dim)
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
        eB_hat = self._out(y)
        return eB_hat

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
    
    def visualize(self, z):
        # Create a list of indices for the dimensions of 'z'
        dimensions = list(range(z.size(1)))

        # Convert 'z' to a NumPy array for plotting
        z_values = z.squeeze().cpu().detach().numpy()

        # Create a line plot to visualize the values across dimensions
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, z_values, marker='o', linestyle='-')
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.title("Latent Vector Visualization")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # E_L = Encoder_LSTM(inp=45).to(DEVICE)
    # D_L = Decoder_LSTM(inp=45)
    # E_l = Encoder_latent(inp = 512)
    # D_l = Decoder_latent()
    # model = MTGVAE(E_L, D_L, E_l, D_l).to(DEVICE)

    # inp = torch.randn(1, 30, 45).to(DEVICE)
    # out, mean, log_var = model(inp, 30, 60)
    # print(out.shape)
    from sklearn.manifold import TSNE
    import numpy as np

    latent_vectors = []  # List to store the latent vectors
    means = []           # List to store the mean values

    # Generate some input data and move it to the same device as the model
    batch_size = 1000
    input_data = torch.randn(batch_size, 512).to(DEVICE)

    # Instantiate the Encoder_latent module and get the latent vectors and means
    encoder_latent = Encoder_latent(inp=512).to(DEVICE)
    latent_z, mean, log_var = encoder_latent(input_data)

    latent_vectors.extend(latent_z.detach().cpu().numpy())  # Move the latent vectors back to CPU and store in the list
    means.extend(mean.detach().cpu().numpy())            # Convert tensors to numpy arrays and store in the list

    # Convert the lists to numpy arrays
    latent_vectors = np.array(latent_vectors)
    means = np.array(means)

    # Reduce the dimensionality of the latent vectors to 2 using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=means[:, 0], cmap='viridis')  # Color points based on mean values
    plt.colorbar(label='Mean Value')
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()




