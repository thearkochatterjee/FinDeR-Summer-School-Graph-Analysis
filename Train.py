from zmq import device
import DataGen

#import pandas
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

graphs, labels = DataGen.GenerateGraphs(100, 10)
matrices = DataGen.convert2Matrix(graphs)
matrices = DataGen.toarray(matrices)
matrices = np.stack(matrices)

# DataGen.plotGraphs(graphs,labels,10)

class Encoder(nn.Module):
    def __init__(self, num_latent,num_hidden):
        super().__init__()
        self.num_latent = num_latent
        self.num_hidden = num_hidden
        
        self.encode = nn.Sequential(
            nn.Conv2d(1,1,3,stride=1), # 1 channel, kernel size =3, stride =1, reduce size from 10x10 to 8x8
            nn.ReLU(),
            nn.Conv2d(1,1,3,stride=1), # 1 channel, kernel size =3, stride =1, reduce size from 8x8 to 6x6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_hidden,num_latent)
        )

        self.encode2mu = nn.Sequential(
            nn.Linear(num_latent,num_latent)
        )

        self.encode2logvar = nn.Sequential(
            nn.Linear(num_latent,num_latent)
        )
        
    def forward(self, X):
        encoded = self.encode(X)
        encodedmu = self.encode2mu(encoded)
        encodedlogvar = self.encode2logvar(encoded)
        return encodedmu, encodedlogvar
    
class Decoder(nn.Module):
    def __init__(self, num_latent,num_hidden):
        super().__init__()
        self.num_latent = num_latent
        self.num_hidden = num_hidden
        
        self.decode = nn.Sequential(
            nn.Linear(num_latent,num_hidden),
            nn.ReLU(),
            nn.Unflatten(1, torch.Size([1,6,6])),
            nn.ConvTranspose2d(1,1,3,stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1,1,3,stride=1),
            nn.ReLU()
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded
    
class AutoEncoder(nn.Module):
    def __init__(self, num_latent,num_hidden):
        super().__init__()
        self.num_latent = num_latent
        self.num_hidden = num_hidden
        
        self.encoder = Encoder(num_latent = self.num_latent,
                               num_hidden = self.num_hidden)
        self.decoder = Decoder(num_latent = self.num_latent,
                               num_hidden = self.num_hidden)
        
    def forward(self, X):
        encodedmu, encodedsigma = self.encoder(X)
        z = self.reparameterize(encodedmu,encodedsigma)
        decoded = self.decoder(z)
        return decoded, encodedmu, encodedsigma  # <- return a tuple of two values

    def reparameterize(self,mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# num_points, D_orig = X_D.shape
num_latent = 16
num_hidden = 36

model = AutoEncoder(num_latent,num_hidden)

# create an optimizer object
# Adam optimizer with learning rate
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# mean-squared error loss
# criterion = nn.MSELoss() + nn.KLDivLoss()

def lossfcn(input,output,mu,logvar):
    someloss = nn.MSELoss()
    mse = someloss(input, output)
    kld = -0.5*torch.sum(1+logvar-mu**2-logvar.exp())
    return mse + kld

matrices = matrices[:,np.newaxis,:,:]
X_torch = torch.from_numpy(matrices)
X_torch = X_torch.float()

epochs=200
for epoch in range(epochs):
    for i in range(len(graphs)):
        optimizer.zero_grad()
        Xi = X_torch[i,:,:,:]
        Xi = Xi.unsqueeze(1)
        decoded, encodedmu, encodesigma = model(Xi)

        # compute training reconstruction loss
        # train_loss = criterion(decoded, X_torch)
        train_loss = lossfcn(Xi,decoded,encodedmu,encodesigma)
        
        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

    # display the epoch training loss
    if epoch%5 == 0:
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, train_loss.item()))
    

# reconstruction of the data
output, _, _ = model(X_torch)
# make everything symmetric
output = output.detach().numpy()
outputSymmetric = 0.5*(output+np.transpose(output,(0,1,3,2)))
outRounded=np.round(outputSymmetric)
outRounded=outRounded.astype(int)

# not it's a symmetric integer matrix

# now transferring back into graphs
matrix_list = []
for j in range(outRounded.shape[0]):
    matrix_list.append(outRounded[j,0,:,:])

graphs = DataGen.convert2Graph(matrix_list)

# plotting some graphs
DataGen.plotGraphs(graphs,num_plots=10)