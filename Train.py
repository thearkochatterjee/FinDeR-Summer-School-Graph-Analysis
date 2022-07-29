from zmq import device
import DataGen

import extractLowerTriangle

#import pandas
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

graphs, labels = DataGen.GenerateGraphs(1000, 10)
matrices = DataGen.convert2Matrix(graphs) # list of matrices
adjacencyMatricesAsVecs = extractLowerTriangle.listOfMatsto2D(matrices)

numVars = adjacencyMatricesAsVecs.shape[0]
# DataGen.plotGraphs(graphs,labels,10)

class Encoder(nn.Module):
    def __init__(self, num_latent,num_hidden):
        super().__init__()
        self.num_latent = num_latent
        self.num_hidden = num_hidden
        
        self.encode = nn.Sequential(
            nn.Linear(numVars,self.num_hidden), 
            nn.Tanh(),
            nn.Linear(self.num_hidden,self.num_hidden), 
            nn.Tanh(),
            nn.Linear(self.num_hidden,self.num_hidden),
            nn.Tanh(),
            nn.Linear(self.num_hidden,self.num_latent)
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
            nn.Tanh(),
            nn.Linear(self.num_hidden,self.num_hidden),
            nn.Tanh(),
            nn.Linear(self.num_hidden,self.num_hidden),
            nn.Tanh(),
            nn.Linear(self.num_hidden,numVars),
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
num_hidden = 32

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
    return mse# + kld

X_torch = torch.from_numpy(adjacencyMatricesAsVecs.T).float()
epochs=10000
for epoch in range(epochs):
    optimizer.zero_grad()
    decoded, encodedmu, encodesigma = model(X_torch)

    # compute training reconstruction loss
    # train_loss = criterion(decoded, X_torch)
    train_loss = lossfcn(X_torch,decoded,encodedmu,encodesigma)
        
    # compute accumulated gradients
    train_loss.backward()

    # perform parameter update based on current gradients
    optimizer.step()

    # display the epoch training loss
    if epoch%500 == 0:
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, train_loss.item()))
    

# reconstruction of the data
output, _, _ = model(X_torch)
output = output.detach().numpy()
output = np.transpose(output)
output = np.round(output)
output = output.astype(int)
listOfMats = extractLowerTriangle.listOfMatsFrom2DObj(output)
graphs = DataGen.convert2Graph(listOfMats)

DataGen.plotGraphs(graphs,num_plots=10)