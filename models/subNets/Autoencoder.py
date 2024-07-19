import torch
import torch.nn as nn
from torch.nn import functional as F
# VAE model
class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
    def forward(self,z_mean,z_log_var):
        epsilon=torch.randn(z_mean.shape)
        epsilon=epsilon.to('cuda')
        return z_mean+(z_log_var/2).exp()*epsilon
class VaeEncoder(nn.Module):
    def __init__(self,original_dim,intermediate_dim,latent_dim):
        super(VaeEncoder, self).__init__()
        self.Dense=nn.Linear(original_dim,intermediate_dim)
        self.z_mean=nn.Linear(intermediate_dim,latent_dim)
        self.z_log_var=nn.Linear(intermediate_dim,latent_dim)
        self.sample=Sample()
    def forward(self,x):
        o = F.relu(self.Dense(x))
        z_mean = self.z_mean(o)
        z_log_var = self.z_log_var(o)
        o = self.sample(z_mean,z_log_var)
        return o,z_mean,z_log_var
class VaeDecoder(nn.Module):
    def __init__(self,original_dim,intermediate_dim,latent_dim):
        super(VaeDecoder, self).__init__()
        self.Dense=nn.Linear(latent_dim,intermediate_dim)
        self.out=nn.Linear(intermediate_dim,original_dim)
        self.sigmoid=nn.Sigmoid()
    def forward(self,z):
        o=F.relu(self.Dense(z))
        o=self.out(o)
        return self.sigmoid(o)
class Vae(nn.Module):
    def __init__(self,original_dim,intermediate_dim,latent_dim):
        super(Vae, self).__init__()
        self.encoder=VaeEncoder(original_dim,intermediate_dim,latent_dim)
        self.decoder=VaeDecoder(original_dim,intermediate_dim,latent_dim)
    def forward(self,x):
        o,mean,var=self.encoder(x)
        return o,x,self.decoder(o),mean,var


