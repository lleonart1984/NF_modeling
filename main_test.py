import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import NF.modeling as modeling


class DatasetSIGGRAPH:
    """
    Eric https://blog.evjang.com/2018/01/nf2.html
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """
    def __init__(self, device):
        with open('./Datasets/siggraph.pkl', 'rb') as f:
            xy = np.array(pickle.load(f), dtype=np.float32)
            # XY -= np.mean(XY, axis=0)  # center
        self.xy = torch.from_numpy(xy).to(device)
        self.device = device

    def sample(self, n):
        return self.xy[np.random.randint(self.xy.shape[0], size=n)].to(device)


import os


torch.set_num_threads(12)
device = torch.device('cpu')

ds = DatasetSIGGRAPH(device)

prior = torch.distributions.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))

layers = []
for i in range(12):
    layers.append(modeling.GlowBlock(2))
    layers.append(
        modeling.AffineCoupling(
            2, [i % 2],
            lambda in_dim, out_dim: modeling.ConstantModule(out_dim, 1.0),
            lambda in_dim, out_dim: modeling.ConstantModule(out_dim, 0.0)
        ))

flow_model = modeling.NormalizingFlowsModel(2, layers, prior).to(device)

optimizer = torch.optim.Adamax(flow_model.parameters(), lr=1e-3, weight_decay=1e-3)

flow_model.train()
for e in range(10000):  # 1000 epochs
    x = ds.sample(1024)  # get batch

    z, ll = flow_model(x)
    loss = -torch.mean(ll)

    flow_model.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 100 == 0:
        print(loss.item())

flow_model.eval()

x = ds.sample(1024)
z, _ = flow_model(x)  # sample latent space
z_ = prior.sample((1024,))  # sample prior

z = z.detach().cpu().numpy()
z_ = z_.detach().cpu().numpy()

plt.scatter(z[:, 0], z[:, 1], c='g', s=5)
plt.scatter(z_[:, 0], z_[:, 1], c='r', s=5)

plt.show()
