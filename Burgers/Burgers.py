import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from tqdm import tqdm

np.random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList([])
        self.linear.append(nn.Linear(2, 20))
        self.linear.append(nn.Linear(20, 20))
        self.linear.append(nn.Linear(20, 20))
        self.linear.append(nn.Linear(20, 20))
        self.linear.append(nn.Linear(20, 20))
        self.linear.append(nn.Linear(20, 20))
        self.linear.append(nn.Linear(20, 20))
        self.linear.append(nn.Linear(20, 20))
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x, t):
        out = torch.cat([x, t], dim=1)
        for layer in self.linear:
            out = torch.tanh(layer(out))
        out = self.output_layer(out)
        return out

class PINN(nn.Module):
    def __init__(self, dnn):
        super().__init__()
        self.dnn = dnn

    def forward(self, x, t):
        u = self.dnn(x, t)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        f = u_t + u * u_x - (0.01 / np.pi) * u_xx
        return f


data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = (np.real(data['usol']).T)

X, T = np.meshgrid(x, t)
grid = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

grid_bc1 = np.hstack((x, np.zeros_like(x)))  # (x,0)
u_bc1 = -np.sin(np.pi * x)
grid_bc2 = np.hstack((np.ones_like(t), t))  # (1,t)
u_bc2 = np.zeros_like(t)
grid_bc3 = np.hstack((-np.ones_like(t), t))  # (-1,t)
u_bc3 = np.zeros_like(t)

Exact = Exact.flatten()[:, None]

grid_bc = np.vstack([grid_bc1, grid_bc2, grid_bc3])
u_bc = np.vstack([u_bc1, u_bc2, u_bc3])

grid_x_bc = torch.tensor(grid_bc[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc = torch.tensor(grid_bc[:, 1:2], requires_grad=True).float().to(device)
grid_x = torch.tensor(grid[:, 0:1], requires_grad=True).float().to(device)
grid_t = torch.tensor(grid[:, 1:2], requires_grad=True).float().to(device)
u_bc = torch.tensor(u_bc).float().to(device)

dnn = DNN().to(device)
model = PINN(dnn)
optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-3)

criterion = torch.nn.MSELoss()

dnn.train()
with tqdm(range(5000)) as bar:
    for epoch in bar:
        optimizer.zero_grad()

        f = model(grid_x, grid_t)
        loss_bc = criterion(u_bc, dnn(grid_x_bc, grid_t_bc))
        loss_f = torch.mean(f ** 2)
        loss = loss_bc + loss_f
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss=loss.item())

dnn.eval()
with torch.no_grad():
    u = dnn(grid_x, grid_t).cpu().numpy()

error = np.linalg.norm(Exact - u, 2) / np.linalg.norm(Exact, 2)
print('Error u: {:}'.format(error))

u = griddata(grid, u.flatten(), (X, T), method='cubic')

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    grid_bc[:, 1],
    grid_bc[:, 0],
    'kx', label='Data (%d points)' % (u_bc.shape[0]),
    markersize=4,
    clip_on=False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize=20)
ax.tick_params(labelsize=15)

plt.show()
