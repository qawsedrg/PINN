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
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = X_star
u_train = np.vstack([uu1, uu2, uu3])

x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
u = torch.tensor(u_train).float().to(device)

dnn = DNN().to(device)
model = PINN(dnn)
optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-3)

criterion = torch.nn.MSELoss()

dnn.train()
with tqdm(range(5000)) as bar:
    for epoch in bar:
        optimizer.zero_grad()

        f_pred = model(x_f, t_f)
        loss_u = criterion(u, dnn(x_u, t_u))
        loss_f = torch.mean(f_pred ** 2)
        loss = loss_u + loss_f
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss=loss.item())

x = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
t = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
dnn.eval()
u_pred = dnn(x, t)
u_pred = u_pred.detach().cpu().numpy()
x = x.detach().cpu().numpy()
t = t.detach().cpu().numpy()

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    'kx', label='Data (%d points)' % (u_train.shape[0]),
    markersize=4,  # marker size doubled
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
ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)

plt.show()
