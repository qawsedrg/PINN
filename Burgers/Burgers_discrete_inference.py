import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

np.random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DNN(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.linear = nn.ModuleList([])
        self.linear.append(nn.Linear(1, 50))
        self.linear.append(nn.Linear(50, 50))
        self.linear.append(nn.Linear(50, 50))
        self.linear.append(nn.Linear(50, 50))
        self.linear.append(nn.Linear(50, 50))
        # q intervals = q+1 time steps
        self.output_layer = nn.Linear(50, q + 1)

    def forward(self, x):
        out = x
        for layer in self.linear:
            out = torch.tanh(layer(out))
        out = self.output_layer(out)
        return out


class PINN(nn.Module):
    def __init__(self, dnn):
        super().__init__()
        self.dnn = dnn

    def forward(self, x, dt, IRK_weights):
        u = self.dnn(x)
        u_prime = u[:, :-1]
        dummy = torch.ones_like(u_prime, requires_grad=True).to(device)
        u_x = torch.autograd.grad(
            u_prime, x,
            grad_outputs=dummy,
            retain_graph=True,
            create_graph=True
        )[0]
        # torch.sum(this u_x,dim=-1)=original u_x
        # to separate the grad
        u_x = torch.autograd.grad(
            u_x, dummy,
            grad_outputs=torch.ones([u_prime.shape[0], 1]).to(device),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=dummy,
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_xx, dummy,
            grad_outputs=torch.ones([u_prime.shape[0], 1]).to(device),
            retain_graph=True,
            create_graph=True
        )[0]
        f = - u_prime * u_x + (0.01 / np.pi) * u_xx
        # -dt... to return from the ending point to the starting point
        return u - dt * torch.matmul(f, IRK_weights.T)


q = 500
q = max(q, 1)
lb = np.array([-1.0])
ub = np.array([1.0])

N = 250

data = scipy.io.loadmat('./burgers_shock.mat')

t = data['t'].flatten()[:, None]  # T x 1
x = data['x'].flatten()[:, None]  # N x 1
Exact = np.real(data['usol']).T  # T x N

idx_t0 = 10
idx_t1 = 90
dt = t[idx_t1] - t[idx_t0]

# Initial data
noise_u0 = 0.0
idx_x = np.random.choice(Exact.shape[1], N, replace=False)
# choose N samples
x0 = x[idx_x, :]
u0 = Exact[idx_t0:idx_t0 + 1, idx_x].T
# nosiy data
u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

# Boundary data
x1 = np.vstack((lb, ub))

# Test data
x_star = x

tmp = np.float32(np.loadtxt('../IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2))
IRK_weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))

x1_ = torch.tensor(x1).float().to(device)
x0_ = torch.tensor(x0, requires_grad=True).float().to(device)
u0_ = torch.tensor(u0, requires_grad=True).float().to(device)
x_star_ = torch.tensor(x_star).float().to(device)
IRK_weights_ = torch.tensor(IRK_weights).float().to(device)
dt_ = torch.tensor(dt).float().to(device)

dnn = DNN(q).to(device)
model = PINN(dnn).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

plt.ion()
fig = plt.figure(figsize=(7, 4))
with tqdm(range(10000)) as bar:
    for epoch in bar:
        dnn.train()
        model.train()
        optimizer.zero_grad()

        # model learns the pattern of u at t=0.9
        # loss1 : bord
        # loss2 : value of u at t = 0.1 by Runge-Kutta
        loss1 = torch.mean(dnn(x1_) ** 2)
        loss2 = criterion(u0_, model(x0_, dt_, IRK_weights_))
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss=loss.item())

        if epoch % 500 == 0:
            dnn.eval()
            model.eval()
            with torch.no_grad():
                U1_pred = dnn(x_star_).cpu().numpy()
            error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
            print('Error: %e' % (error))

            ax = fig.add_subplot(111)
            ax.axis('off')

            ####### Row 0: h(t,x) ##################
            gs0 = gridspec.GridSpec(1, 2)
            gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.1, left=0.15, right=0.85, wspace=0)
            ax = plt.subplot(gs0[:, :])

            h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                          extent=[t.min(), t.max(), x_star.min(), x_star.max()],
                          origin='lower', aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, cax=cax)

            line = np.linspace(x.min(), x.max(), 2)[:, None]
            ax.plot(t[idx_t0] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax.plot(t[idx_t1] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')
            leg = ax.legend(frameon=False, loc='best')
            ax.set_title('$u(t,x)$', fontsize=10)

            ####### Row 1: h(t,x) slices ##################
            gs1 = gridspec.GridSpec(1, 2)
            gs1.update(top=1 - 1 / 2 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

            ax = plt.subplot(gs1[0, 0])
            ax.plot(x, Exact[idx_t0, :], 'b-', linewidth=2)
            ax.plot(x0, u0, 'rx', linewidth=2, label='Data')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$u(t,x)$')
            ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize=10)
            ax.set_xlim([lb - 0.1, ub + 0.1])
            ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

            ax = plt.subplot(gs1[0, 1])
            ax.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
            ax.plot(x_star, U1_pred[:, -1], 'r--', linewidth=2, label='Prediction')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$u(t,x)$')
            ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize=10)
            ax.set_xlim([lb - 0.1, ub + 0.1])

            ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
            plt.pause(0.1)
            fig.clf()