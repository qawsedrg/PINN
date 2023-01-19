import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
    def __init__(self, dnn, parameters):
        super().__init__()
        self.dnn = dnn
        self.param = nn.ParameterList([torch.nn.Parameter(parameter) for parameter in parameters])

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
        f = u_t + self.param[0] * u * u_x - torch.exp(self.param[1]) * u_xx
        return f


data = scipy.io.loadmat('../Data/burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact_ori = np.real(data['usol']).T

X, T = np.meshgrid(x, t)
grid = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

grid_bc1 = np.hstack((x, np.zeros_like(x)))  # (x,0)
u_bc1 = -np.sin(np.pi * x)
grid_bc2 = np.hstack((np.ones_like(t), t))  # (1,t)
u_bc2 = np.zeros_like(t)
grid_bc3 = np.hstack((-np.ones_like(t), t))  # (-1,t)
u_bc3 = np.zeros_like(t)

Exact = Exact_ori.flatten()[:, None]

grid_bc = np.vstack([grid_bc1, grid_bc2, grid_bc3])
u_bc = np.vstack([u_bc1, u_bc2, u_bc3])

grid_x_bc = torch.tensor(grid_bc[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc = torch.tensor(grid_bc[:, 1:2], requires_grad=True).float().to(device)
grid_x = torch.tensor(grid[:, 0:1], requires_grad=True).float().to(device)
grid_t = torch.tensor(grid[:, 1:2], requires_grad=True).float().to(device)
u_bc = torch.tensor(u_bc).float().to(device)

dnn = DNN().to(device)
lambda_1 = torch.tensor([0.0], requires_grad=True).float().to(device)
lambda_2 = torch.tensor([-6.0], requires_grad=True).float().to(device)
parameters = [lambda_1, lambda_2]
model = PINN(dnn, parameters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

criterion = torch.nn.MSELoss()

plt.ion()
fig1 = plt.figure(figsize=(7, 4))
fig2 = plt.figure(figsize=(7, 4))

with tqdm(range(5000)) as bar:
    for epoch in bar:
        dnn.train()
        model.train()
        optimizer.zero_grad()

        f = model(grid_x, grid_t)
        loss_u = criterion(torch.tensor(Exact).float().to(device), dnn(grid_x, grid_t))
        loss_f = torch.mean(f ** 2)
        loss = loss_u + loss_f
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss=loss.item())

        # eval and draw

        if epoch % 10 == 0:
            dnn.eval()
            model.eval()
            lambdas = [parameter.detach().cpu().numpy() for parameter in model.param]
            error_lambda_1 = np.abs(lambdas[0] - 1.0) * 100
            error_lambda_2 = np.abs(np.exp(lambdas[1]) - (0.01 / np.pi)) / (0.01 / np.pi) * 100
            print('Error l1: %.5f%%' % (error_lambda_1), 'Error l2: %.5f%%' % (error_lambda_2))
            with torch.no_grad():
                u = dnn(grid_x, grid_t).cpu().numpy()

            error = np.linalg.norm(Exact - u, 2) / np.linalg.norm(Exact, 2)
            print('Error u: {:}'.format(error))

            u = griddata(grid, u.flatten(), (X, T), method='cubic')

            ax1 = fig1.add_subplot(111)

            h = ax1.imshow(u.T, interpolation='nearest', cmap='rainbow',
                           extent=[t.min(), t.max(), x.min(), x.max()],
                           origin='lower', aspect='auto')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig1.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax1.plot(
                grid_bc[:, 1],
                grid_bc[:, 0],
                'kx', label='Data (%d points)' % (u_bc.shape[0]),
                markersize=4,
                clip_on=False,
                alpha=1.0
            )

            line = np.linspace(x.min(), x.max(), 2)[:, None]
            ax1.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax1.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax1.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax1.set_xlabel('$t$', size=20)
            ax1.set_ylabel('$x$', size=20)
            ax1.legend(
                loc='upper center',
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
                prop={'size': 15}
            )
            ax1.set_title('$u(t,x)$', fontsize=20)
            ax1.tick_params(labelsize=15)

            ax2 = fig2.add_subplot(111)

            gs1 = gridspec.GridSpec(1, 3)
            gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

            ax2 = plt.subplot(gs1[0, 0])
            ax2.plot(x, Exact_ori[25, :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[25, :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_title('$t = 0.25$', fontsize=15)
            ax2.axis('square')
            ax2.set_xlim([-1.1, 1.1])
            ax2.set_ylim([-1.1, 1.1])

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            ax2 = plt.subplot(gs1[0, 1])
            ax2.plot(x, Exact_ori[50, :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[50, :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.axis('square')
            ax2.set_xlim([-1.1, 1.1])
            ax2.set_ylim([-1.1, 1.1])
            ax2.set_title('$t = 0.50$', fontsize=15)
            ax2.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=5,
                frameon=False,
                prop={'size': 15}
            )

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            ax2 = plt.subplot(gs1[0, 2])
            ax2.plot(x, Exact_ori[75, :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[75, :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.axis('square')
            ax2.set_xlim([-1.1, 1.1])
            ax2.set_ylim([-1.1, 1.1])
            ax2.set_title('$t = 0.75$', fontsize=15)

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            plt.pause(0.1)
            fig1.clf()
            fig2.clf()

plt.ioff()
