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
        self.linear.append(nn.Linear(2, 100))
        self.linear.append(nn.Linear(100, 100))
        self.linear.append(nn.Linear(100, 100))
        self.linear.append(nn.Linear(100, 100))
        self.output_layer = nn.Linear(100, 2)

    def forward(self, x, t):
        out = torch.cat([x, t], dim=1)
        for layer in self.linear:
            out = torch.tanh(layer(out))
        out = self.output_layer(out)
        return out[:, 0:1], out[:, 1:2]


class PINN(nn.Module):
    def __init__(self, dnn):
        super().__init__()
        self.dnn = dnn

    def forward(self, x, t):
        u_r, u_i = self.dnn(x, t)
        u_t_r = torch.autograd.grad(
            u_r, t,
            grad_outputs=torch.ones_like(u_r),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x_r = torch.autograd.grad(
            u_r, x,
            grad_outputs=torch.ones_like(u_r),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx_r = torch.autograd.grad(
            u_x_r, x,
            grad_outputs=torch.ones_like(u_x_r),
            retain_graph=True,
            create_graph=True
        )[0]
        u_t_i = torch.autograd.grad(
            u_i, t,
            grad_outputs=torch.ones_like(u_i),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x_i = torch.autograd.grad(
            u_i, x,
            grad_outputs=torch.ones_like(u_i),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx_i = torch.autograd.grad(
            u_x_i, x,
            grad_outputs=torch.ones_like(u_x_i),
            retain_graph=True,
            create_graph=True
        )[0]
        f_r = -u_t_i + 0.5 * u_xx_r + (u_r ** 2 + u_i ** 2) * u_r
        f_i = u_t_r + 0.5 * u_xx_i + (u_r ** 2 + u_i ** 2) * u_i
        return f_r, f_i


# u->real
# v->img

data = scipy.io.loadmat('../Data/NLS.mat')
t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact_ori_i = np.imag(data['uu']).T
Exact_ori_r = np.real(data['uu']).T
Exact_ori = np.sqrt(Exact_ori_i ** 2 + Exact_ori_r ** 2)
interval_t = t[1][0]

X, T = np.meshgrid(x, t)
grid = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

grid_bc1 = np.hstack((x, np.zeros_like(x)))  # (x,0)
u_bc1 = 2 / np.cosh(x)
grid_bc2 = np.hstack((np.ones_like(t) * 5, t))  # (5,t)
u_bc2 = -np.ones_like(t)
grid_bc3 = np.hstack((-np.ones_like(t) * 5, t))  # (-5,t)
u_bc3 = -np.ones_like(t)

# no real effect
grid_bc = np.vstack([grid_bc1, grid_bc2, grid_bc3])
u_bc = np.vstack([u_bc1, u_bc2, u_bc3])

Exact_r = Exact_ori_r.flatten()[:, None]
Exact_i = Exact_ori_i.flatten()[:, None]
Exact = Exact_ori.flatten()[:, None]

grid_x_bc1 = torch.tensor(grid_bc1[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc1 = torch.tensor(grid_bc1[:, 1:2], requires_grad=True).float().to(device)
grid_x_bc2 = torch.tensor(grid_bc2[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc2 = torch.tensor(grid_bc2[:, 1:2], requires_grad=True).float().to(device)
grid_x_bc3 = torch.tensor(grid_bc3[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc3 = torch.tensor(grid_bc3[:, 1:2], requires_grad=True).float().to(device)
grid_x = torch.tensor(grid[:, 0:1], requires_grad=True).float().to(device)
grid_t = torch.tensor(grid[:, 1:2], requires_grad=True).float().to(device)
grid_x_bc = torch.tensor(grid_bc[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc = torch.tensor(grid_bc[:, 1:2], requires_grad=True).float().to(device)
u_bc = torch.tensor(u_bc).float().to(device)
u_bc1 = torch.tensor(u_bc1).float().to(device)

dnn = DNN().to(device)
model = PINN(dnn)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = torch.nn.MSELoss()

plt.ion()
fig1 = plt.figure(figsize=(7, 4))
fig2 = plt.figure(figsize=(7, 4))

with tqdm(range(10000)) as bar:
    for epoch in bar:
        dnn.train()
        model.train()
        optimizer.zero_grad()

        f_r, f_i = model(grid_x, grid_t)

        u1_r, u1_i = dnn(grid_x_bc1, grid_t_bc1)
        u2_r, u2_i = dnn(grid_x_bc2, grid_t_bc2)
        u3_r, u3_i = dnn(grid_x_bc3, grid_t_bc3)
        u_x2_r = torch.autograd.grad(
            u2_r, grid_x_bc2,
            grad_outputs=torch.ones_like(u2_r),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x3_r = torch.autograd.grad(
            u3_r, grid_x_bc3,
            grad_outputs=torch.ones_like(u3_r),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x2_i = torch.autograd.grad(
            u2_i, grid_x_bc2,
            grad_outputs=torch.ones_like(u2_i),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x3_i = torch.autograd.grad(
            u3_i, grid_x_bc3,
            grad_outputs=torch.ones_like(u3_i),
            retain_graph=True,
            create_graph=True
        )[0]

        loss_bc = criterion(u_bc1, u1_r) + torch.mean(u1_i ** 2)
        loss_bc += criterion(u2_i, u3_i) + criterion(u2_r, u3_r)
        loss_bc += criterion(u_x2_i, u_x3_i) + criterion(u_x2_r, u_x3_r)
        loss_f = torch.mean(f_r ** 2) + torch.mean(f_i ** 2)
        loss = loss_bc + loss_f
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss_bc=loss_bc.item(), loss_f=loss_f.item())

        if epoch % 100 == 0:
            dnn.eval()
            with torch.no_grad():
                u_r, u_i = dnn(grid_x, grid_t)
                u_r = u_r.cpu().numpy()
                u_i = u_i.cpu().numpy()
            u = np.sqrt(u_r ** 2 + u_i ** 2)

            error_r = np.linalg.norm(Exact_r - u_r, 2) / np.linalg.norm(Exact_r, 2)
            error_i = np.linalg.norm(Exact_i - u_i, 2) / np.linalg.norm(Exact_i, 2)
            error = np.linalg.norm(Exact - u, 2) / np.linalg.norm(Exact, 2)
            print('Error u_i: {:}'.format(error_i))
            print('Error u_r: {:}'.format(error_r))
            print('Error u: {:}'.format(error))

            u = griddata(grid, u.flatten(), (X, T), method='cubic')

            #####################################
            ############     ax1     ############
            #####################################
            
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
            ax1.plot(t[int(0.60 // interval_t)] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax1.plot(t[int(0.80 // interval_t)] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax1.plot(t[int(1.00 // interval_t)] * np.ones((2, 1)), line, 'w-', linewidth=1)

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

            #####################################
            ############     ax2     ############
            #####################################
            
            gs1 = gridspec.GridSpec(1, 3)
            gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

            ax2 = fig2.add_subplot(gs1[0, 0])
            ax2.plot(x, Exact_ori[int(0.60 // interval_t), :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[int(0.60 // interval_t), :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_title('$t = 0.60$', fontsize=15)
            ax2.set_xlim([x.min(), x.max()])
            ax2.set_ylim([Exact_ori.min(), Exact_ori.max()])

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            ax2 = fig2.add_subplot(gs1[0, 1])
            ax2.plot(x, Exact_ori[int(0.80 // interval_t), :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[int(0.80 // interval_t), :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_xlim([x.min(), x.max()])
            ax2.set_ylim([Exact_ori.min(), Exact_ori.max()])
            ax2.set_title('$t = 0.80$', fontsize=15)
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

            ax2 = fig2.add_subplot(gs1[0, 2])
            ax2.plot(x, Exact_ori[int(1.00 // interval_t), :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[int(1.00 // interval_t), :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_xlim([x.min(), x.max()])
            ax2.set_ylim([Exact_ori.min(), Exact_ori.max()])
            ax2.set_title('$t = 1.00$', fontsize=15)

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            plt.pause(0.1)
            fig1.clf()
            fig2.clf()
plt.ioff()
