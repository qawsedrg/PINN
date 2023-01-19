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


proportion = 1
t_max = 1
t_max_reg = 0.5
# bound de regularization non partout?

data = scipy.io.loadmat('../Data/burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact_ori = np.real(data['usol']).T

X, T = np.meshgrid(x, t)
grid = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

t_bc = t[np.random.choice(np.sum(t < t_max).astype(np.int), (proportion * np.sum(t < t_max)).astype(np.int),
                          replace=False), :]
x_bc = x[np.random.choice(x.shape[0], int(proportion * x.shape[0]), replace=False), :]
grid_bc1 = np.hstack((x_bc, np.zeros_like(x_bc)))  # (x,0)
u_bc1 = -np.sin(np.pi * x_bc)
grid_bc2 = np.hstack((np.ones_like(t_bc), t_bc))  # (1,t)
u_bc2 = np.zeros_like(t_bc)
grid_bc3 = np.hstack((-np.ones_like(t_bc), t_bc))  # (-1,t)
u_bc3 = np.zeros_like(t_bc)

Exact = Exact_ori.flatten()[:, None]

grid_bc = np.vstack([grid_bc1, grid_bc2, grid_bc3])
u_bc = np.vstack([u_bc1, u_bc2, u_bc3])

grid_x_bc = torch.tensor(grid_bc[:, 0:1], requires_grad=True).float().to(device)
grid_t_bc = torch.tensor(grid_bc[:, 1:2], requires_grad=True).float().to(device)
# [25600,1]
grid_x = torch.tensor(grid[:, 0:1], requires_grad=True).float().to(device)
grid_t = torch.tensor(grid[:, 1:2], requires_grad=True).float().to(device)
u_bc = torch.tensor(u_bc).float().to(device)

grid_x_choiced = torch.tensor(grid_x[grid_t < t_max_reg, None], requires_grad=True).float().to(device)
grid_t_choiced = torch.tensor(grid_t[grid_t < t_max_reg, None], requires_grad=True).float().to(device)

dnn = DNN().to(device)
model = PINN(dnn)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

criterion = torch.nn.MSELoss()

plt.ion()
fig1 = plt.figure("Sol", figsize=(7, 5))
fig2 = plt.figure("Error", figsize=(7, 5))
fig3 = plt.figure("Error-x", figsize=(7, 5))
fig4 = plt.figure("Error-t", figsize=(7, 5))

t_num_intervals = 5
x_num_intervals = 5
error_t_slice_list = np.array([]).reshape(0, t_num_intervals)
error_x_slice_list = np.array([]).reshape(0, x_num_intervals)
eval_epoches = []
with tqdm(range(5000)) as bar:
    for epoch in bar:
        dnn.train()
        model.train()
        optimizer.zero_grad()

        f = model(grid_x_choiced, grid_t_choiced)
        loss_bc = criterion(u_bc, dnn(grid_x_bc, grid_t_bc))
        loss_f = torch.mean(f ** 2)
        loss = loss_bc + loss_f
        loss.backward()

        optimizer.step()
        # scheduler.step()
        # bar.set_postfix(loss=loss.item(),lr=scheduler.get_last_lr())
        bar.set_postfix(loss=loss.item())
        # eval and draw

        if epoch % 50 == 0:
            eval_epoches.append(epoch)
            dnn.eval()
            model.eval()
            with torch.no_grad():
                u = dnn(grid_x, grid_t).cpu().numpy()

            # [25600,1]
            error = np.linalg.norm(Exact - u, 2) / np.linalg.norm(Exact, 2)
            print('Error u: {:}'.format(error))

            # X.shape=T.shape prod(X.shape)=prod(u.shape)
            # (x,t)
            Exact_reformed = np.reshape(Exact, X.shape).T
            u_reformed = np.reshape(u, X.shape).T
            Exact_reformed_splitted_x = np.array_split(Exact_reformed, x_num_intervals, axis=0)
            Exact_reformed_splitted_t = np.array_split(Exact_reformed, t_num_intervals, axis=1)
            u_reformed_splitted_x = np.array_split(u_reformed, x_num_intervals, axis=0)
            u_reformed_splitted_t = np.array_split(u_reformed, t_num_intervals, axis=1)
            error_t_slice = np.array([np.linalg.norm(e - u, 2) / np.linalg.norm(e, 2) for (e, u) in
                                      zip(Exact_reformed_splitted_t, u_reformed_splitted_t)])[None, :]
            error_x_slice = np.array([np.linalg.norm(e - u, 2) / np.linalg.norm(e, 2) for (e, u) in
                                      zip(Exact_reformed_splitted_x, u_reformed_splitted_x)])[None, :]
            error_t_slice_list = np.concatenate((error_t_slice_list, error_t_slice), axis=0)
            error_x_slice_list = np.concatenate((error_x_slice_list, error_x_slice), axis=0)

            ax3 = fig3.add_subplot(111)
            ax3.plot(eval_epoches, error_x_slice_list, label=list(map(lambda x: (round(x[0], 2), round(x[1], 2)),
                                                                      zip(np.linspace(np.min(x), np.max(x),
                                                                                      x_num_intervals + 1)[:-1],
                                                                          np.linspace(np.min(x), np.max(x),
                                                                                      x_num_intervals + 1)[1:]))))
            ax3.legend(loc=1)
            ax3.set_yscale("log")

            ax4 = fig4.add_subplot(111)
            ax4.plot(eval_epoches, error_t_slice_list, label=list(map(lambda x: (round(x[0], 2), round(x[1], 2)),
                                                                      zip(np.linspace(np.min(t), np.max(t),
                                                                                      t_num_intervals + 1)[:-1],
                                                                          np.linspace(np.min(t), np.max(t),
                                                                                      t_num_intervals + 1)[1:]))))
            ax4.legend(loc=1)
            ax4.set_yscale("log")

            # griddata(coordinate, value, (points at which to interpolate))
            error = griddata(grid, ((Exact - u) ** 2).flatten(), (X, T), method='cubic') / np.linalg.norm(Exact, 2) ** 2
            # error1=np.sqrt(np.sum(error1))
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

            #####################################
            ############     ax2     ############
            #####################################
            ax2 = fig2.add_subplot(111)

            h = ax2.imshow(error.T, interpolation='nearest', cmap='rainbow',
                           extent=[t.min(), t.max(), x.min(), x.max()],
                           origin='lower', aspect='auto')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig2.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax2.plot(
                grid_bc[:, 1],
                grid_bc[:, 0],
                'kx', label='Data (%d points)' % (u_bc.shape[0]),
                markersize=4,
                clip_on=False,
                alpha=1.0
            )

            line = np.linspace(x.min(), x.max(), 2)[:, None]
            ax2.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax2.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax2.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax2.set_xlabel('$t$', size=20)
            ax2.set_ylabel('$x$', size=20)
            ax2.legend(
                loc='upper center',
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
                prop={'size': 15}
            )
            ax2.set_title('$u(t,x)$', fontsize=20)
            ax2.tick_params(labelsize=15)

            plt.pause(0.01)
            fig1.clf()
            fig2.clf()
            fig3.clf()
            fig4.clf()

plt.ioff()
