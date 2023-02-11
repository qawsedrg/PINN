import matplotlib.pyplot as plt
import numpy as np
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
        self.linear.append(nn.Linear(100, 100))
        self.output_layer = nn.Linear(100, 1)

    def forward(self, x, t):
        out = torch.cat([x, t], dim=1)
        for layer in self.linear:
            out = torch.tanh(layer(out))
        out = self.output_layer(out)
        return out


class PINN(nn.Module):
    def __init__(self, dnn, c):
        super().__init__()
        self.dnn = dnn
        self.c = c

    def forward(self, x, t):
        u = self.dnn(x, t)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
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
        f = u_tt - self.c ** 2 * u_xx
        return f


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


for t_max_reg in np.linspace(0.1, 1, 10):
    T = 1
    L = 1
    c = 0.5
    n_max = 5
    n = np.arange(1, n_max + 1)
    lam = 2 * L / n_max
    omiga = n_max * np.pi * c / L
    k = omiga / c
    phi_n = np.zeros((n_max, 1))
    C_n = np.zeros((n_max, 1))
    C_n[-1, :] = 1


    def alpha(x):
        alpha_n = C_n * np.cos(phi_n)
        return np.sum(alpha_n * np.sin(np.einsum("n,xj->nx", n, x) * np.pi / L), axis=0)[:, None]


    def beta(x):
        beta_n = n[:, None] * np.pi * c / L * C_n * np.sin(phi_n)
        return np.sum(beta_n * np.sin(np.einsum("n,xj->nx", n, x) * np.pi / L), axis=0)[:, None]


    # 20 point per period
    t = np.linspace(0, T, int(n_max * T / (2 * L / c) * 20))[:, None]
    x = np.linspace(0, L, n_max * 20)[:, None]
    Exact_ori = np.sum(
        C_n[:, :, None] * np.einsum("nt,nx->nxt", np.cos(np.einsum("n,tj->nt", n, t) * np.pi * c / L - phi_n),
                                    np.sin(np.einsum("n,xj->nx", n, x) * np.pi / L)), axis=0).T

    X, T = np.meshgrid(x, t)
    grid = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    grid_bc1 = np.hstack((x, np.zeros_like(x)))  # (x,0)
    u_bc1 = alpha(x)
    grid_bc2 = np.hstack((np.zeros_like(t) + L, t))  # (L,t)
    u_bc2 = np.zeros_like(t)
    grid_bc3 = np.hstack((np.zeros_like(t), t))  # (0,t)
    u_bc3 = np.zeros_like(t)

    Exact = Exact_ori.flatten()[:, None]

    grid_bc = np.vstack([grid_bc1, grid_bc2, grid_bc3])
    u_bc = np.vstack([u_bc1, u_bc2, u_bc3])

    grid_x_bc = torch.tensor(grid_bc[:, 0:1], requires_grad=True).float().to(device)
    grid_t_bc = torch.tensor(grid_bc[:, 1:2], requires_grad=True).float().to(device)
    grid_x_bc1 = torch.tensor(grid_bc1[:, 0:1], requires_grad=True).float().to(device)
    grid_t_bc1 = torch.tensor(grid_bc1[:, 1:2], requires_grad=True).float().to(device)
    grid_x = torch.tensor(grid[:, 0:1], requires_grad=True).float().to(device)
    grid_t = torch.tensor(grid[:, 1:2], requires_grad=True).float().to(device)
    u_bc = torch.tensor(u_bc).float().to(device)
    u_t_bc = beta(x)
    u_t_bc = torch.tensor(u_t_bc).float().to(device)

    dnn = DNN().to(device)
    model = PINN(dnn, c)
    ema = EMA(model, 0.999)
    ema.register()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()

    plt.ion()
    fig1 = plt.figure("Sol", figsize=(7, 5))
    fig2 = plt.figure("Error", figsize=(7, 5))
    fig3 = plt.figure("Error-x", figsize=(7, 5))
    fig4 = plt.figure("Error-t", figsize=(7, 5))

    mask_x = torch.logical_or(grid_x >= 0.5, grid_x <= 0.5)
    mask_t = torch.logical_or(grid_t >= 1.1, grid_t <= t_max_reg)
    mask_train = torch.logical_and(mask_x, mask_t)
    grid_x_train = torch.tensor(grid_x[mask_train, None], requires_grad=True).float().to(device)
    grid_t_train = torch.tensor(grid_t[mask_train, None], requires_grad=True).float().to(device)

    t_num_intervals = 5
    x_num_intervals = 5
    error_t_slice_list = np.array([]).reshape(0, t_num_intervals)
    error_x_slice_list = np.array([]).reshape(0, x_num_intervals)
    eval_epoches = []
    with tqdm(range(100000)) as bar:
        for epoch in bar:
            dnn.train()
            model.train()
            optimizer.zero_grad()

            f = model(grid_x_train, grid_t_train)
            loss_bc = criterion(u_bc, dnn(grid_x_bc, grid_t_bc))
            u = dnn(grid_x_bc1, grid_t_bc1)
            u_t = torch.autograd.grad(
                u, grid_t_bc1,
                grad_outputs=torch.ones_like(u),
                retain_graph=True,
                create_graph=True
            )[0]
            loss_bc += criterion(u_t_bc, u_t)
            loss_f = torch.mean(f ** 2)
            loss = loss_bc + loss_f
            loss.backward()

            optimizer.step()
            # scheduler.step()

            ema.update()

            # bar.set_postfix(loss=loss.item(),lr=scheduler.get_last_lr())
            bar.set_postfix(loss=loss.item())
            # eval and draw

            if epoch % 100 == 0:
                eval_epoches.append(epoch)
                dnn.eval()
                model.eval()
                ema.apply_shadow()
                with torch.no_grad():
                    u = dnn(grid_x, grid_t).cpu().numpy()
                ema.restore()

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

                np.save("./error_t_slice_list_{:.1f}_t_max_reg.npy".format(t_max_reg), np.array(error_t_slice_list))
                np.save("./error_x_slice_list_{:.1f}_t_max_reg.npy".format(t_max_reg), np.array(error_x_slice_list))

                # griddata(coordinate, value, (points at which to interpolate))
                error = griddata(grid, ((Exact - u) ** 2).flatten(), (X, T), method='cubic') / np.linalg.norm(Exact,
                                                                                                              2) ** 2
                # error1=np.sqrt(np.sum(error1))
                u = griddata(grid, u.flatten(), (X, T), method='cubic')

                t1 = int(0.25 * len(t))
                t2 = int(0.5 * len(t))
                t3 = int(0.75 * len(t))
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
                ax1.plot(t[t1] * np.ones((2, 1)), line, 'w-', linewidth=1)
                ax1.plot(t[t2] * np.ones((2, 1)), line, 'w-', linewidth=1)
                ax1.plot(t[t3] * np.ones((2, 1)), line, 'w-', linewidth=1)

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
                ax2.plot(t[t1] * np.ones((2, 1)), line, 'w-', linewidth=1)
                ax2.plot(t[t2] * np.ones((2, 1)), line, 'w-', linewidth=1)
                ax2.plot(t[t3] * np.ones((2, 1)), line, 'w-', linewidth=1)

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
                fig1.savefig("sol_{:}_t_max_reg.png".format(t_max_reg))
                fig2.savefig("error_{:}_t_max_reg.png".format(t_max_reg))
                fig3.savefig("error_x_{:}_t_max_reg.png".format(t_max_reg))
                fig4.savefig("error_t_{:}_t_max_reg.png".format(t_max_reg))
                fig1.clf()
                fig2.clf()
                fig3.clf()
                fig4.clf()

    plt.ioff()
