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
        self.output_layer = nn.Linear(50, q)

    def forward(self, x):
        out = x
        for layer in self.linear:
            out = torch.tanh(layer(out))
        out = self.output_layer(out)
        return out


class PINN1(nn.Module):
    def __init__(self, dnn, parameters):
        super().__init__()
        self.dnn = dnn
        self.param = nn.ParameterList([torch.nn.Parameter(parameter) for parameter in parameters])

    def forward(self, x, dt, IRK_alpha):
        u = self.dnn(x)
        dummy = torch.ones_like(u, requires_grad=True).to(device)
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=dummy,
            retain_graph=True,
            create_graph=True
        )[0]
        # torch.sum(this u_x,dim=-1)=original u_x
        # to separate the grad
        u_x = torch.autograd.grad(
            u_x, dummy,
            grad_outputs=torch.ones([u.shape[0], 1]).to(device),
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
            grad_outputs=torch.ones([u.shape[0], 1]).to(device),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=dummy,
            retain_graph=True,
            create_graph=True
        )[0]
        u_xxx = torch.autograd.grad(
            u_xxx, dummy,
            grad_outputs=torch.ones([u.shape[0], 1]).to(device),
            retain_graph=True,
            create_graph=True
        )[0]
        f = - self.param[0] * u * u_x - self.param[1] * u_xxx
        return u - dt * torch.matmul(f, IRK_alpha.T)


class PINN2(nn.Module):
    def __init__(self, dnn, parameters):
        super().__init__()
        self.dnn = dnn
        self.param = nn.ParameterList([torch.nn.Parameter(parameter) for parameter in parameters])

    def forward(self, x, dt, IRK_alpha, IRK_beta):
        u = self.dnn(x)
        dummy = torch.ones_like(u, requires_grad=True).to(device)
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=dummy,
            retain_graph=True,
            create_graph=True
        )[0]
        # torch.sum(this u_x,dim=-1)=original u_x
        # to separate the grad
        u_x = torch.autograd.grad(
            u_x, dummy,
            grad_outputs=torch.ones([u.shape[0], 1]).to(device),
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
            grad_outputs=torch.ones([u.shape[0], 1]).to(device),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=dummy,
            retain_graph=True,
            create_graph=True
        )[0]
        u_xxx = torch.autograd.grad(
            u_xxx, dummy,
            grad_outputs=torch.ones([u.shape[0], 1]).to(device),
            retain_graph=True,
            create_graph=True
        )[0]
        f = - self.param[0] * u * u_x - self.param[1] * u_xxx
        return u + dt * torch.matmul(f, (IRK_beta - IRK_alpha).T)


q = 500
q = max(q, 1)

N = 200

data = scipy.io.loadmat('../Data/KdV.mat')

t_star = data['tt'].flatten()[:, None]  # T x 1
x_star = data['x'].flatten()[:, None]  # N x 1
Exact = np.real(data['uu']).T  # T x N

idx_t0 = 40
idx_t1 = 160
dt = t_star[idx_t1] - t_star[idx_t0]

noise = 0.0

idx_x = np.random.choice(Exact.shape[1], N, replace=False)
x0 = x_star[idx_x, :]
u0 = Exact[idx_t0, idx_x][:, None]
u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

idx_x = np.random.choice(Exact.shape[1], N, replace=False)
x1 = x_star[idx_x, :]
u1 = Exact[idx_t1, idx_x][:, None]
u1 = u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])

# Domain bounds
lb = x_star.min(0)
ub = x_star.max(0)

tmp = np.float32(np.loadtxt('../IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2))
IRK_weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))
IRK_alpha = IRK_weights[0:-1, :]
IRK_beta = IRK_weights[-1, :]

x1_ = torch.tensor(x1, requires_grad=True).float().to(device)
u1_ = torch.tensor(u1, requires_grad=True).float().to(device)
x0_ = torch.tensor(x0, requires_grad=True).float().to(device)
u0_ = torch.tensor(u0, requires_grad=True).float().to(device)
x_star_ = torch.tensor(x_star).float().to(device)
IRK_weights_ = torch.tensor(IRK_weights).float().to(device)
IRK_alpha_ = torch.tensor(IRK_alpha).float().to(device)
IRK_beta_ = torch.tensor(IRK_beta).float().to(device)
dt_ = torch.tensor(dt).float().to(device)

lambda_1 = torch.tensor([0.0], requires_grad=True).float().to(device)
lambda_2 = torch.tensor([0.0], requires_grad=True).float().to(device)
parameters = [lambda_1, lambda_2]

dnn = DNN(q).to(device)
model1 = PINN1(dnn, parameters).to(device)
model2 = PINN2(dnn, parameters).to(device)

optimizer = torch.optim.Adam(model1.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

plt.ion()
fig = plt.figure(figsize=(7, 4))
with tqdm(range(10000)) as bar:
    for epoch in bar:
        dnn.train()
        model1.train()
        model2.train()
        optimizer.zero_grad()

        # periodic(T) bound not given
        # how to define the bound even given when t<T
        loss1 = criterion(u1_, model2(x1_, dt_, IRK_alpha_, IRK_beta_))
        loss2 = criterion(u0_, model1(x0_, dt_, IRK_alpha_))
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss1=loss1.item(), loss2=loss2.item())

        if epoch % 100 == 0:
            dnn.eval()
            model1.eval()
            model2.eval()
            with torch.no_grad():
                U1_pred = dnn(x_star_).cpu().numpy()
            error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
            print('Error: %e' % (error))

            lambdas = [parameter.detach().cpu().numpy() for parameter in model1.param]
            lambda_1_value = lambdas[0]
            lambda_2_value = lambdas[1]
            error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
            error_lambda_2 = np.abs(lambda_2_value - 0.0025) / 0.0025 * 100
            print(lambda_1_value, lambda_2_value)
            print('Error l1: %.5f%%' % (error_lambda_1), 'Error l2: %.5f%%' % (error_lambda_2))

            ax = fig.add_subplot(111)
            ax.axis('off')

            ####### Row 0: h(t,x) ##################
            gs0 = gridspec.GridSpec(1, 2)
            gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.1, left=0.15, right=0.85, wspace=0)
            ax = fig.add_subplot(gs0[:, :])

            h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                          extent=[t_star.min(), t_star.max(), x_star.min(), x_star.max()],
                          origin='lower', aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, cax=cax)

            line = np.linspace(x_star.min(), x_star.max(), 2)[:, None]
            ax.plot(t_star[idx_t0] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax.plot(t_star[idx_t1] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')
            leg = ax.legend(frameon=False, loc='best')
            ax.set_title('$u(t,x)$', fontsize=10)

            ####### Row 1: h(t,x) slices ##################
            gs1 = gridspec.GridSpec(1, 2)
            gs1.update(top=1 - 1 / 2 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

            ax = fig.add_subplot(gs1[0, 0])
            ax.plot(x_star, Exact[idx_t0, :], 'b-', linewidth=2)
            ax.plot(x0, u0, 'rx', linewidth=2, label='Data')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$u(t,x)$')
            ax.set_title('$t = %.2f$' % (t_star[idx_t0]), fontsize=10)
            ax.set_xlim([lb - 0.1, ub + 0.1])
            ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

            ax = fig.add_subplot(gs1[0, 1])
            ax.plot(x_star, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
            ax.plot(x_star, U1_pred[:, -1], 'r--', linewidth=2, label='Prediction')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$u(t,x)$')
            ax.set_title('$t = %.2f$' % (t_star[idx_t1]), fontsize=10)
            ax.set_xlim([lb - 0.1, ub + 0.1])

            ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
            plt.pause(0.1)
            fig.clf()
