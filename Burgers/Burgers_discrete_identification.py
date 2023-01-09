import numpy as np
import scipy
import torch
import torch.nn as nn
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
        f = - self.param[0] * u * u_x + torch.exp(self.param[1]) * u_xx
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
        f = - self.param[0] * u * u_x + torch.exp(self.param[1]) * u_xx
        return u + dt * torch.matmul(f, (IRK_beta - IRK_alpha).T)


skip = 80

N0 = 199
N1 = 201

data = scipy.io.loadmat('./burgers_shock.mat')

t_star = data['t'].flatten()[:, None]
x_star = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])

idx_t = 10

######################################################################
######################## Noiseles Data ###############################
######################################################################
noise = 0.0

idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
x0 = x_star[idx_x, :]
u0 = Exact[idx_x, idx_t][:, None]
u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
x1 = x_star[idx_x, :]
u1 = Exact[idx_x, idx_t + skip][:, None]
u1 = u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])

dt = np.asscalar(t_star[idx_t + skip] - t_star[idx_t])
q = int(np.ceil(0.5 * np.log(np.finfo(float).eps) / np.log(dt)))

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
IRK_alpha_ = torch.tensor(IRK_alpha).float().to(device)
IRK_beta_ = torch.tensor(IRK_beta).float().to(device)
dt_ = torch.tensor(dt).float().to(device)

lambda_1 = torch.tensor([0.0], requires_grad=True).float().to(device)
lambda_2 = torch.tensor([-6.0], requires_grad=True).float().to(device)
parameters = [lambda_1, lambda_2]

dnn = DNN(q).to(device)
model1 = PINN1(dnn, parameters).to(device)
model2 = PINN2(dnn, parameters).to(device)

optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

with tqdm(range(10000)) as bar:
    for epoch in bar:
        dnn.train()
        model1.train()
        model2.train()
        optimizer.zero_grad()

        loss1 = criterion(u1_, model2(x1_, dt_, IRK_alpha_, IRK_beta_))
        loss2 = criterion(u0_, model1(x0_, dt_, IRK_alpha_))
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        bar.set_postfix(loss=loss.item())

        if epoch % 500 == 0:
            dnn.eval()
            model1.eval()
            model2.eval()

            lambdas = [parameter.detach().cpu().numpy() for parameter in model1.param]
            lambda_1_value = lambdas[0]
            lambda_2_value = lambdas[1]
            error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
            error_lambda_2 = np.abs(np.exp(lambda_2_value) - (0.01 / np.pi)) / (0.01 / np.pi) * 100
            print('Error l1: %.5f%%' % (error_lambda_1), 'Error l2: %.5f%%' % (error_lambda_2))
