import scipy.io as scio
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sol(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList([])
        self.linear.append(nn.Linear(2, 8))
        self.linear.append(nn.Linear(8, 16))
        self.linear.append(nn.Linear(16, 32))
        self.linear.append(nn.Linear(32, 16))
        self.linear.append(nn.Linear(16, 8))
        self.output_layer = nn.Linear(8, 1)

    def forward(self, t, x):
        grid_t, grid_x = torch.meshgrid(t, x, indexing='ij')
        grid_t = torch.tensor(grid_t, requires_grad=True)
        grid_x = torch.tensor(grid_x, requires_grad=True)
        out = torch.stack([grid_t, grid_x], dim=-1)
        for layer in self.linear:
            out = F.relu(layer(out))
        out = self.output_layer(out)
        return torch.squeeze(out), grid_t, grid_x


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.u = Sol()

    def forward(self, t, x):
        u, grid_t, grid_x = self.u(t, x)
        u_t = autograd.grad(u.sum(), grid_t, create_graph=True)[0]
        u_x = autograd.grad(u.sum(), grid_x, create_graph=True)[0]
        u_xx = autograd.grad(u_x.sum(), grid_x, create_graph=True)[0]
        f = u_t + u * u_x - (0.01 / torch.pi) * u_xx
        return f


u = Sol()
u = u.to(device)
f = PINN()
f = f.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(u.parameters(), lr=1e-5)

data = scio.loadmat("./burgers_shock.mat")
x = torch.tensor(data['x'], requires_grad=True).to(device)[:, -1].float()
t = torch.tensor(data['t'], requires_grad=True).to(device)[:, -1].float()
usol = torch.tensor(data['usol']).to(device)

x_one = torch.ones_like(x).to(device)
x_minusone = -torch.ones_like(x).to(device)
t_zero = torch.zeros_like(t).to(device)

u_bc1 = -torch.sin(torch.pi * x).to(device)  # (0,x)
u_bc2 = torch.zeros_like(t).to(device)  # (t,1)
u_bc3 = torch.zeros_like(t).to(device)  # (t,-1)

for epoch in range(100000):
    optimizer.zero_grad()
    loss_boundary = 0
    loss_boundary += criterion(u_bc1, u(t_zero, x)[0][0, :])
    loss_boundary += criterion(u_bc2, u(t, x_one)[0][:, -1])
    loss_boundary += criterion(u_bc3, u(t, x_minusone)[0][:, 0])
    loss_pdf = torch.sum(f(t, x) ** 2)
    loss = loss_boundary + loss_pdf
    loss.backward(retain_graph=True)
    optimizer.step()
    print(loss.item())
print("")
