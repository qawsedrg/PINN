import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

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
        self.c=c

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
        f = u_tt - self.c**2 * u_xx
        return f

T=1
L=1
c=0.5
n_max=5
n=np.arange(1, n_max + 1)
lam=2*L/n_max
omiga=n_max*np.pi*c/L
k=omiga/c
#phi_n=np.random.randn(n_max,1)
#C_n=np.random.randn(n_max,1)
phi_n=np.zeros((n_max,1))
C_n=np.zeros((n_max,1))
C_n[-1,:]=1
def alpha(x):
    alpha_n = C_n * np.cos(phi_n)
    return np.sum(alpha_n*np.sin(np.einsum("n,xj->nx",n,x)*np.pi/L),axis=0)[:,None]
def beta(x):
    beta_n = n[:,None] * np.pi * c / L * C_n * np.sin(phi_n)
    return np.sum(beta_n*np.sin(np.einsum("n,xj->nx",n,x)*np.pi/L),axis=0)[:,None]

# 20 point per period
t=np.linspace(0,T,int(n_max*T/(2*L/c)*20))[:,None]
x=np.linspace(0,L,n_max*20)[:,None]
Exact_ori = np.sum(C_n[:,:,None]*np.einsum("nt,nx->nxt",np.cos(np.einsum("n,tj->nt",n,t)*np.pi*c/L-phi_n),np.sin(np.einsum("n,xj->nx",n,x)*np.pi/L)),axis=0).T

X, T = np.meshgrid(x, t)
grid = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

grid_bc1 = np.hstack((x, np.zeros_like(x)))  # (x,0)
u_bc1 = alpha(x)
grid_bc2 = np.hstack((np.zeros_like(t)+L, t))  # (L,t)
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
u_t_bc=beta(x)
u_t_bc = torch.tensor(u_t_bc).float().to(device)

dnn = DNN().to(device)
model = PINN(dnn,c)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = torch.nn.MSELoss()

plt.ion()
fig1 = plt.figure("sol",figsize=(7, 4))
fig2 = plt.figure("t",figsize=(7, 4))
fig3 = plt.figure("Error", figsize=(7, 4))
fig4 = plt.figure("truth", figsize=(7, 4))

mask_x=torch.logical_or(grid_x>=0.5,grid_x<=0.5)
mask_t=torch.logical_or(grid_t>=1.1,grid_t<=0.3)
mask_train=torch.logical_and(mask_x,mask_t)
grid_x_train=torch.tensor(grid_x[mask_train,None], requires_grad=True).float().to(device)
grid_t_train=torch.tensor(grid_t[mask_train,None], requires_grad=True).float().to(device)

# dataloader creat great IO cost cannot be used

with tqdm(range(10000000)) as bar:
    for epoch in bar:
        dnn.train()
        model.train()
        optimizer.zero_grad()

        mask=torch.randint(0,5,(len(grid_x_train),))==0

        f = model(grid_x_train[mask], grid_t_train[mask])
        loss_bc = criterion(u_bc, dnn(grid_x_bc, grid_t_bc))
        u = dnn(grid_x_bc1, grid_t_bc1)
        u_t=torch.autograd.grad(
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
        bar.set_postfix(loss_bc=loss_bc.item(),loss_f=loss_f.item())

        # eval and draw

        if epoch % 500 == 0:
            dnn.eval()
            model.eval()
            with torch.no_grad():
                u = dnn(grid_x, grid_t).cpu().numpy()

            error = np.linalg.norm(Exact - u, 2) / np.linalg.norm(Exact, 2)
            print('Error u: {:}'.format(error))

            error = griddata(grid, ((Exact - u) ** 2).flatten(), (X, T), method='cubic') / np.linalg.norm(Exact, 2) ** 2
            u = griddata(grid, u.flatten(), (X, T), method='cubic')

            t1=int(0.25*len(t))
            t2=int(0.5*len(t))
            t3=int(0.75*len(t))
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

            # ax2 = fig2.add_subplot(111)

            gs1 = gridspec.GridSpec(1, 3)
            gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

            ax2 = fig2.add_subplot(gs1[0, 0])
            ax2.plot(x, Exact_ori[t1, :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[t1, :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_title('$t = 0.t1$', fontsize=15)
            ax2.set_xlim([x.min(), x.max()])
            ax2.set_ylim([Exact_ori.min(), Exact_ori.max()])

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            ax2 = fig2.add_subplot(gs1[0, 1])
            ax2.plot(x, Exact_ori[t2, :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[t2, :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_xlim([x.min(), x.max()])
            ax2.set_ylim([Exact_ori.min(), Exact_ori.max()])
            ax2.set_title('$t = 0.t2$', fontsize=15)
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
            ax2.plot(x, Exact_ori[t3, :], 'b-', linewidth=2, label='Exact')
            ax2.plot(x, u[t3, :], 'r--', linewidth=2, label='Prediction')
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$u(t,x)$')
            ax2.set_xlim([x.min(), x.max()])
            ax2.set_ylim([Exact_ori.min(), Exact_ori.max()])
            ax2.set_title('$t = 0.t3$', fontsize=15)

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(15)

            #####################################
            ############     ax3     ############
            #####################################

            ax3 = fig3.add_subplot(111)

            h = ax3.imshow(error.T, interpolation='nearest', cmap='rainbow',
                           extent=[t.min(), t.max(), x.min(), x.max()],
                           origin='lower', aspect='auto')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig3.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax3.plot(
                grid_bc[:, 1],
                grid_bc[:, 0],
                'kx', label='Data (%d points)' % (u_bc.shape[0]),
                markersize=4,
                clip_on=False,
                alpha=1.0
            )

            line = np.linspace(x.min(), x.max(), 2)[:, None]
            ax3.plot(t[t1] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax3.plot(t[t2] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax3.plot(t[t3] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax3.set_xlabel('$t$', size=20)
            ax3.set_ylabel('$x$', size=20)
            ax3.legend(
                loc='upper center',
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
                prop={'size': 15}
            )
            ax3.set_title('$u(t,x)$', fontsize=20)
            ax3.tick_params(labelsize=15)

            #####################################
            ############     ax4     ############
            #####################################
            
            ax4 = fig4.add_subplot(111)

            h = ax4.imshow(griddata(grid, Exact.flatten(), (X, T), method='cubic').T, interpolation='nearest', cmap='rainbow',
                           extent=[t.min(), t.max(), x.min(), x.max()],
                           origin='lower', aspect='auto')
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig4.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax4.plot(
                grid_bc[:, 1],
                grid_bc[:, 0],
                'kx', label='Data (%d points)' % (u_bc.shape[0]),
                markersize=4,
                clip_on=False,
                alpha=1.0
            )

            line = np.linspace(x.min(), x.max(), 2)[:, None]
            ax4.plot(t[t1] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax4.plot(t[t2] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax4.plot(t[t3] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax4.set_xlabel('$t$', size=20)
            ax4.set_ylabel('$x$', size=20)
            ax4.legend(
                loc='upper center',
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
                prop={'size': 15}
            )
            ax4.set_title('$u(t,x)$', fontsize=20)
            ax4.tick_params(labelsize=15)

            plt.pause(0.1)
            fig1.clf()
            fig2.clf()
            fig3.clf()
            fig4.clf()

plt.ioff()
