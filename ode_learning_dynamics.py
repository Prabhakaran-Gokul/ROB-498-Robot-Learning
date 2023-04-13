import os
import argparse
import time
import numpy as np

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from learning_state_dynamics import *

# parser = argparse.ArgumentParser('ODE demo')
# parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
# parser.add_argument('--data_size', type=int, default=1000)
# parser.add_argument('--batch_time', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=20)
# parser.add_argument('--niters', type=int, default=2000)
# parser.add_argument('--test_freq', type=int, default=20)
# parser.add_argument('--viz', action='store_true')
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', action='store_true')
# = parser.parse_)

method='dopri5'
data_size=1000 
batch_time=11 
batch_size=20 
niters=2000 
test_freq=20 
viz=True 
gpu=False 
adjoint=True

def read_params(flag=False):
    
    global method, data_size, batch_time, batch_size, niters, test_freq, viz, gpu, adjoint
    
    if flag:
        config = yaml.load('config.yaml')
        method = config['method']
        data_size = config['data_size']
        batch_time = config['batch_time']
        batch_size = config['batch_size']
        niters = config['niters']
        test_freq = config['test_freq']
        viz = config['viz']
        gpu = config['gpu']
        adjoint = ['adjoint']



if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method = 'dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEDynamicsModel(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(ODEDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(self.state_dim + action_dim, 50),
            nn.Tanh(),
            nn.Linear(50, self.state_dim + action_dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, input_state):
        print(input_state[0].shape, input_state[1].shape)

        input_state = torch.cat((input_state[0], input_state[1]), -1)
        print("in: ", input_state.shape)
        out = self.net(input_state)
        print("out: ", out.shape)
        return out
        # return torch.cat((out, torch.zeros(size=(len(input_state), 3))), -1)


def train_step(model, train_loader, optimizer, loss_func) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0.

    model.train()
    for batch_idx, data in enumerate(train_loader):
        # --- Your code here
        state = data["state"]
        action = data["action"]
        next_state = data["next_state"]
        optimizer.zero_grad()
        loss = loss_func(model, state, action, next_state)
        loss.backward()
        optimizer.step()
        # ---
        train_loss += loss.item()
    return train_loss/len(train_loader)


def val_step(model, val_loader, loss_func) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    # --- Your code here

    model.eval()
    # ---
    for batch_idx, data in enumerate(val_loader):
        loss = None
        # --- Your code here
        state = data["state"]
        action = data["action"]
        next_state = data["next_state"]
        loss = loss_func(model, state, action, next_state)

        # ---
        val_loss += loss.item()
    return val_loss/len(val_loader)


def train_model(model, train_dataloader, val_dataloader, optimizer, loss_func, num_epochs=1000):
    """
    Trains the given model for `num_epochs` epochs. Use Adam as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = None
        val_loss_i = None
        # --- Your code here
        train_loss_i = train_step(model=model, train_loader=train_dataloader, optimizer=optimizer, loss_func=loss_func)
        val_loss_i = val_step(model=model, val_loader=val_dataloader, loss_func=loss_func)
        # ---
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
        # if epoch_i % test_freq == 0:
        #     with torch.no_grad():
        #         pred_y = odeint(model, true_y0, t)
        #         loss = torch.mean(torch.abs(pred_y - true_y))
        #         print('Iter {:04d} | Total Loss {:.6f}'.format(epoch_i, loss.item()))
        #         visualize(true_y, pred_y, model, epoch_i)
        #         epoch_i += 1
    return train_losses, val_losses


def train_ode(lr=1e-3, num_epochs=1000):
    pushing_multistep_ODE_dynamics_model = ODEDynamicsModel(3, 3).to(device)
    collected_data = np.load(os.path.join("./", 'collected_data.npy'), allow_pickle=True)
    train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=100)

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = MultiStepLoss(pose_loss, discount=0.9)
    # ii = 0
    
    optimizer = optim.RMSprop(pushing_multistep_ODE_dynamics_model.parameters(), lr=lr)

    train_losses, val_losses = train_model(model=pushing_multistep_ODE_dynamics_model,
                                           train_dataloader=train_loader,
                                           val_dataloader=val_loader,
                                           optimizer=optimizer,
                                           loss_func=pose_loss,
                                           num_epochs=num_epochs)
    

    # save model:
    save_path = os.path.join("./", 'pushing_multi_step_residual_dynamics_model.pt')
    torch.save(pushing_multistep_ODE_dynamics_model.state_dict(), save_path)


    # plot train loss and test loss:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    axes[0].plot(train_losses)
    axes[0].grid()
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_yscale('log')
    axes[1].plot(val_losses)
    axes[1].grid()
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_yscale('log')

    plt.show()



if __name__ == '__main__':
    train_ode()
