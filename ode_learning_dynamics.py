import os
import argparse
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from learning_state_dynamics import *

from panda_pushing_env import PandaPushingEnv
from visualizers import GIFVisualizer, NotebookVisualizer
from learning_state_dynamics import PushingController, free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE
from IPython.display import Image

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
gpu_device=0
gpu=True
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
        gpu_device = config['gpu_device']
        gpu = config['gpu']
        adjoint = ['adjoint']



if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(gpu_device) if torch.cuda.is_available() and gpu else 'cpu')

class ODEDynamicsModel(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(ODEDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(self.state_dim + action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.state_dim + action_dim)
            ).to(device)

    def forward(self, t, input_state):
        out = self.net(input_state.to(device))
        
        return out


class ODEDynamicsModelWrapper(nn.Module):
    """
    Wraps ODEDynamicsModel class to take in and return the correct dimensions
    """
    def __init__(self, ode_dynamics_model: ODEDynamicsModel) -> None:
        super(ODEDynamicsModelWrapper, self).__init__()
        self.ode_dynamics_model = ode_dynamics_model
        # self.t = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        self.t = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float32).to(device)
        

    def forward(self, state, action):
        # combine state and action to pass it in to ode dynamics model
        state = state.to(device)
        action = action.to(device)
        input_to_ode = torch.cat((state, action), -1).to(device)
        out = odeint(self.ode_dynamics_model,
                     input_to_ode, 
                     self.t)
        
        # retrieve the next state prediction
        out = out[1]
        out = out[:, :3]
        
        return out


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
        state = data["state"].to(device)
        action = data["action"].to(device)
        next_state = data["next_state"].to(device)
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

    return train_losses, val_losses


def train_ode(lr=1e-3, num_epochs=20):
    pushing_multistep_ODE_dynamics_model = ODEDynamicsModel(3, 3).to(device)
    pushing_multistep_ODE_dynamics_model_wrapper = ODEDynamicsModelWrapper(pushing_multistep_ODE_dynamics_model).to(device)
    collected_data = np.load(os.path.join("./", 'collected_data.npy'), allow_pickle=True)
    train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=100)

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = MultiStepLoss(pose_loss, discount=0.9)

    optimizer = optim.Adam(pushing_multistep_ODE_dynamics_model_wrapper.parameters(), lr=lr)

    train_losses, val_losses = train_model(model=pushing_multistep_ODE_dynamics_model_wrapper,
                                           train_dataloader=train_loader,
                                           val_dataloader=val_loader,
                                           optimizer=optimizer,
                                           loss_func=pose_loss,
                                           num_epochs=num_epochs)
    

    # save model:
    model_save_path = os.path.join("./", 'pushing_multi_step_ode_dynamics_model.pt')
    torch.save(pushing_multistep_ODE_dynamics_model_wrapper.state_dict(), model_save_path)

    # plot train loss and test loss:
    plot_losses(train_losses, val_losses)

    return pushing_multistep_ODE_dynamics_model_wrapper


def obstacle_free_controller():
    model = ODEDynamicsModelWrapper(ode_dynamics_model=ODEDynamicsModel(3, 3))
    model.load_state_dict(torch.load(os.path.join("./", 'pushing_multi_step_ode_dynamics_model.pt'), map_location=torch.device("cuda:0")))
    model = model.to(device)

    # Control on an obstacle free environment
    visualizer = GIFVisualizer()

    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, model, free_pushing_cost_function, num_samples=100, horizon=10)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 20

    for i in tqdm(range(num_steps_max)):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_FREE
    target_state = target_state
    end_state = end_state
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')
    
    Image(visualizer.get_gif())


def obstacle_controller():
    model = ODEDynamicsModelWrapper(ode_dynamics_model=ODEDynamicsModel(3, 3))
    model.load_state_dict(torch.load(os.path.join("./", 'pushing_multi_step_ode_dynamics_model.pt'), map_location=torch.device("cuda:0")))
    model = model.to(device)
    # Control on an obstacle free environment

    visualizer = GIFVisualizer()

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
    env.reset()


    state_0 = env.reset()
    state = state_0

    num_steps_max = 20

    for i in tqdm(range(num_steps_max)):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            break


    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')

    Image(filename=visualizer.get_gif())


def eval_learned_dynamics():
    # load pretrained model
    ODE_model = ODEDynamicsModelWrapper(ode_dynamics_model=ODEDynamicsModel(3, 3))
    ODE_model.load_state_dict(torch.load(os.path.join("./", 'pushing_multi_step_ode_dynamics_model.pt'), map_location=torch.device("cuda:0")))

    # Initialize the simulation environment
    env = PandaPushingEnv(visualizer=None, 
                          render_non_push_motions=True,  
                          camera_heigh=800, 
                          camera_width=800)
    
    state = env.reset()
    pred_state = state.copy()
    gt_states = [state]
    pred_states = [pred_state]


    # Perform a sequence of random actions:
    for i in tqdm(range(30)):
        action_i = env.action_space.sample()
        
        pred_state = ODE_model(torch.tensor(pred_state).reshape(1, -1).to(device), 
                               torch.tensor(action_i).reshape(1, -1).to(device))
        gt_state, reward, done, info = env.step(action_i)

        pred_states.append(pred_state.detach().cpu().numpy().reshape(-1))
        gt_states.append(gt_state)

        if done:
            break

    plot_trajectory(pred_states, gt_states)


# Plot functions

def plot_losses(train_losses, val_losses):
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


def plot_trajectory(pred_states, gt_states):
    pred_states = np.array(pred_states)
    gt_states = np.array(gt_states)
    
    pred_xs = pred_states[:, 0]
    gt_xs = gt_states[:, 0]

    plt.plot(pred_xs, color="r")
    plt.plot(gt_xs, color="g")

    plt.show()


if __name__ == '__main__':
    # train_ode()
    # obstacle_free_controller() 
    # obstacle_controller() 
    eval_learned_dynamics()
