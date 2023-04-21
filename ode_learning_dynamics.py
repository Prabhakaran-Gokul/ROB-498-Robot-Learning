import os
import argparse
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from learning_state_dynamics import *
from torchdiffeq import odeint_adjoint as odeint_adj
from torchdiffeq import odeint as odeint_norm


from panda_pushing_env import PandaPushingEnv
from visualizers import GIFVisualizer
from learning_state_dynamics import PushingController, free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE
from IPython.display import Image


def read_params(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    method = config['method']
    train = config['train']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    viz = config['viz']
    device = config['device']
    adjoint = config['adjoint']
    num_steps_max = config['num_steps_max']
    lr = float(config['lr'])
    num_pts_for_intp = config['num_pts_for_intp']

    return train, num_epochs, batch_size, viz, device, adjoint, num_steps_max, lr


def read_grid_search_params(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    methods = config['methods']
    num_pts_for_intp = config['num_pts_for_intp']
    hidden_layers = config['hidden_layers']
    hidden_layer_neurons = config['hidden_layer_neurons']

    return methods, num_pts_for_intp, hidden_layers, hidden_layer_neurons 


class ODEDynamicsModel(nn.Module):

    def __init__(self, state_dim, action_dim, f, device='cpu',
                 hidden_layer=1,
                 hidden_layer_neuron=100):
        
        super(ODEDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.activation_func = nn.ReLU()

        # create neural network layers
        modules = []
        input_layer = nn.Linear(self.state_dim + action_dim, hidden_layer_neuron)
        modules.append(input_layer)
        modules.append(self.activation_func)
        for layer in range(hidden_layer):
            modules.append(nn.Linear(hidden_layer_neuron, hidden_layer_neuron))
            modules.append(self.activation_func)
        output_layer = nn.Linear(hidden_layer_neuron, self.state_dim + action_dim)
        modules.append(output_layer)
        self.net = nn.Sequential(*modules).to(device)

        # self.net = nn.Sequential(
        #     nn.Linear(self.state_dim + action_dim, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, self.state_dim + action_dim)
        #     ).to(device)
        log_var = [module for module in self.net.modules() if not isinstance(module, nn.Sequential)]
        f.write("architecture:  " + str(log_var) + "\n")

    def forward(self, t, input_state):
        out = self.net(input_state.to(self.device))
        
        return out


class ODEDynamicsModelWrapper(nn.Module):
    """
    Wraps ODEDynamicsModel class to take in and return the correct dimensions
    """
    def __init__(self,
                 odeint, 
                 ode_dynamics_model: ODEDynamicsModel, 
                 f, device='cpu', 
                 method='dopri5', 
                 num_pts_for_intp=4) -> None:
        
        super(ODEDynamicsModelWrapper, self).__init__()
        self.ode_dynamics_model = ode_dynamics_model
        self.t = torch.linspace(start=0, end=1, 
                                steps=num_pts_for_intp+1, 
                                dtype=torch.float32, device=device)
        
        self.t = self.t[:-1]

        f.write("timesteps:  " + str(self.t) + "\n")
        
        self.device = device
        self.method = method
        self.odeint = odeint

    def forward(self, state, action):
        # combine state and action to pass it in to ode dynamics model
        state = state.to(self.device)
        action = action.to(self.device)
        input_to_ode = torch.cat((state, action), -1).to(self.device)

        out = self.odeint(self.ode_dynamics_model,
                     input_to_ode, 
                     self.t,
                     method = self.method)
        
        # retrieve the next state prediction
        out = out[1]
        out = out[:, :3]
        
        return out


def train_step(model, train_loader, optimizer, loss_func, device) -> float:
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


def val_step(model, val_loader, loss_func, device) -> float:
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


def train_model(model, train_dataloader, val_dataloader, 
                optimizer, loss_func, num_epochs=1000, 
                device='cpu'):
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
        train_loss_i = train_step(model=model, train_loader=train_dataloader, optimizer=optimizer, loss_func=loss_func, device=device)
        val_loss_i = val_step(model=model, val_loader=val_dataloader, loss_func=loss_func, device=device)
        # ---
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses


def train_ode(dir, f, 
              method, odeint, 
              lr=1e-3, num_epochs=10, 
              device='cpu', batch_size=100, 
              viz=True, num_pts_for_intp=4,
              hidden_layer=1, hidden_layer_neuron=100):
    
    pushing_multistep_ODE_dynamics_model = ODEDynamicsModel(3, 3, f,
                                                            hidden_layer=hidden_layer,
                                                            hidden_layer_neuron=hidden_layer_neuron,
                                                            device=device)
    pushing_multistep_ODE_dynamics_model_wrapper = ODEDynamicsModelWrapper(odeint, 
                                                                           pushing_multistep_ODE_dynamics_model, 
                                                                           f, 
                                                                           method=method, 
                                                                           num_pts_for_intp=num_pts_for_intp, device=device
                                                                           )
    f.write("training_integration_method:  "+method+"\n")
    curr_dir, _ = os.path.split(os.path.realpath(__file__))
    collected_data = np.load(os.path.join(curr_dir, 'collected_data.npy'), allow_pickle=True)
    train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=batch_size)

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1).to(device)
    pose_loss = MultiStepLoss(pose_loss, discount=0.9).to(device)

    optimizer = optim.Adam(pushing_multistep_ODE_dynamics_model_wrapper.parameters(), lr=lr)
    f.write("learning rate:  " + str(lr) + "\n")
    f.write("optimizer:  " + str(type(optimizer)) + "\n")

    train_losses, val_losses = train_model(model=pushing_multistep_ODE_dynamics_model_wrapper,
                                           train_dataloader=train_loader,
                                           val_dataloader=val_loader,
                                           optimizer=optimizer,
                                           loss_func=pose_loss,
                                           num_epochs=num_epochs,
                                           device=device)

    # save model:
    # model_save_path = os.path.join("./", 'pushing_multi_step_ode_dynamics_model.pt')
    model_log_path = os.path.join(dir, 'pushing_multi_step_ode_dynamics_model.pt')
    # torch.save(pushing_multistep_ODE_dynamics_model_wrapper.state_dict(), model_save_path)
    torch.save(pushing_multistep_ODE_dynamics_model_wrapper.state_dict(), model_log_path)

    # plot train loss and test loss:
    plot_losses(train_losses, val_losses, dir, f, viz)

    return pushing_multistep_ODE_dynamics_model_wrapper


def obstacle_free_controller(dir, f, method, odeint, device='cpu', num_pts_for_intp=4, num_steps_max=20, hidden_layer=1, hidden_layer_neuron=100):
    ode_dynamics_model = ODEDynamicsModel(3, 3, f, hidden_layer=hidden_layer, hidden_layer_neuron=hidden_layer_neuron)
    model = ODEDynamicsModelWrapper(odeint, ode_dynamics_model=ode_dynamics_model, num_pts_for_intp=num_pts_for_intp, f=f, method=method)
    f.write("obstacle_free_integration_method:  " + method + "\n")
    model.load_state_dict(torch.load(os.path.join(dir, 'pushing_multi_step_ode_dynamics_model.pt'), map_location=torch.device(device)))
    model = model.to(device)

    # Control on an obstacle free environment
    visualizer = GIFVisualizer()

    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, model, free_pushing_cost_function, num_samples=100, horizon=10)
    env.reset()

    state_0 = env.reset()
    state = state_0
    is_valid_action = False


    num_steps_max = num_steps_max
    f.write("max_steps:  " + str(num_steps_max) + "\n")
    for i in tqdm(range(num_steps_max)):
        while not is_valid_action:
            action = controller.control(state)
            is_valid_action = env.check_action_valid(action)
        is_valid_action = False
        
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
    f.write("Goal Reached:  "+str(goal_reached)+"\n")

    
    Image(visualizer.get_gif(dir=dir, obs=0))


def obstacle_controller(dir, f, 
                        method, odeint, 
                        device='cpu', num_pts_for_intp=4, 
                        num_steps_max=20, hidden_layer=1, 
                        hidden_layer_neuron=100):

    ode_dynamics_model = ODEDynamicsModel(3, 3, f, hidden_layer=hidden_layer, hidden_layer_neuron=hidden_layer_neuron)
    model = ODEDynamicsModelWrapper(odeint, ode_dynamics_model=ode_dynamics_model, num_pts_for_intp=num_pts_for_intp, f=f, method=method)
    f.write("obstacle_integration_method:  "+ method +"\n")
    model.load_state_dict(torch.load(os.path.join(dir, 'pushing_multi_step_ode_dynamics_model.pt'), map_location=torch.device(device)))
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

    num_steps_max = num_steps_max
    f.write("max_steps_obstacle:  "+str(num_steps_max)+ "\n")
    is_valid_action = False

    for i in tqdm(range(num_steps_max)):
        while not is_valid_action:
            action = controller.control(state)
            is_valid_action = env.check_action_valid(action)
        is_valid_action = False
        
        state, reward, done, _ = env.step(action)

        if done:
            break


    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE
    
    f.write("Goal Reached_obstacle:  "+str(goal_reached)+"\n")
    print(f'GOAL REACHED: {goal_reached}')

    Image(filename=visualizer.get_gif(dir, obs=1))


def eval_learned_dynamics(dir, f, 
                          method, odeint, device='cpu', 
                          num_pts_for_intp=4,
                          hidden_layer=1,
                          hidden_layer_neuron=100,
                          viz='True'):
    # load pretrained model
    ode_dynamics_model = ODEDynamicsModel(3, 3, f, hidden_layer=hidden_layer, hidden_layer_neuron=hidden_layer_neuron)
    model = ODEDynamicsModelWrapper(odeint, ode_dynamics_model=ode_dynamics_model, num_pts_for_intp=num_pts_for_intp, f=f, method=method)
    pose_error_func = SE2PoseLoss(block_width=0.1, block_length=0.1).to(device)

    f.write("eval_integration_method:  " + method + "\n")

    model.load_state_dict(torch.load(os.path.join(dir, 'pushing_multi_step_ode_dynamics_model.pt'), map_location=torch.device(device)))

    # Initialize the simulation environment
    env = PandaPushingEnv(visualizer=None, 
                          render_non_push_motions=True,  
                          camera_heigh=800, 
                          camera_width=800)
    
    state = env.reset()
    pred_state = state.copy()
    gt_states = [state]
    pred_states = [pred_state]
    print(torch.from_numpy(pred_state).reshape(1, -1), torch.from_numpy(pred_state).reshape(1, -1).shape)
    se2_errors = [pose_error_func(torch.from_numpy(pred_state).reshape(1, -1), torch.from_numpy(state).reshape(1, -1))]


    # Perform a sequence of random actions:
    for i in tqdm(range(30)):
        action_i = env.action_space.sample()
        
        pred_state = model(torch.tensor(pred_state).reshape(1, -1).to(device), 
                               torch.tensor(action_i).reshape(1, -1).to(device))
        gt_state, reward, done, info = env.step(action_i)

        pred_states.append(pred_state.detach().cpu().numpy().reshape(-1))
        gt_states.append(gt_state)
        se2_errors.append(
            pose_error_func(
                torch.tensor(pred_state).reshape(1, -1),
                torch.tensor(state).reshape(1, -1)
                )
        )

        if done:
            break

    plot_trajectory(pred_states, gt_states, se2_errors, dir, viz)


# Plot functions

def plot_losses(train_losses, val_losses, dir, f, viz):
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

    plt.savefig(dir+"/Train_val_loss.png")
    if viz:
        plt.show()


def plot_trajectory(pred_states, gt_states, se2_errors, dir, viz):
    pred_states = np.array(pred_states)
    gt_states = np.array(gt_states)

    fig, axes = plt.subplots(4, sharex=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.figure(figsize=(20, 6))
    state_variables = ["x", "y", "theta"]

    for idx, var in enumerate(state_variables):
        pred = pred_states[:, idx]
        gt = gt_states[:, idx]
        axes[idx].plot(pred, color='r')
        axes[idx].plot(gt, color='g')
        axes[idx].grid()
        axes[idx].set_title(f'Ground Truth {var} vs Predicted {var}')
        axes[idx].set_xlabel('Time steps')
        axes[idx].set_ylabel(f'{var} state')

    axes[len(state_variables)].plot(se2_errors)
    axes[len(state_variables)].grid()
    axes[len(state_variables)].set_title(f'SE2 Error between Ground Truth and Predicted Trajectories')
    axes[len(state_variables)].set_xlabel('Time steps')
    axes[len(state_variables)].set_ylabel('SE2 Error (m)')

    plt.savefig("Trajectory.png")
    plt.savefig(dir + "/Trajectory.png")
    if viz:
        plt.show()


def run(method, pts_for_intp, hidden_layer, hidden_layer_neuron):
    train, num_epochs, batch_size, viz, device, adjoint, num_steps_max, lr = read_params(config_file="config.yml")
    
    if adjoint:
        odeint = odeint_adj
    else:
        odeint = odeint_norm
    curr_dir, _ = os.path.split(os.path.realpath(__file__))
    dir = os.path.join(curr_dir, "plots", str(datetime.datetime.now()))
    plots_dir = os.path.join(curr_dir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(dir)
    # os.mkdir(dir)

    with open(dir+"/config_log.txt", "a") as f:
        f.write("\n##########ENTER##########\n")
        f.write("num_epochs:  "+str(num_epochs)+"\n")
        f.write("control_epochs:  "+str(num_steps_max)+"\n")
        f.write("device:  "+str(device)+"\n")
        f.write("ode_method:  "+str(odeint)+"\n")

        if train:
            print("Training")
            f.write("\n##########TRAINING##########\n")
            train_ode(dir=dir, f=f, 
                      method=method, odeint=odeint, 
                      lr=lr, num_epochs=num_epochs, 
                      device=device, batch_size=batch_size, 
                      viz=viz, num_pts_for_intp=pts_for_intp,
                      hidden_layer=hidden_layer,
                      hidden_layer_neuron=hidden_layer_neuron)
            
        print("CONTROL - OBSTACLE FREE")

        f.write("\n##########CONTROL - OBSTACLE FREE##########\n")
        obstacle_free_controller(dir=dir, f=f,
                                 method=method, odeint=odeint, 
                                 device=device, num_pts_for_intp=pts_for_intp, 
                                 num_steps_max=num_steps_max,
                                 hidden_layer=hidden_layer,
                                 hidden_layer_neuron=hidden_layer_neuron)         
        print("CONTROL - W/ OBSTACLE")
        
        f.write("\n##########CONTROL - W/ OBSTACLE##########\n")
        obstacle_controller(dir=dir, f=f, 
                            method=method, odeint=odeint, 
                            device=device, num_pts_for_intp=pts_for_intp,
                            num_steps_max=num_steps_max,
                            hidden_layer=hidden_layer,
                            hidden_layer_neuron=hidden_layer_neuron) 
        
        print("EVALUATION")
        f.write("\n##########EVALUATION##########\n")
        eval_learned_dynamics(dir=dir, f=f, 
                              method=method, odeint=odeint, 
                              device = device, num_pts_for_intp=pts_for_intp,
                              hidden_layer=hidden_layer,
                              hidden_layer_neuron=hidden_layer_neuron,
                              viz=viz)


def grid_search():
    methods, num_pts_for_intp, hidden_layers, hidden_layer_neurons = read_grid_search_params(config_file="grid_search.yml")
    for method in methods:
        for pts_for_intp in num_pts_for_intp:
            for hidden_layer in hidden_layers:
                for hidden_layer_neuron in hidden_layer_neurons:
                    run(method, pts_for_intp, hidden_layer, hidden_layer_neuron)


def main(run_grid_search=False):
    if not run_grid_search:
        method = "dopri5"
        pts_for_intp = 4
        hidden_layer = 1
        hidden_layer_neuron = 100
        run(method, pts_for_intp, hidden_layer, hidden_layer_neuron)

    else:
        grid_search()
