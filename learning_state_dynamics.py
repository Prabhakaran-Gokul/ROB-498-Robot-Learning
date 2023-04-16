import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
from torchdiffeq import odeint_adjoint as odeint

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []

    for i in range(num_trajectories):
        trajectory = {"states": np.zeros((trajectory_length+1, 3), dtype=np.float32), 
                    "actions": np.zeros((trajectory_length, 3), dtype=np.float32)}
        # state = env.reset()
        state = env.reset()
        trajectory["states"][0, :] = state
        
        for t in range(trajectory_length):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            trajectory["states"][t+1, :] = state
            trajectory["actions"][t, :] = action

            if done:
                break

        collected_data.append(trajectory)

    # ---
    return collected_data


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    dataset = MultiStepDynamicsDataset(collected_data=collected_data, num_steps=num_steps)
    # print(dataset[11])
    train_data, val_data = random_split(dataset, 
                                        [0.8, 0.2])

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True
                              )
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=True)   
    
    return train_loader, val_loader


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        traj_idx = item // self.trajectory_length
        step_idx = item % self.trajectory_length
        state = self.data[traj_idx]["states"][step_idx]
        
        next_state_array = np.zeros(shape=(self.num_steps, 3), dtype=np.float32) 
        action_array = np.zeros(shape=(self.num_steps, 3), dtype=np.float32)


        for i in range(self.num_steps):
            action = self.data[traj_idx]["actions"][step_idx+i]
            next_state = self.data[traj_idx]["states"][step_idx+1+i]
            
            action_array[i, :] = action
            next_state_array[i, :] = next_state

        sample["state"] = state
        sample["action"] = action_array
        sample["next_state"] = next_state_array

        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        # x_pred, y_pred, theta_pred = pose_pred[:, 0], pose_pred[:, 1], pose_pred[:, 2]
        # x_target, y_target, theta_target = pose_target[:, 0], pose_target[:, 1], pose_target[:, 2]
        pose_target = pose_target.to(pose_pred.device)
        x_pred, y_pred, theta_pred = pose_pred[:, 0], pose_pred[:, 1], pose_pred[:, 2]
        x_target, y_target, theta_target = pose_target[:, 0], pose_target[:, 1], pose_target[:, 2]
        loss_func = nn.MSELoss()
        rg = np.sqrt((self.w**2 + self.l**2) / 12.0)
        # print(x_pred.device, x_target.device, y_pred.device, y_target.device, theta_pred.device, theta_target.device)

        se2_pose_loss = loss_func(x_pred, x_target) + \
                        loss_func(y_pred, y_target) + \
                        rg * loss_func(theta_pred, theta_target)

        # ---
        return se2_pose_loss



class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        assert len(actions.shape) == 3
        B, NUM_STEPS, size = actions.shape
        target_states = target_states.to(state.device)

        next_state_pred = state
        multi_step_loss = 0
        # next_state_pred_arr = torch.zeros(size=(NUM_STEPS, *(state.shape)))
        losses_i = torch.zeros(actions.shape[1])
        for i in range(NUM_STEPS):
            next_state_pred = model.forward(next_state_pred, actions[:, i, :]) 
            # next_state_pred_arr[i] = next_state_pred
            # print(next_state_pred.shape, target_states.shape)
            loss_i = self.loss(next_state_pred, target_states[:, i, :])
            losses_i[i] = loss_i

        lambdas = torch.tensor([self.discount**i for i in range(actions.shape[1])])
        multi_step_loss = lambdas * losses_i
        multi_step_loss = multi_step_loss.sum()

        # multi_step_loss += self.discount * self.loss(next_state_pred_arr, target_states)

        # multi_step_loss /= len(state)

        # ---
        return multi_step_loss



def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    target_pose = target_pose.to(state.device)

    Q = torch.tensor((
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0.1]
                    ), device=state.device)
    cost = torch.sum((state - target_pose) @ Q * (state - target_pose), 1)

    # ---
    return cost.to("cpu")


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    # obstacle_corners = get_corner_coordinates(obstacle_centre.reshape(1, -1), obstacle_dims)
    # unrotated_object_corners = get_corner_coordinates(state[:, :2], torch.tensor([box_size, box_size]))
    # object_corners = rotate_corner_coordinates(unrotated_object_corners, state[:, -1])
    
    # in_collision = torch.full(size=(len(object_corners),), fill_value=-1.0, dtype=torch.float)
    # for i in range(len(object_corners)):
    #     in_collision[i] = is_polygon_intersecting(obstacle_corners.reshape(4, 2), object_corners[i])

    w = box_size/2
    obs_w = obstacle_dims/2
    corner = torch.tensor([[w,w],[-w,-w]]).to(state.device)
    corner_obs = torch.stack((obs_w,-1*obs_w), dim=0)
    obs_dim = obstacle_centre + corner_obs
    in_collision = torch.zeros(state.shape[0]).to(state.device)

    for i in range(state.shape[0]):
      obj_dim = corner + state[i,:2]
      if(obj_dim[0,0]<obs_dim[1,0] or
      obj_dim[1,0]>obs_dim[0,0] or
      obj_dim[0,1]<obs_dim[1,1] or
      obj_dim[1,1]>obs_dim[0,1]):
        in_collision[i] = 0
      else:
        in_collision[i] = 1

    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    target_pose = target_pose.to(state.device)
    # --- Your code here
    in_collision = collision_detection(state)
    Q = torch.tensor((
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0.1]
                ), device=state.device)
    cost = torch.sum((state - target_pose) @ Q * (state - target_pose), 1) + 100 * in_collision

    # ---
    return cost.to("cpu")


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        # state = state.to("cuda:0")
        # action = action.to("cuda:0")
        next_state = self.model(state, action)

        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        # state_tensor = torch.from_numpy(state[:3])
        state_tensor = torch.from_numpy(state)
        print(state_tensor.device)
        print("control")

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.detach().numpy()
        # print(action)


        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here

def get_corner_coordinates(centre, dims):
    # print(centre.shape)
    B, _ = centre.shape
    corners = torch.zeros(B, 4, 2)
    cx, cy = centre[:, 0].reshape(-1, 1), centre[:, 1].reshape(-1, 1)
    w, l = dims[0], dims[1]
    
    corners[:, 0, :] = torch.cat((cx - w/2, cy + l/2), -1)
    corners[:, 1, :] = torch.cat((cx + w/2, cy + l/2), -1)
    corners[:, 2, :] = torch.cat((cx + w/2, cy - l/2), -1)
    corners[:, 3, :] = torch.cat((cx - w/2, cy - l/2), -1)

    return corners

def rotate_corner_coordinates(corners, angles):
    # rotate corners by the given angles
    rotated_corners = torch.zeros_like(corners)
    angles = angles.reshape(-1, 1)
    # print("rotate")
    # print(corners.shape, angles.shape)

    rotated_corners[:, :, 0] = corners[:, :, 0] * torch.cos(angles) -\
                               corners[:, :, 1] * torch.sin(angles)
    rotated_corners[:, :, 1] = corners[:, :, 0] * torch.sin(angles) +\
                               corners[:, :, 1] * torch.cos(angles)
    
    return rotated_corners

def is_polygon_intersecting(obstacle_corners, object_corners):
    polygons = [obstacle_corners, object_corners]
    minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None

    for i in range(len(polygons)):

        # for each polygon, look at each edge of the polygon, and determine if it separates
        # the two shapes
        polygon = polygons[i]
        for i1 in range(len(polygon)):

            # grab 2 vertices to create an edge
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            # find the line perpendicular to this edge
            normal = { 'x': p2[1] - p1[1], 'y': p1[0] - p2[0] }

            minA, maxA = None, None
            # for each vertex in the first shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            for j in range(4):
                projected = normal['x'] * obstacle_corners[j][0] + normal['y'] * obstacle_corners[j][1]
                if (minA is None) or (projected < minA): 
                    minA = projected

                if (maxA is None) or (projected > maxA):
                    maxA = projected

            # for each vertex in the second shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            minB, maxB = None, None
            for j in range(4): 
                projected = normal['x'] * object_corners[j][0] + normal['y'] * object_corners[j][1]
                if (minB is None) or (projected < minB):
                    minB = projected

                if (maxB is None) or (projected > maxB):
                    maxB = projected

            # if there is no overlap between the projects, the edge we are looking at separates the two
            # polygons, and we know there is no overlap
            if (maxA < minB) or (maxB < minA):
                return 0.0

    return 1.0

# ---
# ============================================================
