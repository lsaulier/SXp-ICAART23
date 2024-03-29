
import random
import numpy as np
import torch
from DQN import normalize

class Agent:

    def __init__(self, id, env, random=False, view_range=5, wave_type="square", wave_range=3, debug_position=None):
        self.id = id
        self.env = env
        self.actions = self.env.get_actionspace().n
        #  Type of wave and its range
        self.wave_type = wave_type
        self.wave_range = wave_range
        #  View range
        self.view_range = view_range
        #  Initial state
        if debug_position is not None:
            self.position = debug_position
        else:
            if random:
                #  Get a random position among all available places
                self.position = env.get_random_position()
            else:
                #  Get a random position among all "S" available places
                self.position = env.get_starting_position()
        # Put the agent on its initial position
        self.env.map[self.position[0]][self.position[1]] = 2
        self.observation = None
        self.dead = False

        return

    #  Get the information in range of the agent
    #  Input: position (int list) and a specific map (int list list)
    #  Output: neighbourhood of an agent (int list list)
    def view(self, position, optional_map=None):
        neighbourhood = []
        if optional_map is None:
            map = self.env.map
        else:
            map = optional_map
        #  Center index on agent's position
        view_range_index = self.convert_index(self.view_range//2)
        for i in range(len(view_range_index)):
            line = []
            #  Deal with cells out of bounds
            if position[0] + view_range_index[i] < 0 or position[0] + view_range_index[i] >= len(map):
                line = [0 for _ in range(self.view_range)]
            else:
                for j in range(len(view_range_index)):
                    #  Deal with cells out of bounds
                    if position[1] + view_range_index[j] < 0 or position[1] + view_range_index[j] >= len(map):
                        line.append(0)
                    #  Get information from cells (tree or drone or empty)
                    else:
                        line.append(map[position[0] + view_range_index[i]][position[1] + view_range_index[j]])

            neighbourhood.append(line)

        return neighbourhood

    #  Center index on agent's position. Function used for extracting sub part of an agent's state.
    #  Input: index's limit of view range (int)
    #  Output: index list (int list)
    def convert_index(self, limit):
        idx_list = [-limit]
        while idx_list[-1] < limit:
            idx_list.append(idx_list[-1] + 1)
        idx_list[-1] = limit
        return idx_list

    #  Choose the action to perform using the epsilon rate and DQN
    #  Input: NN (DQN), epsilon rate (float) and device type (String)
    #  Output: action (int)
    def chooseAction(self, net, epsilon=0.0, device="cpu"):
        #  Exploratory move
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        #  Greedy move
        else:
            #  Convert into tensor
            view = normalize(self.observation[0], 3, "view")
            feat = normalize(self.observation[1], 9, "feats")
            view = np.array(view, copy=False)
            feat = np.array(feat, copy=False)
            tens_view = torch.tensor(view, dtype=torch.float32, device=device)
            tens_feat = torch.tensor(feat, dtype=torch.float32, device=device)
            #  Merge view and position of the agent into one tensor
            state = (tens_view.unsqueeze(0).unsqueeze(0), tens_feat)
            #  Get best action according to DQN
            q_vals_v = net(state)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action

    #  Compute the state importance of a specific observation
    #  Input: policy (DQN), state (int) and device (str)
    #  Output: score (float)
    def stateImportance(self, net, observation, device):
        #  Convert into tensor
        view = normalize(observation[0], 3, "view")
        feat = normalize(observation[1], 9, "feats")
        view = np.array(view, copy=False)
        feat = np.array(feat, copy=False)
        tens_view = torch.tensor(view, dtype=torch.float32, device=device)
        tens_feat = torch.tensor(feat, dtype=torch.float32, device=device)
        #  Merge view and position of the agent into one tensor
        state = (tens_view.unsqueeze(0).unsqueeze(0), tens_feat)
        #  Get best action according to DQN
        q_vals_v = net(state)
        max_q_vals, _ = torch.max(q_vals_v, dim=1)
        min_q_vals, _ = torch.min(q_vals_v, dim=1)
        return max_q_vals.item() - min_q_vals.item()

    #  Get a 2D arrays representing the neighbourhood of the agent and its position
    #  Input: None
    #  Output: an observation (int list list)
    def get_obs(self):
        return self.observation

    #  Set a 2D arrays representing the neighbourhood of the agent and its position
    #  Input: an observation (int list list)
    #  Output: None
    def set_obs(self, obs):
        self.observation = obs
        return

    #  Get done info of the agent
    #  Input: None
    #  Output: done info (boolean)
    def get_dead(self):
        return self.dead

    #  Set done info of the agent
    #  Input: done info (boolean)
    #  Output: None
    def set_dead(self, dead):
        self.dead = dead
        return

class WindAgent:

    def __init__(self, id, env, behaviour, view_range=5, wave_type="square", wave_range=3):
        self.id = id
        self.env = env
        self.actions = self.env.get_actionspace().n - 1  # No "stop" action
        #  Type of Agent's wave and its range
        self.wave_type = wave_type
        self.wave_range = wave_range
        #  View range
        self.view_range = view_range
        #  Initial state (neighbourhood + position + Agent last action)
        self.position = None
        self.observation = [[[0 for n in range(self.view_range)] for n in range(self.view_range)], [0, 1, 0]]
        #  Type of wind agent
        self.behaviour = behaviour
        self.dead = False

    #  Get the information in range of the agent
    #  Input: position (int list) and a specific map (int list list)
    #  Output: neighbourhood of an agent (int list list)
    def view(self, position, optional_map=None):
        neighbourhood = []
        if optional_map is None:
            map = self.env.map
        else:
            map = optional_map
        #  Center index on agent's position
        view_range_index = self.convert_index(self.view_range//2)
        for i in range(len(view_range_index)):
            line = []
            #  Deal with cells out of bounds
            if position[0] + view_range_index[i] < 0 or position[0] + view_range_index[i] >= len(map):
                line = [0 for _ in range(self.view_range)]
            else:
                for j in range(len(view_range_index)):
                    #  Deal with cells out of bounds
                    if position[1] + view_range_index[j] < 0 or position[1] + view_range_index[j] >= len(map):
                        line.append(0)
                    #  Get information from cells (tree or drone or empty)
                    else:
                        line.append(map[position[0] + view_range_index[i]][position[1] + view_range_index[j]])

            neighbourhood.append(line)

        return neighbourhood

    #  Center index on agent's position. Function used for extracting sub part of an agent's state.
    #  Input: index's limit of view range (int)
    #  Output: index list (int list)
    def convert_index(self, limit):
        idx_list = [-limit]
        while idx_list[-1] < limit:
            idx_list.append(idx_list[-1] + 1)
        idx_list[-1] = limit
        return idx_list

    #  Choose the action to perform using the epsilon rate and DQN
    #  Input: NN (DQN), epsilon rate (float) and device type (String)
    #  Output: action (int)
    def chooseAction(self, net, epsilon=0.0, device="cpu"):
        #  Exploratory move
        if np.random.random() < epsilon:
            action = self.env.wind_action_space.sample()
        #  Greedy move
        else:
            #  Convert into tensor
            view = normalize(self.observation[0], 3, "view")
            feat = normalize(self.observation[1], 9, "feats")
            view = np.array(view, copy=False)
            feat = np.array(feat, copy=False)
            tens_view = torch.tensor(view, dtype=torch.float32, device=device)
            tens_feat = torch.tensor(feat, dtype=torch.float32, device=device)
            #  Merge view and position of the agent into one tensor
            state = (tens_view.unsqueeze(0).unsqueeze(0), tens_feat)
            #  Get best action according to net
            q_vals_v = net(state)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action

    #  Get a 2D arrays representing the neighbourhood of the agent and its position
    #  Input: None
    #  Output: an observation (int list list)
    def get_obs(self):
        return self.observation

    #  Set a 2D arrays representing the neighbourhood of the agent and its position
    #  Input: an observation (int list list)
    #  Output: None
    def set_obs(self, obs):
        self.observation = obs
        return


# Method(s) to handle trade-off exploration/exploitation

#  Build a list of values of exploratory rate. Decrease with times
#  Input : number of episodes (int), starting exploratory rate (float), exploratory rate decay (float),
#  minimum exploratory rate (float)
#  Output : list of exploratory rate (np.array)
def expRate_schedule(nE, exp_rate_start=1.0, exp_rate_decay=.9999, exp_rate_min=1e-4):
    x = np.arange(nE) + 1
    y = np.full(nE, exp_rate_start)
    y = np.maximum((exp_rate_decay ** x) * exp_rate_decay, exp_rate_min)
    return y
