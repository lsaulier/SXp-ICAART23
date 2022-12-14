
import gym
from gym import Env
from gym import utils
from gym.spaces import Discrete
from colorize import colorize, yellow
from math import sqrt
import numpy as np
import random
from copy import deepcopy

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STOP = 4

#  Action space
Actions = [LEFT, DOWN, RIGHT, UP, STOP]

#  Set of maps
MAPS = {
    "10x10": ["S---T---TS",
              "-T--------",
              "----------",
              "-----T----",
              "----------",
              "--T-------",
              "--------TT",
              "----------",
              "------T---",
              "S--------S"],

    "30x30": ["S-------TTT---S--T------TT---S",
              "--------------------S---------",
              "-TT--S-----------------------S",
              "--------TT------S---TTT-------",
              "--------------------T----S----",
              "S-----------T-----------------",
              "------------T-----------------",
              "---TT---STT------T---S--TT----",
              "-TT---------------------------",
              "S------------T--S------------S",
              "---------S-----------TT-------",
              "--------TT--------------------",
              "--S-----TT---S---------S-----T",
              "---------T------------T------T",
              "------------------------------",
              "S---T----S-------T------TT---S",
              "-------------------S----------",
              "---S---------T-------------T--",
              "------T-------------TTT-------",
              "-----TTTT----S----T------S----",
              "S------T----------T-----------",
              "------------------------------",
              "----TT--S--------T---S--TT----",
              "--------------S---------------",
              "S---T-------------------------",
              "----T--------TTT-S-----------S",
              "------S------TT----------TT---",
              "------------TTT---------------",
              "------------------------------",
              "S-----T-------S---------TT---S"]

}

class DroneAreaCoverage(Env):

    def __init__(self, map_name="10x10", windless=True, wind_probas=[0.3, 0.2, 0.4, 0.1], wind_agent_train=False):
        #  Only updated when render function is called
        self.render_map = np.asarray(MAPS[map_name], dtype="c")
        self.nRow, self.nCol = len(self.render_map), len(self.render_map[0])
        #  Action space
        self.action_space, self.wind_action_space = Discrete(len(Actions)), Discrete(len(Actions) - 1)
        #  Presence or not during training of wind agent(s)
        self.wind_agent_train = wind_agent_train
        #  Initialize map
        self.init_map()
        #  List of last actions
        self.last_actions = None
        #  Transition probability
        self.windless = windless
        if self.windless:
            self.P = None
        else:
            self.P = wind_probas

    #  Initialize the map (int list list) based on the render map (the map does not contain drones for now)
    #  Input: None
    #  Output: None
    def init_map(self):
        map = []
        for i in range(len(self.render_map)):
            line = []
            for j in range(len(self.render_map[0])):
                if bytes(self.render_map[i, j]) == b"T":
                    line.append(3)  # Tree
                else:
                    line.append(1)  # Otherwise

            map.append(line)
        self.map = map
        return

    #  Initialize observations of Agents and Wind Agents
    #  Input: agents (Agent list), put agents in pre-defined position or not (boolean), distinction between Agent and
    #  Wind Agent (boolean) and use of debugging positions (boolean)
    #  Output: None
    def initObs(self, agents, rand, wind=False, debug_position=False):
        for agent in agents:
            #  Agent: only set an observation
            if not wind or debug_position:
                agent.set_obs([agent.view(agent.position), agent.position])
            # Wind Agent: set position, change map and set observation
            else:
                #  Position
                if rand:
                    agent.position = self.get_random_position()
                else:
                    agent.position = self.get_starting_position()
                #  Map's update
                self.map[agent.position[0]][agent.position[1]] = 2
        #  Wind Agents observations
        if wind:
            for agent in agents:
                feats = [agent.position[0], agent.position[1], random.randint(0, self.action_space.n - 1)]
                agent.set_obs([agent.view(agent.position), feats])
        return

    #  Initialize position of Agents
    #  Input: agents (Agent list), put agents in pre-defined position or not (boolean)
    #  Output: None
    def initPos(self, agents, rand):
        for agent in agents:
            #  Set position
            if rand:
                agent.position = self.get_random_position()
            else:
                agent.position = self.get_starting_position()
            # Map's update
            self.map[agent.position[0]][agent.position[1]] = 2
        return

    #  Update coordinates of a drone
    #  Input: coordinates and an action (int)
    #  Output: new coordinates (int)
    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nRow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.nCol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return [row, col]

    #  Reset environment and agents/wind agents observation
    #  Input: agents (Agent list or WindAgent list), put agents in pre-defined position or not (boolean),
    #  type of agent (String)
    #  Output: None
    def reset(self, agents, rand, behaviour=None):
        # Initialize the map
        self.init_map()
        positions = []
        #  Select a position for each agent
        for agent in agents:
            if not rand:  # Start at a predefined position "S"
                position = self.get_starting_position()
            else:  # Start in any position except trees and drones positions
                position = self.get_random_position()
            #  Reset dead, position, map position value
            agent.dead = False
            agent.position = position
            self.map[position[0]][position[1]] = 2
            positions.append(position)
        #  Set new states for each agent
        for i in range(len(positions)):
            if behaviour is None:
                agents[i].set_obs([agents[i].view(positions[i]), positions[i]])
            else:
                feats = [positions[i][0], positions[i][1], random.randint(0, self.wind_action_space.n - 1)]
                agents[i].set_obs([agents[i].view(positions[i]), feats])
        return

    #  Perform action for all agents, update the map and their observations and check collisions
    #  Input: agents (Agent list) and their actions (int list), use of step() for P-scenario (boolean), wind agent
    #  (WindAgent) and its NN (DQN), device (String), type of transition function (String) and list of fixed transitions
    #  for debugging code (int list list)
    #  Output: the old, temporary (before the wind moves the agent) and new states (int list list), list
    #  used to know if it's the end of an episode (boolean list) and a copy of the current map (int list list)
    def step(self, agents, actions, most_probable_transitions=False, wind_agent=None, wind_net=None, device="cpu", move="all", fixed_transition=[]):
        #  Initialization
        states = []
        temp_states = []
        new_states = []
        new_positions = []
        map_copy = None
        #  Get agents states
        for agent in agents:
            states.append(agent.get_obs())
        #  Update map and positions
        for i in range(len(actions)):
            self.map[states[i][1][0]][states[i][1][1]] = 1
            new_position = self.inc(states[i][1][0], states[i][1][1], actions[i])
            new_positions.append(new_position)
            #  Modify the map for winds prediction when computing SXp
            if wind_net is not None:
                wind_agent.env.map[states[i][1][0]][states[i][1][1]] = 1

        #  Modify the map for winds prediction when computing SXp
        if wind_net is not None:
            for position in new_positions:
                if wind_agent.env.map[position[0]][position[1]] != 3:
                    wind_agent.env.map[position[0]][position[1]] = 2
        #  Store temporary states for wind agents training
        if self.wind_agent_train:
            map_copy = deepcopy(self.map)
            #  Drone positions
            for i in range(len(new_positions)):
                if map_copy[new_positions[i][0]][new_positions[i][1]] != 3:
                    map_copy[new_positions[i][0]][new_positions[i][1]] = 2
            #  Drone temporary states
            for i in range(len(new_positions)):
                temp_state = [agents[0].view(new_positions[i], map_copy), deepcopy(new_positions[i])]
                temp_state[1].append(actions[i])
                temp_states.append(temp_state)
        #  Due to transition probabilities, new positions change
        if not self.windless:
            for i in range(len(new_positions)):
                #  Transition / Wind's action choice
                if most_probable_transitions:  # P-scenario
                    action = np.argmax(self.P)
                elif fixed_transition:  # Study special configurations and compare Agents
                    action = fixed_transition[i]
                elif wind_net is not None:  # HE/FE-scenario
                    feats = [new_positions[i][0], new_positions[i][1], actions[i]]
                    wind_agent.set_obs([wind_agent.view(new_positions[i]), feats])
                    action = wind_agent.chooseAction(wind_net, device=device)
                    for position in new_positions:
                        if wind_agent.env.map[position[0]][position[1]] != 3:
                            wind_agent.env.map[position[0]][position[1]] = 1

                else:  # Agent training process
                    action = np.random.choice(4, p=self.P)

                #  Change position if actions are not opposite and not 'stop' or if action is 'stop'
                if move == "all":
                    if actions[i] == 4 or not (actions[i] - 2 == action or action - 2 == actions[i]):
                        new_positions[i] = self.inc(new_positions[i][0], new_positions[i][1], action)
                else:
                    if actions[i] != 4 and not(actions[i] - 2 == action or action - 2 == actions[i]):
                        new_positions[i] = self.inc(new_positions[i][0], new_positions[i][1], action)

        #  Update self.map, agents.dead and dones
        dones = self.collisions(agents, new_positions)
        #  Update observations of agents
        for i in range(len(agents)):
            agents[i].position = new_positions[i]
            agents[i].set_obs([agents[i].view(agents[i].position), agents[i].position])
            new_states.append(agents[i].get_obs())

        #  Update for rendering the environment
        self.last_actions = actions

        return states, temp_states, new_states, dones, map_copy

    #  Perform action for a type of Wind agents, update the map and their observation and check collisions
    #  Input: wind agents (WindAgent list), their actions (int list), type of transition (String),
    #  agents (only used in correlTrain()) (Agent list), their NN (DQN) and the device (String)
    #  Output: the old, temporary (before the wind moves the agent) and new states (int list list), list
    #  used to know if it's the end of an episode (boolean list) and a copy of the current map (int list list)
    def windStep(self, wind_agents, actions, move="all", agents=[], net=None, device="cpu"):
        #  Initialization
        states = []
        new_states = []
        new_positions = []
        #  Get wind agents states
        for agent in wind_agents:
            states.append(agent.get_obs())

        #  Update map and positions
        for i in range(len(actions)):
            last_agent_action = states[i][1][2]
            #  Transition: move only if Agent action is 'stop' or both actions are not opposite
            if move == "all":
                if last_agent_action == 4 or not (last_agent_action - 2 == actions[i] or actions[i] - 2 == last_agent_action):
                    self.map[states[i][1][0]][states[i][1][1]] = 1
                    new_position = self.inc(states[i][1][0], states[i][1][1], actions[i])
                    new_positions.append(new_position)
                else:
                    new_positions.append([states[i][1][0], states[i][1][1]])
            #  Transition: move only if Agent action is not stop
            else:
                if last_agent_action != 4 and not(last_agent_action - 2 == actions[i] or actions[i] - 2 == last_agent_action):
                    self.map[states[i][1][0]][states[i][1][1]] = 1
                    new_position = self.inc(states[i][1][0], states[i][1][1], actions[i])
                    new_positions.append(new_position)
                else:
                    new_positions.append([states[i][1][0], states[i][1][1]])

        #  Update self.map, wind_agents.dead and dones
        dones = self.collisions(wind_agents, new_positions)
        #  Update observations of wind_agents
        for i in range(len(wind_agents)):
            #  Choice of agent action
            if agents:
                last_agent_action = agents[i].chooseAction(net, epsilon=0.0, device=device)
            else:
                last_agent_action = random.randint(0, self.action_space.n - 1)
            feats = [new_positions[i][0], new_positions[i][1], last_agent_action]
            wind_agents[i].set_obs([wind_agents[i].view(new_positions[i]), feats])
            new_states.append(wind_agents[i].get_obs())

        #  Update for rendering environment
        self.last_actions = actions

        return states, None, new_states, dones, None

    #  Check collisions and update map values and dead attributes of agents
    #  Input : agents (Agent list or WindAgent list), agents positions (int list list)
    #  Output : list used to know if it's the end of an episode (boolean list)
    def collisions(self, agents, positions):
        unique_positions = []
        dones = []
        i = 0
        for position in positions:
            if agents[i].dead:  # Special case, only used for SXp
                dones.append(True)
            else:
                #  Fill unique_positions
                if position not in unique_positions:
                    unique_positions.append(position)

                #  Check collision with a tree
                if self.map[position[0]][position[1]] == 3:
                    dones.append(True)
                    agents[i].dead = True
                else:
                    dones.append(False)
                    self.map[position[0]][position[1]] = 2

            i += 1
        #  Check collision between 2 drones
        if len(unique_positions) < len(positions):
            #  Extract coordinates of collision
            for i in range(len(positions)):
                if positions.count(positions[i]) > 1:
                    dones[i] = True
                    agents[i].dead = True
                    self.map[positions[i][0]][positions[i][1]] = 1
        return dones

    #  List all agents which have an imperfect cover
    #  Input: agents (Agent list)
    #  Output: imperfect cover agents (Agent list)
    def agentsImperfectCover(self, agents):
        imprfct_agents = []
        for agent in agents:
            view = agent.get_obs()[0]
            #  Get only the wave range matrix
            index_range = (agent.view_range - agent.wave_range) // 2
            sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]
            #  Another drone in range or a tree in coverage area zone
            if sum([sub_list.count(2) for sub_list in view]) > 1 or sum([sub_list.count(3) for sub_list in sub_view]) > 0:
                imprfct_agents.append(agent)

        return imprfct_agents

    #  Display to the user the current state of the environment. There are two different ways for rendering the
    #  environment. It depends on the number of agents on the map.
    #  Input: agents (Agent list)
    #  Output: None
    def render(self, agents):
        small_agents_nbr = len(agents) <= 8
        imprfct_agents = []
        # Determine number of imperfect agents (if there is more than 8 agents)
        if not small_agents_nbr:
            imprfct_agents = self.agentsImperfectCover(agents)
        if small_agents_nbr or len(imprfct_agents) <= 8:
            colors = ["blue", "green", "red", "yellow", "cyan", "magenta", "gray", "black"]
        else:
            colors = [str(i) for i in range(1, 256)]

        #  Print current actions of each drone
        if self.last_actions is not None:
            str_actions = ["Left", "Down", "Right", "Up", "Stop"]
            string = " "
            cpt = 0
            for i in range(len(agents)):
                if imprfct_agents and agents[i] in imprfct_agents or not imprfct_agents:
                    #  Display action (or Dead)
                    if agents[i].dead:
                        string += colorize("Dead", colors[i], small=small_agents_nbr)
                    else:
                        string += colorize(str_actions[self.last_actions[i]], colors[i], small=small_agents_nbr)
                    string += " "
                    #  Avoid a long line of actions
                    if cpt % 9 == 0 and cpt not in [len(self.last_actions) - 1, 0]:
                        string += "\n\n "
                    cpt += 1

            print(string)
        print()
        #  Update render_map to the current map
        render_map = self.render_map.tolist()
        render_map = [[c.decode("utf-8") for c in line] for line in render_map]

        for i in range(len(agents)):
            if not agents[i].dead:
                position = agents[i].get_obs()[1]
                render_map[position[0]][position[1]] = "D"
                #  Colorize covered cells
                if imprfct_agents and agents[i] not in imprfct_agents:
                    render_map[position[0]][position[1]] = yellow(render_map[position[0]][position[1]])
                else:
                    self.color_coverageArea(render_map, position, agents[i], colors[i%len(colors)], small=small_agents_nbr)

        #  Render
        for line in range(self.nRow):
            row_str = ""
            for column in range(self.nCol):
                row_str = row_str + render_map[line][column]
            print(row_str)

        print()
        return

    #  Color in render() the coverage area of the agent
    #  Input: render map (np array), position of the agent (int list), the agent (Agent), a color (String)
    #  and a different choice of color (boolean)
    #  Output: None
    def color_coverageArea(self, map, position, agent, color, small):
        #  Color each cell under constraints
        wave_range_index = agent.convert_index(agent.wave_range//2)
        for i in wave_range_index:
            if position[0] + i >= 0 and position[0] + i < len(self.map):  # Out of bounds condition
                for j in wave_range_index:
                    if position[1] + j >= 0 and position[1] + j < len(self.map):   # Out of bounds condition
                        # Cell is neither occupied by a tree nor a drone
                        if map[position[0] + i][position[1] + j] != "T" and map[position[0] + i][position[1] + j] != "D":
                            map[position[0] + i][position[1] + j] = colorize(map[position[0] + i][position[1] + j], color, small=small)

        return

    #  Compute the maximum reachable cumulative reward
    #  Input: agents (Agent list), list used to know if it's the end of an episode (boolean list) and
    #  reward type (String)
    #  Output: reward (int)
    def max_reward(self, agents, dones=None, reward_type="B"):
        max_reward = 0
        for i in range(len(agents)):
            if dones is None:
                if reward_type == "A":
                    max_reward += 1
                else:
                    max_reward += len(agents) - 1
            else:
                if not dones[i]:
                    if reward_type == "A":
                        max_reward += 1
                    else:
                        max_reward += len(agents) - 1
        return max_reward

    #  At a timestep, compute the reward for each agent
    #  Input: agents (Agent list), actions (int list), list used to know if it's the end of an episode (boolean list)
    #  and reward's type (String)
    #  Output: list of reward (float list)
    def getReward(self, agents, actions, dones, reward_type="B"):
        rewards = []
        i = 0
        for agent in agents:
            #  Crash of an agent
            if dones[i]:
                if reward_type == "A":
                    rewards.append(-1)
                else:
                    rewards.append(-(len(agents)-1))
            else:
                #  Initialization
                reward = 0
                max_cells_highlighted = (agent.wave_range * agent.wave_range - 1)
                max_agents_inrange = agent.view_range**2 - 1
                max_agents_inrange = min(max_agents_inrange, len(agents)-1)
                view = agent.get_obs()[0]
                #  Get only the wave range matrix for computing reward (3x3 matrix)
                index_range = (agent.view_range - agent.wave_range) // 2
                sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]
                #  Count highlighted cells in 3x3 matrix
                cells_highlighted = sum(sub_list.count(1) for sub_list in sub_view)
                if reward_type == "A":
                    #  Penality if at least one different drone is in view range
                    if sum([sub_list.count(2) for sub_list in view]) > 1:
                        reward = -1
                    #  Penality Stop action without perfect cover
                    if self.windless:
                        if actions[i] == 4 and cells_highlighted != max_cells_highlighted:
                            reward = -1
                    #  Cover reward
                    if reward == 0:
                        #  Perfect cover
                        if max_cells_highlighted == cells_highlighted:
                            reward = 1

                else:
                    #  Penality for each drone(s) in view range
                    reward -= (sum([sub_list.count(2) for sub_list in view]) - 1) / max_agents_inrange * (len(agents)-1)
                    #  Cover reward
                    if max_cells_highlighted == cells_highlighted:
                        #  Perfect cover
                        reward += len(agents)-1
                    else:
                        reward += (cells_highlighted / max_cells_highlighted) * (len(agents)-2)

                rewards.append(reward)

            i += 1
        return rewards


    #  At a timestep, compute the reward for each wind agent of a specific type
    #  Input: agents (Agent list), list used to know if it's the end of an episode (boolean list) and
    #  reward's type (String)
    #  Output: list of reward (float list)
    def getWindReward(self, agents, dones, reward_type="B"):
        rewards = []
        for i in range(len(agents)):
            agent = agents[i]
            #  Crash of an agent
            if dones[i]:
                if reward_type == "A":
                    if agent.behaviour == "hostile":
                        rewards.append(1)
                    else:
                        rewards.append(-1)
                else:
                    if agent.behaviour == "hostile":
                        rewards.append((len(agents)-1))
                    else:
                        rewards.append(-(len(agents)-1))
            else:
                #  Initialization
                reward = 0
                max_cells_highlighted = (agent.wave_range * agent.wave_range - 1)
                max_agents_inrange = agent.view_range**2 - 1
                max_agents_inrange = min(max_agents_inrange, len(agents)-1)
                view = agent.get_obs()[0]
                #  Get only the wave range matrix for computing reward (3x3 matrix)
                index_range = (agent.view_range - agent.wave_range) // 2
                sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]
                #  Count highlighted cells in 3x3 matrix
                cells_highlighted = sum(sub_list.count(1) for sub_list in sub_view)
                if reward_type == "A":
                    #  Penality if at least one different drone is in view range
                    if sum([sub_list.count(2) for sub_list in view]) > 1:
                        if agent.behaviour == "hostile":
                            reward = 1
                        else:
                            reward = -1
                    #  Cover reward
                    if reward == 0:
                        #  Perfect cover
                        if max_cells_highlighted == cells_highlighted:
                            if agent.behaviour == "hostile":
                                reward = -1
                            else:
                                reward = 1
                else:
                    #  Penality for each drone(s) in view range
                    if agent.behaviour == "hostile":
                        reward += (sum([sub_list.count(2) for sub_list in view]) - 1) / max_agents_inrange * (len(agents)-1)
                    else:
                        reward -= (sum([sub_list.count(2) for sub_list in view]) - 1) / max_agents_inrange * (len(agents)-1)
                    #  Cover Reward
                    if max_cells_highlighted == cells_highlighted:
                        #  Perfect cover
                        if agent.behaviour == "hostile":
                            reward -= len(agents) - 1
                        else:
                            reward += len(agents) - 1
                    else:
                        #  Imperfect cover
                        if agent.behaviour == "hostile":
                            reward -= (cells_highlighted / max_cells_highlighted) * (len(agents) - 2)
                        else:
                            reward += (cells_highlighted / max_cells_highlighted) * (len(agents) - 2)

                rewards.append(reward)

        return rewards

    #  Set a list of actions used for render method
    #  Input : actions (int list)
    #  Output : None
    def set_lastactions(self, actions):
        self.last_actions = actions
        return

    #  Get the action space
    #  Input : None
    #  Output : action space (Discrete)
    def get_actionspace(self):
        return self.action_space

    #  Get the list of predefined available starting free positions "S" and randomly select one
    #  Input: None
    #  Output: a randomly-chosen position (int list)
    def get_starting_position(self):
        positions = []
        for i in range(len(self.render_map)):
            for j in range(len(self.render_map[0])):
                if bytes(self.render_map[i, j]) == b"S" and self.map[i][j] != 2:  # 2: drone is already here
                    positions.append([i, j])
        return random.choice(positions)

    #  Get a random position in the map where a drone can be
    #  Input: None
    #  Output: a position (int list)
    def get_random_position(self):
        i, j = random.randint(0, len(self.map)-1), random.randint(0, len(self.map)-1)
        while bytes(self.render_map[i, j]) == b"T" or self.map[i][j] == 2:  # 2: drone already here | T: tree
            i, j = random.randint(0, len(self.map)-1), random.randint(0, len(self.map)-1)
        return [i, j]

