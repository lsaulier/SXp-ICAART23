#  WARNING : this version of the environment only works when there is only one starting state S in the map
#  Need updates to allow several starting states

import gym
from gym import Env
from gym import utils
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#  Set of maps
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],

    "6x6": [
        "SFFFFF",
        "FFHFFF",
        "FFHFFF",
        "FFFHHF",
        "FFFFFF",
        "HFFFFG"
    ],

    "7x7": [
        "SFFFFFH",
        "FFFFFFH",
        "FHHFFFH",
        "FHHFFFH",
        "FFFFFFF",
        "FFFFFFF",
        "HHHHFFG"
    ],

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]

}

#  Class used for coloring the current state in the render function
class bcolors:
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

class MyFrozenLake(Env):

    def __init__(self, desc=None, map_name="4x4", is_slippery=True, behaviour=None):
        #  Map
        desc = MAPS[map_name]
        self.desc = np.asarray(desc, dtype="c")
        #  Action space
        self.action_space = Discrete(4)
        #  Dimension of the map
        self.nRow, self.nCol = self.desc.shape
        # Number of Actions, States
        nA, nS = 4, self.nRow*self.nCol
        #  Initial state
        self.state = 0
        #  Behaviour
        self.behaviour = behaviour
        #  State space
        if behaviour is None:
            self.observation_space = Discrete(self.nRow*self.nCol)
        else:
            self.observation_space = Discrete(self.nRow * self.nCol * nA)
        # Last action (useful for the render)
        self.lastaction = None
        #  Probability matrix for wind
        if self.behaviour is not None:
            self.P = {s: {a: [] for a in range(nA - 1)} for s in range(nS * nA)}
        #  Probability matrix for the agent
        else:
            self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        #  Get the goal position (only one goal allowed)
        row_goal, col_goal = None, None
        for i in range(len(self.desc)):
            for j in range(len(self.desc[0])):
                if bytes(self.desc[i, j]) in b"G":
                    row_goal, col_goal = i, j
        self.goal_position = [row_goal, col_goal]

        #  Choose a reward for WindAgent, depending on the wind's and cell's type
        #  Input: type of cell (bytes)
        #  Output: reward (float)
        def wind_reward(type_cell):
            if type_cell == b"H":
                if self.behaviour == "Favorable":
                    return 0.0  # -1.0
                else:
                    return 1.0  # 1.0
            elif type_cell == b"G":
                if self.behaviour == "Favorable":
                    return 1  # 10.0
                else:
                    return -1  # -10.0
            else:
                return 0.0

        #  Update the probability matrix
        #  Input: coordinates (int), action (int), and the possible future action (int)
        #  Last argument only used for wind probability matrix
        #  Output: new state (int), reward (float) and end of episode (boolean)
        def update_probability_matrix(row, col, action, future_action=None):
            newrow, newcol = self.inc(row, col, action)
            if future_action is not None:
                newstate = self.to_s(newrow, newcol) * nA + future_action
            else:
                newstate = self.to_s(newrow, newcol)
            newletter = self.desc[newrow, newcol]
            done = bytes(newletter) in b"GH"
            #  Change reward whether it's an agent or the wind
            if self.behaviour is not None:
                reward = wind_reward(bytes(newletter))
            else:
                reward = float(newletter == b"G")

            return newstate, reward, done

        # Fill the probability matrix
        for row in range(self.nRow):
            for col in range(self.nCol):

                if self.behaviour is not None:
                    for a in range(nA):
                        real_state_env = self.to_s(row, col)
                        s = real_state_env * nA + a
                        i = 0
                        for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            li = self.P[s][i]
                            letter = self.desc[row, col]
                            if letter in b"GH":
                                li.append((1.0, s, 0, True))
                            else:
                                for j in range(nA):
                                    li.append((1.0 / 4.0, *update_probability_matrix(row, col, b, j)))

                            i += 1
                else:
                    s = self.to_s(row, col)
                    for a in range(4):
                        li = self.P[s][a]
                        letter = self.desc[row, col]
                        if letter in b"GH":
                            li.append((1.0, s, 0, True))
                        else:
                            if is_slippery:
                                for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                    li.append(
                                        (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                    )
                            else:
                                li.append((1.0, *update_probability_matrix(row, col, a)))


        return

    #  From coordinates to state
    #  Input: coordinates (int)
    #  Output: a state (int)
    def to_s(self, row, col):
        return row * self.nCol + col

    #  Update coordinates
    #  Input: coordinates (int) and action (int)
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
        return (row, col)

    #  The agent performs a step in the environment and arrive in a particular state
    #  according to the probability matrix, the current state, the action of both the agent and the wind
    #  Input: action (int), index of the action (int) and already chosen new state (int).
    #  Two last arguments are only used for the wind
    #  Output: new state (int), the reward (int), done flag (bool) and probability
    #  to come into the new state (float)
    def step(self, action, index_action=None, new_state=None):

        #  Case of the wind's step
        if index_action is not None:

            transitions = self.P[self.state][index_action]
            #  Get the transition which lead to the already chosen new state
            for transition in transitions:
                if new_state == transition[1]:
                    p, s, r, d = transition

        #  Case of the agent's step
        else:

            transitions = self.P[self.state][action]
            #  Random choice among possible transitions
            i = np.random.choice(len(transitions))
            p, s, r, d = transitions[i]
        #  Updates
        self.state = s
        self.lastaction = action
        return (s, r, d, {"prob": p})

    #  Reset the environment
    #  Input: None
    #  Output: initial state (int)
    def reset(self):
        self.state = 0
        self.lastaction = None
        return self.state

    #  Display to the user the current state of the environment
    #  Input:  None
    #  Output: None
    def render(self):

        #  Print the current action
        if self.lastaction != None:
            print("    ({})".format(["Left", "Down", "Right", "Up"][self.lastaction]))

        #  Get the current position depending on the agent's type
        if self.behaviour is not None:
            row, col = (self.state // 4) // self.nCol, (self.state // 4) % self.nCol
        else:
            row, col = self.state // self.nCol, self.state % self.nCol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        #  Highlight current position in red
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        #  Render
        for line in range(self.nRow):
            row_str = ""
            for column in range(self.nCol):
                row_str = row_str + desc[line][column]
            print(row_str)

        return

    #  Update the current state and action. Function Used during a SXp and a training of a wind agent
    #  Input: action (int) and a state (int)
    #  Output: None
    def update(self, action, nextState):
        self.state = nextState
        self.lastaction = action
        return

    #  Set a state
    #  Input: state (int)
    #  Output: None
    def setObs(self, obs):
        self.state = obs
        return

    #  Get the current state
    #  Input: None
    #  Output: state (int)
    def getObs(self):
        return self.state

    #  Get reward given an action performed from a state and a reached state
    #  Input: state, action and reached state (int)
    #  Output: reward (float)
    def getReward(self, state, action, reached_state):
        for transition in self.P[state][action]:
            if transition[1] == reached_state:
                #  Get reward
                return transition[2]

