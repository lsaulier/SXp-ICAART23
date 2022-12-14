
import numpy as np
import json
import time
import os

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class Agent:

    def __init__(self, name, env, lr=.2, exp_rate=1.0, decay_gamma=0.95, model_env_wind_list=None):
        #  Version
        self.name = name
        #  Parameters for updating Q
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        #  Environment
        self.env = env
        #  Actions
        self.actions = 4
        #  States
        self.states = self.env.nRow*self.env.nCol
        #  Q table
        self.Q = np.zeros((self.states, self.actions))
        #  Initialise couple list of model,env of the wind
        if model_env_wind_list is not None:
            self.wind_list = model_env_wind_list

    #  Train agent and wind agents simultaneously
    #  Input: number of training episode (int)
    #  Output: None
    def train(self, nE):
        #  Set list of exploratory rates
        expRate_schedule = self.expRate_schedule(nE)
        #  Training loop
        for episode in range(nE):

            #  Display the training progression
            if episode % 500 == 0:
                print("Episodes : " + str(episode))

            #  Flag to stop the episode
            done = False
            while not done:

                #  Useful to keep for updating Q
                current_state = self.env.state
                #  Choose an action
                self.exp_rate = expRate_schedule[episode]
                action = self.chooseAction(current_state)
                #  Execute it
                new_state, reward, done, _ = self.env.step(action)
                #  Update Q value
                self.updateQ(current_state, action, new_state, reward)

                #  Learn the wind Q tables
                if self.wind_list:
                    for mod_env in self.wind_list:

                        #  Upgrade the current state of the wind
                        wind_state = current_state * self.actions + action
                        mod_env[1].update(action, wind_state) # action is not really necessary
                        #  Choose an action
                        mod_env[0].exp_rate = expRate_schedule[episode]
                        wind_action, index_wind_action = mod_env[0].chooseAction(wind_state)
                        #  Execute it
                        new_wind_state, reward, _, _ = mod_env[1].step(wind_action, index_wind_action)
                        #  Update Q value
                        mod_env[0].updateQ(wind_state, index_wind_action, new_wind_state, reward)

            #  Reset environment(s)
            self.env.reset()

            if self.wind_list:
                for mod_env in self.wind_list:
                    mod_env[1].reset()

        print("End of training")
        return

    #  Train the model
    #  Input: number of training episode (int)
    #  Output: None
    def correlatedTrain(self, nE):
        start_time = time.time()
        #  Set list of exploratory rates
        expRate_schedule = self.expRate_schedule(nE)

        # ------------------------------- AGENT TRAINING -----------------------------------
        print("Starting Agent's training")
        # Training loop
        for episode in range(nE):

            #  Flag to stop the episode
            done = False
            while not done:
                #  Useful to keep for updating Q
                current_state = self.env.state
                #  Choose an action
                self.exp_rate = expRate_schedule[episode]
                action = self.chooseAction(current_state)
                #  Execute it
                new_state, reward, done, _ = self.env.step(action)
                #  Update Q value
                self.updateQ(current_state, action, new_state, reward)
            #  Reset environment
            self.env.reset()

        print("End of Agent's training !")
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))

        self.exp_rate = 0.0  # Agent's training is now done, there is no need to explore, only use the Q table

        # ------------------------------- HOSTILE AGENT TRAINING -----------------------------------
        print("Starting Hostile Agent's training")
        start_time = time.time()
        hostile_model, hostile_env = self.wind_list[1][0], self.wind_list[1][1]
        # Training loop
        for episode in range(nE):

            #  Flag to stop the episode
            done = False
            i = 0
            while not done:
                #  First step: need an initial state based on Agent's Q values
                if not i:
                    agent_action = self.chooseAction(self.env.state)
                    #  Upgrade the current state of the wind
                    wind_state = agent_action
                    hostile_env.update(agent_action, wind_state)
                else:
                    wind_state = hostile_env.state

                #  Choose an action
                hostile_model.exp_rate = expRate_schedule[episode]
                wind_action, index_wind_action = hostile_model.chooseAction(wind_state)
                new_wind_state = self.chooseNewState(wind_state, wind_action)
                #  Execute it
                new_wind_state, reward, done, _ = hostile_env.step(wind_action, index_action=index_wind_action, new_state=new_wind_state)
                #  Update Q value
                hostile_model.updateQ(wind_state, index_wind_action, new_wind_state, reward)
                i+=1

            #  Reset environment
            hostile_env.reset()

        print("End of Hostile Agent's training !")
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))

        # ------------------------------- FAVORABLE AGENT TRAINING -----------------------------------
        print("Starting Favorable Agent's training")
        start_time = time.time()
        favorable_model, favorable_env = self.wind_list[0][0], self.wind_list[0][1]
        # Training loop
        for episode in range(nE):

            #  Flag to stop the episode
            done = False
            i = 0
            while not done:
                #  First step: need an initial state based on Agent's Q values
                if not i:
                    agent_action = self.chooseAction(self.env.state)
                    #  Upgrade the current state of the wind
                    wind_state = agent_action
                    favorable_env.update(agent_action, wind_state)  # action is not really necessary
                else:
                    wind_state = favorable_env.state

                #  Choose an action
                favorable_model.exp_rate = expRate_schedule[episode]
                wind_action, index_wind_action = favorable_model.chooseAction(wind_state)
                new_wind_state = self.chooseNewState(wind_state, wind_action)
                #  Execute it
                new_wind_state, reward, done, _ = favorable_env.step(wind_action, index_action=index_wind_action, new_state=new_wind_state)
                #  Update Q value
                favorable_model.updateQ(wind_state, index_wind_action, new_wind_state, reward)
                i += 1

            #  Reset environment
            favorable_env.reset()

        print("End of Favorable Agent's training !")
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))

        return

    #  Choose an action to perform from a state
    #  Input: current state (int)
    #  Output: action (int)
    def chooseAction(self, state):
        #  Exploratory move
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.randint(0, self.actions)
            return action
        #  Greedy move
        else:
            action = np.argmax(self.Q[state])
            return action

    #  Choose the new state for a wind agent
    #  Input: current state (int) and an action (int)
    #  Output: new state (int)
    def chooseNewState(self, state, action):
        #  Perform action
        env = self.env
        new_row, new_col = env.inc((state // self.actions) // env.nCol, (state // self.actions) % env.nCol, action)
        new_state = env.to_s(new_row, new_col)
        #  Set list of reachable states
        reachable_state_list = [new_state * self.actions + i for i in range(self.actions)]
        #  Select one state using Agent's Q-table
        index_new_state = np.argmax(self.Q[new_state])
        return reachable_state_list[index_new_state]

    #  Update a value in the Q table
    #  Input: state (int), action (int), reached state (int), reward obtained (float)
    #  Output:  None
    def updateQ(self, state, action, new_state, reward):

        # Updating rule
        self.Q[state, action] = self.Q[state, action] + self.lr * \
                                (reward + self.decay_gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])

        return

    #  Save the current Q table in a JSON file
    #  Input: directory path (String)
    #  Output: None
    def save(self, path):
        q_function_list = self.Q.tolist()
        with open(path + os.sep + 'Q_' + self.name, 'w') as fp:
            json.dump(q_function_list, fp)

        return

    #  Load a Q table from a JSON file
    #  Input: directory path (string)
    #  Output: None
    def load(self, path):
        with open(path + os.sep + 'Q_' + self.name, 'r') as fp:
            q_list = json.load(fp)
            self.Q = np.array(q_list)
            print("Q function loaded")

        return

    #  Build a list of values of exploratory rate which decrease over episodes
    #  Input: number of episodes (int), minimum exploratory rate (float)
    #  Output: list of exploratory rate (np.array)
    def expRate_schedule(self, nE, exp_rate_min=0.05):
        x = np.arange(nE) + 1
        exp_rate_decay = exp_rate_min**(1 / nE)
        y = [max((exp_rate_decay**x[i]), exp_rate_min) for i in range(len(x))]
        return y

    #  Predict an action from a given state
    #  Input: state (int)
    #  Output: state (int)
    def predict(self, observation):
        return np.argmax(self.Q[observation]), None

    #  Return the max Q value from a given state
    #  Input: state (int)
    #  Output: Q-value (float)
    def getValue(self, observation):
        return np.max(self.Q[observation])

    #  Set for a specific state a value in Q-table. It's only used for SXp (deal with sparse reward)
    #  Input: state (int) and a value (float)
    #  Output: Q-value (float)
    def setValue(self, observation, value):
        self.Q[observation][:] = value
        return

class WindAgent:

    def __init__(self, name, env, lr=.2, exp_rate=1.0, decay_gamma=0.95):
        #  Version
        self.name = name
        #  Parameters for updating Q
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        #  Environment
        self.env = env
        #  Actions
        self.actions = 4
        #  States
        self.states = self.env.nRow * self.env.nCol
        #  Q table
        self.Q = np.zeros((self.states * self.actions, self.actions - 1))

    #  Choose an action to perform from a state
    #  Input: current state (int)
    #  Output: action (int) and its index in list of possible actions (int)
    def chooseAction(self, state):
        #  Set list of possible actions due to agent's move
        previous_action = state % 4
        possible_action = [(previous_action - 1) % 4, previous_action, (previous_action + 1) % 4]
        #  Exploratory move
        if np.random.uniform(0, 1) <= self.exp_rate:
            index_action = np.random.randint(0, self.actions-1)
            return possible_action[index_action], index_action
        #  Greedy move
        else:
            index_action = np.argmax(self.Q[state])
            return possible_action[index_action], index_action

    #  Update a value in the Q table
    #  Input: state (int), action (int), reached state (int), reward obtained (float)
    #  Output:  None
    def updateQ(self, state, action, new_state, reward):

        #  Updating rule
        self.Q[state, action] = self.Q[state, action] + self.lr * \
                                (reward + self.decay_gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])

        return

    #  Save the current Q table in a JSON file
    #  Input: directory path (String)
    #  Output: None
    def save(self, path):
        #  Distinct files with the behaviour of the wind
        if self.env.behaviour == "Favorable":
            type_env = "F"
        else:
            type_env = "H"
        q_function_list = self.Q.tolist()
        with open(path + os.sep + 'Q_Wind' + type_env + "_" + self.name, 'w') as fp:
            json.dump(q_function_list, fp)

        return

    #  Load a Q table from a JSON file
    #  Input: directory path (String)
    #  Output: None
    def load(self, path):
        #  Distinct files with the behaviour of the wind
        if self.env.behaviour == "Favorable":
            type_env = "F"
        else:
            type_env = "H"
        with open(path + os.sep + 'Q_Wind' + type_env + "_" + self.name, 'r') as fp:
            q_list = json.load(fp)
            self.Q = np.array(q_list)
            print("Wind' Q function loaded")

        return

    #  Predict an action from a given state
    #  Input: state (int)
    #  Output: action (int)
    def predict(self, observation):
        #  Set list of possible actions due to agent's move
        previous_action = observation % self.actions
        possible_action = [(previous_action - 1) % self.actions, previous_action, (previous_action + 1) % self.actions]
        index_action = np.argmax(self.Q[observation])
        return possible_action[index_action], None