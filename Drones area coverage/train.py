
import random
import time
import numpy as np
import argparse
import collections
import os
from matplotlib import pyplot as plt

from DQN import DQN, ExperienceBuffer, calc_loss
from agent import Agent, WindAgent
from env import DroneAreaCoverage

import torch.optim as optim
import torch

#  Experience tuple used for DQN's training
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

#  Compute a training of an agent or a wind agent
#
#  Input: filepaths of logs, models and logfile (String), environment (DroneAreaCoverage), agents (Agent list),
#  NN and target NN (DQN), buffer (ExperienceBuffer), learning rate (float), epsilon values limits and it's decay rate
#  (float), type of device (String), maximum length of an episode (int), timestep where the NN training begins (int),
#  frequency of target NN update (int), batch size (int), discount factor (loss calculation) (float),
#  time limit of training (int), use of random starting position at the beginning of an episode (boolean),
#  use of double-Q learning extension (boolean), type of transition function (String) and agent (String)
#
#  Output: averages of losses, rewards (float list list), ended time episode (int list list), the best average of
#  cumulative reward obtained during the training (float) and the file name of the best NN produced (String)
def agentTrain(log_dir, model_dir, filename, env, agents, net, tgt_net, buffer, learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move, behaviour=None):
    # Initialization
    timestep = 0  # Global timestep
    epstep = 0  # Episode step
    nb_episode = 0  # Number of episodes
    start_time = time.time()
    file = open(log_dir + filename + ".txt", "a")
    filename_last_net = None

    max_global_reward = env.max_reward(agents, reward_type="B")
    print("Global max reward : {}".format(max_global_reward))
    #  Best mean reward
    best_m_reward = None

    losses = []  # Store loss values
    total_rewards = []  # Store all rewards obtained at the end of an episode
    ended_time_horizon = []  # An episode end with time horizon limit (bool)
    mean_losses = [[], []]  #  Store averages of 100 losses and timestep where they were calculated
    mean_rewards = [[], []] #  Store averages of 100 cumulative reward and timestep where they were calculated
    mean_time_horizon = [[], []] #  Store averages of 100 ended time episode and timestep where they were calculated

    #  Initialize optimizer and epsilon rate
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    epsilon_rate = epsilon_final ** (1 / epsilon_decay)

    while True:
        timestep += 1
        epstep += 1
        epsilon = max(epsilon_final, epsilon_rate ** (timestep+1))  # Trade-off explore/exploit
        actions = []

        #  Display infos -----------------------------------------------------------------------------------------------
        if timestep % 20000 == 0:
            print("Timestep : {}  -- Current epsilon : {}".format(timestep, epsilon))

        #  Choose action -----------------------------------------------------------------------------------------------
        for agent in agents:
            actions.append(agent.chooseAction(net, epsilon, device=device))

        #  Step and update observation ---------------------------------------------------------------------------------
        if behaviour is None:
            states, _, new_states, dones, _, _ = env.step(agents, actions, move=move)
        else:
            states, _, new_states, dones, _ = env.windStep(agents, actions, move=move)

        #  Extract reward based on position and neighborhood (local reward) --------------------------------------------
        if behaviour is None:
            rewards = env.getReward(agents, actions, dones, reward_type="B")
        else:
            rewards = env.getWindReward(agents, dones, reward_type="B")

        #  Save experiences --------------------------------------------------------------------------------------------
        for i in range(len(actions)):
            exp = Experience(states[i], actions[i], rewards[i], dones[i], new_states[i])
            buffer.append(exp)

        #  Store the last cumulative reward of an episode and reset env ------------------------------------------------
        if epstep // time_horizon == 1 or dones.count(True) >= 1:
            break_bool, best_m_reward, file, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon, filename_last_net = save(model_dir,
                sum(rewards), best_m_reward, max_global_reward, env, agents, net, file, total_rewards,
                ended_time_horizon, epstep, nb_episode, timestep, start_time, time_limit, random_starting_position, filename, filename_last_net)
            #  Store infos
            if m_reward is not None:
                mean_rewards[0].append(m_reward)
                mean_rewards[1].append(timestep)
                mean_time_horizon[0].append(m_endTimeHorizon)
                mean_time_horizon[1].append(timestep)
            if break_bool:
                break

        #  Skip updates if there is not enough experiences
        if len(buffer) < replay_start:
            continue

        #  Update target NN(s) weights ---------------------------------------------------------------------------------
        if timestep % sync_target == 0:
            tgt_net.load_state_dict(net.state_dict())

        #  Update NN(s) ------------------------------------------------------------------------------------------------
        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss, _ = calc_loss(batch, net, tgt_net, gamma, device=device, double=ddqn)
        losses.append(loss.item())
        #  Store infos
        if len(losses) >= 100:
            mean_losses[0].append(np.mean(losses[-100:]))
            mean_losses[1].append(timestep)
            losses = []
        loss.backward()
        optimizer.step()

    file.close()
    return mean_losses, mean_rewards, mean_time_horizon, best_m_reward, filename_last_net

#  Compute a training of wind agent based on an already learnt agent's DQN
#
#  Input: filepaths of logs, models and logfile (String), environment (DroneAreaCoverage), agents (Agent list),
#  NN and target NN (DQN), buffer (BufferExperience), learning rate (float), epsilon values limits and it's decay rate (float), type of device
#  (String), maximum length of an episode (int), timestep where the NN training begins (int),
#  frequency of target NN update (int), batch size (int), discount factor (loss calculation) (float),
#  time limit of training (int), use of random starting position at the beginning of an episode (boolean),
#  use of double-Q learning extension (boolean), type of transition function (String)
#
#  Output: averages of losses, rewards (float list list), ended time episode (int list list)
#  and the best average of cumulative reward obtained during the training (float)
def correlTrain(log_dir, model_dir, filename, env, agents, wind_agents, net, wind_net, wind_tgt_net, buffer, learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move):

    # Initialization ---------------------------------------------------------------------------------------------------
    timestep = 0  # Global timestep
    epstep = 0  # Episode step
    nb_episode = 0  # Number of episodes
    start_time = time.time()
    file = open(log_dir + filename + ".txt", "a")

    max_global_reward = env.max_reward(agents, reward_type="B")
    print("Global max reward : {}".format(max_global_reward))
    #  Best mean reward
    best_m_reward = None

    losses = []  # Store loss values
    total_rewards = []  # Store all rewards obtained at the end of an episode
    ended_time_horizon = []  # An episode end with time horizon limit (bool)
    mean_losses = [[], []]  # Store averages of 100 losses and timestep where they were calculated
    mean_rewards = [[], []]  # Store averages of 100 cumulative reward and timestep where they were calculated
    mean_time_horizon = [[], []]  # Store averages of 100 ended time episode and timestep where they were calculated

    #  Initialize optimizer and epsilon rate
    optimizer = optim.Adam(wind_net.parameters(), lr=learning_rate)
    epsilon_rate = epsilon_final ** (1 / epsilon_decay)

    while True:
        timestep += 1
        epstep += 1
        epsilon = max(epsilon_final, epsilon_rate ** (timestep+1))  # Trade-off explore/exploit
        actions = []
        #  Display infos -----------------------------------------------------------------------------------------------
        if timestep % 20000 == 0:
            print("Timestep : {}  -- Current epsilon : {}".format(timestep, epsilon))

        #  Choose action -----------------------------------------------------------------------------------------------
        for agent in agents:
            actions.append(agent.chooseAction(net, epsilon=0.0, device=device))

        #  Step and update observation ---------------------------------------------------------------------------------
        states, temp_states, new_states, dones, _, _ = env.step(agents, actions, move=move)

        #  Create experiences for wind agents --------------------------------------------------------------------------
        wind_actions = []

        for i in range(len(temp_states)):
            #  Update observation
            new_obs = [temp_states[i][0], [temp_states[i][1][0], temp_states[i][1][1], actions[i]]]
            wind_agents[i].set_obs(new_obs)
            #  Choose action
            wind_action = wind_agents[i].chooseAction(wind_net, epsilon, device=device)
            wind_actions.append(wind_action)

        #  Step
        wind_states, _, new_wind_states, wind_dones, _ = env.windStep(wind_agents, wind_actions, move=move, agents=agents, net=net, device=device)

        #  Extract reward based on position and neighborhood (local reward)
        wind_rewards = env.getWindReward(wind_agents, wind_dones, reward_type="B")

        #  Save experiences --------------------------------------------------------------------------------------------
        for i in range(len(wind_actions)):
            exp = Experience(wind_states[i], wind_actions[i], wind_rewards[i], wind_dones[i], new_wind_states[i])
            buffer.append(exp)

        #  Store the last cumulative reward of an episode --------------------------------------------------------------
        if epstep // time_horizon == 1 or wind_dones.count(True) >= 1:
            break_bool, best_m_reward, file, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon, _ = save(model_dir, sum(wind_rewards), best_m_reward, max_global_reward, env, agents, wind_net, file, total_rewards, ended_time_horizon, epstep, nb_episode, timestep, start_time, time_limit, random_starting_position, filename, None)
            #  Store infos
            if m_reward is not None:
                mean_rewards[0].append(m_reward)
                mean_rewards[1].append(timestep)
                mean_time_horizon[0].append(m_endTimeHorizon)
                mean_time_horizon[1].append(timestep)

            #  Reset wind agent
            for wa in wind_agents:
                wa.dead = False
            if break_bool:
                break
        #  Update agents obs
        else:
            for i in range(len(agents)):
                obs = [new_wind_states[i][0], new_wind_states[i][1][:2]]
                agents[i].set_obs(obs)
                agents[i].dead = False

        #  Skip updates if there is not enough experiences
        if len(buffer) < replay_start:
            continue

        #  Update target NN(s) weights ---------------------------------------------------------------------------------
        if timestep % sync_target == 0:
            wind_tgt_net.load_state_dict(wind_net.state_dict())

        #  Update NN(s) ------------------------------------------------------------------------------------------------
        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss, _ = calc_loss(batch, wind_net, wind_tgt_net, gamma, device=device, double=ddqn)

        #  Store infos
        losses.append(loss.item())
        if len(losses) >= 100:
            mean_losses[0].append(np.mean(losses[-100:]))
            mean_losses[1].append(timestep)
            losses = []
        loss.backward()
        optimizer.step()

    file.close()
    return mean_losses, mean_rewards, mean_time_horizon, best_m_reward


#  Save data into lists and a file, reset agents and store the current NN when a better cumulative reward is
#  reached. save() can stop the training process. It's called at the end of an episode.
#
#  Input: directory path to store NN (String), current cumulative reward, the best one reached and the maximum one
#  (float), environment (DroneAreaCoverage), agents (Agent list), current NN (DQN), file to write data (String),
#  list of cumulative reward and ended time horizon (float list), ended timestep of the episode (int), number of
#  finished episode (int), current timestep of the training (int), real starting time (float), timestep limit (int),
#  use of random starting position at the beginning of an episode (boolean), filename to use (String) and
#  filename used for NN file (String)
#
#  Output: end of training process (boolean), the best current cumulative reward (float), file to write infos, list of
#  cumulative reward and ended time horizon (float list), ended timestep of the episode (int), number of finished
#  episode (int), the last average of the cumulative reward and ende time horizon (float),
#  and the filename of the last NN saved (DQN)
def save(dir, cum_reward, best_m_reward, max_global_reward, env, agents, net, file, total_rewards, ended_time_horizon, epstep, nb_episode, timestep, start_time, time_limit, random_starting_position, filename, filename_last_net):

    #  Store infos about completeness of an episode
    ended_time_horizon.append(epstep)
    total_rewards.append(cum_reward)

    nb_episode += 1
    #  Reset agents
    if len(agents[0].get_obs()[1]) == 2:
        env.reset(agents, rand=random_starting_position)
    else:
        env.reset(agents, rand=random_starting_position, behaviour="wind")
    epstep = 0

    if len(total_rewards) >= 100:
        #  Write mean reward in file
        m_reward = np.mean(total_rewards[-100:])
        m_endTimeHorizon = np.mean(ended_time_horizon[-100:])
        total_rewards = []
        ended_time_horizon = []

        file.write(
            "Global timestep {} ---- Episodes achieved : {} ---- Mean of reward of the last 100 ep : {} ---- Mean end with time horizon {} \n".format(
                timestep, nb_episode, m_reward, m_endTimeHorizon))

        #  Alternative end----------------------------------------------------------------------------------------------
        if timestep >= time_limit:
            #  Save and display time information
            final_time_s = time.time() - start_time
            time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                             final_time_s % 60
            file.write(
                "Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
                    time_hour, time_minute, time_s))
            print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
                time_hour, time_minute, time_s))
            print("alternative end")
            return True, best_m_reward, file, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon, filename_last_net

        #  Save NN with best mean reward--------------------------------------------------------------------------------
        if best_m_reward is None or best_m_reward < m_reward:
            #  Save in a dat file the current NN
            torch.save(net.state_dict(), dir + os.sep + filename + "-best_%.2f.dat" % m_reward)
            filename_last_net = dir + os.sep + filename + "-best_%.2f.dat" % m_reward
            #  Save and display the new best reward
            if best_m_reward is not None:
                print(dir)
                print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
                file.write("New Best reward ---------------------------------------------- {} \n".format(m_reward))
            best_m_reward = m_reward

            #  End------------------------------------------------------------------------------------------------------
            if m_reward >= max_global_reward - 0.1:
                print("Problem solved ! \n Mean reward in the last 500 episode {} \n Total steps {}, "
                      "episodes {} \n".format(m_reward, timestep, nb_episode))
                #  Save and display time information
                final_time_s = time.time() - start_time
                time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                                 final_time_s % 60
                file.write(
                    "Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
                        time_hour, time_minute, time_s))
                print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
                    time_hour, time_minute, time_s))

                return True, best_m_reward, file, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon, filename_last_net

        return False, best_m_reward, file, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon, filename_last_net

    return False, best_m_reward, file, total_rewards, ended_time_horizon, epstep, nb_episode, None, None, filename_last_net


#  Write data into file
#  Input: file name (String), data (float list list)
#  Output: None
def storeData(filename, data):
    file = open(filename + ".txt", "a")
    file.write(str(data) + "\n")
    file.close()
    return


#  Store data in files. It's used at the end of a training to save the evolution of loss value, cumulative reward and
#  ended time horizon
#  Input: file names (String list), data to store in files (float list list), data labels (String list)
#  and directory path (Sting)
#  Output: None
def saveInfos(filenames, datas, ylabels, dir):
    for i in range(len(filenames)):
        print("First {} : {} Last : {}".format(ylabels[i], datas[i][0][0], datas[i][0][-1]))
        storeData(dir + os.sep + filenames[i], datas[i])
    return


if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    #  Directory Paths
    parser.add_argument('-model', '--model_dir', default="", help="Agent's model directory", type=str, required=True)
    parser.add_argument('-log', '--log_dir', default="", help="Agent's log directory", type=str, required=True)

    #  Hyper-parameters
    parser.add_argument('-limit', '--timestep_limit', default=100000, help="Limits for training", type=int, required=True)

    parser.add_argument('-map', '--map_name', default="10x10", help="Map's name", type=str, required=False)
    parser.add_argument('-agents', '--number_agents', default=4, help="Number of agents in the map", type=int, required=False)
    parser.add_argument('-horizon', '--time_horizon', default=20, help="Time horizon of an episode", type=int, required=False)
    parser.add_argument('-rand', '--random_starting_position', action="store_true", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.add_argument('-no_rand', '--no_random_starting_position', action="store_false", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.set_defaults(random_starting_position=True)

    parser.add_argument('-batch', '--batch_size', default=32, help="Batch size", type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, help="Learning rate", type=float, required=False)
    parser.add_argument('-df', '--discount_factor', default=.99, help="Discount factor", type=float, required=False)
    parser.add_argument('-eps_dec', '--epsilon_decay', default=50000, help="Number of steps where espilon decrease", type=int, required=False)
    parser.add_argument('-eps_f', '--epsilon_final', default=0.1, help="Epsilon' final value", type=float, required=False)

    parser.add_argument('-move', '--step_move', default="stop", help="Type of transition with wind", type=str, required=False)

    parser.add_argument('-sync', '--sync_target', default=1000, help="Synchronize target net at each n steps", type=int, required=False)
    parser.add_argument('-replay', '--replay_size', default=20000, help="Size of replay memory", type=int, required=False)
    parser.add_argument('-replay_s', '--replay_starting_size', default=10000, help="From which number of experiences NN training process start", type=int, required=False)

    parser.add_argument('-w', '--wind', action="store_false", dest='windless', help="Wind's presence in the environment", required=False)
    parser.add_argument('-no_w', '--no_wind', action="store_true", dest='windless', help="Wind's presence in the environment", required=False)
    parser.set_defaults(windless=False)
    parser.add_argument('-cor', '--correlated', action="store_true", dest='correlated',
                        help="Use already trained Agent policy to train Wind Agent", required=False)
    parser.add_argument('-no_cor', '--no_correlated', action="store_false", dest='correlated',
                        help="Use already trained Agent policy to train Wind Agent", required=False)
    parser.set_defaults(correlated=True)
    parser.add_argument('-ddqn', '--double_dqn', action="store_true", dest='ddqn',
                        help="Use Double DQN upgrade", required=False)
    parser.add_argument('-no_ddqn', '--no_double_dqn', action="store_false", dest='ddqn',
                        help="Use Double DQN upgrade", required=False)
    parser.set_defaults(ddqn=False)
    parser.add_argument('-view', '--view_range', default=5, help="View range of a drone", type=int, required=False)

    args = parser.parse_args()

    # Get arguments
    log_dir = args.log_dir
    model_dir = args.model_dir

    time_limit = args.timestep_limit

    map = args.map_name
    number_agents = args.number_agents
    time_horizon = args.time_horizon
    random_starting_position = args.random_starting_position

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    gamma = args.discount_factor
    epsilon_decay = args.epsilon_decay
    epsilon_final = args.epsilon_final

    sync_target = args.sync_target
    replay_size = args.replay_size
    replay_start = args.replay_starting_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    windless = args.windless
    correlated = args.correlated
    ddqn = args.ddqn

    move = args.step_move
    view_range = args.view_range

    AGENT = "Agent"
    HOSTILE = "Hostile"
    FAVORABLE = "Favorable"
    agent_dir_model = model_dir + os.sep + AGENT
    hostile_dir_model = model_dir + os.sep + HOSTILE
    favorable_dir_model = model_dir + os.sep + FAVORABLE
    agent_dir_log = log_dir + os.sep + AGENT
    hostile_dir_log = log_dir + os.sep + HOSTILE
    favorable_dir_log = log_dir + os.sep + FAVORABLE

    filename = "tl"+ str(time_limit) + "e" + str(epsilon_decay) + "s" + str(sync_target) + "th" + str(time_horizon) + "ddqn" + str(ddqn)

    #  Create Dirs to store logs and models
    if not os.path.exists(log_dir):  # log dir
        os.mkdir(log_dir)
    if not os.path.exists(model_dir):  # model dir
        os.mkdir(model_dir)
    if not os.path.exists(agent_dir_model):  # agent subdir
        os.mkdir(agent_dir_model)
    if not os.path.exists(agent_dir_log):  # agent subdir
        os.mkdir(agent_dir_log)
    if not windless:
        if not os.path.exists(hostile_dir_model):  # hostile subdir
            os.mkdir(hostile_dir_model)
        if not os.path.exists(favorable_dir_model):  # favorable subdir
            os.mkdir(favorable_dir_model)
        if not os.path.exists(hostile_dir_log):  # hostile subdir
            os.mkdir(hostile_dir_log)
        if not os.path.exists(favorable_dir_log):  # favorable subdir
            os.mkdir(favorable_dir_log)

    #  Initialization --------------------------------------------------------------------------------------------------

    #  Environment
    if windless:
        wind_agent_train = False
    else:
        wind_agent_train = True  # False

    env = DroneAreaCoverage(map_name=map, windless=windless, wind_agent_train=wind_agent_train)
    #  Agents
    agents = []
    wind_agents = [[], []]
    for i in range(number_agents):
        agent = Agent(i+1, env, view_range=view_range, random=random_starting_position)
        agents.append(agent)

        if not windless:
            h_agent = WindAgent(number_agents * 1 + i + 1, env, "hostile", view_range=view_range)
            f_agent = WindAgent(number_agents * 2 + i + 1, env, "favorable", view_range=view_range)
            wind_agents[0].append(h_agent)
            wind_agents[1].append(f_agent)

    env.initObs(agents, random_starting_position)

    #  Main policy
    net = DQN(np.array(agents[0].observation[0]).shape, np.array(agents[0].observation[1]).shape, agents[0].actions).to(device)
    tgt_net = DQN(np.array(agents[0].observation[0]).shape, np.array(agents[0].observation[1]).shape, agents[0].actions).to(device)

    if not windless:
        #  Hostile wind policy
        h_net = DQN(np.array(wind_agents[0][0].observation[0]).shape, np.array(wind_agents[0][0].observation[1]).shape, wind_agents[0][0].actions).to(device)
        h_tgt_net = DQN(np.array(wind_agents[0][0].observation[0]).shape, np.array(wind_agents[0][0].observation[1]).shape, wind_agents[0][0].actions).to(device)

        #  Favorable wind policy
        f_net = DQN(np.array(wind_agents[1][0].observation[0]).shape, np.array(wind_agents[1][0].observation[1]).shape, wind_agents[1][0].actions).to(device)
        f_tgt_net = DQN(np.array(wind_agents[1][0].observation[0]).shape, np.array(wind_agents[1][0].observation[1]).shape, wind_agents[1][0].actions).to(device)

        wind_nets = [[h_net, h_tgt_net], [f_net, f_tgt_net]]
        #  Initialize ExperienceBuffers
        buffers = [ExperienceBuffer(replay_size), ExperienceBuffer(replay_size), ExperienceBuffer(replay_size)]
    else:
        #  Initialize ExperienceBuffers
        buffers = [ExperienceBuffer(replay_size)]

    #  Train -----------------------------------------------------------------------------------------------------------

    #  Save infos of Agent's training
    filenames = ['losses', 'rewards', 'time_horizons']
    ylabels = ["Loss", "Mean reward", "Mean ended time horizon"]

    #  Train of Agents, Hostile Agent and Favourable Agents
    if not windless:
        #  Save infos of environment-agent's training
        hostile_filenames = ['h_losses', 'h_rewards', 'h_time_horizons']
        favorable_filenames = ['f_losses', 'f_rewards', 'f_time_horizons']

        #  Agents train ------------------------------------------------------------------------------------------------
        losses, rewards, timeH, best_reward, filename_last_net = agentTrain(agent_dir_log, agent_dir_model, filename, env, agents, net, tgt_net, buffers[0], learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move)
        datas = [losses, rewards, timeH]
        saveInfos(filenames, datas, ylabels, agent_dir_log)
        net.load_state_dict(torch.load(filename_last_net))

        #  Don't use the Agents learnt policy
        if not correlated:
            #  Hostile Agents train ------------------------
            env.init_map()
            env.initObs(wind_agents[0], random_starting_position, wind=True)
            losses, rewards, timeH, best_reward, _ = agentTrain(hostile_dir_log, hostile_dir_model, filename, env, wind_agents[0], h_net, h_tgt_net, buffers[1], learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move, behaviour="hostile")
            datas = [losses, rewards, timeH]
            saveInfos(hostile_filenames, datas, ylabels, agent_dir_log)
            #  Favorable Agents train ----------------------
            env.init_map()
            env.initObs(wind_agents[1], random_starting_position, wind=True)
            losses, rewards, timeH, best_reward, _ = agentTrain(favorable_dir_log, favorable_dir_model, filename, env, wind_agents[1], f_net, f_tgt_net, buffers[2], learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move, behaviour="favorable")
            datas = [losses, rewards, timeH]
            saveInfos(favorable_filenames, datas, ylabels, agent_dir_log)
        #  Use the Agents learnt policy
        else:
            env.windless = True
            #  Hostile Agents train ------------------------
            env.init_map()
            env.initPos(agents, random_starting_position)
            env.initObs(agents, random_starting_position)
            losses, rewards, timeH, best_reward = correlTrain(hostile_dir_log, hostile_dir_model, filename, env, agents, wind_agents[0], net, h_net, h_tgt_net, buffers[1], learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move)

            datas = [losses, rewards, timeH]
            saveInfos(hostile_filenames, datas, ylabels, hostile_dir_log)

            #  Favorable Agents train ----------------------
            env.init_map()
            env.initPos(agents, random_starting_position)
            env.initObs(agents, random_starting_position)
            losses, rewards, timeH, best_reward = correlTrain(favorable_dir_log, favorable_dir_model, filename, env, agents, wind_agents[1], net, f_net, f_tgt_net, buffers[2], learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move)
            datas = [losses, rewards, timeH]
            saveInfos(favorable_filenames, datas, ylabels, favorable_dir_log)
    #  Train of Agents
    else:
        losses, rewards, timeH, best_reward, _ = agentTrain(agent_dir_log, agent_dir_model, filename, env, agents, net, tgt_net, buffers[0], learning_rate, epsilon_final, epsilon_decay, device, time_horizon, replay_start, sync_target, batch_size, gamma, time_limit, random_starting_position, ddqn, move)
        datas = [losses, rewards, timeH]
        saveInfos(filenames, datas, ylabels, agent_dir_log)

