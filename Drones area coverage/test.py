from DQN import DQN
from agent import Agent, WindAgent
from env import DroneAreaCoverage
from SXp import SXp, SXpMetric
import torch
import os
import argparse
import numpy as np
import sys
from copy import deepcopy

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    #  Paths
    parser.add_argument('-model', '--model_dir', default="Models"+os.sep+"Agent"+os.sep+"tl1600000e750000s50000th22ddqnTrue-best_11.69.dat", help="Agent's model", type=str, required=False)
    parser.add_argument('-model_h', '--model_hostile_dir', default="Models"+os.sep+"Hostile"+os.sep+"tl1600000e750000s50000th22ddqnTrue_H-best_-5.68.dat", help="Hostile agent's model", type=str, required=False)
    parser.add_argument('-model_f', '--model_favourable_dir', default="Models"+os.sep+"Favourable"+os.sep+"tl1600000e750000s50000th22ddqnTrue_F-best_11.86.dat", help="Favourable agent's model", type=str, required=False)

    parser.add_argument('-map', '--map_name', default="10x10", help="Map's name", type=str, required=False)
    parser.add_argument('-agents', '--number_agents', default=4, help="Number of agents in the map", type=int, required=False)
    parser.add_argument('-horizon', '--time_horizon', default=20, help="Time horizon of an episode", type=int, required=False)
    parser.add_argument('-rand', '--random_starting_position', action="store_true", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.add_argument('-no_rand', '--no_random_starting_position', action="store_false", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.set_defaults(random_starting_position=True)
    parser.add_argument('-move', '--step_move', default="stop", help="Type of transition with wind", type=str, required=False)
    parser.add_argument('-view', '--view_range', default=5, help="View range of a drone", type=int, required=False)
    parser.add_argument('-w', '--wind', action="store_false", dest='windless', help="Wind's presence in the environment", required=False)
    parser.add_argument('-no_w', '--no_wind', action="store_true", dest='windless', help="Wind's presence in the environment", required=False)
    parser.set_defaults(windless=False)
    parser.add_argument('-mm', '--minmax_reward', action="store_true", dest='mm_reward', help="Compute FE/HE-scores with max/min reward instead of last-step reward", required=False)
    parser.add_argument('-no_mm', '--no_minmax_reward', action="store_false", dest='mm_reward', help="Compute FE/HE-scores with max/min reward instead of last-step reward", required=False)
    parser.set_defaults(mm_reward=False)
    parser.add_argument('-spec', '--specific', action="store_true", dest="spec", help="Start from a specific configuration", required=False)
    parser.add_argument('-no_spec', '--no_specific', action="store_false", dest="spec", help="Do not start from a specific configuration", required=False)
    parser.set_defaults(spec=False)
    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores in case of starting from a specific state", type=str, required=False)
    parser.add_argument('-scenarios', '--nb_scenarios', default=1000, help="Number of randomly-generated scenarios given a policy, for a SXp score", type=int, required=False)
    parser.add_argument('-spec_conf', '--specific_starting_configuration', default="[[1, 7], [4, 2], [3, 6], [6, 6]]", help="Specific state", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=6, help="Length of SXps", type=int, required=False)
    args = parser.parse_args()

    # Get arguments
    PATHFILE_MODEL = args.model_dir
    PATHFILE_H_MODEL = args.model_hostile_dir
    PATHFILE_F_MODEL = args.model_favourable_dir
    MAP_NAME = args.map_name
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUMBER_AGENTS = args.number_agents
    VIEW_RANGE = args.view_range
    WINDAGENT = not args.windless
    RANDOM_STARTING_POSITION = args.random_starting_position
    MOVE = args.step_move
    LIMIT = args.time_horizon
    MM_REWARD = args.mm_reward
    SPECIAL_CASE = args.spec
    K = args.length_k
    # only used if SPECIFIC_STATE
    NUMBER_SCENARIOS = args.nb_scenarios
    CSV_FILENAME = args.csv_filename
    RENDER = args.render

    #  Fill configuration list (convert string into int list)
    temp_configuration = args.specific_starting_configuration
    SPECIFIC_CONFIGURATION = []
    sub_l = []
    for elm in list(temp_configuration):
        if elm not in ['[', ',', ' ', ']']:
            sub_l.append(int(elm))
            if len(sub_l) == 2:
                SPECIFIC_CONFIGURATION.append(sub_l)
                sub_l = []
    #  Initialization --------------------------------------------------------------------------------------------------

    #  Environment
    if WINDAGENT:
        env = DroneAreaCoverage(map_name=MAP_NAME, windless=False, wind_agent_train=False)
    else:
        env = DroneAreaCoverage(map_name=MAP_NAME, windless=True)
    #  Agents
    agents = []
    wind_agents = []
    for i in range(NUMBER_AGENTS):
        if SPECIAL_CASE:
            agent = Agent(i+1, env, view_range=VIEW_RANGE, debug_position=SPECIFIC_CONFIGURATION[i])
        else:
            agent = Agent(i + 1, env, view_range=VIEW_RANGE)
        agents.append(agent)

    if WINDAGENT:
        h_agent = WindAgent(NUMBER_AGENTS * 1 + NUMBER_AGENTS + 1, env, "hostile", view_range=5)
        f_agent = WindAgent(NUMBER_AGENTS * 2 + NUMBER_AGENTS + 1, env, "favorable", view_range=5)
        wind_agents.append(h_agent)
        wind_agents.append(f_agent)

    #  Compute SXp's from a specific configuration
    if SPECIAL_CASE:
        env.initObs(agents, RANDOM_STARTING_POSITION, debug_position=True)
        env.set_lastactions([0, 0, 0, 0]) # useless information
    else:
        env.initObs(agents, RANDOM_STARTING_POSITION)

    #  Load net(s) -----------------------------------------------------------------------------------------------------
    net = DQN(np.array(agent.observation[0]).shape, np.array(agent.observation[1]).shape, agent.actions).to(DEVICE)
    net.load_state_dict(torch.load(PATHFILE_MODEL, map_location=DEVICE))

    if WINDAGENT:
        h_net = DQN(np.array(wind_agents[0].observation[0]).shape, np.array(wind_agents[0].observation[1]).shape, wind_agents[0].actions).to(DEVICE)
        h_net.load_state_dict(torch.load(PATHFILE_H_MODEL, map_location=DEVICE))

        f_net = DQN(np.array(wind_agents[1].observation[0]).shape, np.array(wind_agents[1].observation[1]).shape, wind_agents[1].actions).to(DEVICE)
        f_net.load_state_dict(torch.load(PATHFILE_F_MODEL, map_location=DEVICE))

    #  Test ------------------------------------------------------------------------------------------------------------

    if not SPECIAL_CASE:
        env.reset(agents, rand=RANDOM_STARTING_POSITION)
    env.render(agents)

    #  Get max/min reward for normalizing P-score
    max_reward = env.max_reward(agents, reward_type="B")
    extremum_reward = [-max_reward, max_reward]

    if SPECIAL_CASE:
        dones = [False, False, False, False]
        actions = [0, 0, 0, 0]
        rewards = env.getReward(agents, actions, dones, reward_type="B")
        #  Display infos
        print("Dones True : {}".format(dones.count(True)))
        print("Rewards : {}".format(rewards))
        print("Cumulative reward : {}".format(sum(rewards)))
        print('-------')

        # SXp
        if WINDAGENT:
            CSV_FILENAME = "Metrics" + os.sep + "New tests" + os.sep + CSV_FILENAME
            SXpMetric(env, agents, wind_agents, K, net, [h_net, f_net], DEVICE, move=MOVE, number_scenarios=NUMBER_SCENARIOS,
                           extremum_reward=extremum_reward, csv_filename=CSV_FILENAME, render=RENDER, concise=MM_REWARD)
        else:
            SXp(env, agents, None, K, net, None, DEVICE, move=MOVE)
    else:
        cpt = 0
        while cpt <= LIMIT:

            # SXp
            if WINDAGENT:
                SXp(env, agents, wind_agents, K, net, [h_net, f_net], DEVICE, number_scenarios=NUMBER_SCENARIOS, move=MOVE, extremum_reward=extremum_reward, render=True)
            else:
                SXp(env, agents, None, K, net, None, DEVICE, move=MOVE)
            #  Choose action
            actions = []
            for agent in agents:
                action = agent.chooseAction(net, epsilon=0, device=DEVICE)
                actions.append(action)
            #  Step
            _, _, _, dones, _ = env.step(agents, actions, move=MOVE)
            #  Render
            env.render(agents)
            #  Extract rewards
            rewards = env.getReward(agents, actions, dones, reward_type="B")
            #  Check end of episode
            if dones.count(True) == len(dones):
                break

            #  Display infos
            print("Dones True : {}".format(dones.count(True)))
            print("Rewards : {}".format(rewards))
            print("Cumulative reward : {}".format(sum(rewards)))
            print('-------')

            cpt += 1