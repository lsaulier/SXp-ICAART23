import os
import sys
import numpy as np
import argparse
from env import MyFrozenLake
from agent import Agent, WindAgent
from SXp import SXp, SXpMetric

# States to test for 4x4 map : 0, 4, 8, 9, 10, 13, 14
# States to test for 8x8 map : 1, 7, 8, 11, 12, 14, 21, 22, 23, 25, 26, 31, 34, 36, 39, 44, 53, 55, 57, 62

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-scenarios', '--nb_scenarios', default=10000, help="Number of randomly-generated scenarios given a policy, for a SXp score", type=int, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-spec', '--specific', action="store_true", dest="spec", help="Start from a specific state", required=False)
    parser.add_argument('-no_spec', '--no_specific', action="store_false", dest="spec", help="Do not start from a specific state", required=False)
    parser.set_defaults(spec=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps", type=int, required=False)
    parser.add_argument('-spec_obs', '--specific_strating_observation', default=4, help="Specific state", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores in case of starting from a specific state", type=str, required=False)
    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    parser.add_argument('-mm_value', '--maxmin_value', dest="maxmin_value", action="store_true", help="Compute FE/HE-scores with max/min value instead of last-step value", required=False)
    parser.add_argument('-no_mm_value', '--no_maxmin_value', action="store_false", dest="maxmin_value", help="Compute FE/HE-scores with max/min value instead of last-step value", required=False)
    parser.set_defaults(maxmin_value=False)
    args = parser.parse_args()

    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    NUMBER_SCENARIOS = args.nb_scenarios
    SPECIFIC_STATE = args.spec
    K = args.length_k
    MM_VALUE = args.maxmin_value
    NUMBER_EPISODES = args.nb_episodes
    # only used if SPECIFIC_STATE
    SPECIFIC_STATE_OBS = args.specific_strating_observation
    CSV_FILENAME = args.csv_filename
    RENDER = args.render


    # Paths to store Q tables
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
    favorable_agent_Q_dirpath = "Q-tables" + os.sep + "Favorable"
    hostile_agent_Q_dirpath = "Q-tables" + os.sep + "Hostile"

    #  Paths to store new SXP's scores
    if MAP_NAME == "4x4":
        CSV_FILENAME = "Metrics"+ os.sep + "7 reachable states - 4x4" + os.sep + "New tests" + os.sep + CSV_FILENAME
    elif MAP_NAME == "8x8":
        CSV_FILENAME = "Metrics" + os.sep + "20 random states - 8x8" + os.sep + "New tests" + os.sep + CSV_FILENAME
    else:
        CSV_FILENAME = "Metrics" + os.sep + "Other" + os.sep + CSV_FILENAME

    #  Envs initialisation
    env_windH = MyFrozenLake(behaviour="Hostile", map_name=MAP_NAME)
    env_windF = MyFrozenLake(behaviour="Favorable", map_name=MAP_NAME)
    env = MyFrozenLake(map_name=MAP_NAME)

    #  Agents initialization
    windAgent_H = WindAgent(POLICY_NAME, env_windH)
    windAgent_F = WindAgent(POLICY_NAME, env_windF)
    agent = Agent(POLICY_NAME, env)

    #  Load Q table(s)
    windAgent_H.load(hostile_agent_Q_dirpath)
    windAgent_F.load(favorable_agent_Q_dirpath)
    agent.load(agent_Q_dirpath)

    line, column = env.nRow, env.nCol
    actions = ["Left", "Down", "Right", "Up"]
    wind_agents = [windAgent_H, windAgent_F]

    # Start from a specific state and compute SXp with scores
    if SPECIFIC_STATE:
        env.setObs(SPECIFIC_STATE_OBS)
        env.render()
        # Compute SXp
        SXpMetric(env, SPECIFIC_STATE_OBS, agent, K, wind_agents, number_scenarios=NUMBER_SCENARIOS, csv_filename=CSV_FILENAME, render=RENDER, mm_value=MM_VALUE)

    # Start at initial state and compute SXp in each state, depending on user's choice
    else:
        sum_reward = 0
        misses = 0
        steps_list = []
        nb_episode = NUMBER_EPISODES

        # test loop
        for episode in range(1, nb_episode + 1):
            obs = env.reset()
            done = False
            score = 0
            steps = 0
            while not done:

                steps += 1
                env.render()
                # Compute SXp
                SXp(env, obs, agent, K, wind_agents, number_scenarios=NUMBER_SCENARIOS)
                action, _ = agent.predict(obs)
                obs, reward, done, info = env.step(action)
                score += reward

                # Store infos
                if done and reward == 1:
                    steps_list.append(steps)
                elif done and reward == 0:
                    misses += 1

            sum_reward += score
            print('Episode:{} Score: {}'.format(episode, score))

        if nb_episode > 1:
            print('Score: {}'.format(sum_reward/nb_episode))
            print('----------------------------------------------')
            print('Average of {:.0f} steps to reach the goal position'.format(np.mean(steps_list)))
            print('Fall {:.2f} % of the times'.format((misses / nb_episode) * 100))
            print('----------------------------------------------')
