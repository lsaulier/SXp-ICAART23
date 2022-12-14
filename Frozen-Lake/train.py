import numpy as np
import argparse
import os
from env import MyFrozenLake
from agent import Agent, WindAgent

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4map_test", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=10000, help="Number of training episodes", type=int, required=False)

    args = parser.parse_args()
    
    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    NB_EPISODES = args.nb_episodes

    # Paths to store Q tables
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
    favorable_agent_Q_dirpath = "Q-tables" + os.sep + "Favorable"
    hostile_agent_Q_dirpath = "Q-tables" + os.sep + "Hostile"


    #  Envs initialisation
    env_windF = MyFrozenLake(behaviour="Favorable", map_name=MAP_NAME)
    env_windH = MyFrozenLake(behaviour="Hostile", map_name=MAP_NAME)
    env = MyFrozenLake(map_name=MAP_NAME)

    #  Agents initialization
    windAgent_F = WindAgent(POLICY_NAME, env_windF)
    windAgent_H = WindAgent(POLICY_NAME, env_windH)
    agent = Agent(POLICY_NAME, env, model_env_wind_list=[[windAgent_F, env_windF], [windAgent_H, env_windH]])

    #  Train
    agent.correlatedTrain(NB_EPISODES)
    print("End of training")

    #  Save Q tables
    agent.save(agent_Q_dirpath)
    windAgent_H.save(hostile_agent_Q_dirpath)
    windAgent_F.save(favorable_agent_Q_dirpath)

    # Delete agents
    del agent
    del windAgent_H
    del windAgent_F
