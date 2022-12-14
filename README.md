The implemented code for the *Experimental Results* of the paper *"Reinforcement Learning explained via Reinforcement 
Learning: towards explainable policies through predictive explanation"* is divided into two folders, one for each 
problem. Before testing our implementations, it's *necessary* to install packages of requirements.txt using the 
following pip command: 

```bash
pip install -r requirements.txt
```

Afterward, before running any files, the user must be in the directory of the problem:
```bash
cd .\Frozen-Lake\
cd '.\Drones area coverage\'
```

Find below the main commands to use:
```bash
#####  Frozen Lake  #####
# Train of Agent and WindAgents for 4x4 map (not required command)
python train.py
# Test a policy trained in a 4x4 map. The user can ask at each timestep, SXp's of length 5 and their associated scores.
python test.py
#####  Drone Coverage  #####
# Train of Agent and WindAgents with a steplimit of 40000 (not required command)
python train.py -model "Test_Models" -log "Test_Logs" -limit 40000
# Test a trained policy. The user can ask at each timestep, SXp's of length 6 and their associated scores.
python test.py
```


# Frozen Lake (FL) #

## File Description ##

The Frozen Lake folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent and WindAgent instances, and store learnt Q-tables into JSON files


* **test.py**: parameterized python file which loads learnt Q-tables and tests them in both ways :
    * A classic sequence loop (named *user mode*) which starts in the initial state of the chosen map. The learnt agent's policy is used. 
      At each time-step, the user can ask for SXp's. Explanation scores can be computed when all SXp's are displayed.
    * A specific computation of the three SXp's from a particular state. In this case, explanation scores are 
      necessarily computed.


* **agent.py**: regroups two classes: *Agent* for classic RL agents and *WindAgent* which is used to represent hostile/favourable environment-agents. These classes include Q-tables.


* **env.py**: contains a class *MyFrozenLake* which represent the Frozen Lake environment (inspired by https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)


* **SXp.py**: performs SXp's and compute their scores with the quality evaluation function f representing a weighted addition of the last-step reward of a scenario and it's last maximum Q-value.


* **meansMetric.py**: computes average of SXp's scores. There are two different average we can extract from this file:
    * Average of scores based on *n* runs of explanation scores starting from the *same* specific state
    * Average scores based on *m* explanation scores starting from *m* different states. 
      This average cannot be computed without producing the first dot average explained for the *m* different states.


* **Q-tables folder**: contains all learnt Q-tables for Agent, Hostile agent and Favourable agent. Each filename starts 
by either *Q_*, *Q_WindF* or *Q_WindH* depending on the type of trained agent, i.e. an Agent, a favourable WindAgent or 
a hostile WindAgent respectively.


* **Metrics folder**: contains all CSV files produced for P/HE/FE-scores. Specific folders exist for user experiments. 
A convention that must be respected by the user is the names of CSV files. Indeed, they must start with the map's 
name and ends with the state number (e.g. "4x4_0.csv", "8x8_22.csv")


By default, running **train.py** starts a training of Agent and WindAgent of 10 000 episodes each, on the 4x4 map and **test.py**
runs a classic testing loop on the 4x4 map for the policy explained in the paper. The user can ask for SXp's of length 5 
and their associated scores , at each time-step. To test SXp's for a new policy, it's necessary to execute sequentially **train.py**, then 
**test.py** with appropiate parameters. If the user wants to compute SXp's from a specific state with **test.py**, he must change parameters *-spec*, *-spec_obs* and *-csv*. 
To compute average scores based on *n* scores in a same CSV file from a specific state, you need to run **meansMetric.py** and
provides 3 parameters: CSV filename (*-csv*), states number (in that case,*-s=1*) and map's name (e.g. *-map "4x4"*). 
**meansMetric.py** can also be used to compute an overall average based on *m* CSV files representing explanation scores of *m*
different starting states. Accordingly, the value of parameter *-csv* is not important. By default, the average 
is computed with these starting states use for the paper (hence, we firstly need to compute explanation score from all starting states):

* 4x4 map: [0, 4, 8, 9, 10, 13, 14]
* 8x8 map: [1, 7, 8, 11, 12, 14, 21, 22, 23, 25, 26, 31, 34, 36, 39, 44, 53, 55, 57, 62]

Otherwise, you must change the states list parameter *-st_list*.

## Examples ##

In order to test SXp's from different specific states, the followings lists provide available states (i.e. states that are neither a hole nor a goal state):

* 4x4 map : [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
* 8x8 map : [0-18, 20-28, 30-34, 36-40, 43-45, 47-48, 50-51, 53, 55-58, 60-62]

The number of training episodes must be greater than or equal to 10000, 50000 for the training in the 4x4 map and the other
maps respectively. The followings bash commands are examples of use of **train.py**, **test.py** and **meansMetric.py** files.

**Train:**
```bash
#  Train Agent and WindAgents on 4x4 map with 10000 episodes and save Q-tables in JSON files with names starting with "4x4_test"
python train.py
#  Train Agent and WindAgents on 8x8 map with 50000 episodes and save Q-tables in JSON files with names starting with "8x8_test"
python train.py -map "8x8" -policy "8x8_test" -ep 50000
```
**Test:**
```bash
#####  Test in user mode a policy of a 4x4 map:  ##### 

#  Test a learnt policy on a 4x4 map over 1 episode. SXp's of length 5 can be computed at each time-step. SXp's scores are based on 10000 randomly-generated scenarios
python test.py
#  Test a learnt policy on a 4x4 map over 5 episodes. SXp's of length 3 can be computed at each time-step. SXp's scores are based on 10 randomly-generated scenarios 
python test.py -k 3 -ep 5 -scenarios 10

#####  Test SXp's from a specific state and compute associated scores  ##### 

#  Compute SXp's of length 5, starting from state 22 on 8x8 map. The used policies are "8x8" (three distinct policies for Agent and WindAgents) and scores calculated are saved in "8x8_22.csv" file
python test.py -spec -spec_obs 22 -k 5 -policy "8x8" -map "8x8" -csv "8x8_22.csv"
#  Compute SXp's of length 1, starting from state 10 on 4x4 map. The used policies are "4x4" and scores calculated are saved in "4x4_10.csv" file
python test.py -spec -spec_obs 10 -k 1 -policy "4x4" -csv "4x4_10.csv"
#  Compute SXp's of length 1, starting from state 13 on 4x4 map. The used policies are "4x4" and scores calculated are saved in "4x4_13.csv" file
python test.py -spec -spec_obs 13 -k 1 -policy "4x4" -csv "4x4_13.csv"
```
**SXp's score average:**
```bash
#####   Compute score average for SXp's starting from a specific state. The csv file may contains n scores due to n run of computing SXp's scores  ##### 

#  Compute average from a csv file representing SXp's scores from the starting state 10 in 4x4 map
python meansMetric.py -csv "4x4_10.csv"
#  Compute average from a csv file representing SXp's scores from the starting state 13 in 4x4 map
python meansMetric.py -csv "4x4_13.csv"

#####  Compute score average for SXp's starting from s different states (followings two lines reproduce scores presented in the paper)  ##### 

#  Compute average from 7 csv files representing SXp's scores from 7 different starting states in 4x4 map
python meansMetric.py -s 7 -p 'Metrics\7 reachable states - 4x4\Last step reward'
#  Compute average from 20 csv files representing SXp's scores from 20 different starting states in 8x8 map
python meansMetric.py -s 20 -p 'Metrics\20 random states - 8x8\Last step reward' -map "8x8"
#  Compute average from 2 csv files representing SXp's scores from 2 different starting states in 4x4 map
python meansMetric.py -s 2 -st_list "[10, 13]" 
```
# Drone Coverage (DC) #

## File Description ##

The Drone Coverage folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent and WindAgent instances, save info into text files and neural networks into *dat* files.


* **test.py**: parameterized python file which loads learnt neural networks and tests them in both ways :
    * A classic sequence loop which starts in the initial configuration of the chosen map. The learnt agent's policy is used. 
      At each time-step, the user can ask for SXp's. Explanation scores can be computed when all SXp's are displayed.
    * A specific computation of the three SXp's from a particular state. In this case, explanation scores are 
      necessarily computed.


* **agent.py**: regroups two classes: *Agent* for classic RL agents and *WindAgent* which is used to represent hostile/favourable
environment-agents.


* **env.py**: contains a class DroneCoverageArea which represent the Drone Coverage environment


* **DQN.py**: this file is divided into 3 parts and it's inspired by https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py
and https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py. 
It contains a *DQN* class which is the neural network used to approximate *Q*, an *ExperienceBuffer* class to store agents 
experiences and functions to calculate the DQN's loss.


* **SXp.py**: performs SXp's and compute their scores with the quality evaluation function f representing the last-step reward of a scenario


* **meansMetric.py**: computes average of SXp's scores. There are two different average we can extract from this file:
    * Average of scores based on *n* runs of explanation scores starting from the *same* specific configuration
    * Average scores based on *m* explanation scores starting from *m* different states. 
      This average cannot be computed without producing the first dot average explained for the *m* different configurations.


* **colorize.py**: tool file only used for rendering the environment.


* **Models folder**: contains already learnt policies for agent, hostile agent and favourable agent. 
The names of the produced DQNs show partially the hyper-parameter values. In order, values correspond to the timestep 
limit of training, the timestep where epsilon reaches the value of 0.1 (exploration rate), timestep frequency of 
synchronization of the target neural network, the time horizon of an episode and the use of double-Q learning extension.


* **Metrics folder**: contains all CSV files produced for P/HE/FE-scores. Specific folders exist for user experiments. 
A convention that must be respected by the user is the names of CSV files. Indeed, they must start with the number of
the configurations produced (starts at 1) and ends with the maximum cumulative reward reached by the policy during the 
training process (e.g. "1-11.69.csv", "2-11.69.csv", "1-5.19.csv") 

## Examples ##

In order to test SXp's from a specific configuration, the user must provide a list of coordinates without any coordinate corresponding to a tree or another drone position.
The following list gives all the tree positions (i.e. positions the user cannot use):
[[0,4], [0,8], [1,1], [3,5], [5,2], [6,8], [6,9]]

The number of training timesteps must be greater than or equal to 40000 according to the other default parameters values. 
The followings bash commands are examples of use of **train.py**, **test.py** and **meansMetric.py** files.

**Train:**
```bash
#  Train Agents and WindAgents on 10x10 map with a timestep limit of 40000. It saves info into "Test_Logs" folder and 
#  neural networks into "Test_Models" folder
python train.py -model "Test_Models" -log "Test_Logs" -limit 40000
#  Train Agents on 10x10 map with a timestep limit of 30000. It saves info into "Test_Logs" folder and 
#  neural networks into "Test_Models" folder. The transition function is deterministic since there is no wind (hence we 
#  cannot compute SXp's for this policy). The batch size is set to 16 
python train.py -model "Test_Models" -log "Test_Logs" -limit 30000 -no_w -batch 16
```
**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test a learnt policy (by default, the one presented in the paper) on a 10x10 map with 4 agents. Agents start at random positions. SXp's of length 6 can be computed at each time-step. SXp's scores are based on 1000 randomly-generated scenarios.
python test.py
#  Test a learnt policy (by default, the one presented in the paper) on a 10x10 map with 4 agents. Agents start at pre-defined positions (S cells). SXp's of length 4 can be computed at each time-step. SXp's scores are based on 100 randomly-generated scenarios.
python test.py -k 4 -no_rand -scenarios 100

#####  Test SXp's from a specific configuration and compute associated scores  ##### 

#  Compute SXp's of length 3, starting from configuration [[1,5], [6,6], [7,7], [8,8]]. The default policies are used and scores are saved in '1-11.69.csv' file.
python test.py -csv '1-11.69.csv' -k 3 -spec -spec_conf "[[1,5], [6,6], [7,7], [8,8]]"
#  Idem with a different csv filename and configuration
python test.py -csv '2-11.69.csv' -k 3 -spec -spec_conf "[[1,9], [6,6], [7,3], [8,2]]" 
#  Compute SXp's of length 5, starting from configuration [[4,1], [6,7], [9,3], [2,4]]. Different policies are used.
python test.py -csv '1-11.58.csv' -k 5 -spec -spec_conf "[[4,1], [6,7], [9,3], [2,4]]" -model 'Models\Agent\tl1950000e750000s50000th22ddqnTrue-best_11.58.dat' -model_h 'Models\Hostile\tl1950000e750000s50000th22ddqnTrue_H-best_-5.29.dat' -model_f 'Models\Favourable\tl1950000e750000s50000th22ddqnTrue_F-best_11.93.dat'
```
**SXp's score average:**
```bash
#####  Compute score average for SXp's starting from a specific configuration. The csv file may contains n scores due to n run of computing SXp's scores  ##### 

#  Compute average from the default csv file "1-11.69.csv" representing SXp's scores from the first configuration used in Test section
python meansMetric.py
#  Compute average from the csv file "2-11.69.csv" representing SXp's scores from the second configuration used in Test section
python meansMetric.py -csv '2-11.69.csv'

#####  Compute score average for SXp's starting from s different states (following line reproduce scores presented in the paper)  ##### 

#  Compute average from 30 csv files representing SXp's scores from 30 different starting configurations
python meansMetric.py -conf 30 -p 'Metrics\30 random configs - policy explained 11.69'
#  Compute average from 2 csv files representing SXp's scores from 2 different starting configurations
python meansMetric.py -conf 2
```
## Remarks ##

In each **test.py** python file, there is a boolean parameter *max_min_reward* (*max_min_value*) corresponding to a different
way of computing explanation scores. Set to *False*, the quality evaluation function f is the last step reward
(a weighted addition of Q and R) for the DC problem (resp. FL problem). This f is used  for computing SXp's scores in the paper.
If this parameter is set to *True*, f is the maximum reward (value)  reached during the scenario and the minimum reward (value) reached during the scenario 
for FE/HE scenarios respectively. In the user mode, the user can ask for the last version of f by answering *yes* to 
questions:*"Do you want a mm_value version of HE/FE-scenario?"*. A technical detail, whenever a path
must be given as a parameter, the user need to care about the directory separator. 
Another detail, for the DC problem, if CUDA is available, the training and use of DQN will be with the GPU, otherwise with the CPU.