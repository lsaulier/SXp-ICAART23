import math
import os
from copy import deepcopy
import csv


#  Normalize a value between 0 and 1
#  Input: value to normalize (float), min, max reachable values (float) and normalisation for summary problem (boolean)
#  Output: normalized value (float)
def normalize(value, mini=0, maxi=1.0000001, summary=False):
    if summary and mini == 0 and maxi == 1:
        return 1.0
    else:
        return (value - mini) / (maxi - mini)


#  Compute SXp's quality scores after simulating n randomly-generated scenarios
#  Input: HE/P/FE-scenarios rewards (float list), environment (DroneCoverageAreaEnv), agents (Agent list), the number of
#  steps to look forward (int), number of scenarios to simulate (int), model (DQN), type of transition function (String)
#  device (String), the way of computing SXp quality score (boolean), min/max rewards (float list) and timestep from
#  which the HE/FE-scenario's reward is extracted (int)
#  Output: HE/P/FE quality scores (float)
def computeMetric(xp_reward, env, agents, k, number_scenarios, model, move, device, concise, extremum_reward, step_limit_HE=0, step_limit_FE=0):
    he_reward = xp_reward[0]
    p_reward = xp_reward[1]
    fe_reward = xp_reward[2]
    # Store rewards of n scenarios
    rewards = []
    rewards_he = []
    rewards_fe = []

    #  Simulate the n scenarios
    for i in range(number_scenarios):
        r, he_r, fe_r = scenario(env, agents, k, model, device, concise, move)
        rewards.append(r)
        if concise:
            rewards_he.append(he_r)
            rewards_fe.append(fe_r)

    #  Decide which rewards to use for comparison
    if concise:
        rs_H = rewards_he if rewards_he and len(rewards_he) == len(rewards) else rewards
        rs_F = rewards_fe if rewards_fe and len(rewards_fe) == len(rewards) else rewards
    else:
        rs_H = rewards
        rs_F = rewards

    print("Reward P_scenario scenario : {}".format(p_reward))
    print("Step limit : {}, Reward HE-scenario : {}".format(step_limit_HE, he_reward))
    print("Step limit : {}, Reward FE-scenario : {}".format(step_limit_FE, fe_reward))

    #  Compute the scores
    he_score, p_score, fe_score = metrics(he_reward, fe_reward, p_reward, rs_H, rs_F, rewards, number_scenarios, extremum_reward)
    return he_score, p_score, fe_score


#  Compute HE/P/FE quality scores
#  Input: HE/FE/P rewards (float), lists of rewards from the randomly-generated scenarios (float list),
#  number of produced scenarios (int) and min/max reward (float)
#  Output: HE/P/FE quality scores (float)
def metrics(he_reward, fe_reward, p_reward, scenar_h_rewards, scenar_f_rewards, scenar_last_rewards, number_scenarios, extremum_reward):
    #  Used for cardinality
    he_l = [he_reward <= scenar_h_rewards[i] for i in range(number_scenarios)]
    fe_l = [fe_reward >= scenar_f_rewards[i] for i in range(number_scenarios)]
    #  Average normalized reward of n randomly-generated scenarios
    mean_rewards = normalize(sum(scenar_last_rewards) / len(scenar_last_rewards), extremum_reward[0],
                                   extremum_reward[1])
    # Compute HE/P/FE-scores
    return he_l.count(1) / number_scenarios, abs(p_reward - mean_rewards), fe_l.count(1) / number_scenarios


#  Compute a scenario based on the already learnt model
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), the number of steps to look forward (int),
#  model (DQN), device (String), the way of computing SXp quality score (boolean) and the type of
#  transition function (String)
#  Output: cumulative rewards for computing HE/P/FE quality scores (float)
def scenario(env, agents, k, model, device, concise, move="all"):
    sum_r_he = None
    sum_r_fe = None
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    for agent in agents_copy:
        agent.env = env_copy

    #  Sequence loop
    for i in range(k):
        actions = []
        #  Choose actions
        for agent in agents_copy:
            actions.append(agent.chooseAction(model, device=device))
        #  Step
        _, _, _, dones, _, _ = env_copy.step(agents_copy, actions, move=move)
        #  Rewards
        rewards = env_copy.getReward(agents_copy, actions, dones, reward_type="B")

        #  Save best/worst reward encountered during the scenario if mm_reward
        if concise:
            #  Update sum_r_he and sum_r_fe
            if sum_r_he is None or sum_r_he > sum(rewards):
                sum_r_he = sum(rewards)
            if sum_r_fe is None or sum_r_fe < sum(rewards):
                sum_r_fe = sum(rewards)
        #  Check end condition
        if dones.count(True) == len(dones) and i != k - 1:
            #  Save best/worst reward encountered during the scenario if mm_reward
            if concise:
                if sum_r_he is None or sum_r_he > sum(rewards):
                    sum_r_he = sum(rewards)
                if sum_r_fe is None or sum_r_fe < sum(rewards):
                    sum_r_fe = sum(rewards)
            break

    if sum_r_fe is None and sum_r_he is None:
        return sum(rewards), sum(rewards), sum(rewards)
    else:
        return sum(rewards), sum_r_he, sum_r_fe


#  Compute a P-scenario
#  Input: environment (DroneCoverageArea), agents (Agent list), the number of steps to look forward (int),
#  the model (DQN), min/max rewards (float list), device (String), type of scenario's transition (int), type of
#  transition function (String), use of render(), compression ratio (int) and balance between impact of importance
#  and gap scores (float)
#  Output: normalized reward of P-scenario and the path probability linked to the scenario (float)
def P_scenario(env, agents, k, model, extremum_reward, device, most_probable_transition=1, move="all", render=False,
               summary=0, lbda=1.0):
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    path_proba = [1.0, 1.0, 1.0, 1.0]
    for agent in agents_copy:
        agent.env = env_copy
    #  Init obs_list to store SXp' observations (used for SXp summary)
    if summary:
        obs_list = [[agent.get_obs() for agent in agents_copy]]
        action_list = []
        dead_list = [[agent.get_dead() for agent in agents_copy]]
    #  Sequence loop
    for i in range(k):
        actions = []
        #  Choose actions
        for agent in agents_copy:
            actions.append(agent.chooseAction(model, device=device))
        #  Get the most probable transition
        if env.windless:
            _, _, _, dones, _, pr = env_copy.step(agents_copy, actions, move=move)
        else:
            _, _, _, dones, _, pr = env_copy.step(agents_copy, actions,
                                                 most_probable_transitions=most_probable_transition, move=move)
        #  Update path probability
        path_proba = [a * b for a, b in zip(path_proba, pr)]
        #  Extract rewards
        rewards = env_copy.getReward(agents_copy, actions, dones, reward_type="B")
        #  Store obs and actions for summarized SXp
        if summary:
            action_list.append(actions)
            obs_list.append([agent.get_obs() for agent in agents_copy])
            dead_list.append([agent.get_dead() for agent in agents_copy])
        #  Render
        if not summary:
            if render:
                env_copy.render(agents_copy)
        #  Check end condition
        if dones.count(True) == len(dones) and i != k - 1:
            break

    # Summarized SXp
    if summary:
        summarizedSXp(agents_copy, obs_list, action_list, dead_list, summary, lbda, model, env_copy, True, device)

    return normalize(sum(rewards), extremum_reward[0], extremum_reward[1]), path_proba


#  Compute a HE/FE-scenario, depending on the environment type
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), wind agents (WindAgent list), the number of steps to
#  look forward (int), agent's model and wind agent's model (DQN), device (String), the way of computing SXp quality
#  score (boolean), type of transition function (String), use of render(), compression ratio (int) and balance between
#  impact of importance and gap scores (float)
#  Output: HE/FE-scenario's reward (float), the step from where the reward was reached (int) and the path probability
#  linked to the scenario (float)
def E_scenario(env, agents, wind_agent, k, model, wind_model, device, concise, move="all", render=False,
               summary=0, lbda=1.0):
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    path_proba = [1.0, 1.0, 1.0, 1.0]
    for agent in agents_copy:
        agent.env = env_copy
    #  Init obs_list to store SXp' observations (used for SXp summary)
    if summary:
        obs_list = [[agent.get_obs() for agent in agents_copy]]
        action_list = []
        dead_list = [[agent.get_dead() for agent in agents_copy]]
    if concise:
        best_reward = None
        envs_agents = []
        n = 0
    #  Sequence loop
    for i in range(k):
        #  Choose actions
        actions = []
        for agent in agents_copy:
            action = agent.chooseAction(model, device=device)
            actions.append(action)
        #  Step using wind model as transition
        _, _, _, dones, _, pr = env_copy.step(agents_copy, actions, wind_agent=wind_agent,
                                          wind_net=wind_model, device=device, move=move)
        #  Update path probability
        path_proba = [a * b for a, b in zip(path_proba, pr)]
        #  Extract reward
        rewards = env_copy.getReward(agents_copy, actions, dones, reward_type="B")
        sum_reward = sum(rewards)
        #  Store obs and actions for summarized SXp
        if summary:
            action_list.append(actions)
            obs_list.append([agent.get_obs() for agent in agents_copy])
            dead_list.append([agent.get_dead() for agent in agents_copy])

        #  Render
        if not summary:
            if concise:
                if best_reward is not None:
                    if wind_agent.behaviour == "hostile":
                        condition = best_reward > sum_reward
                    else:
                        condition = best_reward < sum_reward
                #  Unroll configurations depending on condition
                if best_reward is None or condition:
                    best_reward = sum_reward
                    #  Render
                    if envs_agents:
                        if render:
                            for env, agents in envs_agents:
                                env.render(agents)
                                pass
                        envs_agents = []

                    if render:
                        env_copy.render(agents_copy)
                    n = i + 1
                else:
                    #  Store the environment
                    envs_agents.append((deepcopy(env_copy), deepcopy(agents_copy)))
            else:
                if render:
                    env_copy.render(agents_copy)
        #  Check end condition
        if dones.count(True) == len(dones) and i != k - 1:
            #print("H/F-E-scenario : Help, a drone is crashed")
            break

    # Summarized SXp
    if summary:
        summarizedSXp(agents_copy, obs_list, action_list, dead_list, summary, lbda, model, env_copy, True, device)

    if concise:
        return best_reward, n, path_proba
    else:
        return sum_reward, i+1, path_proba


#  Compute argmax of a couple list
#  Input: iterable (tuple list)
#  Output: key corresponding to the greatest value (int)
def argmax(pairs):
    return max(pairs, key=lambda x: x[1])[0]


#  Compute argmax of a list
#  Input: scores (float list)
#  Output: key corresponding to the greatest value (int)
def argmax_index(values):
    return argmax(enumerate(values))


#  Get min/max values of a list of configuration importance scores
#  Input: scores list (float list)
#  Output: min/max value (float)
def getImportanceExtremum(imp_list):
    min_sum = math.inf
    max_sum = - math.inf
    # Simultaneously search best/worst scores
    for _, elm in enumerate(imp_list):
        #  Update max
        if elm > max_sum:
            max_sum = elm
        #  Update min
        if elm < min_sum:
            min_sum = elm
    # Specific case: only one solution is possible, hence min and max are the same
    if min_sum == max_sum:
        min_sum = 0
        max_sum = 1

    return min_sum, max_sum


#  Get min/max gap value when selecting m elements among a list of n elements
#  Input: number of elements to select (int) and total number of elements (int)
#  Output: min/max value (int)
def getGapExtremum(m, n):
    # Worst case: each element is direct neighbors except 1 at the opposite
    max_gap = (m - 2) + (n - m + 1)**2
    # Best case: elements are evenly spread out
    mean_elm_partition = (n-m) / (m - 1)
    max_elm_part = math.ceil(mean_elm_partition)
    min_elm_part = math.floor(mean_elm_partition)
    nbr_max_part = math.floor((mean_elm_partition - min_elm_part) * (m-1))
    min_gap = nbr_max_part * (max_elm_part + 1)**2 + (m - 1 - nbr_max_part) * (min_elm_part + 1)**2
    # Specific case: only one solution is possible, hence min and max are the same
    if min_gap == max_gap:
        min_gap = 0
        max_gap = 1
    return min_gap, max_gap


#  Get the selection of m states using the dynamic programming method
#  Input: configuration importance score list (float list), configuration list (int list list list), number of element
#  to select (int), balance between impact of importance and gap scores (float)
#  Output: selected states (int list list list) and associated indexes (int list)
def getImportantStates_DynProg(imp_list, obs_list, m, lbda):
    # Summary problem init
    m = m + 2
    n = len(imp_list)
    g = {i: None for i in range(2, m + 1)}
    # Get importance score extremums
    min_imp, max_imp = getImportanceExtremum(imp_list)
    imp_dict = {i + 1: score for i, score in enumerate(imp_list)}
    # Init g[2] (only one answer is possible: select first and last element)
    l = []
    imp_init_score = normalize(imp_dict[1], min_imp, max_imp, summary=True)
    for i in range(2, n+1):  # look all possible couples
        l.append(([1, i], imp_init_score + normalize(imp_dict[i], min_imp, max_imp, summary=True), (i-1)**2))
    g[2] = l
    # Iterate over m to fill g
    for i in range(3, m + 1):
        l = []
        for j in range(i, n + 1):
            # Get gap extremums
            min_gap, max_gap = getGapExtremum(i, j)
            # Compute normalized importance and gap scores
            imp_scores = [(1/i) * (g[i - 1][p][1] + normalize(imp_dict[j], min_imp, max_imp, summary=True)) for p in range(j - i + 1)]
            gap_scores = [normalize(g[i - 1][p][2] + (j - (p + i - 1))**2, min_gap, max_gap, summary=True) for p in range(j - i + 1)]
            # Compute g scores
            values = [imp - lbda * gap for imp, gap in zip(imp_scores, gap_scores)]
            # Extract the best element to add in the selection according to g scores
            tmp = argmax_index(values)  # (i-1): lie in range (m-1, n-1) | -1 : python list indexing
            idx, imp, gap = deepcopy(g[i - 1][tmp])
            #  Update idx, imp, gap
            idx.append(j)
            imp += normalize(imp_dict[j], min_imp, max_imp, summary=True)
            gap += (j - idx[-2])**2
            l.append((idx, imp, gap))
        # Update g
        g[i] = l
    # Extract indexes
    idxs = g[m][-1][0]
    idxs = [idx - 1 for idx in idxs]
    return [obs_list[i] for i in idxs], idxs


#  Find a loop in a list of configurations
#  Input: configuration's list (int list list list)
#  Output: loop (int list list) and index of the first element in the loop (int)
def findLoop(obs_list):
    d = {}
    for i, elm in enumerate(obs_list):
        #  Loop is found
        if d.get(str(elm)):
            return obs_list[d.get(str(elm)):i], d.get(str(elm))
        #  Create a key
        else:
            d[str(elm)] = i
    return [], -1


#  Render a part of the SXp
#  Input: agents (Agent list), environment (DroneAreaCoverage), configuration' list (int list list list), dones' list
#  (bool list), actions' list (int list list)
#  Output: None
def renderSummarizedSXP(agents, env, obs_list, dead_list, action_list=None):
    for idx, observations in enumerate(obs_list):
        # Set for each agent their observation, last action and done value
        for j, agent in enumerate(agents):
            agent.set_obs(observations[j])
            agent.set_dead(dead_list[idx][j])
        if action_list is None:
            env.set_lastactions(None)
        else:
            env.set_lastactions(action_list[idx])
        # Rendering
        env.render(agents)
    return


#  Summarize a scenario using dynamic programming solution
#  Input: number of state to highlight (int), balance between impact of importance and gap scores (float), agents
#  (Agent list), configuration' list (int list list list), actions' list (int list list), policy (DQN) and device (str)
#  Output: important observations (int list list), important actions (int list list)
#  indexes of important observations (int list)
def summarize(m, lbda, agents, obs_list, action_list, model, device):
    # Compute and store configuration importance score
    importance_list = []
    for observations in obs_list:
        importance_list.append(sum([agents[0].stateImportance(model, obs, device) for obs in observations])
                               / len(observations))
    # Dyn. prog. solution
    important_obs, important_obs_idx = getImportantStates_DynProg(importance_list, obs_list, m, lbda)
    important_obs = important_obs[1:-1]
    important_obs_idx = important_obs_idx[1:-1]
    #  Get associated actions
    important_action = [action_list[idx-1] for idx in important_obs_idx]

    return important_obs, important_action, important_obs_idx


#  Manage the render and computation of a summarized SXp
#  Input: agents (Agent list), configurations' list (int list list list), actions' list (int list list), dones' list
#  (bool list list), compression ratio (int), balance between impact of importance and gap scores (float), policy (DQN),
#  environment (DroneAreaCoverage), display or not (bool), device (str)
#  Output: None
def summarizedSXp(agents, obs_list, action_list, dead_list, summary, lbda, model, env, render, device):
    last_state_render = True
    #  First configuration render
    if render:
        print('First configuration')
        renderSummarizedSXP(agents, env, [obs_list[0]], [dead_list[0]])
    #  Loop detection
    loop, i = findLoop(obs_list)
    #  A loop was detected
    if i >= 0:
        #  First part of the scenario is summarized or not, depending on a threshold
        if i > 1:
            if len(obs_list[1:(i-1)]) < summary:
                #  Render
                if render:
                    renderSummarizedSXP(agents, env, obs_list[1:(i-1)], dead_list[1:(i-1)], action_list[:(i-2)])
            else:
                n_obs = len(obs_list[1:(i-1)]) // summary
                important_obs, important_actions, idx_list = summarize(n_obs, lbda, agents, obs_list[:i], action_list[:i], model, device)
                if render:
                    renderSummarizedSXP(agents, env, important_obs,
                                    [elm for i, elm in enumerate(dead_list[:i]) if i in idx_list], important_actions)

            #  Last configuration of first part render
            if render:
                renderSummarizedSXP(agents, env, [obs_list[i-1]], [dead_list[i-1]], [action_list[i-2]])

        #  Specific case: first state is part of the loop
        if i == 0:
            loop = loop[1:]
            i = 1
        #  Depending on a threshold, we either directly display states or summarize them
        print('Loop of length {} strating from timestep {} detected!'.format(len(loop), i + 1))
        print('Repeated  {} times'.format((len(obs_list) - i) // len(loop)))
        if (len(loop) - 2) <= summary:
            print('Small loop rendering')
            #  Render
            if render:
                renderSummarizedSXP(agents, env, loop, dead_list[i:i + len(loop)], action_list[i-1:i-1 + len(loop)])
                last_state_render = False
        else:
            #  First loop configuration render
            if render:
                renderSummarizedSXP(agents, env, [loop[0]], [dead_list[i]], [action_list[i-1]])
            print('Loop summary')
            #  Number of states to highlight
            n_obs = (len(loop) - 2) // summary
            important_obs, important_actions, idx_list = summarize(n_obs, lbda, agents, loop, action_list[i:i + len(loop)], model, device)
            if render:
                renderSummarizedSXP(agents, env, important_obs,
                                    [elm for i, elm in enumerate(dead_list[i:i + len(loop)]) if i in idx_list], important_actions)
            #  Last loop configuration render
            if render:
                renderSummarizedSXP(agents, env, [loop[-1]], [dead_list[i + len(loop)]], [action_list[i + len(loop) - 1]])

    else:
        #  Number of states to highlight
        n_obs = (len(obs_list) - 2) // summary
        important_obs, important_actions, idx_list = summarize(n_obs, lbda, agents, obs_list, action_list, model, device)
        if render:
            renderSummarizedSXP(agents, env, important_obs,
                                [elm for i, elm in enumerate(dead_list) if i in idx_list],
                                important_actions)
            if obs_list[-1] in important_obs:
                last_state_render = False
    #  Last configuration render
    if render and last_state_render:
        print('Last configuration')
        renderSummarizedSXP(agents, env, [obs_list[-1]], [dead_list[-1]], [action_list[-1]])

    return


#  Compute SXps from a specific states and verify how good they are with quality scores and store them in a CSV file
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), wind agents (WindAgent list), the number of steps to
#  look forward (int), agent's model and wind agent's models (DQN), device (String), type of transition function
#  (String), number of scenarios randomly-generated (int), min/max rewards (float list), csv to store scores (String),
#  use of render() (boolean), way of computing SXp quality score (boolean), compression ratio (int) and balance between
#  impact of importance and gap scores (float)
#  Output: None
def SXpMetric(env, agents, wind_agents, k, model, wind_models, device, move="all", number_scenarios=1000, extremum_reward=[], csv_filename="", render=False, concise=False, summary=0, lbda=1.0):
    #  ------------------------ HE-scenario ----------------------------------------
    print("HE-scenario -------------")
    reward_HE, step_limit_HE, proba_HE = E_scenario(env, agents, wind_agents[0], k, model, wind_models[0], device, concise, move=move, render=render, summary=summary, lbda=lbda)
    print("Reward obtained : {}, with step : {}".format(reward_HE, step_limit_HE))

    #  ------------------------ P-scenario ----------------------------------------
    print("P-scenario -------------")
    reward_P, proba_P = P_scenario(env, agents, k, model, extremum_reward, device, move=move, render=render, summary=summary, lbda=lbda)
    print("Reward obtained : {}".format(reward_P))

    #  ------------------------ FE-scenario ----------------------------------------
    print("FE-scenario -------------")
    reward_FE, step_limit_FE, proba_FE = E_scenario(env, agents, wind_agents[1], k, model, wind_models[1], device, concise, move=move, render=render, summary=summary, lbda=lbda)
    print("Reward obtained : {}, with step : {}".format(reward_FE, step_limit_FE))

    #  ------------------------ Metrics F/H/PXp ----------------------------------------
    mean_P = sum(proba_P) / len(proba_P)
    mean_FE = sum(proba_FE) / len(proba_FE)
    mean_HE = sum(proba_HE) / len(proba_HE)

    if csv_filename and csv_filename.split(os.sep)[-1][:2] == "rp":
        # Compute representativeness scores
        rp_P = mean_P / mean_P
        rp_FE = min(1.0, max(0.0, round(mean_FE / mean_P, 5)))
        rp_HE = min(1.0, max(0.0, round(mean_HE / mean_P, 5)))
        #  ------------------------ Store in CSV ----------------------------------------
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([[agent.position for agent in agents], rp_P, rp_FE, rp_HE])

    else:
        print("Probas: P {}, FE {}, HE {}".format(proba_P, proba_FE, proba_HE))
        print("Means: P {}, FE {}, HE {}".format(mean_P, mean_FE, mean_HE))
        # Compute representativeness scores
        print('Representativeness scores:')
        print('P-scenario: {}'.format(mean_P / mean_P))
        print('FE-scenario: {}'.format(min(1.0, max(0.0, round(mean_FE / mean_P, 5)))))
        print('HE-scenario: {}'.format(min(1.0, max(0.0, round(mean_HE / mean_P, 5)))))
        # Compute quality scores
        HE_score, P_score, FE_score = computeMetric([reward_HE, reward_P, reward_FE], env, agents, k, number_scenarios, model, move, device, concise, extremum_reward, step_limit_HE, step_limit_FE)
        print("For HE-scenario, percentage of worse scenarios over {} scenarios : {}".format(number_scenarios, HE_score))
        print("Cumulative reward difference between P-scenario and the mean reward of  {} scenarios : {}".format(number_scenarios, P_score))
        print("For FE-scenario, percentage of better scenarios over {} scenarios : {}".format(number_scenarios, FE_score))

        #  ------------------------ Store in CSV ----------------------------------------
        if csv_filename:
            with open(csv_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([P_score, HE_score, FE_score])

    return


#  Provide different SXps from a starting state depending on user's choice
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), wind agents (WindAgent list), the number of steps to
#  look forward (int), agent's model and wind agent's models (DQN), device (String), type of transition function
#  (String), number of scenarios randomly-generated (int), min/max rewards (float list), use of render() (boolean),
#  compression ratio (int) and balance between impact of importance and gap scores (float)
#  Output: None
def SXp(env, agents, wind_agents, k, model, wind_models, device, move="all", number_scenarios=1000, extremum_reward=[],
        render=False, summary=0, lbda=1.0):
    answer = False
    concise = False
    good_answers = ["yes", "y"]
    while not answer:

        question = "Do you want an explanation?"
        explanation = input(question)

        if explanation in good_answers:

            #  ------------------------ HE-scenario ----------------------------------------
            question_HE = "Do you want a HE-scenario of the agent's move? "
            explanation_HE = input(question_HE)

            # Provide HE-scenario
            if explanation_HE in good_answers:

                question_boosted = "Do you want a mm_value version of HE-scenario?"
                boosted_case = input(question_boosted)
                mm_reward_HE = boosted_case in good_answers
                print("------------- HE-scenario -------------")
                reward_HE, step_limit_HE, proba_HE = E_scenario(env, agents, wind_agents[0], k, model, wind_models[0], device,
                                                      mm_reward_HE, move=move, render=render, summary=summary, lbda=lbda)

            #  ------------------------ P-scenario ----------------------------------------
            question_P = "Do you want a P-scenario of the agent's move? "
            explanation_P = input(question_P)

            # Provide P-scenario
            if explanation_P in good_answers:
                print("------------- P-scenario -------------")
                reward_P, proba_P = P_scenario(env, agents, k, model, extremum_reward, device, move=move, render=render,
                                      summary=summary, lbda=lbda)

            #  ------------------------ FE-scenario ----------------------------------------
            question_FE = "Do you want a FE-scenario of the agent's move? "
            explanation_FE = input(question_FE)

            # Provide FE-scenario
            if explanation_FE in good_answers:

                question_boosted = "Do you want a mm_value version of FE-scenario?"
                boosted_case = input(question_boosted)
                mm_reward_FE = boosted_case in good_answers

                print("------------- FE-scenario -------------")
                reward_FE, step_limit_FE, proba_FE = E_scenario(env, agents, wind_agents[1], k, model, wind_models[1], device,
                                                      mm_reward_FE, move=move, render=render, summary=summary, lbda=lbda)

            #  ------------------------ Metrics ----------------------------------------
            if explanation_HE in good_answers and explanation_P in good_answers and explanation_FE in good_answers:
                question_metric = "Do you want a metric score for these explanations ?"
                answer_metric = input(question_metric)

                if answer_metric in good_answers:
                    # Compute representativeness scores
                    mean_P = sum(proba_P) / len(proba_P)
                    mean_FE = sum(proba_FE) / len(proba_FE)
                    mean_HE = sum(proba_HE) / len(proba_HE)
                    print("Probas: P {}, FE {}, HE {}".format(proba_P, proba_FE, proba_HE))
                    print("Means: P {}, FE {}, HE {}".format(mean_P, mean_FE, mean_HE))

                    print('Representativeness scores:')
                    print('P-scenario: {}'.format(mean_P / mean_P))
                    print('FE-scenario: {}'.format(min(1.0, max(0.0, round(mean_FE / mean_P, 5)))))
                    print('HE-scenario: {}'.format(min(1.0, max(0.0, round(mean_HE / mean_P, 5)))))

                    #  Quality of scenarios
                    HE_score, P_score, FE_score = computeMetric([reward_HE, reward_P, reward_FE], env, agents, k, number_scenarios, model, move, device, concise, extremum_reward, step_limit_HE, step_limit_FE)
                    print("For HE-scenario, percentage of worse scenarios over {} scenarios : {}".format(number_scenarios, HE_score))
                    print("Cumulative reward difference between P-scenario and the mean reward of  {} scenarios : {}".format(number_scenarios, P_score))
                    print("For FE-scenario, percentage of better scenarios over {} scenarios : {}".format(number_scenarios, FE_score))


            print("Go back to the current state of the problem!")
            env.render(agents)

        else:
            pass

        answer = True

    return

