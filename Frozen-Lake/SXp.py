import os
from copy import deepcopy
import csv
import math


#  Normalize a value between 0 and 1
#  Input: value to normalize (float), min, max reachable value (float) and normalisation for summary problem (boolean)
#  Output: normalized value (float)
def normalize(value, mini=0, maxi=1.0000001, summary=False):
    if summary and mini == 0 and maxi == 1:
        return 1.0
    else:
        return (value - mini) / (maxi - mini)


# Check if the reached state is a terminal one
# Input: environment, state (int)
# Output: a boolean
def terminalState(env, obs):
    i, j = int(obs // env.nCol), int(obs % env.nCol)  # extract coordinates
    return env.desc[i, j].decode('UTF-8') == 'H' or env.desc[i, j].decode('UTF-8') == 'G'


#  Generate n randomly-generated scenarios and compute quality scores for SXps
#  Input: SXps values (float list), environment (MyFrozenLake), current state (int), scenario's length (int),
#  number of scenarios to generate (int), agent (Agent), the way of computing SXp score (boolean) and
#  timestep from which the HE/FE-scenario's value is extracted (int)
#  Output: quality scores (float list)
def computeMetric(xp_value, env, obs, k, number_scenarios, model, mm_value, step_limit_HE=0, step_limit_FE=0):

    HE_value = xp_value[0]
    psxp_value = xp_value[1]
    FE_value = xp_value[2]

    # Store values of n scenarios
    values = []
    values_HE = []
    values_FE = []

    #  Simulate the n scenarios
    for i in range(number_scenarios):
        v, HE_v, FE_v = scenario(env, obs, k, model, mm_value)
        values.append(v)
        if mm_value:
            values_HE.append(HE_v)
            values_FE.append(FE_v)

    #  Decide which values to use for comparison
    if mm_value:
        vs_H = values_HE if values_HE and len(values_HE) == len(values) else values
        vs_F = values_FE if values_FE and len(values_FE) == len(values) else values
    else:
        vs_H = values
        vs_F = values

    print("Reward P scenario : {}".format(psxp_value))
    print("Step limit : {}, Reward HE-scenario : {}".format(step_limit_HE, HE_value))
    print("Step limit : {}, Reward FE-scenario : {}".format(step_limit_FE, FE_value))

    #  Compute scores
    HE_score, psxp_score, FE_score = metrics(HE_value, FE_value, vs_H, vs_F, number_scenarios, psxp_value, scenar_last_values=values)
    return HE_score, psxp_score, FE_score


#  Compute SXps quality scores
#  Input: HE/FE-scenario's value (float), values of randomly-generated scenarios for HE/FE-scenario (float list),
#  number of randomly-generated scenarios (int), P-scenario's value (float), compute P-score (boolean) and
#  values of randomly-generated scenarios for P-scenario (float list)
#  Output: HE/FE/P quality score (float list)
def metrics(HE_value, FE_value, scenar_h_values, scenar_f_values, number_scenarios, psxp_value=0, psxp=True, scenar_last_values=[]):
    # Used for computing cardinality
    HE_l = [HE_value <= scenar_h_values[i] for i in range(number_scenarios)]
    FE_l = [FE_value >= scenar_f_values[i] for i in range(number_scenarios)]

    if psxp:
        mean_values = sum(scenar_last_values) / len(scenar_last_values)
        mean_values = normalize(mean_values)
        print("P-score metric : Sum scenar rewards {} --- len {} --- nomalize mean {}".format(sum(scenar_last_values),
                                                                                         len(scenar_last_values),
                                                                                         mean_values))
        # Compute SXps quality scores
        return HE_l.count(1) / number_scenarios, abs(psxp_value - mean_values), FE_l.count(1) / number_scenarios
    else:
        # Compute SXps quality scores
        return HE_l.count(1) / number_scenarios, None, FE_l.count(1) / number_scenarios


#  Compute a scenario of length k (stop earlier in case of a terminal state reached)
#  Input:  environment (MyFrozenLake), current state (int), length of scenario (int), agent (Agent)
#  and the way of computing SXp score (boolean)
#  Output: f value for P/HE/FE-scenario (float) and all values encountered (float list)
def scenario(env, obs, k, model, mm_value):
    f_value_HE = None
    f_value_FE = None
    env_copy = deepcopy(env)
    #  Sequence loop
    for i in range(k):
        #  Save obs to get reward
        last_obs = obs
        #  Action choice
        action, _ = model.predict(obs)
        #  Step
        obs, _, done, _ = env_copy.step(action)
        #  Extract value and reward
        value = model.getValue(obs)
        reward = env.getReward(last_obs, action, obs)
        f = value + 1.0000001 * reward
        if mm_value:
            if f_value_HE is None or f_value_HE > f:
                f_value_HE = f
            if f_value_FE is None or f_value_FE < f:
                f_value_FE = f
        # Check whether the scenario ends or not
        if done and i != k - 1:
            if mm_value:

                if f_value_HE is None or f_value_HE > f:
                    f_value_HE = f

                if f_value_FE is None or f_value_FE < f:
                    f_value_FE = f

            break

    if not mm_value:
        return f, f, f
    else:
        return f, f_value_HE, f_value_FE


#  Compute a P scenario
#  Input: environment (MyFrozenLake), the current state (int), the number of steps to look forward (int),
#  agent, render (boolean), compression ratio (int) and balance between impact of importance and gap scores (float)
#  Output: normalized f value for P-scenario (float)
def P_scenario(env, obs, k, model, render=False, summary=0, lbda=1.0):

    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    path_proba = 1.0
    #  Init obs_list to store SXp' observations
    if summary:
        obs_list = [obs]
        action_list = []
    # Sequence loop
    for i in range(k):
        #  Save obs to get reward
        last_obs = obs
        #  Action choice
        action, _ = model.predict(obs)
        #  Step
        obs, _, done, _ = env_copy.step(action)
        #  Update path_proba
        path_proba *= (1.0 / 3.0)
        # Store obs for summarized SXp
        if summary:
            obs_list.append(obs)
            action_list.append(action)
        #  Render
        if render and not summary:
            env_copy.render()
        #  Extract value and reward
        value = model.getValue(obs)
        reward = env_copy.getReward(last_obs, action, obs)
        f = value + 1.0000001 * reward
        # Check whether the scenario ends or not
        if done and i != k - 1:
            break

    # Summarized SXp
    if summary:
        summarizedSXp(obs_list, action_list, summary, lbda, model, env_copy, render, scenario='P')

    print("Last-step normalised f value of P-scenario: {}".format(normalize(f)))
    return normalize(f), path_proba


#  Compute a HE/FE-scenario, depending on the environment type
#  Input: environment (MyFrozenLake), current state (int), number of steps to look forward (int),
#  agent, wind agent, way of computing SXp quality score (boolean), render (boolean), compression ratio (int) and
#  balance between impact of importance and gap scores (float)
#  Output: f value of a state (float) and step number of the state where the value comes from (int)
def E_scenario(env, obs, k, model, wind_model, mm_value, render=False, summary=0, lbda=1.0):
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    path_proba = 1.0
    #  Init obs_list to store SXp' observations
    if summary:
        obs_list = [obs]
        action_list = []

    if mm_value:
        best_f = None
        envs_tmp = []
        n = 0
    #  Sequence loop
    for i in range(k):
        #  Save obs to get reward
        last_obs = obs
        #  Actions choice
        action, _ = model.predict(obs)
        #  Step
        wind_action, _ = wind_model.predict(obs * model.actions + action)
        #  Update path_proba
        path_proba *= (1.0 / 3.0)
        #  New state of the agent
        new_row, new_col = env_copy.inc(obs // env_copy.nCol, obs % env_copy.nCol, wind_action)
        obs = env_copy.to_s(new_row, new_col)
        # Store obs for summarized SXp
        if summary:
            obs_list.append(obs)
            action_list.append(action)
        #  Extract value and reward
        value = model.getValue(obs)
        reward = env_copy.getReward(last_obs, action, obs)
        f = value + 1.0000001*reward

        #  Render
        if not summary:
            env_copy.update(action, obs)
            if mm_value:
                if best_f is not None:

                    if wind_model.env.behaviour == "Hostile":
                        condition = best_f > f
                    else:
                        condition = best_f < f
                if best_f is None or condition:
                    best_f = f

                    #  Render
                    if envs_tmp:
                        if render:
                            for env in envs_tmp:
                                env.render()
                                pass
                        envs_tmp = []
                    if render:
                        env_copy.render()
                    n = i + 1
                else:
                    # Store the environment
                    envs_tmp.append(deepcopy(env_copy))
            else:
                if render:
                    env_copy.render()

        #  Check whether the scenario ends or not
        done = terminalState(env_copy, obs)
        if done and i != k - 1:
            break

    # Summarized SXp
    if summary:
        summarizedSXp(obs_list, action_list, summary, lbda, model, env_copy, render)

    if mm_value:
        print("Best f value encounter : {}, at time-step: {}".format(best_f, n))
        return best_f, n, path_proba
    else:
        print("Last-step f value from state {} is : {}".format(obs, f))
        return f, i+1, path_proba


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
#  Input: state importance score list (float list), state list (int list), number of element to select (int), balance
#  between impact of importance and gap scores (float)
#  Output: selected states (int list) and associated indexes (int list)
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


#  Find a loop in a list of observation
#  Input: observation's list (int list)
#  Output: loop (int list) and index of the first element in the loop (int)
def findLoop(obs_list):
    d = {}
    for i, elm in enumerate(obs_list):
        #  Loop is found
        if d.get(elm):
            return obs_list[d.get(elm):i], d.get(elm)
        #  Create a key
        else:
            d[elm] = i
    return [], -1


#  Render a part of the SXp
#  Input: environment (MyFrozenLake), observations' list (int list), actions' list (int list)
#  Output: None
def renderSummarizedSXP(env, obs_list, action_list=None):
    for idx, observation in enumerate(obs_list):
        # Set agent's observation and last action
        if action_list is None:
            env.update(None, observation)
        else:
            env.update(action_list[idx], observation)
        # Rendering
        env.render()
    return


#  Summarize a scenario using dynamic programming solution
#  Input: number of state to highlight (int), balance between impact of importance and gap scores (float), observations'
#  list (int list), actions' list (int list), agent (Agent)
#  Output: important observations (int list), important actions (int list), indexes of important observations (int list)
def summarize(m, lbda, obs_list, action_list, model):
    #  Compute and store stateImportance of each state
    importance_list = []
    for obs in obs_list:
        importance_list.append(model.stateImportance(obs))
    # Dyn. prog. solution
    important_obs, important_obs_idx = getImportantStates_DynProg(importance_list, obs_list, m, lbda)
    important_obs = important_obs[1:-1]
    important_obs_idx = important_obs_idx[1:-1]
    #  Get associated actions
    important_action = [action_list[idx-1] for idx in important_obs_idx]

    return important_obs, important_action, important_obs_idx


#  Manage the render and computation of a summarized SXp
#  Input: observations' list (int list), actions' list (int list), compression ratio (int), balance between impact of
#  importance and gap scores (float), agent (Agent), environment (MyFrozenLake), display or not (bool),
#  type of scenario (str)
#  Output: None
def summarizedSXp(obs_list, action_list, summary, lbda, model, env, render, scenario='E'):
    last_state_render = True
    #  First configuration render
    if render:
        print('First configuration')
        renderSummarizedSXP(env, [obs_list[0]], [action_list[0]])

    #  Loop detection
    loop, i = findLoop(obs_list) if scenario != 'P' else ([], -1)
    #  A loop was detected
    if i >= 0:
        #  First part of the scenario is summarized or not, depending on a threshold
        if i > 1:
            if len(obs_list[1:(i-1)]) < summary:
                #  Render
                if render:
                    renderSummarizedSXP(env, obs_list[1:(i-1)], action_list[:(i-2)])
            else:
                n_obs = len(obs_list[1:(i-1)]) // summary
                important_obs, important_actions, idx_list = summarize(n_obs, lbda, obs_list[:i], action_list[:i], model)
                if render:
                    renderSummarizedSXP(env, important_obs, important_actions)
            #  Last configuration of first part render
            if render:
                renderSummarizedSXP(env, [obs_list[i-1]], [action_list[i-2]])

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
                renderSummarizedSXP(env, loop, action_list[i-1:i-1+ len(loop)])
                last_state_render = False
        else:
            #  First loop configuration render
            if render:
                renderSummarizedSXP(env, [loop[0]], [action_list[i-1]])
            print('Loop summary')
            #  Number of states to highlight
            n_obs = (len(loop) - 2) // summary
            important_obs, important_actions, idx_list = summarize(n_obs, lbda, loop, action_list[i:i + len(loop)], model)
            if render:
                renderSummarizedSXP(env, important_obs, important_actions)
            #  Last loop configuration render
            if render:
                renderSummarizedSXP(env, [loop[-1]], [action_list[i + len(loop) - 1]])

    else:
        #  Number of states to highlight
        n_obs = (len(obs_list) - 2) // summary
        important_obs, important_actions, idx_list = summarize(n_obs, lbda, obs_list, action_list, model)
        if render:
            renderSummarizedSXP(env, important_obs, important_actions)
            if obs_list[-1] in important_obs:
                last_state_render = False

    #  Last configuration render
    if render and last_state_render:
        print('Last configuration')
        renderSummarizedSXP(env, [obs_list[-1]], [action_list[-1]])

    return


#  Compute SXps from a specific states and verify how good they are with quality scores and store them in a CSV file
#  Input: environment (MyFrozenLake), current state (int), agent (Agent), number of steps to look forward (int),
#  wind agents (windAgent list), number of randomly-generated scenarios for calculating quality scores (int),
#  csv file (String), render (boolean), way of computing SXp quality score (boolean), compression ratio (int) and
#  balance between impact of importance and gap scores (float)
#  Output: (no return)
def SXpMetric(env, obs, model, k, wind_models, number_scenarios=5, csv_filename="", render=False, mm_value=False,
              summary=0, lbda=1.0):
    #  ------------------------ HE-scenario ----------------------------------------
    print("------------- HE-scenario -------------")
    f_value_HE, step_limit_HE, proba_HE = E_scenario(env, obs, k, model, wind_models[0], mm_value, render=render,
                                                     summary=summary, lbda=lbda)
    #  ------------------------ P-scenario ----------------------------------------
    print("------------- P-scenario -------------")
    f_value_P, proba_P = P_scenario(env, obs, k, model, render=render, summary=summary, lbda=lbda)
    #  ------------------------ FE-scenario ----------------------------------------
    print("------------- FE-scenario -------------")
    f_value_FE, step_limit_FE, proba_FE = E_scenario(env, obs, k, model, wind_models[1], mm_value, render=render,
                                                     summary=summary, lbda=lbda)
    #  ------------------------ Metrics F/H/PXp ----------------------------------------
    if csv_filename and csv_filename.split(os.sep)[-1][:2] == "rp":
        # Compute representativeness scores
        rp_P = proba_P / proba_P
        rp_FE = min(1.0, max(0.0, round(proba_FE / proba_P, 5)))
        rp_HE = min(1.0, max(0.0, round(proba_HE / proba_P, 5)))
        #  ------------------------ Store in CSV ----------------------------------------
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([obs, rp_P, rp_FE, rp_HE])
    else:
        # Compute representativeness scores
        print("Probas: P {}, FE {}, HE {}".format(proba_P, proba_FE, proba_HE))
        print('Representativeness scores:')
        print('P-scenario: {}'.format(proba_P / proba_P))
        print('FE-scenario: {}'.format(min(1.0, max(0.0, round(proba_FE / proba_P, 5)))))
        print('HE-scenario: {}'.format(min(1.0, max(0.0, round(proba_HE / proba_P, 5)))))
        # Compute quality scores
        HE_score, P_score, FE_score = computeMetric([f_value_HE, f_value_P, f_value_FE], env, obs, k, number_scenarios, model, mm_value, step_limit_HE, step_limit_FE)
        print("For FE-scenario, percentage of better scenarios over {} scenarios : {}".format(number_scenarios, FE_score))
        print("For HE-scenario, percentage of worse scenarios over {} scenarios : {}".format(number_scenarios, HE_score))
        print("Cumulative reward difference between P scenario and the mean reward of  {} scenarios : {}".format(number_scenarios, P_score))

        #  ------------------------ Store in CSV ----------------------------------------
        if csv_filename:
            with open(csv_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([P_score, HE_score, FE_score])

    return


#  Provide different SXps from a starting state depending on user's choice
#  Input: environment (MyFrozenLake), starting state (int), agent (Agent), number of step to look forward (int),
#  wind agents (windAgent list), number of randomly-generated scenarios for calculating metric scores (int)
#  render (boolean), compression ratio (int) and balance between impact of importance and gap scores (float)
#  Output: None
def SXp(env, obs, model, k, wind_models, number_scenarios=5, render=True, summary=0, lbda=1.0):
    answer = False
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
                #  Choose a mm_value version of HE-scenario
                question_boosted = "Do you want a mm_value version of HE-scenario?"
                boosted_case = input(question_boosted)
                mm_f_value_HE = boosted_case in good_answers

                # Compute HE-scenario
                print("------------- HE-scenario -------------")
                f_value_HE, step_limit_HE, proba_HE = E_scenario(env, obs, k, model, wind_models[0], mm_f_value_HE, render=render
                                                       , summary=summary, lbda=lbda)

            #  ------------------------ P-scenario ----------------------------------------
            question_P = "Do you want a P of the agent's move? "
            explanation_P = input(question_P)

            # Provide P-scenario
            if explanation_P in good_answers:
                print("------------- P-scenario -------------")
                f_value_P, proba_P = P_scenario(env, obs, k, model, render=render, summary=summary, lbda=lbda)

            #  ------------------------ FE-scenario ----------------------------------------
            question_FE = "Do you want a FE-scenario of the agent's move? "
            explanation_FE = input(question_FE)

            # Provide FE-scenario
            if explanation_FE in good_answers:

                #  Choose a mm_value version of FE-scenario
                question_boosted = "Do you want a mm_value version of FE-scenario?"
                boosted_case = input(question_boosted)
                mm_f_value_FE = boosted_case in good_answers

                # Compute FE-scenario
                print("------------- FE-scenario -------------")
                f_value_FE, step_limit_FE, proba_FE = E_scenario(env, obs, k, model, wind_models[1], mm_f_value_FE, render=render
                                                       , summary=summary, lbda=lbda)

            #  ------------------------ Metrics ----------------------------------------
            if explanation_HE in good_answers and explanation_P in good_answers and explanation_FE in good_answers:
                question_metric = "Do you want a metric score for these explanations ?"
                answer_metric = input(question_metric)

                if answer_metric in good_answers:
                    #  Compute representativeness scores
                    confidence_P = proba_P / proba_P
                    confidence_FE = min(1.0, max(0.0, round(proba_FE / proba_P, 5)))
                    confidence_HE = min(1.0, max(0.0, round(proba_HE / proba_P, 5)))
                    print("Probas: P {}, FE {}, HE {}".format(proba_P, proba_FE, proba_HE))

                    print('Representativeness scores:')
                    print('P-scenario: {}'.format(confidence_P))
                    print('FE-scenario: {}'.format(confidence_FE))
                    print('HE-scenario: {}'.format(confidence_HE))

                    # Compute quality scores
                    HE_score, P_score, FE_score = computeMetric([f_value_HE, f_value_P, f_value_FE], env, obs, k, number_scenarios, model, mm_f_value_FE, step_limit_HE, step_limit_FE)
                    print("For HE-scenario, percentage of worse scenarios over {} scenarios : {}".format(number_scenarios, HE_score))
                    print("Cumulative reward difference between P scenario and the mean reward of  {} scenarios : {}".format(number_scenarios, P_score))
                    print("For FE-scenario, percentage of better scenarios over {} scenarios : {}".format(number_scenarios, FE_score))

            print("Go back to the current state of the problem!")
            env.render()

        else:
            pass

        answer = True

    return
