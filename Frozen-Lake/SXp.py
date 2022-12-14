
from copy import deepcopy
import numpy as np
import csv

# Check if the reached state is a terminal one
# Input: environment, state (int)
# Output: a boolean
def terminalState(env, obs):
    i, j = int(obs // env.nCol), int(obs % env.nCol)  # extract coordinates
    return env.desc[i, j].decode('UTF-8') == 'H' or env.desc[i, j].decode('UTF-8') == 'G'

#  Generate n randomly-generated scenarios and compute scores for SXps
#  Input: SXps values (float list), environment (MyFrozenLake), current state (int), scenario's length (int),
#  number of scenarios to generate (int), agent (Agent), the way of computing SXp score (boolean) and
#  timestep from which the HE/FE-scenario's value is extracted (int)
#  Output: metric's scores (float list)
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

#  Compute SXps scores
#  Input: HE/FE-scenario's value (float), values of randomly-generated scenarios for HE/FE-scenario (float list),
#  number of randomly-generated scenarios (int), P-scenario's value (float), compute P-score (boolean) and
#  values of randomly-generated scenarios for P-scenario (float list)
#  Output: HE/FE/P-score (float list)
def metrics(HE_value, FE_value, scenar_h_values, scenar_f_values, number_scenarios, psxp_value=0, psxp=True, scenar_last_values=[]):
    # Used for computing cardinality
    HE_l = [HE_value <= scenar_h_values[i] for i in range(number_scenarios)]
    FE_l = [FE_value >= scenar_f_values[i] for i in range(number_scenarios)]

    if psxp:
        mean_values = sum(scenar_last_values) / len(scenar_last_values)
        mean_values = normalizedValue(mean_values)
        print("P-score metric : Sum scenar rewards {} --- len {} --- nomalize mean {}".format(sum(scenar_last_values),
                                                                                         len(scenar_last_values),
                                                                                         mean_values))
        # Compute SXps scores
        return HE_l.count(1) / number_scenarios, abs(psxp_value - mean_values), FE_l.count(1) / number_scenarios
    else:
        # Compute SXps scores
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
#  agent and render (boolean)
#  Output: normalized f value for P-scenario (float)
def P_scenario(env, obs, k, model, render=False):

    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    # Sequence loop
    for i in range(k):
        #  Save obs to get reward
        last_obs = obs
        #  Action choice
        action, _ = model.predict(obs)
        #  Step
        obs, _, done, _ = env_copy.step(action)
        #  Render
        if render:
            env_copy.render()
        #  Extract value and reward
        value = model.getValue(obs)
        reward = env_copy.getReward(last_obs, action, obs)
        f = value + 1.0000001 * reward
        # Check whether the scenario ends or not
        if done and i != k - 1:
            break
    print("Last-step normalised f value of P-scenario: {}".format(normalizedValue(f)))
    return normalizedValue(f)

#  Normalize a value between 0 and 1
#  Input: value to normalize (float) and min, max reachable value (float)
#  Output: normalized value (float)
def normalizedValue(value, mini=0, maxi=1.0000001):
    return ((value - mini) / (maxi - mini))

#  Compute a HE/FE-scenario, depending on the environment type
#  Input: environment (MyFrozenLake), current state (int), number of steps to look forward (int),
#  agent, wind agent, list of blocking states (int list list) and render (boolean)
#  Output: f value of a state (float) and step number of the state where the value comes from (int)
def E_scenario(env, obs, k, model, wind_model, mm_value, render=False):
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)

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
        #  New state of the agent
        new_row, new_col = env_copy.inc(obs // env_copy.nCol, obs % env_copy.nCol, wind_action)
        obs = env_copy.to_s(new_row, new_col)
        #  Extract value and reward
        value = model.getValue(obs)
        reward = env_copy.getReward(last_obs, action, obs)
        f = value + 1.0000001*reward

        #  Render
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

    if mm_value:
        print("Best f value encounter : {}, at time-step: {}".format(best_f, n))
        return best_f, n
    else:
        print("Last-step f value from state {} is : {}".format(obs, f))
        return f, i+1

#  Compute SXps from a specific states and verify how good they are with scores and store them in a CSV file
#  Input: environment (MyFrozenLake), current state (int), agent (Agent), number of steps to look forward (int),
#  wind agents (windAgent list), number of randomly-generated scenarios for calculating metric scores (int),
#  csv file (String), render (boolean) and way of computing SXp score (boolean)
#  Output: (no return)
def SXpMetric(env, obs, model, k, wind_models, number_scenarios=5, csv_filename="", render=False, mm_value=False):
    #  ------------------------ HE-scenario ----------------------------------------
    print("------------- HE-scenario -------------")
    f_value_HE, step_limit_HE = E_scenario(env, obs, k, model, wind_models[0], mm_value, render=render)
    #  ------------------------ P-scenario ----------------------------------------
    print("------------- P-scenario -------------")
    f_value_P = P_scenario(env, obs, k, model, render=render)
    #  ------------------------ FE-scenario ----------------------------------------
    print("------------- FE-scenario -------------")
    f_value_FE, step_limit_FE = E_scenario(env, obs, k, model, wind_models[1], mm_value, render=render)
    #  ------------------------ Metrics F/H/PXp ----------------------------------------
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
#  and render (boolean)
#  Output: display a SXp(s), depending on the user's choice (no return)
def SXp(env, obs, model, k, wind_models, number_scenarios=5, render=True):
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
                f_value_HE, step_limit_HE = E_scenario(env, obs, k, model, wind_models[0], mm_f_value_HE, render=render)

            #  ------------------------ P-scenario ----------------------------------------
            question_P = "Do you want a P of the agent's move? "
            explanation_P = input(question_P)

            # Provide P-scenario
            if explanation_P in good_answers:
                print("------------- P-scenario -------------")
                f_value_P = P_scenario(env, obs, k, model, render=render)

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
                f_value_FE, step_limit_FE = E_scenario(env, obs, k, model, wind_models[1], mm_f_value_FE, render=render)

            #  ------------------------ Metrics ----------------------------------------
            if explanation_HE in good_answers and explanation_P in good_answers and explanation_FE in good_answers:
                question_metric = "Do you want a metric score for these explanations ?"
                answer_metric = input(question_metric)

                if answer_metric in good_answers:
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
