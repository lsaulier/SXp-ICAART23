import sys
import csv
import numpy as np
import argparse
import os

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-csv', '--csv_filename', default="4x4_0.csv", help="CSV filename to compute average of explanation scores",
                        type=str, required=False)
    parser.add_argument('-s', '--nb_states', default=1, help="Number of explanation scores of SXp's from starting states used for an average", type=int,
                        required=False)
    parser.add_argument('-st_list', '--states_list', default=" ", help="Starting states list", type=str, required=False)
    parser.add_argument('-p', '--path', default=" ", help="Path to use old results", type=str, required=False)
    args = parser.parse_args()

    # Get arguments
    MAP = args.map_name
    PATH = args.path
    CSV_FILENAME = args.csv_filename
    STATES_NUMBER = args.nb_states
    STATES_LIST = args.states_list
    #  Store infos
    datas = []
    final_scores = []
    #  Get from the CSV file datas and compute means
    if STATES_NUMBER == 1:

        #  Paths to store new SXP's scores
        if MAP == "4x4":
            if PATH != " ":
                CSV_FILENAME = PATH + os.sep + CSV_FILENAME
            else:
                CSV_FILENAME = "Metrics" + os.sep + "7 reachable states - 4x4" + os.sep + "New tests" + os.sep + CSV_FILENAME
        elif MAP == "8x8":
            if PATH != " ":
                CSV_FILENAME = PATH + os.sep + CSV_FILENAME
            else:
                CSV_FILENAME = "Metrics" + os.sep + "20 random states - 8x8" + os.sep + "New tests" + os.sep + CSV_FILENAME
        else:
            if PATH != " ":
                CSV_FILENAME = PATH + os.sep + CSV_FILENAME
            else:
                CSV_FILENAME = "Metrics" + os.sep + "Other" + os.sep + CSV_FILENAME

        with open(CSV_FILENAME, 'r') as f:
            for line in f:
                #  Extract scores
                if not line.isspace():
                    data = [float(d) for d in line.rstrip().split('\t')[0].split(',')]
                    print(data)
                    datas.append(data)
            #  Compute average of scores
            final_score = np.mean(datas, axis=0)
            print("Final Score: {}".format(final_score))
            final_scores.append(final_score)

        # Write in a new file all average score
        new_csv_filename = CSV_FILENAME[:-4] + "_avg.csv"
        with open(new_csv_filename, 'a') as f:
            writer = csv.writer(f)
            for fs in final_scores:
                writer.writerow(fs)
    #  Get from n CSV files average scores and compute an overall average
    else:
        #  Get states depending on MAP, and paths
        if MAP == "4x4":
            if STATES_LIST != " ":
                states = [int(s) for s in STATES_LIST.strip('[]').split(',')]
            else:
                states = [0, 4, 8, 9, 10, 13, 14]

            if PATH != " ":
                path = PATH + os.sep
            else:
                path = "Metrics" + os.sep + "7 reachable states - 4x4" + os.sep + "New tests" + os.sep
            print("Map 4x4 \n Average scores computed from 7 reachable states: {}".format(states))
        elif MAP == "8x8":
            if STATES_LIST != " ":
                states = [int(s) for s in STATES_LIST.strip('[]').split(',')]
            else:
                states = [1, 7, 8, 11, 12, 14, 21, 22, 23, 25, 26, 31, 34, 36, 39, 44, 53, 55, 57, 62]
            if PATH != " ":
                path = PATH + os.sep
            else:
                path = "Metrics" + os.sep + "20 random states - 8x8" + os.sep + "New tests" + os.sep
            print("Map 8x8 \n Average scores computed from 20 randomly-chosen states: {}".format(states))
        else:
            states = [int(s) for s in STATES_LIST.strip('[]').split(',')]
            if PATH != " ":
                path = PATH + os.sep
            else:
                path = "Metrics" + os.sep + "Other" + os.sep
        Scores = [[], [], []]  # P-scores, HE-scores, FE-scores

        for i in range(STATES_NUMBER):
            filename = path + MAP+"_"+str(states[i])+"_avg.csv"
            #  Extract scores from each CSV files
            with open(filename, 'r') as f:
                for line in f:
                    if not line.isspace():
                        data = [float(d) for d in line.rstrip().split('\t')[0].split(',')]
                        for i in range(3):
                            Scores[i].append(data[i])
        #  Display collected scores
        for score in Scores:
            print("Score: {}".format(score))
        #  Calculate average
        print("Final means : \n PSXP : {} \n HEXp : {} \n FEXp : {}".format(sum(Scores[0])/STATES_NUMBER, sum(Scores[1])/STATES_NUMBER, sum(Scores[2])/STATES_NUMBER))
