import sys
import csv
import numpy as np
import argparse
import os

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', '--csv_filename', default="1-11.69.csv", help="CSV filename to compute average of explanation scores",
                        type=str, required=False)
    parser.add_argument('-conf', '--nb_config', default=1, help="Number of explanation scores of SXp's from starting configurations used for an average", type=int,
                        required=False)
    parser.add_argument('-p', '--path', default=" ", help="Path to use old results", type=str, required=False)
    parser.add_argument('-csv_names', '--csv_names', default="-11.69_avg.csv", help="Common part of filenames used for score average over n csv files", type=str,
                        required=False)
    args = parser.parse_args()

    # Get arguments
    PATH = args.path
    CSV_FILENAME = args.csv_filename
    CONFIGS_NUMBER = args.nb_config
    CSV_COMMON = args.csv_names
    #  Store infos
    datas = []
    final_scores = []
    #  Get from the CSV file datas and compute means
    if CONFIGS_NUMBER == 1:

        if PATH != " ":
            CSV_FILENAME = PATH + os.sep + CSV_FILENAME
        else:
            CSV_FILENAME = "Metrics" + os.sep + "New tests" + os.sep + CSV_FILENAME

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

        # Write in a new file all means computed
        new_csv_filename = CSV_FILENAME[:-4] + "_avg.csv"
        with open(new_csv_filename, 'a') as f:
            writer = csv.writer(f)
            for fs in final_scores:
                writer.writerow(fs)
    #  Get from n CSV files average scores and compute an overall average
    else:
        Scores = [[], [], []]  # PSXp_scores, HEXp_scores, FEXp_scores

        if PATH != " ":
            path = PATH + os.sep
        else:
            path = "Metrics" + os.sep + "New tests" + os.sep

        for i in range(CONFIGS_NUMBER):
            #  Extract scores from each CSV files
            filename = path + str(i+1)+ CSV_COMMON
            with open(filename, 'r') as f:
                for line in f:
                    if not line.isspace():
                        data = [float(d) for d in line.rstrip().split('\t')[0].split(',')]
                        for k in range(3):
                            Scores[k].append(data[k])
        #  Display collected scores
        for score in Scores:
            print("Score: {}".format(score))
        #  Calculate average
        print("Final means : \n P-scenario : {} \n HE-scenario : {} \n FE-scenario : {}".format(sum(Scores[0])/CONFIGS_NUMBER, sum(Scores[1])/CONFIGS_NUMBER, sum(Scores[2])/CONFIGS_NUMBER))
