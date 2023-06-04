# plot graphs for the data in parent directory saved in csv files with first row as column names
# data with 'iteration_reward' in the name will be plotted as a line graph
# data with 'scores' will be plotted as a scatter graph with a line for the average
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_graphs():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data")
    graphs_dir = os.path.join(data_dir, "graphs")
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file))
            if "iteration_reward" in file:
                plt.plot(df["generation"], df["reward"])
                plt.title(file)
                plt.xlabel("Generation")
                plt.ylabel("Reward")
                plt.savefig(os.path.join(graphs_dir, file[:-4] + ".png"))
                plt.clf()
            elif "scores" in file:
                plt.scatter(df["run"], df["score"], s=1)
                # plot average line for all runs
                plt.plot(np.unique(df["run"]), np.poly1d(np.polyfit(df["run"], df["score"], 1))(np.unique(df["run"])), color="red")
                
                plt.title(file)
                plt.xlabel("Run")
                plt.ylabel("score")
                plt.savefig(os.path.join(graphs_dir, file[:-4] + ".png"))
                plt.clf()
            else:
                print(f"File {file} is not a csv file with 'iteration_reward' or 'scores' in its name")
        
if __name__ == '__main__':
    plot_graphs()
