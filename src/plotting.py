import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_stats(algorithm, env, spec, comb, param_tuned, path, temp_index=0):
    results_list=[]
    for file in os.listdir(path):
        if algorithm in file and env in file and spec in file and comb in file and "-"+param_tuned in file:
            with open(os.path.join(path, file), 'rt') as f:
                reader = csv.reader(f)
                # the first line is names of returned items, e g [seed, true_r, final_r]
                list_results = list(reader)[1::]
                list_rewards = []
                for res in list_results:
                    s = res[1]
                    s = s.replace(']', '').replace('[', '').replace(' ', '').split(',')
                    list_rewards.append(float(s[temp_index]))
                list_rewards = np.asarray(list_rewards)

                param_val = file.split('-'+param_tuned+'=', 1)[-1]
                param_val = param_val.split('-')[0]

                results_list.append([float(param_val), np.mean(list_rewards), np.std(list_rewards)])
    results_list = np.asarray(results_list)
    # return a list sorted by the value of param_tuned
    return results_list[results_list[:,0].argsort()]


def plot_params_one_subplot(stats_list_per_env, ax, color_list, env_names,
                            y_min, y_max, comb, title=None, current_subplot=0):
    ticks_string=[]
    for i in stats_list_per_env[0][0][:,0]:
        tick = str(i)
        if tick[len(tick)-2::]=='.0':
            tick = tick[0:len(tick)-2]
        ticks_string.append(tick)

    for j, stats_list in enumerate(stats_list_per_env):
        stats_stack = np.vstack(stats_list)

        for i in range(len(stats_list)):
            c = color_list[i]
            stats = stats_list[i]

            ax.set_ylim(y_min, y_max)
            ax.scatter(np.log2(stats[:,0]), stats[:,1], color=c, edgecolor=c, s=40, label=comb[i]+env_names[i])
            ax.plot(np.log2(stats[:,0]), stats[:,1], color=c)

            plt.tick_params(axis='both', labelsize=12)
            ax.tick_params(axis='both', labelsize='large')
            plt.xticks(np.log2(stats[::2,0]), ticks_string[0::2])

            if current_subplot==0:
                plt.ylabel("Fraction of max R", fontsize=17)
                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                plt.legend(handles, labels, loc="best", fontsize=12, handletextpad=-0.4)

            # xlabel only for the middle subplot when plotting additive vs bayesian
            if current_subplot==1:
                plt.xlabel("Standard deviation", fontsize=21)

        if title is not None: plt.title(title, fontsize=24)


def plot_params_multiple_subplots(env_lists_per_t, titles_list, y_min=0.45, y_max=1.05):
    fig = plt.figure(figsize=(5*len(env_lists_per_t), 3.4))
    for j, stats_list in enumerate(env_lists_per_t):
        ax = plt.subplot(1, len(env_lists_per_t), j+1)
        plot_params_one_subplot(stats_list, ax,
                                 color_list=['blue', 'orange', '#5177d6', '#ffe500', 'deepskyblue', 'coral'],
                                 env_names=[' room', ' room', ' train', ' train', ' batteries', ' batteries'],
                                 comb=['Bayesian,', 'Additive,','Bayesian,', 'Additive,', 'Bayesian,', 'Additive,'],
                                 title=titles_list[j], current_subplot=j, y_min=y_min, y_max=y_max)
    fig.subplots_adjust(top=1.1)
    plt.tight_layout()

    pp = PdfPages('./results/additive-vs-bayesian.pdf')
    pp.savefig()
    pp.close()


if __name__ == "__main__":
    ###############
    # Appendix D  #
    ###############
    # plot Additive vs Bayesian

    # temperature=0 (rational agent)
    avb_stats_list_per_env_t0 = [[get_stats("rlsp", "room", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=0),
                              get_stats("rlsp", "room", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=0),
                              get_stats("rlsp", "train", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=0),
                              get_stats("rlsp", "train", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=0),
                              get_stats("rlsp", "batteries", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=0),
                              get_stats("rlsp", "batteries", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=0)]]
    # temperature=0.1
    avb_stats_list_per_env_t01 = [[get_stats("rlsp", "room", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=1),
                              get_stats("rlsp", "room", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=1),
                              get_stats("rlsp", "train", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=1),
                              get_stats("rlsp", "train", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=1),
                              get_stats("rlsp", "batteries", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=1),
                              get_stats("rlsp", "batteries", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=1)]]
    # temperature=1
    avb_stats_list_per_env_t1 = [[get_stats("rlsp", "room", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=2),
                              get_stats("rlsp", "room", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=2),
                              get_stats("rlsp", "train", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=2),
                              get_stats("rlsp", "train", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=2),
                              get_stats("rlsp", "batteries", "default", "bayesian", "k",  "./results/additive-vs-bayesian", temp_index=2),
                              get_stats("rlsp", "batteries", "default", "additive", "k",  "./results/additive-vs-bayesian", temp_index=2)]]

    env_lists_per_t = [avb_stats_list_per_env_t0, avb_stats_list_per_env_t01, avb_stats_list_per_env_t1]
    titles_list = ['temperature = 0','temperature = 0.1','temperature = 1']

    plt.rcParams["font.family"] = "Times New Roman"
    plot_params_multiple_subplots(env_lists_per_t, titles_list=titles_list)

    ###############
    # Section 5.4 #
    ###############
    # plot robustness to the choice of Alice's planning horizon

    # temperature=0 (rational agent). This is the stat we're plotting. To plot boltzmann-rational agents, replace
    # "stats_list_per_env_t0" by the stat corresponding to the temperature you want to plot in the cell below.
    h_stats_list_per_env_t0 = [[get_stats("rlsp", "train", "default", "additive", "T",  "./results/horizon"),
                             get_stats("rlsp", "room", "default", "additive", "T",  "./results/horizon"),
                             get_stats("rlsp", "batteries", "default", "additive", "T",  "./results/horizon"),
                             get_stats("rlsp", "apples", "default", "additive", "T",  "./results/horizon")]]
    # temperature=0.1
    h_stats_list_per_env_t01 = [[get_stats("rlsp", "room", "default", "additive", "T",  "./results/horizon", temp_index=1),
                             get_stats("rlsp", "train", "default", "additive", "T",  "./results/horizon", temp_index=1),
                             get_stats("rlsp", "batteries", "default", "additive", "T",  "./results/horizon", temp_index=1),
                             get_stats("rlsp", "apples", "default", "additive", "T",  "./results/horizon", temp_index=1)]]
    # temperature=1
    h_stats_list_per_env_t1 = [[get_stats("rlsp", "room", "default", "additive", "T",  "./results/horizon", temp_index=2),
                             get_stats("rlsp", "train", "default", "additive", "T",  "./results/horizon", temp_index=2),
                             get_stats("rlsp", "batteries", "default", "additive", "T",  "./results/horizon", temp_index=2),
                             get_stats("rlsp", "apples", "default", "additive", "T",  "./results/horizon", temp_index=2)]]

    fig = plt.figure(figsize=(4.0, 2.6))
    ax = plt.subplot(1, 1, 1)
    plot_params_one_subplot(h_stats_list_per_env_t0, ax, y_min=0.45, y_max=1.05,
                                    env_names=['train', 'room', 'batteries', 'apples'],
                                    comb=['','','',''],
                                    color_list=['green', 'orange', '#5177d6', 'firebrick'])
    plt.xlabel("Horizon", fontsize=17)
    ax.legend(bbox_to_anchor=(1, 1.051), fontsize=12,  handletextpad=-0.4, borderpad=0.1)
    plt.tight_layout()

    pp = PdfPages('./results/horizon_t0.pdf')
    pp.savefig()
    pp.close()
