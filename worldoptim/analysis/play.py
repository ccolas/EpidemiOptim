import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from worldoptim.utils import plot_stats, get_repo_path
from notebook_utils import setup_for_replay
from worldoptim.environments.models.utils.pyworld import plot_world_state

NB_EPISODES = 1
FOLDER = get_repo_path() + "/data/results/World2Discrete-v0/NSGAII/"
SAVE = True

def play(folder, nb_eps, seed, save=False):
    """
    Replaying script.

    Parameters
    ----------
    folder: str
        path to result folder.
    nb_eps: int
        Number of episodes to be replayed.
    seed: int
    save: bool
        Whether to save figures.

    """

    algorithm, cost_function, env, params = setup_for_replay(folder, seed)

    goal = np.array([10, -80])
    for i_ep in range(nb_eps):
        res, costs = algorithm.evaluate(n=1, best=False, goal=goal)#np.array([0.4]))#p.array([0.5, 1, 1]))
        # print('----------------')
        # for k in res.keys():
        #     print(k + ': {:.2f}'.format(res[k]))
        stats = env.unwrapped.get_data()

        plot_world_state(time=stats['stats_run']['time'], states=stats['world_stats']['states'], show=False)
        if save:
            plt.savefig(folder + 'plots/worldmodel_states_ep_{}_{}.pdf'.format(i_ep, goal))
            plt.close('all')
        plot_stats(t=stats['stats_run2']['time'],
                   states=stats['stats_run2']['to_plot'],
                   labels=stats['stats_run2']['labels'],
                   legends=stats['stats_run2']['legends'],
                   title=stats['title'],
                   time_jump=stats['time_jump'],
                   )
        if save:
            plt.savefig(folder + 'plots/model_states_ep_{}_{}.pdf'.format(i_ep, goal))
            plt.close('all')
        plot_stats(t=stats['stats_run']['time'],
                   states=stats['stats_run']['to_plot'],
                   labels=stats['stats_run']['labels'],
                   legends=stats['stats_run']['legends'],
                   title=stats['title'],
                   time_jump=stats['time_jump'],
                   show=False if save else i_ep == (nb_eps - 1)
                   )
        if save:
            plt.savefig(folder + 'plots/rl_states_ep_{}_{}.pdf'.format(i_ep, goal))
            plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--folder', type=str, default=FOLDER, help='path_to_model')
    add('--nb_eps', type=int, default=NB_EPISODES, help='the number of training episodes')
    add('--seed', type=int, default=np.random.randint(1e6), help='random seed')
    add('--save', type=bool, default=SAVE, help='save figs')
    kwargs = vars(parser.parse_args())
    # play(**kwargs)
    folder = kwargs['folder']
    files = os.listdir(folder)
    for f in sorted(files):
        if f != 'res':
            print(f)
            plot_folder = folder + f + '/plots/'
            os.makedirs(plot_folder, exist_ok=True)
            play(folder + f + '/', kwargs['nb_eps'], kwargs['seed'], kwargs['save'])
