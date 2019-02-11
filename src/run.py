import argparse
import csv
import datetime
import numpy as np
import os
import sys

from scipy.stats import uniform as uniform_distr

from envs.apples import ApplesEnv, ApplesState
from envs.apples_spec import APPLES_PROBLEMS
from envs.batteries import BatteriesEnv, BatteriesState
from envs.batteries_spec import BATTERIES_PROBLEMS
from envs.room import RoomEnv, RoomState
from envs.room_spec import ROOM_PROBLEMS
from envs.train import TrainEnv, TrainState
from envs.train_spec import TRAIN_PROBLEMS

from relative_reachability import relative_reachability_penalty
from rlsp import rlsp
from sampling import sample_from_posterior
from utils import norm_distr, laplace_distr, printoptions
from value_iter import value_iter, evaluate_policy


def print_rollout(env, start_state, policies, last_steps_printed, horizon):
    if last_steps_printed == 0:
        last_steps_printed = horizon

    env.reset(start_state)
    print("Executing the policy from state:")
    env.print_state(env.s); print()
    print('Last {} of the {} rolled out steps:'.format(
        last_steps_printed, horizon))

    for i in range(horizon-1):
        s_num = env.get_num_from_state(env.s)
        a = np.random.choice(env.nA, p=policies[i][s_num,:])
        env.step(a)

        if i>=(horizon-last_steps_printed-1):
            env.print_state(env.s); print()


def forward_rl(env, r_planning, r_true, h=40, temp=0, last_steps_printed=0,
               current_s_num=None, weight=1, penalize_deviation=False,
               relative_reachability=False, print_level=1):
    '''Given an env and R, runs soft VI for h steps and rolls out the resulting policy'''
    current_state = env.get_state_from_num(current_s_num)
    r_s = env.f_matrix @ r_planning
    time_dependent_reward = False

    if penalize_deviation:
        diff = env.f_matrix - env.s_to_f(current_state).T
        r_s -= weight * np.linalg.norm(diff, axis=1)
    if relative_reachability:
        time_dependent_reward = True
        r_r = relative_reachability_penalty(env, h, current_s_num)
        r_s = np.expand_dims(r_s, 0) - weight * r_r

    # For evaluation, plan optimally instead of Boltzmann-rationally
    policies = value_iter(env, 1, r_s, h, temperature=temp, time_dependent_reward=time_dependent_reward)

    # For print level >= 1, print a rollout
    if print_level >= 1:
        print_rollout(env, current_state, policies, last_steps_printed, h)

    return evaluate_policy(env, policies, current_s_num, 1, env.f_matrix @ r_true, h)


PROBLEMS = {
    'room': ROOM_PROBLEMS,
    'apples': APPLES_PROBLEMS,
    'train': TRAIN_PROBLEMS,
    'batteries': BATTERIES_PROBLEMS
}

ENV_CLASSES = {
    'room': RoomEnv,
    'apples': ApplesEnv,
    'train': TrainEnv,
    'batteries': BatteriesEnv
}


def get_problem_parameters(env_name, problem_name):
    if env_name not in ENV_CLASSES:
        raise ValueError('Environment {} is not one of {}'.format(
            env_name, list(ENV_CLASSES.keys())))
    if problem_name not in PROBLEMS[env_name]:
        raise ValueError('Problem spec {} is not one of {}'.format(
            problem_name, list(PROBLEMS[env_name].keys())))

    spec, cur_state, r_task, r_true = PROBLEMS[env_name][problem_name]
    env = ENV_CLASSES[env_name](spec)
    return env, env.get_num_from_state(cur_state), r_task, r_true


def get_r_prior(prior, reward_center, std):
    if prior == "gaussian":
        return norm_distr(reward_center, std)
    elif prior == "laplace":
        return laplace_distr(reward_center, std)
    elif prior == "uniform":
        return None
    else:
        raise ValueError('Unknown prior {}'.format(prior))


def experiment_wrapper(env_name='vases',
                       problem_spec='default',
                       inference_algorithm='rlsp',
                       combination_algorithm='additive',
                       prior='gaussian',
                       horizon=20,
                       evaluation_horizon=0,
                       temperature=1,
                       learning_rate=.1,
                       inferred_weight=1,
                       epochs=200,
                       uniform_prior=False,
                       measures=['final_reward'],
                       n_samples=10000,
                       mcmc_burn_in=1000,
                       step_size=.01,
                       seed=0,
                       std=0.5,
                       print_level=1,
                       soft_forward_rl=False,
                       reward_constant=1.0):
    # Check the parameters so that we fail fast
    assert inference_algorithm in ['rlsp', 'sampling', 'deviation', 'reachability', 'spec']
    assert combination_algorithm in ['additive', 'bayesian']
    assert prior in ['gaussian', 'laplace', 'uniform']
    assert all((measure in ['true_reward', 'final_reward'] for measure in measures))

    if evaluation_horizon==0:
        evaluation_horizon = horizon

    if combination_algorithm == 'bayesian':
        assert inference_algorithm in ['rlsp', 'sampling']

    np.random.seed(seed)
    env, s_current, r_task, r_true = get_problem_parameters(env_name, problem_spec)

    if print_level >= 1:
        print('Initial state:')
        env.print_state(env.init_state)
        print()

    p_0 = env.get_initial_state_distribution(known_initial_state=not uniform_prior)

    deviation = inference_algorithm == "deviation"
    reachability = inference_algorithm == "reachability"
    reward_center = r_task if combination_algorithm == "bayesian" else np.zeros(env.num_features)
    r_prior = get_r_prior(prior, reward_center, std)

    # Infer reward by observing the world state
    if inference_algorithm == "rlsp":
        r_inferred = rlsp(env, s_current, p_0, horizon, temperature, epochs, learning_rate, r_prior)
    elif inference_algorithm == "sampling":
        r_samples = sample_from_posterior(
            env, s_current, p_0, horizon, temperature, n_samples, step_size,
            r_prior, gamma=1, print_level=print_level)
        r_inferred = np.mean(r_samples[mcmc_burn_in::], axis=0)
    elif inference_algorithm in ["deviation", "reachability", "spec"]:
        r_inferred = None
    else:
        raise ValueError('Unknown inference algorithm: {}'.format(inference_algorithm))

    if print_level >= 1 and r_inferred is not None:
        with printoptions(precision=4, suppress=True):
            print(); print('Inferred reward vector: ', r_inferred)

    # Run forward RL to evaluate
    def evaluate(forward_rl_temp):
        if combination_algorithm == "additive":
            r_final = r_task
            if r_inferred is not None:
                r_final = r_task + inferred_weight * r_inferred
            true_reward_obtained = forward_rl(env, r_final, r_true, temp=forward_rl_temp, h=evaluation_horizon, current_s_num=s_current, weight=inferred_weight, penalize_deviation=deviation, relative_reachability=reachability, print_level=print_level)
        elif combination_algorithm == "bayesian":
            assert r_inferred is not None
            assert (not deviation) and (not reachability)
            r_final = r_inferred
            true_reward_obtained = forward_rl(env, r_final, r_true, temp=forward_rl_temp, h=evaluation_horizon, current_s_num=s_current, penalize_deviation=False, relative_reachability=False, print_level=print_level)
        else:
            raise ValueError('Unknown combination algorithm: {}'.format(combination_algorithm))

        best_possible_reward = forward_rl(env, r_true, r_true, temp=forward_rl_temp, h=evaluation_horizon, current_s_num=s_current, print_level=0)

        # Add the reward constant in
        true_reward_obtained += reward_constant * evaluation_horizon
        best_possible_reward += reward_constant * evaluation_horizon

        def get_measure(measure):
            if measure == 'final_reward':
                return r_final
            elif measure == 'true_reward':
                return true_reward_obtained * 1.0 / best_possible_reward
            else:
                raise ValueError('Unknown measure {}'.format(measure))

        return [get_measure(measure) for measure in measures]

    if soft_forward_rl:
        return [evaluate(temp) for temp in [0.1, 0.5, 1, 5, 10]]
    else:
        return [evaluate(0.0)]



# The command line parameters that should be included in the filename of the
# file summarizing the results.
PARAMETERS = [
    ('-e', '--env_name', 'room', None,
     'Environment to run: one of [vases, boxes, room, apples, train, batteries]'),
    ('-p', '--problem_spec', 'default', None,
     'The name of the problem specification to solve.'),
    ('-i', '--inference_algorithm', 'spec', None,
     'Frame condition inference algorithm: one of [rlsp, sampling, deviation, reachability, spec].'),
    ('-c', '--combination_algorithm', 'additive', None,
     'How to combine the task reward and inferred reward for forward RL: one of [additive, bayesian]. bayesian only has an effect if algorithm is rlsp or sampling.'),
    ('-r', '--prior', 'gaussian', None,
     'Prior on the inferred reward function: one of [gaussian, laplace, uniform]. Centered at zero if combination_algorithm is additive, and at the task reward if combination_algorithm is bayesian. Only has an effect if inference_algorithm is rlsp or sampling.'),
    ('-T', '--horizon', '20', int,
     'Number of timesteps we assume the human has been acting.'),
    ('-x', '--evaluation_horizon', '0', int,
     'Number of timesteps we act after inferring the reward.'),
    ('-t', '--temperature', '1.0', float,
     'Boltzmann rationality constant for the human. Note this is temperature, which is the inverse of beta.'),
    ('-l', '--learning_rate', '0.1', float,
     'Learning rate for gradient descent. Applies when inference_algorithm is rlsp.'),
    ('-w', '--inferred_weight', '1', float,
     'Weight for the inferred reward when adding task and inferred rewards. Applies if combination_algorithm is additive.'),
    ('-m', '--epochs', '50', int,
     'Number of gradient descent steps to take.'),
    ('-u', '--uniform_prior', 'False', lambda x: x != 'False',
     'Whether to use a uniform prior over initial states, or to know the initial state. Either true or false.'),
    ('-d', '--dependent_vars', 'final_reward', None,
     'Dependent variables to measure and report'),
    ('-n', '--n_samples', '10000', int,
     'Number of samples to generate with MCMC'),
    ('-b', '--mcmc_burn_in', '1000', int,
     'Number of samples to ignore at the start'),
    ('-z', '--step_size', '0.01', float,
     'Step size for computing neighbor reward functions. Only has an effect if inference_algorithm is sampling.'),
    ('-s', '--seed', '0', int,
     'Random seed.'),
    ('-k', '--std', '0.5', float,
     'Standard deviation for the prior'),
    ('-v', '--print_level', '1', int,
     'Level of verbosity.'),
    ('-f', '--soft_forward_rl', 'False', lambda x: x != 'False',
     'Evaluate with a range of temperatures for soft VI for forward RL if true, else evaluate with hard VI for forward RL'),
    ('-q', '--reward_constant', '1.0', float,
     'Living reward provided when evaluating performance.'),
]

# Writing output for experiments
def get_filename(args):
    # Drop the '--' in front of the names
    param_short_names = [name[1:] for name, _, _, _, _ in PARAMETERS]
    param_names = [name[2:] for _, name, _, _, _ in PARAMETERS]
    param_values = [args.__dict__[name] for name in param_names]

    filename = '{}-' + '={}-'.join(param_short_names) + '={}.csv'
    #time_str = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-')
    time_str = 'res'
    filename = filename.format(time_str, *param_values)
    return args.output_folder + '/' + filename

def write_output(results, indep_var, indep_vals, dependent_vars, args):
    with open(get_filename(args), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[indep_var] + dependent_vars)
        writer.writeheader()
        for indep_val, result in zip(indep_vals, results):
            row = {}
            row[indep_var] = indep_val
            for dependent_var, dependent_val in zip(dependent_vars, result):
                row[dependent_var] = dependent_val
            writer.writerow(row)


# Command-line arguments
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    for name, long_name, default, _, help_str in PARAMETERS:
        parser.add_argument(name, long_name, type=str, default=default, help=help_str)

    # Parameters that shouldn't be included in the filename.
    parser.add_argument('-o', '--output_folder', type=str, default='',
                        help='Output folder')
    return parser.parse_args(args)


def setup_experiment(args):
    indep_vars_dict, control_vars_dict = {}, {}

    for _, var, _, fn, _ in PARAMETERS:
        var = var[2:]
        if var == 'dependent_vars': continue
        if fn is None: fn = lambda x: x

        vals = [fn(x) for x in args.__dict__[var].split(',')]
        if len(vals) > 1:
            indep_vars_dict[var] = vals
        else:
            control_vars_dict[var] = vals[0]

    return indep_vars_dict, control_vars_dict, args.dependent_vars.split(',')


def main():
    if sys.platform == "win32":
        import colorama; colorama.init()

    args = parse_args()
    print(args)
    indep_vars_dict, control_vars_dict, dependent_vars = setup_experiment(args)
    # print(indep_vars_dict, control_vars_dict, dependent_vars)
    # For now, restrict to zero or one independent variables, but it
    # could be generalized to two variables
    if len(indep_vars_dict) == 0:
        indep_var = 'N/A'
        indep_vals = ['N/A']
        results = [[] for _ in range(len(dependent_vars))]
        for condition_result in experiment_wrapper(measures=dependent_vars, **control_vars_dict):
            for i, result in enumerate(condition_result):
                results[i].append(result)
        results = [results]
    elif len(indep_vars_dict) == 1:
        indep_var = next(iter(indep_vars_dict.keys()))
        indep_vals = indep_vars_dict[indep_var]
        results = []
        for indep_val in indep_vals:
            curr_results = [[] for _ in range(len(dependent_vars))]
            experiment_args = control_vars_dict.copy()
            experiment_args[indep_var] = indep_val
            experiment_args['measures'] = dependent_vars
            for condition_result in experiment_wrapper(**experiment_args):
                for i, result in enumerate(condition_result):
                    curr_results[i].append(result)
            results.append(curr_results)
    else:
        raise ValueError('Can only support up to one independent variable (that is, a flag with multiple comma-separated values)')

    if args.output_folder == '' or os.path.isfile(get_filename(args)):
        print(results)
    else:
        write_output(results, indep_var, indep_vals, dependent_vars, args)


if __name__ == '__main__':
    main()
