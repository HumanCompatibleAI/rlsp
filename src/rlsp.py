import numpy as np
from scipy.optimize import check_grad

from value_iter import value_iter
from utils import norm_distr, laplace_distr, printoptions


def compute_g(mdp, policy, p_0, T, d_last_step_list, expected_features_list):
    nS, nA, nF = mdp.nS, mdp.nA, mdp.num_features

    # base case
    G = np.zeros((nS, nF))
    # recursive case
    for t in range(T-1):
        # G(s') = \sum_{s, a} p(a | s) p(s' | s, a) [ p(s) g(s, a) + G_prev[s] ]
        # p(s) is given by d_last_step_list[t]
        # g(s, a) = f(s) - F(s) + \sum_{s'} p(s' | s, a) F(s')
        # Distribute the addition to get three different terms:
        # First term:  p(s) [f(s') - F(s')]
        # Second term: p(s) \sum_{s2} p(s2 | s, a) F(s2)
        # Third term:  G_prev[s]
        g_first = mdp.f_matrix - expected_features_list[t]
        g_second = mdp.T_matrix.dot(expected_features_list[t+1])
        g_second = g_second.reshape((nS, nA, nF))
        g_total = np.expand_dims(g_first, axis=1) + g_second

        prob_s_a = np.expand_dims(d_last_step_list[t].reshape(nS), axis=1) * policy[t]

        G_value = np.expand_dims(prob_s_a, axis=2) * g_total
        G_value = mdp.T_matrix_transpose.dot(G_value.reshape((nS * nA, nF)))

        G_recurse = np.expand_dims(policy[t], axis=-1) * np.expand_dims(G, axis=1)
        G_recurse = mdp.T_matrix_transpose.dot(G_recurse.reshape((nS * nA, nF)))

        G = G_value + G_recurse

    return G


def compute_d_last_step(mdp, policy, p_0, T, gamma=1, verbose=False, return_all=False):
    """Computes the last-step occupancy measure"""
    D, d_last_step_list = p_0, [p_0]
    for t in range(T-1):
        # D(s') = \sum_{s, a} D_prev(s) * p(a | s) * p(s' | s, a)
        state_action_prob = np.expand_dims(D, axis=1) * policy[t]
        D = mdp.T_matrix_transpose.dot(state_action_prob.flatten())

        if verbose is True: print(D)
        if return_all: d_last_step_list.append(D)

    return (D, d_last_step_list) if return_all else D

def compute_feature_expectations(mdp, policy, p_0, T):
    nS, nA, nF = mdp.nS, mdp.nA, mdp.num_features
    expected_features = mdp.f_matrix
    expected_feature_list = [expected_features]
    for t in range(T-2, -1, -1):
        # F(s) = f(s) + \sum_{a, s'} p(a | s) * p(s' | s, a) * F(s')
        future_features = mdp.T_matrix.dot(expected_features).reshape((nS, nA, nF))
        future_features = future_features * np.expand_dims(policy[t], axis=2)
        expected_features = mdp.f_matrix + np.sum(future_features, axis=1)
        expected_feature_list.append(expected_features)
    return expected_features, expected_feature_list[::-1]


def rlsp(mdp, s_current, p_0, horizon, temp=1, epochs=1, learning_rate=0.2,
         r_prior=None, r_vec=None, threshold=1e-3, check_grad_flag=False):
    """The RLSP algorithm"""
    def compute_grad(r_vec):
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s)
        policy = value_iter(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp)
        d_last_step, d_last_step_list = compute_d_last_step(
            mdp, policy, p_0, horizon, return_all=True)
        if d_last_step[s_current] == 0:
            print('Error in om_method: No feasible trajectories!')
            return r_vec

        expected_features, expected_features_list = compute_feature_expectations(
            mdp, policy, p_0, horizon)

        G = compute_g(mdp, policy, p_0, horizon, d_last_step_list, expected_features_list)
        # Compute the gradient
        dL_dr_vec = G[s_current] / d_last_step[s_current]
        # Gradient of the prior
        if r_prior!= None: dL_dr_vec += r_prior.logdistr_grad(r_vec)
        return dL_dr_vec

    def compute_log_likelihood(r_vec):
        policy = value_iter(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp)
        d_last_step = compute_d_last_step(mdp, policy, p_0, horizon)
        log_likelihood = np.log(d_last_step[s_current])
        if r_prior!= None: log_likelihood += np.sum(r_prior.logpdf(r_vec))
        return log_likelihood

    def get_grad(_):
        """dummy function for use with check_grad()"""
        return dL_dr_vec

    if r_vec is None:
        r_vec = 0.01*np.random.randn(mdp.f_matrix.shape[1])
    print('Initial reward vector: {}'.format(r_vec))

    if check_grad_flag: grad_error_list=[]

    for i in range(epochs):
        dL_dr_vec = compute_grad(r_vec)
        if check_grad_flag:
            grad_error_list.append(check_grad(compute_log_likelihood, get_grad, r_vec))

        # Gradient ascent
        r_vec = r_vec + learning_rate * dL_dr_vec

        # with printoptions(precision=4, suppress=True):
        #     print('Epoch {}; Reward vector: {}'.format(i, r_vec))
        #     if check_grad_flag: print('grad error: {}'.format(grad_error_list[-1]))

        if np.linalg.norm(dL_dr_vec) < threshold:
            if check_grad_flag:
                print()
                print('Max grad error: {}'.format(np.amax(np.asarray(grad_error_list))))
                print('Median grad error: {}'.format(np.median(np.asarray(grad_error_list))))
            break

    return r_vec
