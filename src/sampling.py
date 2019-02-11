import numpy as np
from math import exp

from value_iter import value_iter
from rlsp import compute_d_last_step


def sample_from_posterior(
        env, s_current, p_0, h, temp, n_samples, step_size, r_prior, gamma=1,
        print_level=1):
    """
    Algorithm similar to BIRL that uses the last-step OM of a Boltzmann rational
    policy instead of the BIRL likelihood. Samples the reward from the posterior
    p(r | s_T, r_spec) \propto  p(s_T | \theta) * p(r | r_spec).

    This is Algorithm 1 in Appendix C of the paper.
    """

    def log_last_step_om(policy):
        d_last_step = compute_d_last_step(env, policy, p_0, h)
        return np.log(d_last_step[s_current])

    def log_probability(r_vec, verbose=False):
        pi = value_iter(env, gamma, env.f_matrix @ r_vec, h, temp)
        log_p = log_last_step_om(pi)

        log_prior = 0
        if r_prior is not None:
            log_prior = np.sum(r_prior.logpdf(r_vec))

        if verbose:
            print('Log prior: {}\nLog prob:  {}\nTotal:     {}'.format(
                log_prior, log_p, log_p + log_prior))
        return log_p + log_prior

    times_accepted = 0
    samples = []

    if r_prior is None:
        r = .01*np.random.randn(env.num_features)
    else:
        r = 0.1 * r_prior.rvs()

    if print_level >= 1:
        print('Initial reward: {}'.format(r))

    # probability of the initial reward
    log_p = log_probability(r, verbose=(print_level >= 1))

    while len(samples) < n_samples:
        verbose = (print_level >= 1) and (len(samples) % 200 == 199)
        if verbose:
            print('\nGenerating sample {}'.format(len(samples) + 1))

        r_prime = np.random.normal(r, step_size)
        log_p_1 = log_probability(r_prime, verbose=verbose)

        # Accept or reject the new sample
        # If we reject, the new sample is the previous sample
        acceptance_probability = exp(log_p_1-log_p)
        if np.random.uniform() < acceptance_probability:
            times_accepted += 1
            r, log_p = r_prime, log_p_1
        samples.append(r)


        if verbose:
            # Acceptance probability should not be very high or very low
            print('Acceptance probability is {}'.format(acceptance_probability))

    if print_level >= 1:
        print('Done! Accepted {} of samples'.format(times_accepted/n_samples))
    return samples
