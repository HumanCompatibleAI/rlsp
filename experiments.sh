# Commands for the experiments in the paper. These will write to stdout, and are meant to be run individually.
# Most experiments should run in seconds, though some can take minutes (especially with the sampling algorithm).

###############
# Section 5.2 #
###############

# Comparison to baselines (Table 1 and Figure 2)

# Room: Specified reward, deviation, reachability, RLSP
python src/run.py -e room -p default -c additive -i spec -d true_reward,final_reward -T 7 -x 20
python src/run.py -e room -p default -c additive -i deviation -d true_reward,final_reward -T 7 -x 20 -w 0.5
python src/run.py -e room -p default -c additive -i reachability -d true_reward,final_reward -T 7 -x 20
python src/run.py -e room -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 7 -x 20

# Train:
python src/run.py -e train -p default -c additive -i spec -d true_reward,final_reward -T 8 -x 20
python src/run.py -e train -p default -c additive -i deviation -d true_reward,final_reward -T 8 -x 20 -w 0.5
python src/run.py -e train -p default -c additive -i reachability -d true_reward,final_reward -T 8 -x 20
python src/run.py -e train -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 8 -x 20

# Apples:
python src/run.py -e apples -p default -c additive -i spec -d true_reward,final_reward -T 11 -x 20
python src/run.py -e apples -p default -c additive -i deviation -d true_reward,final_reward -T 11 -x 20 -w 0.5
python src/run.py -e apples -p default -c additive -i reachability -d true_reward,final_reward -T 11 -x 20
python src/run.py -e apples -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20

# Batteries, easy:
python src/run.py -e batteries -p easy -c additive -i spec -d true_reward,final_reward -T 11 -x 20
python src/run.py -e batteries -p easy -c additive -i deviation -d true_reward,final_reward -T 11 -x 20 -w 0.5
python src/run.py -e batteries -p easy -c additive -i reachability -d true_reward,final_reward -T 11 -x 20
python src/run.py -e batteries -p easy -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20

# Batteries, hard:
python src/run.py -e batteries -p default -c additive -i spec -d true_reward,final_reward -T 11 -x 20
python src/run.py -e batteries -p default -c additive -i deviation -d true_reward,final_reward -T 11 -x 20 -w 0.5
python src/run.py -e batteries -p default -c additive -i reachability -d true_reward,final_reward -T 11 -x 20
python src/run.py -e batteries -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20

# Far away vase:
python src/run.py -e room -p bad -c additive -i spec -d true_reward,final_reward -T 5 -x 20
python src/run.py -e room -p bad -c additive -i deviation -d true_reward,final_reward -T 5 -x 20 -w 0.5
python src/run.py -e room -p bad -c additive -i reachability -d true_reward,final_reward -T 5 -x 20
python src/run.py -e room -p bad -c additive -i rlsp -d true_reward,final_reward -s 0 -T 5 -x 20

###############
# Section 5.3 #
###############

# Comparison between knowing the s_{-T} vs. using a uniform distribution over s_{-T}
# The commands are the same in the knowing the s_{-T} case; for the uniform distribution we simply add -u True

python src/run.py -e room -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 7 -x 20
python src/run.py -e room -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 7 -x 20 -u True
python src/run.py -e train -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 8 -x 20
python src/run.py -e train -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 8 -x 20 -u True
python src/run.py -e apples -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20
python src/run.py -e apples -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20 -u True
python src/run.py -e batteries -p easy -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20
python src/run.py -e batteries -p easy -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20 -u True
python src/run.py -e batteries -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20
python src/run.py -e batteries -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 11 -x 20 -u True
python src/run.py -e room -p bad -c additive -i rlsp -d true_reward,final_reward -s 0 -T 5 -x 20
python src/run.py -e room -p bad -c additive -i rlsp -d true_reward,final_reward -s 0 -T 5 -x 20 -u True

###############
# Section 5.4 #
###############

# Robustness to the choice of Alice's planning horizon T.
# Simply take the RLSP commands from before and try different values of T, for example:
python src/run.py -e room -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 20 -x 20
python src/run.py -e apples -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 20 -x 20

# It is also possible to run with multiple values of T and collect the results in an output file, see src/run.py for details.

##############
# Appendix C #
##############

# MCMC sampling
# Simply replace -i rlsp with -i sampling:
python src/run.py -e room -p default -c additive -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -T 7 -x 20
python src/run.py -e train -p default -c additive -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -T 8 -x 20
python src/run.py -e apples -p default -c additive -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -T 11 -x 20
python src/run.py -e batteries -p easy -c additive -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -T 11 -x 20
python src/run.py -e batteries -p default -c additive -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -T 11 -x 20
python src/run.py -e room -p bad -c additive -i sampling -d true_reward,final_reward -s 0,1,2,3,4 -T 5 -x 20

##############
# Appendix D #
##############

# Use -c additive for the Additive method, and -c bayesian for the Bayesian method
# Use the -k parameter to control the standard deviation (set to 0.5 by default)
# Note that since the Apples environment has no specified reward, the -c option has no effect on it.
python src/run.py -e room -p default -c additive -i rlsp -d true_reward,final_reward -s 0 -T 7 -x 20 -k 1
python src/run.py -e room -p default -c bayesian -i rlsp -d true_reward,final_reward -s 0 -T 7 -x 20 -k 1
