# Script to generate the plots in the paper. This script will create a "results" folder, and write the experiment
# outputs into it. Hereafter the plots would be generated in the "results" folder using "src/plotting.py" script.
# Running this script would take several (3-6) hours.

###############
# Section 5.4 #
###############

# Robustness to the choice of Alice's planning horizon T.
mkdir -p results/horizon

# room env
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 1 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 2 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 3 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 5 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 10 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 20 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 30 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 50 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 100 -o results/horizon -x 20 -d true_reward,final_reward

# train env
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 1 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 2 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 3 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 5 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 10 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 20 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 30 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 50 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 100 -o results/horizon -x 20 -d true_reward,final_reward

# apples env
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 1 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 2 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 3 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 5 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 10 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 20 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 30 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 50 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e apples -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 100 -o results/horizon -x 20 -d true_reward,final_reward

# batteries env
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 1 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 2 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 3 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 5 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 10 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 20 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 30 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 50 -o results/horizon -x 20 -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -u True -m 1000  -T 100 -o results/horizon -x 20 -d true_reward,final_reward


##############
# Appendix D #
##############

# Option -c additive stands for the Additive method, and -c bayesian for the Bayesian method
# The -k parameter controls the standard deviation (set to 0.5 by default)
mkdir -p results/additive-vs-bayesian

# room env additive
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.05 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 10 -o results/additive-vs-bayesian -d true_reward,final_reward

# train env additive
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.05 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 10 -o results/additive-vs-bayesian -d true_reward,final_reward

# batteries env additive
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.05 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c additive -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 10 -o results/additive-vs-bayesian -d true_reward,final_reward

# room env bayesian
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.05 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 0.5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e room -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 10 -k 10 -o results/additive-vs-bayesian -d true_reward,final_reward

# train env bayesian
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.05 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 0.5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e train -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 8 -k 10 -o results/additive-vs-bayesian -d true_reward,final_reward

# batteries env bayesian
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.05 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 0.5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 1 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 2 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 3 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 5 -o results/additive-vs-bayesian -d true_reward,final_reward
python src/run.py -e batteries -p default -c bayesian -i rlsp -f True -s 0 -l 0.001 -m 1000  -T 11 -k 10 -o results/additive-vs-bayesian -d true_reward,final_reward


######################
# Generate the plots #
######################
python src/plotting.py
